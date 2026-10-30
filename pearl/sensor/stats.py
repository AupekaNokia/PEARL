"""Sensor statistics: update period, box-car window/lag fit, pulse capture, rise time, energy counter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NVML_SUCCESS = 0


@dataclass(frozen=True)
class UpdatePeriodStats:
    """Distribution of intervals between consecutive reading changes (milliseconds)."""

    median_ms: float
    p10_ms: float
    p90_ms: float
    n_intervals: int
    n_changes: int


@dataclass(frozen=True)
class IdleLevel:
    """Reading level during quiet off-phases: median and ``1.4826 * MAD`` (mW)."""

    median_mw: float
    sigma_mw: float
    n: int


@dataclass(frozen=True)
class BoxcarFit:
    """Best model ``value(t) = p_idle + delta_p * duty[t - lag - window, t - lag]``."""

    window_ms: float
    lag_ms: float
    p_idle_mw: float
    delta_p_mw: float
    r2: float
    n_readings: int


def load_probe_run(run_dir: Path | str) -> dict[str, Any]:
    """Load ``meta.json`` and the ``polls`` / ``samples`` / ``edges`` Parquet files of one probe run."""
    path = Path(run_dir)
    if not (path / "meta.json").is_file() or not (path / "polls.parquet").is_file():
        raise FileNotFoundError(f"missing meta.json or polls.parquet in {path}")

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(path / name) if (path / name).is_file() else pd.DataFrame()

    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return {"run_dir": str(path), "meta": meta, "polls": read("polls.parquet"),
            "samples": read("samples.parquet"), "edges": read("edges.parquet")}


def support_matrix(polls: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-source share of successful NVML calls and the dominant error code."""
    out: list[dict[str, Any]] = []
    for source, grp in polls.groupby("source", sort=True):
        failures = grp.loc[grp["nvml_return"] != NVML_SUCCESS, "nvml_return"]
        dom = None if failures.empty else int(failures.mode().iloc[0])
        out.append({
            "source": str(source), "success_share": float((grp["nvml_return"] == NVML_SUCCESS).mean()),
            "n_polls": len(grp), "dominant_error": dom,
            "dominant_error_share": 0.0 if dom is None else float((failures == dom).mean()),
        })
    return out


def _change_points(polls: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Call-midpoint times (ns) and values where ``source`` changes between successful polls."""
    sub = polls.loc[polls["source"] == source].sort_values("poll_idx", kind="stable")
    sub = sub.loc[sub["nvml_return"].eq(NVML_SUCCESS) & sub["value"].notna()]
    v = sub["value"].astype(np.float64).to_numpy()
    t = (sub["t_before_ns"].astype(np.float64).to_numpy() + sub["t_after_ns"].astype(np.float64).to_numpy()) / 2.0
    if v.size == 0:
        return t, v
    keep = np.concatenate([[True], v[1:] != v[:-1]])
    return t[keep], v[keep]


def _period_stats(dts_ms: np.ndarray, n_changes: int) -> UpdatePeriodStats | None:
    """Quantiles of intervals with the first and last interval dropped (partial at run edges)."""
    if dts_ms.size < 3:
        return None
    dts_ms = dts_ms[1:-1]
    return UpdatePeriodStats(float(np.median(dts_ms)), float(np.percentile(dts_ms, 10)),
                             float(np.percentile(dts_ms, 90)), int(dts_ms.size), n_changes)


def update_period(run: dict[str, Any], source: str) -> UpdatePeriodStats | None:
    """Update period of ``source``: sensor timestamp spacing for ``samples``, value changes otherwise."""
    samples = run["samples"]
    if source == "samples" and not samples.empty:
        ts = np.unique(samples["sensor_ts_us"].astype(np.int64))
        return _period_stats(np.diff(ts.astype(np.float64)) * 1e-3, int(ts.size))
    t, _ = _change_points(run["polls"], source)
    t = t.astype(np.int64)
    return _period_stats(np.diff(t.astype(np.float64)) * 1e-6, int(t.size))


def activity_intervals(edges: pd.DataFrame) -> pd.DataFrame:
    """On-phases from ``edges.parquet``: ``block_idx, period_ms, cycle_idx, rise_ns, fall_ns`` sorted by rise."""
    cols = ["block_idx", "period_ms", "cycle_idx", "rise_ns", "fall_ns"]
    if edges.empty:
        return pd.DataFrame(columns=cols)
    wide = edges.pivot_table(index=cols[:3], columns="edge", values="t_ns", aggfunc="first").reset_index()
    if not {"rise", "fall"}.issubset(wide.columns) or wide[["rise", "fall"]].isna().any().any():
        raise ValueError("every cycle needs one rise and one fall edge")
    out = wide.rename(columns={"rise": "rise_ns", "fall": "fall_ns"})[cols]
    out = out.astype({"rise_ns": np.int64, "fall_ns": np.int64}).sort_values("rise_ns", kind="stable")
    r, f = out["rise_ns"].to_numpy(), out["fall_ns"].to_numpy()
    if np.any(f < r) or np.any(r[1:] < f[:-1]):
        raise ValueError("activity intervals overlap or are reversed")
    return out.reset_index(drop=True)


def _on_time_before_ns(intervals: pd.DataFrame, t_ns: np.ndarray) -> np.ndarray:
    t = np.asarray(t_ns, dtype=np.float64)
    if intervals.empty:
        return np.zeros_like(t)
    rise = intervals["rise_ns"].to_numpy(np.float64)
    dur = intervals["fall_ns"].to_numpy(np.float64) - rise
    cum = np.concatenate([[0.0], np.cumsum(dur)])
    k = np.searchsorted(rise, t, side="right")
    last = np.clip(k - 1, 0, None)
    partial = np.where(k > 0, np.clip(t - rise[last], 0.0, dur[last]), 0.0)
    return np.where(k > 0, cum[last] + partial, 0.0)


def duty_in_window(intervals: pd.DataFrame, t_end_ns: np.ndarray, window_ns: float, lag_ns: float = 0.0) -> np.ndarray:
    """Fraction of ``[t_end - lag - window, t_end - lag]`` covered by on-phases."""
    if window_ns <= 0:
        raise ValueError("window_ns must be > 0")
    t1 = np.asarray(t_end_ns, dtype=np.float64) - lag_ns
    return (_on_time_before_ns(intervals, t1) - _on_time_before_ns(intervals, t1 - window_ns)) / window_ns


def realtime_minus_monotonic_ns(polls: pd.DataFrame) -> float:
    """Median ``CLOCK_REALTIME - CLOCK_MONOTONIC`` (ns) from the back-to-back clock reads of every call."""
    t = np.concatenate([polls["t_before_ns"].to_numpy(np.int64), polls["t_after_ns"].to_numpy(np.int64)])
    d = np.concatenate([
        polls["t_realtime_before_ns"].to_numpy(np.int64) - polls["t_before_ns"].to_numpy(np.int64),
        polls["t_realtime_after_ns"].to_numpy(np.int64) - polls["t_after_ns"].to_numpy(np.int64),
    ])
    if d.size == 0:
        raise ValueError("polls contain no clock pairs")
    d = d[np.argsort(t, kind="stable")]
    return float(int(d[0]) + np.median((d - int(d[0])).astype(np.float64)))


def source_readings(polls: pd.DataFrame, samples: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Distinct readings of ``source`` on the host monotonic clock (columns ``t_ns``, ``value_mw``).

    ``samples`` uses sensor timestamps converted from realtime; polled sources use the call
    midpoint of each value change (lagging publication by up to one poll interval).
    """
    if source == "samples":
        if samples.empty:
            return pd.DataFrame(columns=["t_ns", "value_mw"])
        s = samples.drop_duplicates("sensor_ts_us").sort_values("sensor_ts_us")
        t = s["sensor_ts_us"].astype(np.float64).to_numpy() * 1e3 - realtime_minus_monotonic_ns(polls)
        return pd.DataFrame({"t_ns": t, "value_mw": s["value_mw"].astype(np.float64).to_numpy()})
    t, v = _change_points(polls, source)
    return pd.DataFrame({"t_ns": t, "value_mw": v})


def readings_in_span(readings: pd.DataFrame, intervals: pd.DataFrame, pad_s: float) -> pd.DataFrame:
    """Readings within ``pad_s`` of the first rise and last fall."""
    if intervals.empty:
        return readings.iloc[0:0]
    t0 = float(intervals["rise_ns"].iloc[0]) - pad_s * 1e9
    t1 = float(intervals["fall_ns"].iloc[-1]) + pad_s * 1e9
    return readings.loc[(readings["t_ns"] >= t0) & (readings["t_ns"] <= t1)].reset_index(drop=True)


def _since_fall_ns(intervals: pd.DataFrame, t_ns: np.ndarray) -> np.ndarray:
    """Time since the last fall (NaN inside an on-phase or before the first rise)."""
    rise = intervals["rise_ns"].to_numpy(np.float64)
    fall = intervals["fall_ns"].to_numpy(np.float64)
    t = np.asarray(t_ns, dtype=np.float64)
    k = np.searchsorted(rise, t, side="right") - 1
    kk = np.clip(k, 0, None)
    return np.where((k >= 0) & (t >= fall[kk]), t - fall[kk], np.nan)


def idle_level(readings: pd.DataFrame, intervals: pd.DataFrame, *, quiet_after_ms: float = 2000.0) -> IdleLevel:
    """Reading level at least ``quiet_after_ms`` after a fall, or before the first rise."""
    t = readings["t_ns"].to_numpy(np.float64)
    v = readings["value_mw"].to_numpy(np.float64)
    m = (t < float(intervals["rise_ns"].iloc[0])) | (_since_fall_ns(intervals, t) >= quiet_after_ms * 1e6)
    if not np.any(m):
        raise ValueError("no readings in quiet off-phases")
    med = float(np.median(v[m]))
    return IdleLevel(med, 1.4826 * float(np.median(np.abs(v[m] - med))), int(m.sum()))


def boxcar_fit(
    readings: pd.DataFrame, intervals: pd.DataFrame, windows_ms: Sequence[float], lags_ms: Sequence[float]
) -> tuple[BoxcarFit, pd.DataFrame]:
    """Grid-search the averaging window and lag (least-squares level per point); return best fit and grid."""
    t = readings["t_ns"].to_numpy(np.float64)
    y = readings["value_mw"].to_numpy(np.float64)
    if y.size < 3:
        raise ValueError("need at least 3 readings")
    sst = float(np.sum((y - y.mean()) ** 2))
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, float, float] | None = None
    for w in windows_ms:
        for lag in lags_ms:
            x = duty_in_window(intervals, t, w * 1e6, lag * 1e6)
            xm = x.mean()
            sxx = float(np.sum((x - xm) ** 2))
            if sxx == 0.0:
                continue
            b = float(np.sum((x - xm) * (y - y.mean())) / sxx)
            a = float(y.mean() - b * xm)
            sse = float(np.sum((y - a - b * x) ** 2))
            rows.append({"window_ms": float(w), "lag_ms": float(lag), "sse": sse,
                         "r2": 1.0 - sse / sst if sst > 0 else float("nan")})
            if best is None or sse < best[0]:
                best = (sse, float(w), float(lag), a, b)
    if best is None:
        raise ValueError("activity duty is constant for every grid point")
    sse, w, lag, a, b = best
    return BoxcarFit(w, lag, a, b, 1.0 - sse / sst if sst > 0 else float("nan"), int(y.size)), pd.DataFrame(rows)


def pulse_capture(
    readings: pd.DataFrame, intervals: pd.DataFrame, threshold_mw: float, bin_edges_ms: Sequence[float]
) -> pd.DataFrame:
    """Share of off-phase readings above ``threshold_mw`` per bin of time since the preceding fall."""
    v = readings["value_mw"].to_numpy(np.float64)
    d_ms = _since_fall_ns(intervals, readings["t_ns"].to_numpy(np.float64)) * 1e-6
    edges = np.asarray(bin_edges_ms, dtype=np.float64)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_ms >= lo) & (d_ms < hi)
        n = int(m.sum())
        rows.append({"d_lo_ms": float(lo), "d_hi_ms": float(hi), "n": n,
                     "share_elevated": float(np.mean(v[m] > threshold_mw)) if n else float("nan")})
    return pd.DataFrame(rows)


def capture_window_ms(capture: pd.DataFrame, *, level: float = 0.5) -> float | None:
    """First time since fall (interpolated bin centre) where the elevated share drops below ``level``."""
    c = capture.loc[capture["n"] > 0]
    if c.empty:
        return None
    x = ((c["d_lo_ms"] + c["d_hi_ms"]) / 2.0).to_numpy()
    s = c["share_elevated"].to_numpy()
    below = np.nonzero(s < level)[0]
    if s[0] < level or below.size == 0:
        return None
    i = int(below[0])
    return float(x[i - 1] + (s[i - 1] - level) / (s[i - 1] - s[i]) * (x[i] - x[i - 1]))


def rise_times_ms(
    readings: pd.DataFrame, intervals: pd.DataFrame, *, min_on_ms: float, level: float = 0.9, plateau_frac: float = 0.3
) -> np.ndarray:
    """
    Per on-phase of at least ``min_on_ms``: first reading at ``level`` of the step minus the rise time.

    Baseline / plateau are medians over the last ``plateau_frac`` of the preceding off-phase / the on-phase.
    """
    t = readings["t_ns"].to_numpy(np.float64)
    v = readings["value_mw"].to_numpy(np.float64)
    rise = intervals["rise_ns"].to_numpy(np.float64)
    fall = intervals["fall_ns"].to_numpy(np.float64)
    out: list[float] = []
    for i in range(1, rise.size):
        on, off = fall[i] - rise[i], rise[i] - fall[i - 1]
        base_m = (t >= rise[i] - plateau_frac * off) & (t < rise[i])
        plat_m = (t >= fall[i] - plateau_frac * on) & (t <= fall[i])
        if on < min_on_ms * 1e6 or not base_m.any() or not plat_m.any():
            continue
        base, plat = float(np.median(v[base_m])), float(np.median(v[plat_m]))
        if plat <= base:
            continue
        hit = np.nonzero((t >= rise[i]) & (t <= fall[i]) & (v >= base + level * (plat - base)))[0]
        if hit.size:
            out.append((t[hit[0]] - rise[i]) * 1e-6)
    return np.asarray(out, dtype=np.float64)


def coverage_fraction(window_ms: float, update_period_ms: float) -> float:
    """Share of wall time inside some reading's averaging window (``> 1``: windows overlap)."""
    if update_period_ms <= 0:
        raise ValueError("update_period_ms must be > 0")
    return window_ms / update_period_ms


def energy_counter_power(polls: pd.DataFrame) -> pd.DataFrame:
    """Power (W) from the cumulative ``energy`` counter between consecutive counter changes."""
    t, mj = _change_points(polls, "energy")
    t, mj = t[1:], mj[1:]
    if t.size < 2:
        return pd.DataFrame(columns=["t_start_ns", "t_end_ns", "power_w"])
    return pd.DataFrame({"t_start_ns": t[:-1], "t_end_ns": t[1:], "power_w": np.diff(mj) * 1e6 / np.diff(t)})


def energy_counter_vs_usage(polls: pd.DataFrame) -> dict[str, float | int] | None:
    """Ratio of counter-derived power to the mean ``usage`` reading over each counter interval."""
    derived = energy_counter_power(polls)
    usage = polls.loc[(polls["source"] == "usage") & polls["nvml_return"].eq(NVML_SUCCESS)]
    if derived.empty or usage.empty:
        return None
    mids = ((usage["t_before_ns"].astype(np.float64) + usage["t_after_ns"].astype(np.float64)) / 2.0).to_numpy()
    vals = usage["value"].astype(np.float64).to_numpy()
    usage_w = np.array([
        np.mean(vals[m]) / 1000.0 if (m := (mids >= t0) & (mids <= t1)).any() else np.nan
        for t0, t1 in zip(derived["t_start_ns"], derived["t_end_ns"])
    ])
    counter_w = derived["power_w"].astype(np.float64).to_numpy()
    ok = ~np.isnan(usage_w) & (usage_w > 0) & ~np.isnan(counter_w)
    if not ok.any():
        return None
    ratio = pd.Series(counter_w[ok] / usage_w[ok])
    return {
        "n_intervals": int(ok.sum()),
        "median_ratio_counter_over_usage": float(ratio.median()),
        "median_rel_err_vs_usage": float((1.0 - ratio).median()),
        "p10_ratio_counter_over_usage": float(ratio.quantile(0.1)),
        "p90_ratio_counter_over_usage": float(ratio.quantile(0.9)),
    }
