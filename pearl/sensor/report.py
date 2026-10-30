"""
Per-device sensor characterization report: ``sensor_tables.json`` and S1-S3 figures.

Usage
-----
    python -m pearl.sensor.report results/sensor --out results/sensor_report

Tables: S1 call support and update period per source/condition; S2 square-wave box-car window,
lag, pulse capture window, rise time and coverage per source, plus energy-counter vs usage
ratios; S3 sampler-check call durations and stalls. One subfolder per (GPU, driver).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pearl.sensor import stats as st  # noqa: E402

_LOG = logging.getLogger(__name__)

S2_SOURCES = ("usage", "instant", "average", "samples")
BOXCAR_WINDOWS_MS = (1, 2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200,
                     300, 400, 500, 600, 750, 900, 1000, 1100, 1250, 1500, 2000)
BOXCAR_LAGS_MS = tuple(range(0, 51)) + tuple(range(55, 101, 5)) + tuple(range(125, 501, 25))
CAPTURE_SIGMA = 5.0
CAPTURE_MIN_STEP_FRAC = 0.05
QUIET_AFTER_MS = 2000.0
RAW_WINDOW_S = 3.0
RAW_SQUARE_PERIOD_MS = 500
COLORS = {"usage": "C0", "average": "C1", "instant": "C2", "samples": "C3", "energy": "C4"}
MARKERS = {"usage": ("s", 130), "average": ("o", 45), "instant": ("^", 60), "samples": ("D", 50)}


def s2_source_result(run: dict[str, Any], source: str) -> dict[str, Any]:
    """Square-wave box-car fit, pulse capture, rise time, and coverage for one source."""
    square = run["meta"].get("square") or {}
    iv = st.activity_intervals(run["edges"])
    rd = st.readings_in_span(st.source_readings(run["polls"], run["samples"], source), iv,
                             float(square.get("settle_s", 5.0)))
    if len(rd) < 3:
        raise ValueError(f"too few readings for source={source!r}")
    t_upd = float(np.median(np.diff(rd["t_ns"].to_numpy()))) * 1e-6
    fit, _ = st.boxcar_fit(rd, iv, BOXCAR_WINDOWS_MS, BOXCAR_LAGS_MS)
    idle = st.idle_level(rd, iv, quiet_after_ms=QUIET_AFTER_MS)
    threshold = idle.median_mw + max(CAPTURE_SIGMA * idle.sigma_mw, CAPTURE_MIN_STEP_FRAC * fit.delta_p_mw)
    periods = sorted(int(p) for p in square.get("periods_ms", [500]))
    max_d = min(4.0 * max(fit.window_ms, t_upd), periods[-1] / 2.0)
    w_capture = st.capture_window_ms(st.pulse_capture(rd, iv, threshold, np.linspace(0.0, max_d, 41)))
    need_on = max(3.0 * t_upd, 1.5 * fit.window_ms)
    min_on = next((p / 2.0 for p in periods if p / 2.0 >= need_on), None)
    rt = st.rise_times_ms(rd, iv, min_on_ms=min_on) if min_on is not None else np.array([])
    row = {
        "run_dir": run["run_dir"], "poll_ms": run["meta"].get("poll_interval_ms"), "source": source,
        "n_readings": fit.n_readings, "update_ms": t_upd,
        "boxcar_window_ms": fit.window_ms, "boxcar_lag_ms": fit.lag_ms, "boxcar_r2": fit.r2,
        "window_at_grid_edge": fit.window_ms in (BOXCAR_WINDOWS_MS[0], BOXCAR_WINDOWS_MS[-1]),
        "lag_at_grid_edge": fit.lag_ms == BOXCAR_LAGS_MS[-1],
        "capture_window_ms": w_capture,
        "rise_t90_ms": float(np.median(rt)) if rt.size else None,
        "rise_t90_p10_ms": float(np.percentile(rt, 10)) if rt.size else None,
        "rise_t90_p90_ms": float(np.percentile(rt, 90)) if rt.size else None,
        "rise_n_pulses": int(rt.size) if rt.size else None, "rise_min_on_ms": min_on,
        "coverage": st.coverage_fraction(fit.window_ms, t_upd),
        "p_idle_W": fit.p_idle_mw / 1e3, "delta_p_W": fit.delta_p_mw / 1e3,
    }
    return {"readings": rd, "intervals": iv, "fit": fit, "row": row}


def _s1_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support, update = [], []
    for run in runs:
        key = {"run_dir": run["run_dir"], "condition": run["meta"].get("condition"),
               "poll_interval_ms": run["meta"].get("poll_interval_ms")}
        support += [{**key, **s} for s in st.support_matrix(run["polls"])]
        for source in (*S2_SOURCES, "energy"):
            ps = st.update_period(run, source)
            if ps is not None:
                update.append({**key, "source": source, **ps.__dict__})
    return support, update


def _s2(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """S2 rows for every square run, per-source results of the fastest-poll square run, counter rows."""
    square = sorted((r for r in runs if r["meta"].get("condition") == "square"),
                    key=lambda r: float(r["meta"].get("poll_interval_ms", 1e9)))
    rows, primary, energy = [], {}, []
    for i, run in enumerate(square):
        for source in S2_SOURCES:
            try:
                res = s2_source_result(run, source)
            except ValueError as e:
                _LOG.warning("S2 skipped %s %s: %s", run["run_dir"], source, e)
                continue
            rows.append(res["row"])
            if i == 0:
                primary[source] = res
        ec = st.energy_counter_vs_usage(run["polls"])
        if ec is not None:
            energy.append({"run_dir": run["run_dir"], **ec})
    return rows, primary, energy


def _raw_window(run: dict[str, Any]) -> tuple[float, float]:
    t0 = float(np.median(run["polls"]["t_before_ns"].to_numpy(np.float64)))
    if run["meta"].get("condition") == "square":
        iv = st.activity_intervals(run["edges"])
        blk = iv.loc[iv["period_ms"] == RAW_SQUARE_PERIOD_MS]
        if not blk.empty:
            t0 = float(blk["rise_ns"].iloc[len(blk) // 2]) - 0.25e9
    return t0, t0 + RAW_WINDOW_S * 1e9


def plot_s1_raw(runs: list[dict[str, Any]], title: str, out: Path) -> Path | None:
    """3 s windows of every source for idle@10ms, load@10ms and the square run."""
    def pick(cond: str, ms: float | None) -> dict[str, Any] | None:
        m = [r for r in runs if r["meta"].get("condition") == cond
             and (ms is None or float(r["meta"]["poll_interval_ms"]) == ms)]
        return m[0] if m else None

    picks = [r for r in (pick("idle", 10), pick("load", 10), pick("square", None)) if r is not None]
    if not picks:
        return None
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.9 * len(picks)), squeeze=False)
    for ax, run in zip(axes[:, 0], picks):
        t0, t1 = _raw_window(run)
        polls = run["polls"]
        win = polls.loc[(polls["t_before_ns"] >= t0) & (polls["t_after_ns"] <= t1)]
        if run["meta"].get("condition") == "square":
            iv = st.activity_intervals(run["edges"])
            for _, c in iv.loc[(iv["fall_ns"] > t0) & (iv["rise_ns"] < t1)].iterrows():
                ax.axvspan((max(c["rise_ns"], t0) - t0) * 1e-9, (min(c["fall_ns"], t1) - t0) * 1e-9, color="0.9", lw=0)
        for source in ("usage", "average", "instant"):
            sub = win.loc[(win["source"] == source) & (win["nvml_return"] == st.NVML_SUCCESS)]
            mid = (sub["t_before_ns"] + sub["t_after_ns"]) / 2
            ax.step((mid - t0) * 1e-9, sub["value"] / 1e3, where="post", lw=1.2, color=COLORS[source], label=source)
        rd = st.source_readings(polls, run["samples"], "samples")
        rd = rd.loc[(rd["t_ns"] >= t0) & (rd["t_ns"] <= t1)]
        ax.plot((rd["t_ns"] - t0) * 1e-9, rd["value_mw"] / 1e3, ".-", ms=3, lw=0.5, color=COLORS["samples"], label="samples")
        ed = st.energy_counter_power(win)
        if not ed.empty:
            ax.hlines(ed["power_w"], (ed["t_start_ns"] - t0) * 1e-9, (ed["t_end_ns"] - t0) * 1e-9,
                      color=COLORS["energy"], lw=1.5, label="energy dE/dt")
        ax.set_title(f"{run['meta'].get('condition')} @ {run['meta'].get('poll_interval_ms')} ms poll", fontsize=9)
        ax.set_ylabel("power (W)")
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=5)
    axes[-1, 0].set_xlabel("time in window (s)")
    fig.suptitle(f"S1 raw readings - {title}", fontsize=10)
    return _save(fig, out / "s1_raw_traces.png")


def plot_s2_source(res: dict[str, Any], source: str, out: Path) -> Path:
    """Readings of one source with the fitted box-car model overlaid."""
    rd, fit = res["readings"], res["fit"]
    t = rd["t_ns"].to_numpy(np.float64)
    duty = st.duty_in_window(res["intervals"], t, fit.window_ms * 1e6, fit.lag_ms * 1e6)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot((t - t[0]) * 1e-9, rd["value_mw"] / 1e3, ".-", ms=2, lw=0.8, label="readings")
    ax.plot((t - t[0]) * 1e-9, (fit.p_idle_mw + fit.delta_p_mw * duty) / 1e3, color="tab:red", lw=1.2,
            label=f"boxcar W={fit.window_ms:g} ms L={fit.lag_ms:g} ms R2={fit.r2:.2f}")
    ax.set(xlabel="time (s)", ylabel="power (W)", title=f"{source} (update {res['row']['update_ms']:.1f} ms)")
    ax.legend(fontsize=7)
    return _save(fig, out / f"s2_{source}.png")


def plot_s2_coverage(rows: list[dict[str, Any]], out: Path) -> Path:
    """Averaging window (filled) and pulse capture window (open) vs update period, per source."""
    lim = np.array([5.0, 5000.0])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(lim, lim, "k--", lw=0.8, label="coverage = 1")
    for r in rows:
        c, (marker, size) = COLORS[r["source"]], MARKERS[r["source"]]
        ax.scatter(r["update_ms"], r["boxcar_window_ms"], color=c, marker=marker, s=size, zorder=3,
                   label=f"{r['source']}: update {r['update_ms']:.0f} ms, W {r['boxcar_window_ms']:g} ms, "
                         f"coverage {r['coverage']:.2f}")
        if r["capture_window_ms"] is not None:
            ax.scatter(r["update_ms"], r["capture_window_ms"], facecolors="none", edgecolors=c,
                       marker=marker, s=size, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="update period (ms)", ylabel="averaging window (ms)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return _save(fig, out / "s2_coverage.png")


def plot_s3_sampler(rows: list[dict[str, Any]], out: Path) -> Path | None:
    """Median NVML call duration and stall count per sampler-check source."""
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = [str(r["source"]) for r in rows]
    axes[0].bar(labels, [(r["call_duration_ms"] or {}).get("p50_ms") or 0.0 for r in rows], color=["C0", "C1"][: len(rows)])
    axes[0].set(ylabel="median call duration (ms)", title="NVML call duration")
    axes[1].bar(labels, [(r["stalls"] or {}).get("count", 0) for r in rows], color=["C0", "C1"][: len(rows)])
    axes[1].set(ylabel="stall count", title="Poll stalls (> 3x interval)")
    return _save(fig, out / "s3_sampler_timing.png")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def analyze_device(runs: list[dict[str, Any]], sampler: list[dict[str, Any]], out: Path) -> Path:
    """Write ``sensor_tables.json`` and figures for one (GPU, driver) group; return the tables path."""
    out.mkdir(parents=True, exist_ok=True)
    meta = runs[0]["meta"]
    support, update = _s1_rows(runs)
    s2_rows, primary, energy = _s2(runs)
    figures = [plot_s1_raw(runs, f"{meta.get('gpu_name')} driver {meta.get('nvidia_driver')}", out),
               plot_s3_sampler(sampler, out)]
    if primary:
        figures.append(plot_s2_coverage([res["row"] for res in primary.values()], out))
        figures += [plot_s2_source(res, src, out) for src, res in primary.items()]
    tables = {
        "schema_version": 1, "gpu_name": meta.get("gpu_name"), "nvidia_driver": meta.get("nvidia_driver"),
        "s1_support": support, "s1_update_period": update, "s2_square_response": s2_rows,
        "s2_energy_counter_coverage": energy, "s3_sampler_check": sampler,
        "figures": [p.name for p in figures if p is not None],
    }
    path = out / "sensor_tables.json"
    path.write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    return path


def run_report(results_dir: Path, out_dir: Path) -> list[Path]:
    """Discover probe and sampler-check runs under ``results_dir`` and report each (GPU, driver)."""
    groups: dict[tuple[str, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for meta_path in sorted(results_dir.glob("**/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rel = str(meta_path.parent.relative_to(results_dir))
        key = (str(meta.get("gpu_name")), str(meta.get("nvidia_driver")))
        runs, sampler = groups.setdefault(key, ([], []))
        if meta.get("kind") == "sampler_check":
            sampler.append({"run_dir": rel, "source": meta.get("source"), "poll_interval_ms": meta.get("poll_interval_ms"),
                            "call_duration_ms": meta.get("call_duration_ms"), "stalls": meta.get("stalls"),
                            "sensor_readings": meta.get("sensor_readings")})
        elif (meta_path.parent / "polls.parquet").is_file():
            runs.append({**st.load_probe_run(meta_path.parent), "run_dir": rel})
    if not any(runs for runs, _ in groups.values()):
        raise FileNotFoundError(f"no probe runs found under {results_dir}")
    paths = []
    for (gpu, driver), (runs, sampler) in sorted(groups.items()):
        if runs:
            slug = f"{gpu}_{driver}".replace(" ", "_").replace("/", "_").replace(".", "_")
            _LOG.info("Analyzing %s driver %s (%d probe runs)", gpu, driver, len(runs))
            paths.append(analyze_device(runs, sampler, out_dir / slug))
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("results_dir", type=Path, help="Folder containing campaign output(s) of pearl.sensor.probe.")
    p.add_argument("--out", type=Path, required=True, help="Report output directory.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for path in run_report(args.results_dir, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
