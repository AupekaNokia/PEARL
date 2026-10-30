"""Sensor statistics and report on synthetic signals (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pearl.sensor import stats as st
from pearl.sensor.load import parse_square_schedule
from pearl.sensor.report import run_report
from pearl.sensor.sampler_check import call_duration_quantiles_ms, poll_stall_stats, sensor_skipped_repeated

MS = 1_000_000
IDLE_MW, STEP_MW = 50_000.0, 100_000.0


def _edges(period_ms: int = 200, n_cycles: int = 40, t0_ns: int = 3_000 * MS) -> pd.DataFrame:
    rows = []
    for c in range(n_cycles):
        rise = t0_ns + c * period_ms * MS
        rows += [(0, period_ms, c, "rise", rise), (0, period_ms, c, "fall", rise + period_ms * MS // 2)]
    return pd.DataFrame(rows, columns=["block_idx", "period_ms", "cycle_idx", "edge", "t_ns"])


def _boxcar_readings(iv: pd.DataFrame, t_ns: np.ndarray, window_ms: float, lag_ms: float) -> np.ndarray:
    return IDLE_MW + STEP_MW * st.duty_in_window(iv, t_ns, window_ms * MS, lag_ms * MS)


def _polls(source_values: dict[str, np.ndarray], t_ns: np.ndarray) -> pd.DataFrame:
    rows = []
    for source, values in source_values.items():
        for i, (t, v) in enumerate(zip(t_ns, values)):
            rows.append({"poll_idx": i, "source": source, "t_before_ns": int(t), "t_after_ns": int(t),
                         "t_realtime_ns": int(t), "t_realtime_before_ns": int(t), "t_realtime_after_ns": int(t),
                         "nvml_return": 0, "value": float(v), "unit": "mW"})
    return pd.DataFrame(rows)


def test_update_period_drops_edge_intervals() -> None:
    t = np.arange(0, 2_000, 5) * MS
    values = np.floor(np.arange(t.size) / 20)  # changes every 20 polls = 100 ms
    ps = st.update_period({"polls": _polls({"usage": values}, t), "samples": pd.DataFrame()}, "usage")
    assert ps is not None
    assert ps.median_ms == pytest.approx(100.0)
    assert ps.n_changes == 20 and ps.n_intervals == 17


def test_boxcar_fit_recovers_window_and_lag() -> None:
    iv = st.activity_intervals(_edges())
    t = np.arange(3_000, 11_000, 20, dtype=np.float64) * MS
    rd = pd.DataFrame({"t_ns": t, "value_mw": _boxcar_readings(iv, t, 40.0, 10.0)})
    fit, grid = st.boxcar_fit(rd, iv, windows_ms=(10, 20, 40, 80), lags_ms=(0, 5, 10, 20))
    assert (fit.window_ms, fit.lag_ms) == (40.0, 10.0)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.p_idle_mw == pytest.approx(IDLE_MW) and fit.delta_p_mw == pytest.approx(STEP_MW)
    assert len(grid) == 16


def test_capture_window_interpolates_half_level() -> None:
    cap = pd.DataFrame({"d_lo_ms": [0.0, 10.0, 20.0], "d_hi_ms": [10.0, 20.0, 30.0],
                        "n": [5, 5, 5], "share_elevated": [1.0, 0.6, 0.2]})
    assert st.capture_window_ms(cap) == pytest.approx(17.5)
    assert st.coverage_fraction(40.0, 20.0) == 2.0


def test_energy_counter_matches_constant_usage() -> None:
    t = np.arange(0, 2_000, 10) * MS
    energy_mj = np.floor(np.arange(t.size) / 10) * 10 * 10 * 0.1 * 1000  # 100 W, counter updates every 100 ms
    ec = st.energy_counter_vs_usage(_polls({"usage": np.full(t.size, 100_000.0), "energy": energy_mj}, t))
    assert ec is not None
    assert ec["median_ratio_counter_over_usage"] == pytest.approx(1.0)


def test_sampler_check_stats() -> None:
    assert call_duration_quantiles_ms([1 * MS, 2 * MS, 3 * MS])["p50_ms"] == 2.0
    stalls = poll_stall_stats([0, 10 * MS, 20 * MS, 60 * MS], interval_ms=10)
    assert stalls["count"] == 1 and stalls["longest_ms"] == 40.0
    readings = sensor_skipped_repeated([0, 20_000, 40_000, 120_000], [1.0, 1.0, 2.0, 3.0])
    assert readings["n_repeated"] == 1 and readings["n_skipped"] == 1


def test_parse_square_schedule_rejects_bad_periods() -> None:
    assert parse_square_schedule("10,100", 5.0, 1.0, 0).periods_ms == (10, 100)
    with pytest.raises(ValueError):
        parse_square_schedule("1", 5.0, 1.0, 0)
    with pytest.raises(ValueError):
        parse_square_schedule("10000", 5.0, 1.0, 0)


def test_campaign_steps_match_protocol() -> None:
    from pearl.sensor.probe import DEFAULT_TIMING, QUICK_TIMING, build_steps

    steps = build_steps(DEFAULT_TIMING)
    assert [s["step_id"] for s in steps] == ["idle_10ms", "idle_2ms", "load_10ms", "load_2ms", "square_5ms",
                                             "sampler_usage", "sampler_samples"]
    assert steps[4]["square"]["periods_ms"] == [10, 25, 50, 100, 200, 500, 1000, 2000, 4000]
    assert build_steps(QUICK_TIMING)[4]["square"]["periods_ms"] == [10, 100]


def _write_run(run_dir: Path, meta: dict, polls: pd.DataFrame, edges: pd.DataFrame | None = None) -> None:
    run_dir.mkdir(parents=True)
    polls.to_parquet(run_dir / "polls.parquet", index=False)
    pd.DataFrame(columns=["poll_idx", "sensor_ts_us", "value_mw", "value_type"]).to_parquet(
        run_dir / "samples.parquet", index=False)
    if edges is not None:
        edges.to_parquet(run_dir / "edges.parquet", index=False)
    (run_dir / "meta.json").write_text(json.dumps(meta))


def test_report_on_synthetic_campaign(tmp_path: Path) -> None:
    dev = {"gpu_name": "Synthetic GPU", "nvidia_driver": "1.0"}
    edges = _edges()
    iv = st.activity_intervals(edges)
    t_pub = np.arange(0, 14_000, 17, dtype=np.float64) * MS  # not phase-locked to the 200 ms square wave
    pub = _boxcar_readings(iv, t_pub, 40.0, 0.0) + np.random.default_rng(0).normal(0.0, 10.0, t_pub.size)
    t_poll = np.arange(0, 14_000, 5, dtype=np.float64) * MS
    usage = pub[np.searchsorted(t_pub, t_poll, side="right") - 1]
    root = tmp_path / "campaign"
    _write_run(root / "probe/square_5ms", {**dev, "condition": "square", "poll_interval_ms": 5,
               "square": {"periods_ms": [200], "settle_s": 3.0}}, _polls({"usage": usage}, t_poll), edges)
    _write_run(root / "probe/idle_10ms", {**dev, "condition": "idle", "poll_interval_ms": 10},
               _polls({"usage": np.full(100, IDLE_MW)}, np.arange(100) * 10 * MS))
    (root / "sampler_check/usage").mkdir(parents=True)
    (root / "sampler_check/usage/meta.json").write_text(json.dumps(
        {**dev, "kind": "sampler_check", "source": "usage", "call_duration_ms": {"p50_ms": 0.1}, "stalls": {"count": 0}}))

    (tables_path,) = run_report(root, tmp_path / "report")
    tables = json.loads(tables_path.read_text())
    (row,) = tables["s2_square_response"]
    assert row["source"] == "usage"
    assert row["boxcar_window_ms"] == pytest.approx(40.0, abs=10.0)
    assert {"s1_raw_traces.png", "s2_coverage.png", "s2_usage.png", "s3_sampler_timing.png"} <= set(tables["figures"])
    assert all((tables_path.parent / f).is_file() for f in tables["figures"])
    assert tables["s3_sampler_check"][0]["run_dir"] == "sampler_check/usage"
