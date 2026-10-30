"""Compact Parquet trace store: one folder per grid point with segments, power, firings and layers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pearl.collect.sampler import PowerSample

COMBO_FILES = ("segments", "power_trace", "firings", "layers")
IDLE_FILE = "idle_power.parquet"

POWER_SCHEMA = pa.schema([
    pa.field("segment_id", pa.int16()),
    pa.field("time_ns_delta", pa.int64()),
    pa.field("power_mw", pa.uint32()),
    pa.field("nvidia_timestamp_us", pa.int64(), nullable=True),
    pa.field("sm_clock_mhz", pa.uint16(), nullable=True),
    pa.field("mem_clock_mhz", pa.uint16(), nullable=True),
    pa.field("gpu_util_pct", pa.uint8(), nullable=True),
    pa.field("mem_util_pct", pa.uint8(), nullable=True),
])


@dataclass(frozen=True)
class Segment:
    """One measured window: N consecutive forwards with power and hook timings."""

    segment_id: int
    power_samples: Sequence[PowerSample]
    layer_timings: list[dict[str, Any]]
    run_config: dict[str, Any]
    sampler_meta: dict[str, Any]
    wall_time_s: float
    forward_count: int
    trace_margin_ms: float
    workload_start_ns: int
    workload_end_ns: int


@dataclass
class LoadedSegment:
    """Decoded segment: power inside the workload span and firings with absolute ``start_ns``."""

    segment_id: int
    meta: dict[str, Any]
    run_config: dict[str, Any]
    time_ns: np.ndarray
    power_w: np.ndarray
    firings: pd.DataFrame = field(repr=False)


def _deltas(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)
    return np.concatenate([v[:1], np.diff(v)]) if v.size else v


def _bounded(v: float | None, bits: int) -> int | None:
    if v is None:
        return None
    iv = int(round(v))
    return iv if 0 <= iv < (1 << bits) else None


def _power_table(segment_id: int, samples: Sequence[PowerSample]) -> pa.Table:
    t_delta = _deltas(np.array([s.time_ns for s in samples], dtype=np.int64))
    return pa.table({
        "segment_id": [segment_id] * len(samples),
        "time_ns_delta": t_delta.tolist(),
        "power_mw": [int(round(s.power_w * 1000.0)) for s in samples],
        "nvidia_timestamp_us": [s.nvidia_timestamp_us for s in samples],
        "sm_clock_mhz": [_bounded(s.sm_clock_mhz, 16) for s in samples],
        "mem_clock_mhz": [_bounded(s.mem_clock_mhz, 16) for s in samples],
        "gpu_util_pct": [_bounded(s.gpu_util_pct, 8) for s in samples],
        "mem_util_pct": [_bounded(s.mem_util_pct, 8) for s in samples],
    }, schema=POWER_SCHEMA)


def _firings_frame(segment_id: int, timings: list[dict[str, Any]], name_to_idx: dict[str, int]) -> pd.DataFrame:
    rows = [
        (int(block.get("forward_idx", f)), int(lyr.get("step_id") or 0), name_to_idx[lyr["layer_name"]],
         int(lyr["start_ns"]), int(lyr["end_ns"]) - int(lyr["start_ns"]))
        for f, block in enumerate(timings)
        for lyr in block.get("layers") or []
        if lyr["layer_name"] in name_to_idx
    ]
    df = pd.DataFrame(rows, columns=["forward_idx", "step_id", "layer_idx", "start_ns", "duration_ns"])
    df = df.sort_values(["forward_idx", "start_ns", "layer_idx"], kind="stable").reset_index(drop=True)
    df.insert(0, "segment_id", segment_id)
    df["firing_ord"] = np.arange(len(df), dtype=np.int64)
    df["start_ns_delta"] = _deltas(df.pop("start_ns").to_numpy())
    return df.astype({"forward_idx": "int64", "step_id": "int64", "layer_idx": "int64", "duration_ns": "int64"})


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _segment_row(seg: Segment) -> dict[str, Any]:
    m = seg.sampler_meta
    stalls, sensor = m.get("stalls") or {}, m.get("sensor_gaps") or {}
    return {
        "segment_id": seg.segment_id,
        "measured_sm_mean_mhz": _mean([s.sm_clock_mhz for s in seg.power_samples]),
        "measured_mem_mean_mhz": _mean([s.mem_clock_mhz for s in seg.power_samples]),
        "workload_start_ns": seg.workload_start_ns,
        "workload_end_ns": seg.workload_end_ns,
        "trace_margin_ms": seg.trace_margin_ms,
        "forward_count": seg.forward_count,
        "wall_time_s": seg.wall_time_s,
        "sampler_power_source": m.get("power_source"),
        "interval_ms": m.get("interval_ms"),
        "timestamp_midpoint": m.get("timestamp_midpoint"),
        "stall_threshold_ms": stalls.get("threshold_ms"),
        "n_polls": stalls.get("n_polls"),
        "n_stalls": stalls.get("n"),
        "stall_total_ms": stalls.get("total_ms"),
        "stall_max_ms": stalls.get("max_ms"),
        "sensor_median_spacing_ms": sensor.get("median_spacing_ms"),
        "n_sensor_gaps": sensor.get("n"),
        "sensor_gap_total_ms": sensor.get("total_ms"),
        "sensor_gap_max_ms": sensor.get("max_ms"),
        "sampler_meta_json": json.dumps(m, separators=(",", ":")),
        "run_config_json": json.dumps(seg.run_config, separators=(",", ":")),
    }


def _atomic_write(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def write_idle_trace(traces_root: Path, samples: Sequence[PowerSample]) -> None:
    """Write the campaign idle baseline to ``<traces_root>/idle_power.parquet`` (``segment_id = -1``)."""
    _atomic_write(Path(traces_root) / IDLE_FILE, _power_table(-1, samples))


def write_combo(combo_dir: Path, layers: Sequence[tuple[int, str, str]], segments: Sequence[Segment]) -> None:
    """
    Write ``layers``, ``segments``, ``power_trace`` and ``firings`` Parquet files under ``combo_dir``.

    ``layers`` rows are ``(layer_idx, layer_name, layer_type)``; firings of modules absent from
    ``layers`` are dropped. Each file is written to ``*.tmp`` then renamed.
    """
    combo_dir = Path(combo_dir)
    name_to_idx = {name: idx for idx, name, _ in layers}
    layers_df = pd.DataFrame(layers, columns=["layer_idx", "layer_name", "layer_type"]).assign(is_leaf=True)
    firings = pd.concat([_firings_frame(s.segment_id, s.layer_timings, name_to_idx) for s in segments],
                        ignore_index=True)
    _atomic_write(combo_dir / "layers.parquet", pa.Table.from_pandas(layers_df, preserve_index=False))
    _atomic_write(combo_dir / "segments.parquet", pa.Table.from_pylist([_segment_row(s) for s in segments]))
    _atomic_write(combo_dir / "power_trace.parquet",
                  pa.concat_tables([_power_table(s.segment_id, s.power_samples) for s in segments]))
    _atomic_write(combo_dir / "firings.parquet", pa.Table.from_pandas(firings, preserve_index=False))


def load_idle_trace(traces_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(time_ns, power_w)`` arrays of the idle baseline."""
    pt = pd.read_parquet(Path(traces_root) / IDLE_FILE)
    return np.cumsum(pt["time_ns_delta"].to_numpy(np.int64)), pt["power_mw"].to_numpy(np.float64) / 1000.0


def load_combo(combo_dir: Path) -> tuple[pd.DataFrame, list[LoadedSegment]]:
    """
    Read a combo folder written by :func:`write_combo`.

    Returns the ``layers`` table and, per segment, power samples inside
    ``[workload_start_ns, workload_end_ns]`` and firings sorted by ``firing_ord`` with
    decoded ``start_ns``/``end_ns`` columns.
    """
    combo_dir = Path(combo_dir)
    layers = pd.read_parquet(combo_dir / "layers.parquet")
    segments = pd.read_parquet(combo_dir / "segments.parquet")
    power = pd.read_parquet(combo_dir / "power_trace.parquet")
    firings = pd.read_parquet(combo_dir / "firings.parquet")
    out: list[LoadedSegment] = []
    for meta in segments.to_dict("records"):
        seg_id = int(meta["segment_id"])
        pt = power[power["segment_id"] == seg_id]
        t = np.cumsum(pt["time_ns_delta"].to_numpy(np.int64))
        p = pt["power_mw"].to_numpy(np.float64) / 1000.0
        w0, w1 = meta.get("workload_start_ns"), meta.get("workload_end_ns")
        if w0 is not None and w1 is not None and not pd.isna(w0) and not pd.isna(w1):
            keep = (t >= int(w0)) & (t <= int(w1))
            t, p = t[keep], p[keep]
        fir = firings[firings["segment_id"] == seg_id].sort_values("firing_ord", kind="stable")
        fir = fir.assign(start_ns=np.cumsum(fir["start_ns_delta"].to_numpy(np.int64))).reset_index(drop=True)
        fir["end_ns"] = fir["start_ns"] + fir["duration_ns"].astype(np.int64)
        out.append(LoadedSegment(seg_id, meta, json.loads(meta["run_config_json"]), t, p, fir))
    return layers, out
