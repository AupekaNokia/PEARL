"""Background-thread NVML poll overhead and stall check while the main thread runs fp16 matmuls."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pynvml

from pearl.sensor.sources import SAMPLE_COLUMNS, device_meta, nvml_err_value, nvml_scalar

MATMUL_N = 512
STALL_FACTOR = 3.0
SENSOR_GAP_FACTOR = 3.0

SamplerSource = Literal["usage", "samples"]


def call_duration_quantiles_ms(durations_ns: Sequence[int]) -> dict[str, float | int | None]:
    """p50 / p99 / max NVML call durations in milliseconds (empty input gives null fields)."""
    if not durations_ns:
        return {"n": 0, "p50_ms": None, "p99_ms": None, "max_ms": None}
    v = sorted(int(x) for x in durations_ns)
    n = len(v)
    p99 = v[min(n - 1, int(0.99 * (n - 1)))] if n > 1 else v[0]
    return {"n": n, "p50_ms": v[n // 2] / 1e6, "p99_ms": p99 / 1e6, "max_ms": v[-1] / 1e6}


def poll_stall_stats(poll_start_ns: Sequence[int], interval_ms: float) -> dict[str, float | int]:
    """Gaps between poll starts longer than ``STALL_FACTOR x interval_ms``."""
    threshold_ns = STALL_FACTOR * float(interval_ms) * 1e6
    gaps = [b - a for a, b in zip(poll_start_ns, poll_start_ns[1:]) if b - a > threshold_ns]
    return {
        "threshold_ms": STALL_FACTOR * float(interval_ms),
        "count": len(gaps),
        "longest_ms": max(gaps) / 1e6 if gaps else 0.0,
        "total_ms": sum(gaps) / 1e6 if gaps else 0.0,
        "n_polls": len(poll_start_ns),
    }


def sensor_skipped_repeated(sensor_ts_us: Sequence[int], value_mw: Sequence[float]) -> dict[str, float | int | None]:
    """Repeated sensor values and timestamp gaps above ``SENSOR_GAP_FACTOR x`` median spacing."""
    ts = np.asarray(sensor_ts_us, dtype=np.int64)
    vals = np.asarray(value_mw, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    uniq_ts, uniq_idx = np.unique(ts[order], return_index=True)
    if uniq_ts.size < 2:
        return {"n_readings": int(uniq_ts.size), "n_repeated": 0, "n_skipped": 0,
                "median_spacing_ms": None, "gap_threshold_ms": None}
    vals = vals[order][uniq_idx]
    spacing_us = np.diff(uniq_ts.astype(np.float64))
    median_us = float(np.median(spacing_us))
    threshold_us = SENSOR_GAP_FACTOR * median_us
    return {
        "n_readings": int(uniq_ts.size),
        "n_repeated": int(np.sum(np.diff(vals) == 0)),
        "n_skipped": int(np.sum(spacing_us > threshold_us)),
        "median_spacing_ms": median_us / 1000.0,
        "gap_threshold_ms": threshold_us / 1000.0,
    }


@dataclass
class SamplerPollRow:
    poll_idx: int
    t_call_start_ns: int
    t_call_end_ns: int
    lateness_ns: int
    nvml_return: int
    value_mw: float | None


@dataclass
class _Buffers:
    polls: list[SamplerPollRow] = field(default_factory=list)
    samples: list[tuple[int, int, float, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _read(handle: object, source: SamplerSource, cursor_us: int) -> tuple[int, int, float | None, list[tuple[int, float, int]]]:
    """One NVML read: ``(nvml_return, cursor_us, value_mw, new samples as (ts_us, mW, type))``."""
    try:
        if source == "usage":
            return pynvml.NVML_SUCCESS, cursor_us, float(pynvml.nvmlDeviceGetPowerUsage(handle)), []
        value_type, raw = pynvml.nvmlDeviceGetSamples(handle, pynvml.NVML_TOTAL_POWER_SAMPLES, cursor_us)
    except pynvml.NVMLError as e:
        if source == "samples" and e.value == pynvml.NVML_ERROR_NOT_FOUND:
            return pynvml.NVML_SUCCESS, cursor_us, None, []
        return nvml_err_value(e), cursor_us, None, []
    vt = int(value_type)
    new = [(int(s.timeStamp), nvml_scalar(s.sampleValue, vt), vt) for s in raw if int(s.timeStamp) > cursor_us]
    if new:
        cursor_us = max(cursor_us, max(ts for ts, _, _ in new))
    return pynvml.NVML_SUCCESS, cursor_us, (new[-1][1] if new else None), new


def _poll_loop(handle: object, source: SamplerSource, interval_ms: int, duration_s: float,
               buf: _Buffers, stop: threading.Event) -> None:
    interval_ns = interval_ms * 1_000_000
    t_start = time.monotonic_ns()
    t_end = t_start + int(duration_s * 1e9)
    poll_idx = 0
    cursor_us = time.time_ns() // 1000 if source == "samples" else 0
    while time.monotonic_ns() < t_end and not stop.is_set():
        deadline = t_start + poll_idx * interval_ns
        now = time.monotonic_ns()
        if now < deadline:
            time.sleep((deadline - now) / 1e9)
        lateness = time.monotonic_ns() - deadline
        t0 = time.monotonic_ns()
        ret, cursor_us, value, new = _read(handle, source, cursor_us)
        t1 = time.monotonic_ns()
        with buf.lock:
            buf.polls.append(SamplerPollRow(poll_idx, t0, t1, lateness, ret, value))
            buf.samples.extend((poll_idx, ts, mw, vt) for ts, mw, vt in new)
        poll_idx += 1


def _matmul_loop(physical_gpu_id: int, duration_s: float, stop: threading.Event) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for sampler check (CUDA_VISIBLE_DEVICES={physical_gpu_id!r})")
    a = torch.randn(MATMUL_N, MATMUL_N, device="cuda:0", dtype=torch.float16)
    b = torch.randn(MATMUL_N, MATMUL_N, device="cuda:0", dtype=torch.float16)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline and not stop.is_set():
        torch.matmul(a, b)
        torch.cuda.synchronize()


def _write(out_dir: Path, buf: _Buffers, meta: dict[str, Any], source: SamplerSource) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in buf.polls]).to_parquet(out_dir / "sampler_polls.parquet", index=False)
    if source == "samples":
        pd.DataFrame(buf.samples, columns=list(SAMPLE_COLUMNS)).to_parquet(
            out_dir / "sampler_samples.parquet", index=False
        )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def run_sampler_check(*, gpu_id: int, source: SamplerSource, interval_ms: int, duration_s: float, out_dir: Path) -> Path:
    """
    Poll ``source`` on a background thread while the main thread runs a 512x512 fp16 matmul loop.

    Writes ``sampler_polls.parquet`` (+ ``sampler_samples.parquet`` for ``samples``) and ``meta.json``
    with call-duration quantiles, stall statistics, and sensor reading gaps.
    """
    if interval_ms < 1 or duration_s <= 0:
        raise ValueError("interval_ms must be >= 1 and duration_s > 0")
    buf, stop = _Buffers(), threading.Event()
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        dev = device_meta(handle, gpu_id)
        if source == "samples":
            ret = _read(handle, source, 0)[0]
            if ret != pynvml.NVML_SUCCESS:
                raise RuntimeError(f"source='samples' unsupported on GPU {gpu_id} (NVML return {ret})")
        thread = threading.Thread(target=_poll_loop, args=(handle, source, interval_ms, duration_s, buf, stop), daemon=True)
        thread.start()
        try:
            _matmul_loop(gpu_id, duration_s, stop)
        finally:
            stop.set()
            thread.join(timeout=duration_s + 30.0)
        if thread.is_alive():
            raise RuntimeError("sampler-check poll thread did not finish in time")
    finally:
        pynvml.nvmlShutdown()

    meta: dict[str, Any] = {
        "kind": "sampler_check", "source": source, "poll_interval_ms": interval_ms,
        "duration_s": duration_s, "matmul_n": MATMUL_N, "n_polls": len(buf.polls), **dev,
        "call_duration_ms": call_duration_quantiles_ms([r.t_call_end_ns - r.t_call_start_ns for r in buf.polls]),
        "stalls": poll_stall_stats([r.t_call_start_ns for r in buf.polls], interval_ms),
    }
    if source == "samples":
        meta["sensor_readings"] = sensor_skipped_repeated([s[1] for s in buf.samples], [s[2] for s in buf.samples])
    _write(out_dir, buf, meta, source)
    return out_dir
