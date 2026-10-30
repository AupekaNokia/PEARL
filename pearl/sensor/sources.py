"""Five-source NVML poll loop, device metadata, and probe Parquet writers."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import pynvml

SOURCE_ORDER: tuple[str, ...] = ("usage", "instant", "average", "samples", "energy")
EDGE_COLUMNS: tuple[str, ...] = ("block_idx", "period_ms", "cycle_idx", "edge", "t_ns")
SAMPLE_COLUMNS: tuple[str, ...] = ("poll_idx", "sensor_ts_us", "value_mw", "value_type")

_T = TypeVar("_T")
_SCALAR_ATTR = {
    pynvml.NVML_VALUE_TYPE_UNSIGNED_INT: "uiVal",
    pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG: "ulVal",
    pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: "ullVal",
    pynvml.NVML_VALUE_TYPE_SIGNED_INT: "siVal",
    pynvml.NVML_VALUE_TYPE_SIGNED_LONG_LONG: "sllVal",
    pynvml.NVML_VALUE_TYPE_UNSIGNED_SHORT: "usVal",
    pynvml.NVML_VALUE_TYPE_DOUBLE: "dVal",
}


def nvml_err_value(exc: BaseException) -> int:
    """NVML return code carried by ``exc`` (``NVML_ERROR_UNKNOWN`` for other exceptions)."""
    return int(exc.value) if isinstance(exc, pynvml.NVMLError) else int(pynvml.NVML_ERROR_UNKNOWN)


def nvml_scalar(value: Any, value_type: int) -> float | None:
    """Decode an NVML value union according to ``value_type``."""
    attr = _SCALAR_ATTR.get(int(value_type))
    return None if attr is None else float(getattr(value, attr))


def _text(raw: str | bytes) -> str:
    return (raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)).strip()


def _try(fn: Callable[[], _T]) -> _T | None:
    try:
        return fn()
    except pynvml.NVMLError:
        return None


def device_meta(handle: object, gpu_id: int) -> dict[str, Any]:
    """GPU model, driver, and power settings (no host or serial identifiers)."""
    cuda = _try(pynvml.nvmlSystemGetCudaDriverVersion)
    return {
        "physical_gpu_index": gpu_id,
        "gpu_name": _try(lambda: _text(pynvml.nvmlDeviceGetName(handle))),
        "nvidia_driver": _text(pynvml.nvmlSystemGetDriverVersion()),
        "nvml_version": _text(pynvml.nvmlSystemGetNVMLVersion()),
        "cuda_driver_version": None if cuda is None else f"{cuda // 1000}.{(cuda % 1000) // 10}",
        "vbios": _try(lambda: _text(pynvml.nvmlDeviceGetVbiosVersion(handle))),
        "power_limit_mw": _try(lambda: int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))),
        "persistence_mode": _try(lambda: int(pynvml.nvmlDeviceGetPersistenceMode(handle))),
    }


@dataclass
class PollRow:
    poll_idx: int
    source: str
    t_before_ns: int
    t_after_ns: int
    t_realtime_ns: int
    t_realtime_before_ns: int
    t_realtime_after_ns: int
    nvml_return: int
    value: float | None
    unit: str


@dataclass
class ProbeBuffers:
    poll_rows: list[PollRow] = field(default_factory=list)
    sample_rows: list[tuple[int, int, float, int]] = field(default_factory=list)


def _timed(fn: Callable[[], _T]) -> tuple[int, _T | None, tuple[int, int, int, int]]:
    """Run ``fn`` between realtime/monotonic reads; NVML errors become return codes."""
    rt0, m0 = time.time_ns(), time.monotonic_ns()
    try:
        result: _T | None = fn()
        ret = pynvml.NVML_SUCCESS
    except pynvml.NVMLError as e:
        result, ret = None, nvml_err_value(e)
    m1, rt1 = time.monotonic_ns(), time.time_ns()
    return ret, result, (rt0, m0, m1, rt1)


def _field(handle: object, field_id: int) -> tuple[int, float | None]:
    fv = pynvml.nvmlDeviceGetFieldValues(handle, [field_id])[0]
    ret = int(fv.nvmlReturn)
    return ret, (nvml_scalar(fv.value, fv.valueType) if ret == pynvml.NVML_SUCCESS else None)


def poll_once(handle: object, poll_idx: int, cursor_us: int, buf: ProbeBuffers) -> int:
    """Read every source once in :data:`SOURCE_ORDER`; return the updated ``samples`` cursor."""
    t_rt = time.time_ns()

    def add(source: str, ret: int, value: float | None, timing: tuple[int, int, int, int], unit: str) -> None:
        rt0, m0, m1, rt1 = timing
        buf.poll_rows.append(PollRow(poll_idx, source, m0, m1, t_rt, rt0, rt1, ret, value, unit))

    ret, mw, tm = _timed(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
    add("usage", ret, None if mw is None else float(mw), tm, "mW")
    for source, fid in (("instant", pynvml.NVML_FI_DEV_POWER_INSTANT),
                        ("average", pynvml.NVML_FI_DEV_POWER_AVERAGE)):
        ret, fv, tm = _timed(lambda f=fid: _field(handle, f))
        value = None
        if fv is not None:
            ret, value = fv
        add(source, ret, value, tm, "mW")

    ret, got, tm = _timed(lambda: pynvml.nvmlDeviceGetSamples(handle, pynvml.NVML_TOTAL_POWER_SAMPLES, cursor_us))
    newest: float | None = None
    if got is not None:
        vt = int(got[0])
        for s in got[1]:
            ts, newest = int(s.timeStamp), nvml_scalar(s.sampleValue, vt)
            buf.sample_rows.append((poll_idx, ts, newest, vt))
            cursor_us = max(cursor_us, ts)
    add("samples", ret, newest, tm, "mW")

    ret, mj, tm = _timed(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    add("energy", ret, None if mj is None else float(mj), tm, "mJ")
    return cursor_us


def run_poll_loop(
    handle: object,
    *,
    interval_ms: int,
    duration_s: float | None,
    buffers: ProbeBuffers,
    stop: mp.synchronize.Event | None = None,
) -> int:
    """Poll on absolute deadlines until ``duration_s`` elapses or ``stop`` is set; return poll count."""
    if duration_s is None and stop is None:
        raise ValueError("either duration_s or stop is required")
    interval_ns = interval_ms * 1_000_000
    t_start = time.monotonic_ns()
    t_end = None if duration_s is None else t_start + int(duration_s * 1e9)
    poll_idx = cursor_us = 0
    while True:
        now = time.monotonic_ns()
        if (t_end is not None and now >= t_end) or (stop is not None and stop.is_set()):
            return poll_idx
        deadline = t_start + poll_idx * interval_ns
        if now < deadline:
            time.sleep((deadline - now) / 1e9)
        cursor_us = poll_once(handle, poll_idx, cursor_us, buffers)
        poll_idx += 1


def write_outputs(
    out_dir: Path,
    buffers: ProbeBuffers,
    meta: dict[str, Any],
    edges: list[tuple[int, int, int, str, int]] | None = None,
) -> None:
    """Write ``polls.parquet``, ``samples.parquet``, optional ``edges.parquet``, and ``meta.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if edges is not None:
        pd.DataFrame(edges, columns=list(EDGE_COLUMNS)).to_parquet(out_dir / "edges.parquet", index=False)
    pd.DataFrame([asdict(r) for r in buffers.poll_rows]).to_parquet(out_dir / "polls.parquet", index=False)
    pd.DataFrame(buffers.sample_rows, columns=list(SAMPLE_COLUMNS)).to_parquet(
        out_dir / "samples.parquet", index=False
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
