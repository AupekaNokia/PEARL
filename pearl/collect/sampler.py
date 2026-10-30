"""NVML power sampler (``nvmlDeviceGetSamples``) on the host monotonic clock, plus trapezoid energy."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pynvml

_LOG = logging.getLogger(__name__)

STALL_FACTOR = 3.0
SENSOR_GAP_FACTOR = 3.0


@dataclass(frozen=True)
class PowerSample:
    """One sensor reading; ``time_ns`` is the sensor timestamp converted to host monotonic time."""

    time_ns: int
    power_w: float
    nvidia_timestamp_us: int | None = None
    sm_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    gpu_util_pct: float | None = None
    mem_util_pct: float | None = None


def _gaps_over(times_ns: Sequence[int], threshold_ns: float) -> dict[str, float | int]:
    gaps = [b - a for a, b in zip(times_ns, times_ns[1:]) if b - a > threshold_ns]
    return {"n": len(gaps), "total_ms": sum(gaps) / 1e6, "max_ms": max(gaps) / 1e6 if gaps else 0.0}


def _try_float(fn: Any) -> float | None:
    try:
        return float(fn())
    except pynvml.NVMLError:
        return None


class PowerSampler:
    """
    Background thread polling ``nvmlDeviceGetSamples(NVML_TOTAL_POWER_SAMPLES)`` every ``interval_ms``.

    Every new sensor sample becomes one :class:`PowerSample`. Its realtime sensor timestamp is
    mapped to ``time.monotonic_ns()`` (the clock of the layer hooks) using the realtime-monotonic
    offset read right after each NVML call. SM/memory clocks and utilization are read once per poll.
    """

    def __init__(self, gpu_id: int, interval_ms: int = 10) -> None:
        self.gpu_id = gpu_id
        self.interval_ms = max(1, int(interval_ms))
        self._samples: list[PowerSample] = []
        self._poll_starts_ns: list[int] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: object | None = None
        self._cursor_us = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"failed to initialise NVML (driver or GPU missing?): {e}") from e
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            try:
                pynvml.nvmlDeviceGetSamples(self._handle, pynvml.NVML_TOTAL_POWER_SAMPLES, 0)
            except pynvml.NVMLError as e:
                if e.value != pynvml.NVML_ERROR_NOT_FOUND:
                    raise RuntimeError(f"nvmlDeviceGetSamples unsupported on GPU {self.gpu_id}: {e}") from e
        except Exception:
            pynvml.nvmlShutdown()
            raise
        self._cursor_us = time.time_ns() // 1000
        self._samples.clear()
        self._poll_starts_ns.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _read(self) -> list[tuple[float, int]]:
        """New ``(power_w, sensor_ts_us)`` readings since the cursor, oldest first."""
        try:
            _, raw = pynvml.nvmlDeviceGetSamples(self._handle, pynvml.NVML_TOTAL_POWER_SAMPLES, self._cursor_us)
        except pynvml.NVMLError as e:
            if e.value == pynvml.NVML_ERROR_NOT_FOUND:
                return []
            raise
        out = [(float(s.sampleValue.uiVal) / 1000.0, int(s.timeStamp)) for s in raw if int(s.timeStamp) > self._cursor_us]
        out.sort(key=lambda r: r[1])
        if out:
            self._cursor_us = max(ts for _, ts in out)
        return out

    def _poll_loop(self) -> None:
        handle = self._handle
        while not self._stop.wait(timeout=self.interval_ms / 1000.0):
            t_before = time.monotonic_ns()
            self._poll_starts_ns.append(t_before)
            try:
                readings = self._read()
            except pynvml.NVMLError:
                continue
            t_after = time.monotonic_ns()
            rt_minus_mono = time.time_ns() - (t_after + time.monotonic_ns()) // 2
            if not readings:
                continue
            sm = _try_float(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            mem = _try_float(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_u, mem_u = float(util.gpu), float(util.memory)
            except pynvml.NVMLError:
                gpu_u = mem_u = None
            rows = [PowerSample(ts * 1000 - rt_minus_mono, p, ts, sm, mem, gpu_u, mem_u) for p, ts in readings]
            with self._lock:
                self._samples.extend(rows)

    def stop(self, timeout_s: float = 5.0) -> list[PowerSample]:
        """Stop polling, shut NVML down, and return all samples."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._thread = self._handle = None
        st = self.meta()["stalls"]
        if st["n"]:
            _LOG.warning(
                "Sampler stalls: %d poll gap(s) > %.0f ms (max %.1f ms, total %.1f ms) over %d polls",
                st["n"], st["threshold_ms"], st["max_ms"], st["total_ms"], st["n_polls"],
            )
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
        with self._lock:
            return list(self._samples)

    def meta(self) -> dict[str, Any]:
        """Sampler settings, host poll stalls, and sensor timestamp gaps."""
        with self._lock:
            times = [s.time_ns for s in self._samples]
        spacing = sorted(b - a for a, b in zip(times, times[1:]))
        median_ns = spacing[len(spacing) // 2] if spacing else None
        stall_ms = STALL_FACTOR * self.interval_ms
        return {
            "power_source": "samples",
            "interval_ms": self.interval_ms,
            "timestamp_midpoint": False,
            "log_stalls": True,
            "stalls": {"threshold_ms": stall_ms, "n_polls": len(self._poll_starts_ns),
                       **_gaps_over(self._poll_starts_ns, stall_ms * 1e6)},
            "nvidia_timestamp": {
                "clock": "CLOCK_REALTIME",
                "unit": "us",
                "time_ns_monotonic": "sensor timestamp minus (realtime - monotonic), "
                                     "realtime read after each call between two monotonic reads",
            },
            "sensor_gaps": {
                "median_spacing_ms": None if median_ns is None else median_ns / 1e6,
                "threshold_factor": SENSOR_GAP_FACTOR,
                **(_gaps_over(times, SENSOR_GAP_FACTOR * median_ns) if median_ns
                   else {"n": 0, "total_ms": 0.0, "max_ms": 0.0}),
            },
        }


def trapezoid_energy_j(times_ns: np.ndarray, powers_w: np.ndarray, wall_time_s: float | None = None) -> float:
    """
    Trapezoidal energy (J) of a power trace.

    With a single sample and ``wall_time_s`` set, returns the rectangle ``P * wall_time_s``;
    with no sample, 0.
    """
    if times_ns.size == 1 and wall_time_s is not None and wall_time_s > 0:
        return float(powers_w[0]) * wall_time_s
    if times_ns.size < 2:
        return 0.0
    t = times_ns.astype(np.float64) * 1e-9
    p = powers_w.astype(np.float64)
    return float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(t)))
