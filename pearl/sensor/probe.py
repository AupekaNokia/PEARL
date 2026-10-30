"""
NVML sensor characterization campaign: idle / load / square-wave probes and sampler checks.

Usage
-----
    python -m pearl.sensor.probe --gpu 0 --out results/sensor [--quick] [--steps idle_10ms,square_5ms]

Outputs
-------
<out>/<gpu>_<driver>_<timestamp>/
    campaign.json                         schedule, timing, per-step status
    probe/<condition>_<interval>ms/       polls.parquet, samples.parquet, [edges.parquet], meta.json
    sampler_check/<source>/               sampler_polls.parquet, [sampler_samples.parquet], meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import re
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pynvml

from pearl.sensor.load import (
    DEFAULT_SQUARE_PERIODS_MS,
    SQUARE_MATMUL_N,
    SquareSchedule,
    gpu_load_worker,
    square_load_worker,
)
from pearl.sensor.sampler_check import run_sampler_check
from pearl.sensor.sources import SOURCE_ORDER, ProbeBuffers, device_meta, run_poll_loop, write_outputs

_LOG = logging.getLogger(__name__)

Condition = Literal["idle", "load", "square"]


@dataclass(frozen=True)
class Timing:
    probe_duration_s: float = 60.0
    load_warmup_s: float = 5.0
    square_interval_ms: int = 5
    square_block_s: float = 30.0
    square_settle_s: float = 5.0
    square_seed: int = 0
    sampler_duration_s: float = 120.0
    sampler_interval_ms: int = 10


DEFAULT_TIMING = Timing()
QUICK_TIMING = Timing(probe_duration_s=3.0, load_warmup_s=1.0, square_block_s=5.0,
                      square_settle_s=1.0, sampler_duration_s=5.0)


def _join(proc: mp.Process) -> None:
    proc.join(timeout=30.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)


@contextmanager
def _constant_load(gpu_id: int, warmup_s: float) -> Iterator[None]:
    ctx = mp.get_context("spawn")
    ready, stop = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=gpu_load_worker, args=(gpu_id, warmup_s, ready, stop), daemon=True)
    proc.start()
    try:
        if not ready.wait(timeout=warmup_s + 120.0):
            raise RuntimeError("GPU load worker did not become ready in time")
        yield
    finally:
        stop.set()
        _join(proc)


@contextmanager
def _square_load(gpu_id: int, schedule: SquareSchedule) -> Iterator[tuple[Any, Any, Any]]:
    """Yield ``(start, done, results)``: set ``start`` to begin the schedule, ``done`` marks its end."""
    ctx = mp.get_context("spawn")
    ready, start, done, results = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    proc = ctx.Process(target=square_load_worker, args=(gpu_id, schedule, ready, start, done, results), daemon=True)
    proc.start()
    try:
        if not ready.wait(timeout=120.0):
            raise RuntimeError("square-wave worker did not become ready in time")
        yield start, done, results
    finally:
        _join(proc)


def run_probe(
    *,
    gpu_id: int,
    condition: Condition,
    interval_ms: int,
    duration_s: float,
    out_dir: Path,
    warmup_s: float = 5.0,
    square: SquareSchedule | None = None,
) -> Path:
    """
    Poll all five NVML power sources under one load condition and write the probe artifacts.

    With ``condition="square"`` polling lasts until the schedule finishes (``duration_s`` ignored)
    and ``edges.parquet`` records the host-monotonic rise/fall time of every on phase.
    """
    if condition == "square" and square is None:
        raise ValueError("condition='square' requires a SquareSchedule")
    buffers, payload = ProbeBuffers(), None
    load_start_ns = load_stop_ns = None
    with ExitStack() as stack:
        sq = stack.enter_context(_square_load(gpu_id, square)) if condition == "square" else None
        if condition == "load":
            stack.enter_context(_constant_load(gpu_id, warmup_s))
            load_start_ns = time.monotonic_ns()
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            dev = device_meta(handle, gpu_id)
            if sq is not None:
                sq[0].set()
            n_polls = run_poll_loop(handle, interval_ms=interval_ms, buffers=buffers,
                                    duration_s=None if sq else duration_s, stop=sq[1] if sq else None)
            if sq is not None:
                payload = sq[2].get(timeout=60.0)
        finally:
            pynvml.nvmlShutdown()
        if condition == "load":
            load_stop_ns = time.monotonic_ns()

    meta: dict[str, Any] = {
        **dev, "poll_interval_ms": interval_ms, "duration_s": None if sq else duration_s,
        "condition": condition, "warmup_s": warmup_s if condition == "load" else 0.0,
        "source_order": list(SOURCE_ORDER), "n_polls": n_polls,
        "load_start_ns_monotonic": load_start_ns, "load_stop_ns_monotonic": load_stop_ns,
    }
    edges = None
    if square is not None and condition == "square":
        edges = payload["edges"]
        meta["square"] = {**asdict(square), "periods_ms": list(square.periods_ms), "duty": 0.5,
                          "matmul_n": SQUARE_MATMUL_N, "blocks": payload["blocks"]}
    write_outputs(out_dir, buffers, meta, edges)
    _LOG.info("Wrote %s (%d polls)", out_dir, n_polls)
    return out_dir


def build_steps(timing: Timing) -> list[dict[str, Any]]:
    """Ordered suite: idle/load at 10 and 2 ms, square at 5 ms, sampler checks (usage, samples)."""
    steps: list[dict[str, Any]] = [
        {"step_id": f"{c}_{ms}ms", "kind": "probe", "out_dir": f"probe/{c}_{ms}ms",
         "condition": c, "interval_ms": ms, "duration_s": timing.probe_duration_s,
         "warmup_s": timing.load_warmup_s if c == "load" else 0.0}
        for c in ("idle", "load") for ms in (10, 2)
    ]
    sq_ms = timing.square_interval_ms
    periods = DEFAULT_SQUARE_PERIODS_MS if timing.square_block_s >= 20.0 else (10, 100)
    steps.append({"step_id": f"square_{sq_ms}ms", "kind": "probe", "out_dir": f"probe/square_{sq_ms}ms",
                  "condition": "square", "interval_ms": sq_ms, "duration_s": 0.0,
                  "square": {"periods_ms": list(periods), "block_s": timing.square_block_s,
                             "settle_s": timing.square_settle_s, "seed": timing.square_seed}})
    steps += [
        {"step_id": f"sampler_{s}", "kind": "sampler_check", "out_dir": f"sampler_check/{s}", "source": s,
         "interval_ms": timing.sampler_interval_ms, "duration_s": timing.sampler_duration_s}
        for s in ("usage", "samples")
    ]
    return steps


def _run_step(step: dict[str, Any], gpu_id: int, out_dir: Path) -> None:
    if step["kind"] == "sampler_check":
        run_sampler_check(gpu_id=gpu_id, source=step["source"], interval_ms=step["interval_ms"],
                          duration_s=step["duration_s"], out_dir=out_dir)
        return
    sq = step.get("square")
    schedule = None if sq is None else SquareSchedule(tuple(sq["periods_ms"]), sq["block_s"], sq["settle_s"], sq["seed"])
    run_probe(gpu_id=gpu_id, condition=step["condition"], interval_ms=step["interval_ms"],
              duration_s=step["duration_s"], out_dir=out_dir, warmup_s=step["warmup_s"], square=schedule)


def _campaign_dir(out: Path, gpu_id: int) -> tuple[Path, str, str]:
    pynvml.nvmlInit()
    try:
        dev = device_meta(pynvml.nvmlDeviceGetHandleByIndex(gpu_id), gpu_id)
    finally:
        pynvml.nvmlShutdown()
    gpu, driver = str(dev["gpu_name"] or "gpu"), dev["nvidia_driver"]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")
    return out / f"{slug}_{driver.replace('.', '_')}_{time.strftime('%Y%m%d-%H%M%S')}", gpu, driver


def run_campaign(*, gpu_id: int, out: Path, quick: bool = False, only: set[str] | None = None) -> Path:
    """Run every step sequentially (a failed step is logged and recorded); return the campaign folder."""
    timing = QUICK_TIMING if quick else DEFAULT_TIMING
    steps = [s for s in build_steps(timing) if only is None or s["step_id"] in only]
    if not steps:
        raise ValueError(f"no step selected (known: {[s['step_id'] for s in build_steps(timing)]})")
    root, gpu, driver = _campaign_dir(out, gpu_id)
    root.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "physical_gpu_index": gpu_id, "gpu_name": gpu, "nvidia_driver": driver,
              "quick": quick, "timing": asdict(timing),
              "steps": [{**s, "status": "pending", "error": None} for s in steps]}
    path = root / "campaign.json"
    for step in record["steps"]:
        step["status"], step["started_at"] = "running", time.strftime("%Y-%m-%dT%H:%M:%S")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        try:
            _run_step(step, gpu_id, root / step["out_dir"])
            step["status"] = "ok"
        except Exception as e:  # keep the campaign going; the failure is recorded per step
            _LOG.exception("Step %s failed", step["step_id"])
            step["status"], step["error"] = "failed", repr(e)
        step["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--gpu", type=int, default=0, help="Physical NVML GPU index.")
    p.add_argument("--out", type=Path, required=True, help="Output root directory.")
    p.add_argument("--quick", action="store_true", help="Short durations (smoke test, a few minutes).")
    p.add_argument("--steps", default=None, help="Comma-separated subset of step ids (default: all).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    only = None if args.steps is None else {s.strip() for s in args.steps.split(",") if s.strip()}
    print(run_campaign(gpu_id=args.gpu, out=args.out, quick=args.quick, only=only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
