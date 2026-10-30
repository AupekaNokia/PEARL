"""GPU load child processes: constant fp16 matmul and square-wave on/off blocks."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

MATMUL_N = 8192
SQUARE_MATMUL_N = 4096
DEFAULT_SQUARE_PERIODS_MS: tuple[int, ...] = (10, 25, 50, 100, 200, 500, 1000, 2000, 4000)

Edge = tuple[int, int, int, str, int]


def _operands(physical_gpu_id: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pin this process to one physical GPU and allocate two ``n x n`` fp16 matrices."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable in load worker (CUDA_VISIBLE_DEVICES={physical_gpu_id!r})")
    a = torch.randn(n, n, device="cuda:0", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda:0", dtype=torch.float16)
    return a, b


def gpu_load_worker(
    physical_gpu_id: int, warmup_s: float, ready: mp.synchronize.Event, stop: mp.synchronize.Event
) -> None:
    """Constant load: synchronized matmuls until ``stop``; ``ready`` is set after ``warmup_s``."""
    import torch

    a, b = _operands(physical_gpu_id, MATMUL_N)
    deadline = time.monotonic() + warmup_s
    while time.monotonic() < deadline:
        torch.matmul(a, b)
        torch.cuda.synchronize()
    ready.set()
    while not stop.is_set():
        torch.matmul(a, b)
        torch.cuda.synchronize()


@dataclass(frozen=True)
class SquareSchedule:
    """Square-wave plan: one 50 % duty-cycle block per period, separated by idle settles."""

    periods_ms: tuple[int, ...]
    block_s: float
    settle_s: float
    seed: int


def parse_square_schedule(periods_csv: str, block_s: float, settle_s: float, seed: int) -> SquareSchedule:
    """Build a :class:`SquareSchedule`; raise ``ValueError`` on invalid values."""
    try:
        periods = tuple(int(p) for p in periods_csv.split(",") if p.strip())
    except ValueError as e:
        raise ValueError(f"invalid square periods {periods_csv!r}") from e
    if not periods or any(p < 2 for p in periods):
        raise ValueError("square periods need values >= 2 ms")
    if block_s <= 0 or settle_s < 0:
        raise ValueError("square block_s must be > 0 and settle_s >= 0")
    if any(p / 1000.0 > block_s for p in periods):
        raise ValueError("every square period must fit in block_s")
    return SquareSchedule(periods_ms=periods, block_s=block_s, settle_s=settle_s, seed=seed)


def square_block_offsets_ms(schedule: SquareSchedule) -> list[float]:
    """Random start offset per block, uniform in [0, 1000) ms, reproducible from ``seed``."""
    rng = random.Random(schedule.seed)
    return [rng.uniform(0.0, 1000.0) for _ in schedule.periods_ms]


def _square_block(
    a: Any, b: Any, block_idx: int, period_ms: int, block_s: float, edges: list[Edge]
) -> tuple[int, int]:
    """Run one block of on/off cycles, appending host-monotonic edges; return ``(start_ns, n_cycles)``."""
    import torch

    period_ns = period_ms * 1_000_000
    half_ns = period_ns // 2
    block_start = time.monotonic_ns()
    block_end = block_start + int(block_s * 1e9)
    cycle = 0
    while block_start + (cycle + 1) * period_ns <= block_end:
        cycle_start = block_start + cycle * period_ns
        now = time.monotonic_ns()
        if now < cycle_start:
            time.sleep((cycle_start - now) / 1e9)
        rise = time.monotonic_ns()
        while True:
            torch.matmul(a, b)
            torch.cuda.synchronize()
            fall = time.monotonic_ns()
            if fall >= rise + half_ns:
                break
        edges.append((block_idx, period_ms, cycle, "rise", rise))
        edges.append((block_idx, period_ms, cycle, "fall", fall))
        cycle += 1
    return block_start, cycle


def square_load_worker(
    physical_gpu_id: int,
    schedule: SquareSchedule,
    ready: mp.synchronize.Event,
    start: mp.synchronize.Event,
    done: mp.synchronize.Event,
    results: mp.Queue,
) -> None:
    """
    Drive square-wave blocks and send back activity edges and block records.

    Rising edge: monotonic time just before the first kernel of an on phase.
    Falling edge: monotonic time just after the last ``synchronize`` of that on phase.
    """
    import torch

    a, b = _operands(physical_gpu_id, SQUARE_MATMUL_N)
    for _ in range(20):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    ready.set()
    start.wait()

    edges: list[Edge] = []
    blocks: list[dict[str, Any]] = []
    time.sleep(schedule.settle_s)
    for block_idx, (period_ms, offset_ms) in enumerate(zip(schedule.periods_ms, square_block_offsets_ms(schedule))):
        time.sleep(offset_ms / 1000.0)
        t0, n_cycles = _square_block(a, b, block_idx, period_ms, schedule.block_s, edges)
        blocks.append({
            "block_idx": block_idx, "period_ms": period_ms, "offset_ms": offset_ms,
            "start_ns_monotonic": t0, "end_ns_monotonic": time.monotonic_ns(), "n_cycles": n_cycles,
        })
        time.sleep(schedule.settle_s)
    results.put({"edges": edges, "blocks": blocks})
    done.set()
