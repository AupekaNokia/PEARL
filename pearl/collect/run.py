"""Pipeline 2a CLI: idle baseline, then per grid point N hooked ``generate()`` calls under NVML sampling."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pearl.collect.sampler import PowerSample, PowerSampler, trapezoid_energy_j
from pearl.collect.traces import Segment, write_combo, write_idle_trace

_LOG = logging.getLogger(__name__)

METADATA_SCHEMA_VERSION = 2
CAMPAIGN_FILE = "layer_energy_campaign_config.json"


@dataclass(frozen=True)
class Protocol:
    """Measurement settings shared by all grid points."""

    gpu_id: int = 0
    seed: int = 42
    interval_ms: int = 10
    idle_duration_s: float = 10.0
    warmup_steps: int = 10
    forwards_per_measure: int = 10
    trace_margin_ms: float = 150.0
    use_cache: bool = True


def load_config(path: Path) -> tuple[Protocol, list[dict[str, Any]], Path]:
    """Parse a collect YAML into ``(protocol, grids, output_dir)``."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw.get("protocol") or {}) - set(Protocol.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown protocol keys in {path}: {sorted(unknown)}")
    grids = raw.get("grids") or []
    if not grids:
        raise ValueError(f"{path} defines no grids")
    return Protocol(**(raw.get("protocol") or {})), grids, Path(raw.get("output_dir", "results/collect"))


def iter_grid(grids: list[dict[str, Any]], use_cache: bool) -> Iterator[Any]:
    """
    Cartesian product models x batch_sizes x sequence_lengths x max_new_tokens x precisions, per grid.

    An optional ``revisions: {model: commit}`` mapping in a grid pins Hub revisions.
    """
    from pearl.collect.workload import WorkloadSpec

    revisions = {}
    for g in grids:
        revisions.update(g.get("revisions") or {})
        axes = (g["models"], g["batch_sizes"], g["sequence_lengths"], g["max_new_tokens"], g["precisions"])
        for model, bs, seq, tok, prec in itertools.product(*axes):
            yield WorkloadSpec(str(model), int(bs), int(seq), int(tok), str(prec), revisions.get(model), use_cache)


def combo_tag(spec: Any) -> str:
    """Folder name, e.g. ``Qwen__Qwen3-0.6B_bs1_seq32_tok16_fp16``."""
    model = re.sub(r"[^a-zA-Z0-9._-]+", "_", spec.model_name.replace("/", "__").strip())
    return f"{model}_bs{spec.batch_size}_seq{spec.sequence_length}_tok{spec.max_new_tokens}_{spec.precision}"


def run_config(spec: Any, proto: Protocol) -> dict[str, Any]:
    """Per-segment config (same nesting as the historical traces, read by Pipeline 2b)."""
    return {
        "workload": {"task": "text_generation", "device": "cuda", **asdict(spec),
                     "forward_repeats": proto.forwards_per_measure},
        "inference": {"warmup_steps": proto.warmup_steps},
        "sampler": {"interval_ms": proto.interval_ms, "power_source": "samples", "log_stalls": True},
        "system": {"gpu_id": proto.gpu_id, "seed": proto.seed},
    }


def torch_index(gpu_id: int) -> int:
    """Torch device index of NVML GPU ``gpu_id`` (PCI bus order, honouring ``CUDA_VISIBLE_DEVICES``)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return gpu_id
    ids = [v.strip() for v in visible.split(",")]
    if str(gpu_id) not in ids:
        raise ValueError(f"protocol.gpu_id={gpu_id} is not in CUDA_VISIBLE_DEVICES={visible!r} "
                         "(gpu_id is the NVML index; use integer CUDA_VISIBLE_DEVICES entries)")
    return ids.index(str(gpu_id))


def measure_idle(proto: Protocol) -> tuple[list[PowerSample], dict[str, Any]]:
    """Sample the idle GPU for ``idle_duration_s``; return samples and their mean/peak power."""
    sampler = PowerSampler(proto.gpu_id, proto.interval_ms)
    sampler.start()
    time.sleep(proto.idle_duration_s)
    samples = sampler.stop()
    if len(samples) < 2:
        raise RuntimeError(f"idle baseline collected {len(samples)} sample(s); NVML sampling is not working")
    t = np.array([s.time_ns for s in samples])
    p = np.array([s.power_w for s in samples])
    duration_s = float(t[-1] - t[0]) * 1e-9
    energy = trapezoid_energy_j(t, p)
    info = {"duration_s": duration_s, "total_joules": energy, "avg_power_w": energy / duration_s,
            "peak_power_w": float(p.max()), "num_samples": len(samples), "sampler_meta": sampler.meta()}
    _LOG.info("Idle power: %.2f W over %.1f s (%d samples)", info["avg_power_w"], duration_s, len(samples))
    return samples, info


def measure_segment(wl: Any, profiler: Any, proto: Protocol) -> dict[str, Any]:
    """
    One measured window: sampler start, margin, N forwards (hooked), margin, sampler stop.

    ``wall_time_s`` is sampler start-up + forwards + sampler shutdown (margins excluded).
    """
    import torch

    sampler = PowerSampler(proto.gpu_id, proto.interval_ms)
    profiler.reset()
    t_call = time.perf_counter()
    sampler.start()
    start_call_s = time.perf_counter() - t_call
    try:
        time.sleep(proto.trace_margin_ms / 1e3)
        span_start_ns, t0 = time.monotonic_ns(), time.perf_counter()
        torch.cuda.synchronize(wl.device)
        try:
            for _ in range(proto.forwards_per_measure):
                wl.run_forward()
                profiler.mark_forward_end()
        finally:
            torch.cuda.synchronize(wl.device)
            span_end_ns, span_s = time.monotonic_ns(), time.perf_counter() - t0
        time.sleep(proto.trace_margin_ms / 1e3)
    finally:
        t_stop = time.perf_counter()
        samples = sampler.stop()
    return {"samples": samples, "timings": profiler.collect(), "meta": sampler.meta(),
            "wall_time_s": start_call_s + span_s + (time.perf_counter() - t_stop),
            "span": (span_start_ns, span_end_ns)}


def run_combo(spec: Any, proto: Protocol, traces_root: Path) -> None:
    """Load, warm up, index leaves, measure one segment, write the combo traces, unload."""
    import torch

    from pearl.collect.hooks import LeafHookProfiler, build_layer_registry
    from pearl.collect.workload import GenerationWorkload

    wl = GenerationWorkload(spec, torch.device(f"cuda:{torch_index(proto.gpu_id)}"), proto.seed)
    profiler = None
    try:
        wl.load()
        wl.warmup(proto.warmup_steps)
        wl.run_forward()
        layers = build_layer_registry(wl.model, wl.example_input())
        profiler = LeafHookProfiler(wl.model)
        profiler.attach()
        m = measure_segment(wl, profiler, proto)
    finally:
        if profiler is not None:
            profiler.detach()
        wl.unload()
    seg = Segment(0, m["samples"], m["timings"], run_config(spec, proto), m["meta"], m["wall_time_s"],
                  proto.forwards_per_measure, proto.trace_margin_ms, *m["span"])
    write_combo(traces_root / combo_tag(spec), layers, [seg])


def run_campaign(config_path: Path, output_dir: Path | None = None) -> Path:
    """Run the full collection described by ``config_path``; return the ``traces`` root."""
    proto, grids, cfg_out = load_config(config_path)
    session = (output_dir or cfg_out) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S") / "layer_energy"
    traces_root = session / "traces"
    idle_samples, idle_info = measure_idle(proto)
    write_idle_trace(traces_root, idle_samples)
    with (session / CAMPAIGN_FILE).open("w", encoding="utf-8") as f:
        json.dump({"metadata_schema_version": METADATA_SCHEMA_VERSION, "protocol": asdict(proto),
                   "grids": grids, "idle": idle_info}, f, indent=2)
    specs = list(iter_grid(grids, proto.use_cache))
    n_ok = 0
    for i, spec in enumerate(specs, start=1):
        _LOG.info("[%d/%d] %s", i, len(specs), combo_tag(spec))
        try:
            run_combo(spec, proto, traces_root)
            n_ok += 1
        except Exception:  # noqa: BLE001 - one grid point (e.g. OOM) must not abort the campaign
            _LOG.exception("Skipping %s", combo_tag(spec))
    _LOG.info("Collected %d/%d grid points under %s", n_ok, len(specs), traces_root)
    return traces_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="collect YAML (e.g. configs/collect_smoke.yaml)")
    parser.add_argument("--out", type=Path, default=None, help="override output_dir from the config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print(run_campaign(args.config, args.out))


if __name__ == "__main__":
    main()
