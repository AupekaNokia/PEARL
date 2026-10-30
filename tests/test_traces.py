"""CPU tests for Pipeline 2a: trace store round trip, leaf hooks, layer registry, config grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from pearl.collect.sampler import PowerSample, trapezoid_energy_j
from pearl.collect.traces import (
    POWER_SCHEMA,
    Segment,
    load_combo,
    load_idle_trace,
    write_combo,
    write_idle_trace,
)

REPO = Path(__file__).resolve().parents[1]


def _samples(t0: int, n: int, step_ns: int = 10_000_000) -> list[PowerSample]:
    return [PowerSample(t0 + i * step_ns, 100.0 + i, 1_000 + i, 1400.0, 7000.0, 50.0, 20.0) for i in range(n)]


def _timings(t0: int) -> list[dict]:
    def lyr(name: str, step: int, start: int, end: int) -> dict:
        return {"layer_name": name, "layer_type": "Linear", "step_id": step, "start_ns": start, "end_ns": end}

    return [
        {"forward_idx": 0, "layers": [lyr("b", 0, t0 + 300, t0 + 400), lyr("a", 0, t0 + 100, t0 + 200),
                                      lyr("ghost", 0, t0 + 250, t0 + 260), lyr("a", 1, t0 + 500, t0 + 600)]},
        {"forward_idx": 1, "layers": [lyr("a", 0, t0 + 1_000, t0 + 1_100)]},
    ]


def test_trapezoid_energy() -> None:
    t = np.arange(101) * 10_000_000
    assert trapezoid_energy_j(t, np.full(101, 10.0)) == pytest.approx(10.0)
    assert trapezoid_energy_j(t[:1], np.array([5.0]), wall_time_s=2.0) == pytest.approx(10.0)
    assert trapezoid_energy_j(t[:0], np.array([])) == 0.0


def test_combo_round_trip(tmp_path: Path) -> None:
    t0 = 1_000_000_000
    samples = _samples(t0, 20)
    seg = Segment(0, samples, _timings(t0), {"workload": {"model_name": "m"}}, {"power_source": "samples",
                  "interval_ms": 10, "stalls": {"n": 0}}, 0.25, 2, 150.0, t0 + 30_000_000, t0 + 120_000_000)
    combo = tmp_path / "m_bs1_seq16_tok16_fp16"
    write_combo(combo, [(1, "a", "Linear"), (2, "b", "Linear")], [seg])

    assert sorted(p.stem for p in combo.iterdir()) == ["firings", "layers", "power_trace", "segments"]
    assert pq.read_schema(combo / "power_trace.parquet").equals(POWER_SCHEMA, check_metadata=False)
    layers, (loaded,) = load_combo(combo)
    assert layers["layer_name"].tolist() == ["a", "b"]
    assert loaded.run_config == {"workload": {"model_name": "m"}}
    assert loaded.meta["forward_count"] == 2 and loaded.meta["sampler_power_source"] == "samples"
    assert loaded.time_ns.tolist() == [s.time_ns for s in samples[3:13]]
    assert loaded.power_w.tolist() == pytest.approx([s.power_w for s in samples[3:13]])
    fir = loaded.firings
    assert fir["layer_idx"].tolist() == [1, 2, 1, 1]
    assert fir["start_ns"].tolist() == [t0 + 100, t0 + 300, t0 + 500, t0 + 1_000]
    assert (fir["end_ns"] - fir["start_ns"]).tolist() == [100, 100, 100, 100]
    assert fir["step_id"].tolist() == [0, 0, 1, 0] and fir["forward_idx"].tolist() == [0, 0, 0, 1]


def test_idle_round_trip(tmp_path: Path) -> None:
    samples = _samples(5_000, 4)
    write_idle_trace(tmp_path, samples)
    t, p = load_idle_trace(tmp_path)
    assert t.tolist() == [s.time_ns for s in samples]
    assert p.tolist() == pytest.approx([s.power_w for s in samples])


def test_leaf_hooks_step_ids_and_registry() -> None:
    torch = pytest.importorskip("torch")
    from pearl.collect.hooks import LeafHookProfiler, build_layer_registry

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, 2)))
    x = torch.zeros(1, 4)
    assert build_layer_registry(model, x) == [(1, "0", "Linear"), (2, "1.0", "ReLU"), (3, "1.1", "Linear")]

    prof = LeafHookProfiler(model)
    prof.attach()
    for _ in range(2):
        for _ in range(3):
            model(x)
        prof.mark_forward_end()
    prof.detach()
    model(x)
    blocks = prof.collect()
    assert len(blocks) == 2
    for block in blocks:
        layers = block["layers"]
        assert [lyr["step_id"] for lyr in layers] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert [lyr["layer_name"] for lyr in layers[:3]] == ["0", "1.0", "1.1"]
        assert all(lyr["end_ns"] >= lyr["start_ns"] > 0 for lyr in layers)


def test_paper_and_smoke_configs() -> None:
    pytest.importorskip("yaml")
    from pearl.collect.run import combo_tag, iter_grid, load_config

    proto, grids, _ = load_config(REPO / "configs" / "collect_paper.yaml")
    specs = list(iter_grid(grids, proto.use_cache))
    assert len(specs) == 11 * 4 * 3 * 3 * 3 + 7 * 4 * 5 * 3 * 3
    assert (proto.forwards_per_measure, proto.warmup_steps, proto.trace_margin_ms) == (10, 10, 150.0)
    proto, grids, _ = load_config(REPO / "configs" / "collect_smoke.yaml")
    (spec,) = iter_grid(grids, proto.use_cache)
    assert combo_tag(spec) == "Qwen__Qwen3-0.6B_bs1_seq32_tok16_fp16"
