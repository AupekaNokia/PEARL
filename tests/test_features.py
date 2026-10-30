"""CPU tests for Pipeline 2b analytical features, layer roles and the end-to-end Parquet build."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pearl.collect.sampler import PowerSample
from pearl.collect.traces import Segment, write_combo
from pearl.dataset import build as build_mod
from pearl.dataset.features import ANALYTICAL_COLUMNS, dtype_bytes, fine_features
from pearl.dataset.structure import Leaf, ModelStructure, classify_role


def _structure() -> ModelStructure:
    leaves = [
        Leaf("model.embed_tokens", "embedding", 32 * 4, 8 * 4, None, None),
        Leaf("model.layers.0.self_attn.q_proj", "q_proj", 4 * 4, 8 * 4, 4, 4),
        Leaf("model.layers.0.self_attn.o_proj", "o_proj", 4 * 4, 8 * 4, 4, 4),
        Leaf("model.rotary_emb", "rotary", 0, None, None, None),
    ]
    return ModelStructure("toy/model", trace_tokens=8, n_heads=2, n_kv_heads=1, d_head=2, leaves=leaves)


@pytest.mark.parametrize(("name", "module_type", "role"), [
    ("model.layers.0.self_attn.q_proj", "Linear", "q_proj"),
    ("model.layers.0.self_attn.qkv_proj", "Linear", "qkv_proj"),
    ("transformer.h.0.attn.c_proj", "Conv1D", "o_proj"),
    ("transformer.h.0.mlp.c_proj", "Conv1D", "down_proj"),
    ("model.layers.0.mlp.act_fn", "SiLU", "activation"),
    ("model.layers.0.input_layernorm", "Qwen3RMSNorm", "norm"),
    ("model.rotary_emb", "Qwen3RotaryEmbedding", "rotary"),
    ("model.embed_tokens", "Embedding", "embedding"),
    ("lm_head", "Linear", "lm_head"),
])
def test_classify_role(name: str, module_type: str, role: str) -> None:
    assert classify_role(name, module_type) == role


def test_dtype_bytes() -> None:
    assert (dtype_bytes("fp32"), dtype_bytes("BF16"), dtype_bytes("fp16")) == (4, 2, 2)
    with pytest.raises(ValueError, match="unknown precision"):
        dtype_bytes("fp4")


def test_fine_features_prefill_and_decode() -> None:
    f = fine_features(_structure(), batch_size=2, prompt_len=3, max_new_tokens=3, precision="fp16")
    assert len(f) == 4 * 3 and set(ANALYTICAL_COLUMNS) <= set(f.columns)
    row = f.set_index(["layer_name", "decode_step"])
    q0 = row.loc[("model.layers.0.self_attn.q_proj", 0)]
    assert (q0["n_tokens"], q0["q_len"], q0["context_len"]) == (6, 3, 3)
    assert q0["macs"] == 4 * 4 * 6 and q0["FLOPs_analytical"] == 2 * 4 * 4 * 6
    assert (q0["Bytes_input"], q0["Bytes_output"], q0["Bytes_weights"]) == (6 * 4 * 2, 6 * 4 * 2, 16 * 2)
    assert q0["operational_intensity"] == pytest.approx(96 / (48 + 48 + 32))
    assert math.isnan(q0["attention_core_macs"]) and math.isnan(q0["kv_cache_bytes"])
    o2 = row.loc[("model.layers.0.self_attn.o_proj", 2)]
    assert (o2["n_tokens"], o2["q_len"], o2["context_len"]) == (2, 1, 5)
    assert o2["attention_core_macs"] == 2 * 2 * 2 * 1 * 5 * 2
    assert o2["kv_cache_bytes"] == 2 * 2 * 1 * 5 * 2 * 2
    emb = row.loc[("model.embed_tokens", 0)]
    assert emb["macs"] == 0 and emb["Bytes_output"] == 6 * 4 * 2 and emb["operational_intensity"] == 0
    rot = row.loc[("model.rotary_emb", 1)]
    assert rot["Bytes_weights"] == 0 and np.isnan([rot["Bytes_input"], rot["dram_bytes"],
                                                    rot["operational_intensity"]]).all()


def _write_toy_combo(traces: Path, name: str, fwd: int = 2) -> None:
    t0 = 1_000_000_000
    layers = [(1, "model.embed_tokens", "Embedding"), (2, "model.layers.0.self_attn.q_proj", "Linear"),
              (3, "model.layers.0.self_attn.o_proj", "Linear"), (4, "model.rotary_emb", "RotaryEmbedding")]
    timings, cur = [], t0
    for f in range(fwd):
        lyrs = []
        for step in range(2):
            for _, lname, ltype in layers:
                lyrs.append({"layer_name": lname, "layer_type": ltype, "step_id": step,
                             "start_ns": cur, "end_ns": cur + 400_000})
                cur += 500_000
        timings.append({"forward_idx": f, "layers": lyrs})
    samples = [PowerSample(t0 + i * 1_000_000, 100.0 + i) for i in range(-2, (cur - t0) // 1_000_000 + 3)]
    run_config = {"workload": {"model_name": "toy/model", "batch_size": 2, "sequence_length": 3,
                               "max_new_tokens": 2, "precision": "fp16", "revision": None}}
    meta = {"power_source": "samples", "interval_ms": 1}
    seg = Segment(0, samples, timings, run_config, meta, (cur - t0) * 1e-9, fwd, 0.0, t0, cur)
    write_combo(traces / name, layers, [seg])


def test_build_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_mod, "extract_structure", lambda model_id, revision: _structure())
    for session in ("2026-01-01_00-00-00", "2026-01-02_00-00-00"):
        _write_toy_combo(tmp_path / session / build_mod.TRACES_SUBDIR, "toy__model_bs2_seq3_tok2_fp16")
    out = build_mod.build(tmp_path)
    table = pq.read_table(out)
    assert table.schema.equals(build_mod.FINE_SCHEMA)
    df = table.to_pandas()
    assert len(df) == 8 and set(df["session"]) == {"2026-01-02_00-00-00"}
    assert df["firing_idx"].tolist() == list(range(1, 9))
    assert df["decode_step"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert df["phase"].tolist() == ["prefill"] * 4 + ["decode"] * 4
    assert df["role"].tolist()[:4] == ["embedding", "q_proj", "o_proj", "rotary"]
    assert (df["forward_passes"] == 2).all() and (df["n_params"] == _structure().total_params).all()
    assert df["duration_s"].to_numpy() == pytest.approx(np.full(8, 400e-6 * 2), rel=1e-6)
    # 1 ms samples at 100 + i W: forward 0 spans 0-3.9 ms, forward 1 spans 4-7.9 ms
    block_mj = [101 + 102 + 103 + 104 * 0.9, 105 + 106 + 107 + 108 * 0.9]
    assert df["energy_aligned_end_j"].sum() == pytest.approx(np.mean(block_mj) * 1e-3, rel=1e-6)
    assert df["measured_full_forward_j"].iloc[0] == pytest.approx(104.0 * 8e-3 / 2, rel=1e-6)
    meta = json.loads(build_mod.sidecar_path(out).read_text())
    assert (meta["n_runs"], meta["n_rows"], meta["failed"]) == (1, 8, [])
    with pytest.raises(FileExistsError):
        build_mod.build(tmp_path)


def test_build_records_failed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_mod, "extract_structure", lambda model_id, revision: _structure())
    traces = tmp_path / "s1" / build_mod.TRACES_SUBDIR
    _write_toy_combo(traces, "ok")
    _write_toy_combo(traces, "bad")
    seg_path = traces / "bad" / "segments.parquet"
    seg = pd.read_parquet(seg_path)
    cfg = json.loads(seg.loc[0, "run_config_json"])
    cfg["workload"]["batch_size"] = 4
    seg["run_config_json"] = json.dumps(cfg)
    seg["sampler_power_source"] = "usage"
    seg.to_parquet(seg_path)
    out = build_mod.build(tmp_path)
    meta = json.loads(build_mod.sidecar_path(out).read_text())
    assert meta["n_runs"] == 1 and len(meta["failed"]) == 1
    assert "got 'usage'" in meta["failed"][0]["error"]
