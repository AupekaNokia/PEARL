"""CPU tests for the attention-core leaf insertion and its analytical features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pearl.dataset.attention_core import CORE_IDX_OFFSET, _core_rows, core_features, insert_attention_core
from pearl.dataset.attribution import attribute_firings


def _firings(names: list[str], gap_ns: int = 100, n_forwards: int = 1) -> tuple[pd.DataFrame, dict[int, str]]:
    rows, idx = [], {n: i + 1 for i, n in enumerate(names)}
    for f in range(n_forwards):
        t = f * 100_000
        for name in names:
            rows.append({"forward_idx": f, "layer_idx": idx[name], "start_ns": t, "end_ns": t + 100})
            t += 100 + gap_ns
    return pd.DataFrame(rows), {i: n for n, i in idx.items()}


def _leaf_names(fir: pd.DataFrame, names: dict[int, str], core_names: dict[int, str]) -> list[str]:
    return fir["layer_idx"].map({**names, **core_names}).str.rsplit(".", n=1).str[-1].tolist()


def test_insert_attention_core_llama_order() -> None:
    fir, names = _firings([f"model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")])
    out, pre, core_names = insert_attention_core(fir, names)
    assert _leaf_names(out, names, core_names) == ["q_proj", "k_proj", "v_proj", "attention_core", "o_proj"]
    core = out.iloc[3]
    assert core["layer_idx"] == CORE_IDX_OFFSET
    assert (core["start_ns"], core["end_ns"]) == (out.iloc[2]["end_ns"], out.iloc[4]["start_ns"])
    assert pre == {"model.layers.0.self_attn.v_proj"}


def test_insert_attention_core_gemma_pre_core_is_k_norm() -> None:
    fir, names = _firings([f"model.language_model.layers.3.self_attn.{p}"
                           for p in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj")])
    out, pre, core_names = insert_attention_core(fir, names)
    assert core_names[int(out.iloc[5]["layer_idx"])] == "model.language_model.layers.3.self_attn.attention_core"
    assert pre == {"model.language_model.layers.3.self_attn.k_norm"}


def test_insert_attention_core_skips_without_same_layer_attention_leaf() -> None:
    fir, names = _firings(["model.layers.0.input_layernorm", "model.layers.0.self_attn.o_proj"])
    out, pre, _ = insert_attention_core(fir, names)
    assert len(out) == 2 and not pre


def test_insert_attention_core_does_not_cross_forward_blocks() -> None:
    fir, names = _firings(["model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.v_proj"], n_forwards=2)
    out, pre, _ = insert_attention_core(fir, names)
    assert len(out) == 4 and not pre


def test_split_conserves_pre_core_energy_and_keeps_other_rows() -> None:
    fir, names = _firings([f"model.layers.0.self_attn.{p}" for p in ("q_proj", "v_proj", "o_proj")], gap_ns=400)
    t = np.arange(-100, 1500, 50, dtype=np.int64)
    p = 50.0 + t / 100.0
    ref = attribute_firings(fir, t, p)
    split, _, core_names = insert_attention_core(fir, names)
    new = attribute_firings(split, t, p)
    e_ref = dict(zip(ref["layer_idx"].map(names), ref["energy_j"]))
    e_new = dict(zip(new["layer_idx"].map({**names, **core_names}), new["energy_j"]))
    core = e_new["model.layers.0.self_attn.attention_core"]
    assert core > 0.0
    assert e_new["model.layers.0.self_attn.v_proj"] + core == pytest.approx(e_ref["model.layers.0.self_attn.v_proj"])
    for name in ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj"):
        assert e_new[name] == pytest.approx(e_ref[name])


def _o_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "attention_core_macs": [1000.0], "kv_cache_bytes": [64.0],
        "n_tokens": [2.0], "dtype_bytes": [2], "macs": [7.0],
    })


def test_core_features_with_dims() -> None:
    df = core_features(_o_rows(), (4, 2, 8))
    row = df.iloc[0]
    assert row["macs"] == 1000.0 and row["FLOPs_analytical"] == 2000.0
    assert row["Bytes_weights"] == 0.0
    assert row["Bytes_input"] == 2 * 8 * 2 * (4 + 2 * 2)
    assert row["Bytes_output"] == 2 * 8 * 2 * 4
    assert row["dram_bytes"] == row["Bytes_input"] + row["Bytes_output"] + 64.0
    assert row["operational_intensity"] == pytest.approx(1000.0 / row["dram_bytes"])


def test_core_features_without_dims_are_nan() -> None:
    df = core_features(_o_rows(), None)
    assert np.isnan(df.iloc[0]["Bytes_input"]) and np.isnan(df.iloc[0]["dram_bytes"])
    assert df.iloc[0]["macs"] == 1000.0


def test_core_rows_duration_on_reference_scale() -> None:
    merged = pd.DataFrame({
        "layer_name": ["model.layers.0.self_attn.o_proj"], "decode_step": [0], "firing_idx": [5],
        "layer_idx": [4], "energy_new": [1.0], "forward_passes": [10], "duration_s": [0.003],
        "attention_core_macs": [1000.0], "kv_cache_bytes": [64.0], "n_tokens": [2.0], "dtype_bytes": [2],
    })
    core = pd.DataFrame({
        "layer_name": ["model.layers.0.self_attn.attention_core"], "decode_step": [0], "firing_idx": [4],
        "layer_idx": [10_000], "energy_new": [0.5], "duration_new": [2e-4],
    })
    row = _core_rows(merged, core, None).iloc[0]
    assert row["duration_s"] == pytest.approx(2e-4 * 10)
    assert row["energy_aligned_end_j"] == 0.5
