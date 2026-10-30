"""Analytical per-(leaf, generation step) features: MACs, FLOPs, bytes, operational intensity, KV cache."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pearl.dataset.structure import ModelStructure

ANALYTICAL_COLUMNS: tuple[str, ...] = (
    "macs", "FLOPs_analytical", "attention_core_macs", "kv_cache_bytes",
    "Bytes_input", "Bytes_output", "Bytes_weights", "Bytes_total", "dram_bytes",
    "operational_intensity", "n_tokens", "q_len", "context_len",
)

_DTYPE_BYTES: dict[str, int] = {
    "fp32": 4, "float32": 4, "float": 4, "tf32": 4,
    "fp16": 2, "float16": 2, "half": 2, "bf16": 2, "bfloat16": 2,
    "int8": 1, "uint8": 1, "fp8": 1, "e4m3": 1, "e5m2": 1,
}


def dtype_bytes(precision: str) -> int:
    """
    Bytes per element of a precision label (case-insensitive).

    Raises
    ------
    ValueError
        Unknown label.
    """
    key = precision.strip().lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]


def _steps(prompt_len: int, max_new_tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(decode_step, q_len, context_len)``: prefill (``q = S``) then ``T - 1`` decode steps (``q = 1``)."""
    t = np.arange(max(int(max_new_tokens), 1), dtype=np.int64)
    q_len = np.where(t == 0, prompt_len, 1)
    return t, q_len, prompt_len + t


def _floats(values: list[int | None]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _leaf_arrays(structure: ModelStructure) -> dict[str, np.ndarray]:
    """Per-leaf arrays; ``NaN`` where the trace captured nothing."""
    leaves = structure.leaves
    width = [
        lf.out_features if lf.out_features is not None
        else (int(lf.activations) // structure.trace_tokens if lf.activations else None)
        for lf in leaves
    ]
    return {
        "name": np.array([lf.layer_name for lf in leaves], dtype=object),
        "role": np.array([lf.role for lf in leaves], dtype=object),
        "in": _floats([lf.in_features for lf in leaves]),
        "out": _floats([lf.out_features for lf in leaves]),
        "width": _floats(width),
        "params": _floats([lf.params for lf in leaves]),
        "is_o_proj": np.array([lf.role == "o_proj" for lf in leaves], dtype=bool),
    }


def fine_features(
    structure: ModelStructure, batch_size: int, prompt_len: int, max_new_tokens: int, precision: str
) -> pd.DataFrame:
    """
    Analytical features for every traced leaf at every generation step.

    Step 0 is the prefill (``q_len = context_len = S``); step ``t`` in ``1..T-1`` is a
    decode step (``q_len = 1``, ``context_len = S + t``). Matmul leaves get
    ``in * out * B * q_len`` MACs; other leaves get 0 MACs and activation bytes from their
    traced per-token width (``NaN`` when unknown). Attention score/context MACs and KV-cache
    bytes are carried on each block's ``o_proj`` row and are ``NaN`` elsewhere.

    Returns
    -------
    pandas.DataFrame
        Keys ``layer_name, decode_step``, plus ``role`` and :data:`ANALYTICAL_COLUMNS` (float64).
    """
    step, q_len, ctx = _steps(prompt_len, max_new_tokens)
    n_leaf, db, b = len(structure.leaves), float(dtype_bytes(precision)), float(batch_size)
    leaf = {k: np.tile(v, step.size) for k, v in _leaf_arrays(structure).items()}
    q, c = np.repeat(q_len.astype(np.float64), n_leaf), np.repeat(ctx.astype(np.float64), n_leaf)
    is_mm = ~np.isnan(leaf["in"]) & ~np.isnan(leaf["out"])
    n_tokens = b * q
    macs = np.where(is_mm, leaf["in"] * leaf["out"] * n_tokens, 0.0)
    b_in = np.where(is_mm, n_tokens * leaf["in"] * db, n_tokens * leaf["width"] * db)
    b_out = np.where(is_mm, n_tokens * leaf["out"] * db, n_tokens * leaf["width"] * db)
    b_w = leaf["params"] * db
    dram = b_in + b_out + b_w
    attn_ok = leaf["is_o_proj"] & bool(structure.n_heads) & bool(structure.d_head)
    n_heads, d_head = float(structure.n_heads or 0), float(structure.d_head or 0)
    n_kv = float(structure.n_kv_heads or structure.n_heads or 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        oi = np.where(dram == 0, np.nan, macs / dram)
    return pd.DataFrame({
        "layer_name": leaf["name"],
        "decode_step": np.repeat(step, n_leaf),
        "role": leaf["role"],
        "macs": macs,
        "FLOPs_analytical": 2.0 * macs,
        "attention_core_macs": np.where(attn_ok, 2.0 * b * n_heads * q * c * d_head, np.nan),
        "kv_cache_bytes": np.where(attn_ok, 2.0 * b * n_kv * c * d_head * db, np.nan),
        "Bytes_input": b_in,
        "Bytes_output": b_out,
        "Bytes_weights": b_w,
        "Bytes_total": dram,
        "dram_bytes": dram,
        "operational_intensity": oi,
        "n_tokens": n_tokens,
        "q_len": q,
        "context_len": c,
    })
