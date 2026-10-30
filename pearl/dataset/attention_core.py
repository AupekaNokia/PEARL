"""
Rebuild the fine ``aligned_end`` Parquet with a non-hooked ``attention_core`` firing per layer.

Gates (a run failing one is skipped and listed in the sidecar):

- every row except the pre-core leaf matches the reference fine Parquet;
- ``new pre-core + attention_core == reference pre-core`` per layer and step.

Usage::

    python -m pearl.dataset.build results/collect
    python -m pearl.dataset.attention_core results/collect
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pearl.collect.traces import load_combo
from pearl.dataset.attribution import attribute_firings
from pearl.dataset.build import DEFAULT_NAME, FINE_SCHEMA, discover_runs, sidecar_path
from pearl.dataset.structure import StructuralExtractionError, extract_structure

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FORMAT = "fine_attention_core_aligned_end_v1"
OUT_NAME = "fine_attention_core_aligned_end.parquet"

ENERGY_COL = "energy_aligned_end_j"
REF_ENERGY_COL = "energy_aligned_end_ref_j"
CORE_LEAF = "attention_core"
CORE_ROLE = "attention_core"
CORE_TYPE = "AttentionCore"
CORE_IDX_OFFSET = 10_000
REL_TOL = 1e-4
ABS_TOL_J = 1e-7

LAYER_PATH_RE = re.compile(r"^(?:.+\.)?layers\.(\d+)\.(.+)$")
OUT_SCHEMA = pa.schema(
    list(FINE_SCHEMA) + [pa.field(REF_ENERGY_COL, pa.float32()), pa.field("is_pre_core", pa.bool_())]
)
_KEYS = ["layer_name", "decode_step"]

AttnDims = tuple[int, int, int]  # (n_heads, n_kv_heads, d_head)


@dataclass(frozen=True)
class Job:
    """One combo trace folder matched to a reference-Parquet run."""

    combo_dir: str
    session: str
    model_id: str
    config_tag: str
    dims: AttnDims | None
    ref_path: str


def core_name(o_proj_name: str) -> str:
    """``...self_attn.o_proj`` -> ``...self_attn.attention_core``."""
    return o_proj_name[: -len("o_proj")] + CORE_LEAF


def insert_attention_core(
    firings: pd.DataFrame, names: dict[int, str]
) -> tuple[pd.DataFrame, set[str], dict[int, str]]:
    """
    Insert a virtual firing before every ``self_attn.o_proj`` firing of each forward block.

    The firing spans ``[end_ns(previous firing), start_ns(o_proj)]`` when the previous
    firing of the block is a ``self_attn`` leaf of the same layer; otherwise nothing is
    inserted. Core firings get ``layer_idx = CORE_IDX_OFFSET + layer number``.

    Parameters
    ----------
    firings :
        ``forward_idx, layer_idx, start_ns, end_ns`` (plus any other column) in execution order.
    names :
        ``layer_idx -> layer_name``.

    Returns
    -------
    tuple
        ``(firings with core rows, names of the pre-core leaves, core layer_idx -> name)``.
    """
    fir = firings.reset_index(drop=True)
    layer_name = fir["layer_idx"].map(names)
    parts = layer_name.str.extract(LAYER_PATH_RE)
    num, sub = parts[0], parts[1]
    is_attn = sub.str.startswith("self_attn.", na=False)
    hit = (
        sub.eq("self_attn.o_proj")
        & fir["forward_idx"].eq(fir["forward_idx"].shift())
        & is_attn.shift(fill_value=False)
        & num.eq(num.shift())
    ).to_numpy()
    prev_end = fir["end_ns"].shift().fillna(0).to_numpy(np.int64)
    o_start = fir["start_ns"].to_numpy(np.int64)
    core = fir[hit].copy()
    core["layer_idx"] = CORE_IDX_OFFSET + num[hit].astype(np.int64)
    core["start_ns"] = np.minimum(prev_end[hit], o_start[hit])
    core["end_ns"] = o_start[hit]
    if "duration_ns" in core:
        core["duration_ns"] = core["end_ns"] - core["start_ns"]
    core_names = dict(zip(core["layer_idx"].astype(int), layer_name[hit].map(core_name)))
    pre_core = set(layer_name.shift()[hit])
    order = np.concatenate([2 * np.flatnonzero(hit), 2 * np.arange(len(fir)) + 1])
    out = pd.concat([core, fir], ignore_index=True).iloc[np.argsort(order, kind="stable")]
    return out.reset_index(drop=True), pre_core, core_names


def core_features(o_rows: pd.DataFrame, dims: AttnDims | None) -> pd.DataFrame:
    """
    Analytical features of the attention core, derived from the same-step ``o_proj`` rows.

    ``macs`` is the score + context matmul count already carried by ``o_proj``
    (``attention_core_macs``). Bytes read are Q, K, V of the new tokens plus the KV
    cache; bytes written are the context. Byte features are NaN when the model
    exposes no attention dims.
    """
    df = o_rows.copy()
    df["macs"] = df["attention_core_macs"]
    df["FLOPs_analytical"] = 2.0 * df["macs"]
    df["Bytes_weights"] = 0.0
    if dims is None:
        for c in ("Bytes_input", "Bytes_output", "Bytes_total", "dram_bytes", "operational_intensity"):
            df[c] = np.nan
        return df
    h, h_kv, d_h = dims
    per_tok = df["n_tokens"] * d_h * df["dtype_bytes"]
    df["Bytes_input"] = per_tok * (h + 2 * h_kv)
    df["Bytes_output"] = per_tok * h
    df["Bytes_total"] = df["Bytes_input"] + df["Bytes_output"]
    df["dram_bytes"] = df["Bytes_total"] + df["kv_cache_bytes"]
    df["operational_intensity"] = df["macs"] / df["dram_bytes"].replace(0.0, np.nan)
    return df


def _rel_err(a: pd.Series, b: pd.Series) -> float:
    diff = (a - b).abs() - ABS_TOL_J
    return float((diff.clip(lower=0.0) / b.abs().clip(lower=ABS_TOL_J)).max()) if len(a) else 0.0


def _reattribute(job: Job) -> tuple[pd.DataFrame, set[str]]:
    """Insert the core firings and run aligned_end; one row per firing position."""
    layers, segments = load_combo(Path(job.combo_dir))
    if len(segments) != 1:
        raise ValueError(f"expected 1 segment, found {len(segments)}")
    seg = segments[0]
    names = dict(zip(layers["layer_idx"].astype(int), layers["layer_name"]))
    firings, pre_core, core_names = insert_attention_core(seg.firings, names)
    new = attribute_firings(firings, seg.time_ns, seg.power_w)
    new["layer_name"] = new["layer_idx"].astype(int).map({**names, **core_names})
    new = new.rename(columns={"energy_j": "energy_new", "duration_s": "duration_new"})
    return new, pre_core


def _gate(merged: pd.DataFrame, core: pd.DataFrame) -> dict[str, float]:
    """Unchanged-row agreement and pre-core conservation (max relative errors)."""
    other = merged[~merged["is_pre_core"]]
    pre = merged[merged["is_pre_core"]].assign(prefix=lambda d: d["layer_name"].str.extract(r"^(.*self_attn\.)")[0])
    cons = pre.merge(core.assign(prefix=core["layer_name"].str[: -len(CORE_LEAF)]),
                     on=["prefix", "decode_step"], suffixes=("", "_core"), validate="one_to_one")
    return {
        "max_rel_unchanged": _rel_err(other["energy_new"], other[ENERGY_COL]),
        "max_rel_conservation": _rel_err(cons["energy_new"] + cons["energy_new_core"], cons[ENERGY_COL]),
        "n_pre_core_rows": float(len(pre)),
        "n_conserved_rows": float(len(cons)),
    }


def _core_rows(merged: pd.DataFrame, core: pd.DataFrame, dims: AttnDims | None) -> pd.DataFrame:
    o_rows = merged[merged["layer_name"].str.endswith("self_attn.o_proj")].copy()
    o_rows["layer_name"] = o_rows["layer_name"].map(core_name)
    rows = core_features(o_rows, dims).drop(columns=["firing_idx", "layer_idx", "energy_new"])
    rows = rows.merge(core[["layer_name", "decode_step", "firing_idx", "layer_idx", "energy_new", "duration_new"]],
                      on=_KEYS, how="inner", suffixes=("_o", ""), validate="one_to_one")
    rows[ENERGY_COL] = rows["energy_new"]
    # same scale as the reference rows: mean hook duration x forward_passes
    rows["duration_s"] = rows["duration_new"] * rows["forward_passes"]
    rows[REF_ENERGY_COL] = np.nan
    rows["is_pre_core"] = False
    rows["role"], rows["layer_type"] = CORE_ROLE, CORE_TYPE
    return rows


def process_job(job: Job) -> tuple[pa.Table | None, dict[str, Any]]:
    """Re-attribute one run; return ``(table or None if a gate fails, stats)``."""
    stats: dict[str, Any] = {"model_id": job.model_id, "config_tag": job.config_tag, "session": job.session}
    try:
        new, pre_core = _reattribute(job)
        filt = [("model_id", "==", job.model_id), ("config_tag", "==", job.config_tag)]
        ref = pq.read_table(job.ref_path, filters=filt).to_pandas()
        is_core = new["layer_name"].str.endswith(CORE_LEAF)
        merged = ref.merge(new[~is_core], on=_KEYS, how="left", suffixes=("_ref", ""), validate="one_to_one")
        if merged["energy_new"].isna().any() or len(merged) != len(ref):
            raise ValueError(f"{int(merged['energy_new'].isna().sum())} reference firings not re-attributed")
        merged["is_pre_core"] = merged["layer_name"].isin(pre_core)
        stats.update(_gate(merged, new[is_core]))
        stats["pre_core_roles"] = sorted(merged.loc[merged["is_pre_core"], "role"].unique().tolist())
        stats["n_core_rows"] = int(is_core.sum())
    except (OSError, ValueError, KeyError) as exc:
        stats["error"] = repr(exc)
        return None, stats
    if max(stats["max_rel_unchanged"], stats["max_rel_conservation"]) > REL_TOL:
        stats["error"] = "gate failed"
        return None, stats
    merged["layer_idx"] = merged["layer_idx_ref"]
    merged[REF_ENERGY_COL] = merged[ENERGY_COL]
    merged[ENERGY_COL] = np.where(merged["is_pre_core"], merged["energy_new"], merged[ENERGY_COL].astype("float64"))
    out = pd.concat([merged, _core_rows(merged, new[is_core], job.dims)], ignore_index=True)
    out = out.sort_values("firing_idx", kind="stable")
    stats["core_share"] = float(out.loc[out["role"] == CORE_ROLE, ENERGY_COL].sum() / out[ENERGY_COL].sum())
    return pa.Table.from_pandas(out[OUT_SCHEMA.names], schema=OUT_SCHEMA, preserve_index=False, safe=False), stats


def _attn_dims(model_id: str, revision: str | None) -> AttnDims | None:
    try:
        s = extract_structure(model_id, revision)
    except StructuralExtractionError as exc:
        _LOG.warning("structure extraction failed: %s", exc)
        return None
    if s.n_heads and s.d_head:
        return int(s.n_heads), int(s.n_kv_heads or s.n_heads), int(s.d_head)
    return None


def build_jobs(data_dir: Path, ref_path: Path) -> list[Job]:
    """Match combo folders to reference runs (same model, config and session)."""
    ref = pq.read_table(ref_path, columns=["model_id", "config_tag", "session"]).to_pandas()
    ref_runs = set(ref.drop_duplicates().itertuples(index=False, name=None))
    sessions = sorted({s for _, _, s in ref_runs})
    runs = [r for r in discover_runs(data_dir, sessions) if (r.model_id, r.config_tag, r.session) in ref_runs]
    _LOG.info("matched %d combos to %d reference runs", len(runs), len(ref_runs))
    dims: dict[tuple[str, str | None], AttnDims | None] = {}
    for r in runs:
        if (r.model_id, r.revision) not in dims:
            dims[(r.model_id, r.revision)] = _attn_dims(r.model_id, r.revision)
    return [Job(str(r.combo_dir), r.session, r.model_id, r.config_tag, dims[(r.model_id, r.revision)],
                str(ref_path)) for r in runs]


def _iter_results(jobs: Sequence[Job], workers: int) -> Iterator[tuple[pa.Table | None, dict[str, Any]]]:
    if workers <= 1:
        yield from map(process_job, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        yield from ex.map(process_job, jobs, chunksize=1)


def build_attention_core(
    data_dir: Path, *, ref_path: Path | None = None, out_path: Path | None = None,
    workers: int = 1, overwrite: bool = False,
) -> Path:
    """
    Write ``fine_attention_core_aligned_end.parquet`` (one row group per run) and its sidecar.

    ``ref_path`` defaults to ``<data_dir>/fine_firings_aligned_end.parquet`` (the output of
    :mod:`pearl.dataset.build`); ``out_path`` to ``<data_dir>/fine_attention_core_aligned_end.parquet``.

    Raises
    ------
    FileExistsError
        Destination exists and ``overwrite`` is false.
    RuntimeError
        No run passed the gates.
    """
    data_dir = Path(data_dir)
    ref = Path(ref_path) if ref_path is not None else data_dir / DEFAULT_NAME
    out = Path(out_path) if out_path is not None else data_dir / OUT_NAME
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild")
    t0 = time.monotonic()
    jobs = build_jobs(data_dir, ref)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    all_stats: list[dict[str, Any]] = []
    try:
        with pq.ParquetWriter(tmp, OUT_SCHEMA, compression="zstd") as writer:
            for i, (table, stats) in enumerate(_iter_results(jobs, workers), 1):
                all_stats.append(stats)
                if table is None:
                    _LOG.warning("skipped %s %s: %s", stats["model_id"], stats["config_tag"], stats["error"])
                else:
                    writer.write_table(table, row_group_size=max(1, table.num_rows))
                if i % 50 == 0:
                    _LOG.info("%d/%d runs", i, len(jobs))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    ok = [s for s in all_stats if "error" not in s]
    if not ok:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"no run passed the gates ({len(all_stats)} runs, first: {all_stats[:1]})")
    os.replace(tmp, out)
    n_rows = pq.ParquetFile(out).metadata.num_rows
    meta = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attribution": "aligned_end",
        "reference": ref.name,
        "units": {ENERGY_COL: "J per forward", REF_ENERGY_COL: "J per forward, before the split (NaN on core rows)"},
        "gates": {"rel_tol": REL_TOL, "abs_tol_j": ABS_TOL_J,
                  "max_rel_unchanged": max(s["max_rel_unchanged"] for s in ok),
                  "max_rel_conservation": max(s["max_rel_conservation"] for s in ok)},
        "n_runs": len(ok),
        "n_rows": n_rows,
        "build_seconds": round(time.monotonic() - t0, 1),
        "runs": ok,
        "failed": [s for s in all_stats if "error" in s],
    }
    tmp_meta = sidecar_path(out).with_name(sidecar_path(out).name + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_meta, sidecar_path(out))
    _LOG.info("Wrote %s: %d runs, %d rows, %d skipped", out, len(ok), n_rows, len(all_stats) - len(ok))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="collection root (holds <session>/layer_energy/traces/)")
    parser.add_argument("--ref", type=Path, default=None, help=f"reference Parquet (default: <data_dir>/{DEFAULT_NAME})")
    parser.add_argument("--out", type=Path, default=None, help=f"output Parquet (default: <data_dir>/{OUT_NAME})")
    parser.add_argument("--workers", type=int, default=1, help="parallel re-attribution processes")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(build_attention_core(args.data_dir, ref_path=args.ref, out_path=args.out,
                               workers=args.workers, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
