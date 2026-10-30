"""
Build ``fine_firings_aligned_end.parquet`` (one row per leaf firing) directly from collected traces.

Input layout (written by ``python -m pearl.collect.run``)::

    <data_dir>/<session>/layer_energy/traces/<combo>/{layers,segments,power_trace,firings}.parquet

Each run (one combo) becomes one row group: ``aligned_end`` energy per firing position,
averaged over the measured forwards, joined to the analytical features on
``(layer_name, decode_step)``. A ``<stem>.meta.json`` sidecar records the schema version,
included runs, failures and units.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing
import os
import time
from collections import deque
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

from pearl.collect.sampler import trapezoid_energy_j
from pearl.collect.traces import load_combo
from pearl.dataset.attribution import attribute_firings
from pearl.dataset.features import ANALYTICAL_COLUMNS, dtype_bytes, fine_features
from pearl.dataset.structure import ModelStructure, StructuralExtractionError, extract_structure

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FORMAT = "fine_firings_aligned_end_v1"
DEFAULT_NAME = "fine_firings_aligned_end.parquet"
TRACES_SUBDIR = Path("layer_energy") / "traces"

_S, _I32, _I64, _F32 = pa.string(), pa.int32(), pa.int64(), pa.float32()
FINE_SCHEMA = pa.schema(
    [(c, _S) for c in ("model_id", "config_tag", "session")]
    + [(c, _I32) for c in ("firing_idx", "layer_idx", "decode_step")]
    + [(c, _S) for c in ("phase", "layer_name", "role", "layer_type")]
    + [(c, _I32) for c in ("batch_size", "prompt_len", "max_new_tokens")]
    + [("precision", _S), ("dtype_bytes", _I32), ("forward_passes", _I32), ("n_params", _I64)]
    + [(c, _F32) for c in ANALYTICAL_COLUMNS]
    + [("energy_aligned_end_j", _F32), ("duration_s", _F32), ("measured_full_forward_j", _F32)]
)
UNITS = {
    "energy_aligned_end_j": "J per forward",
    "measured_full_forward_j": "J per forward (trapezoid over the measured window / forward_passes)",
    "duration_s": "s, mean hook duration of the firing x forward_passes",
}


@dataclass(frozen=True)
class Run:
    """One collected grid point (combo folder)."""

    session: str
    combo_dir: Path
    model_id: str
    revision: str | None
    batch_size: int
    prompt_len: int
    max_new_tokens: int
    precision: str

    @property
    def config_tag(self) -> str:
        return f"bs{self.batch_size}_seq{self.prompt_len}_tok{self.max_new_tokens}_{self.precision}"


def _read_run(session: str, combo_dir: Path) -> Run:
    seg = pd.read_parquet(combo_dir / "segments.parquet", columns=["run_config_json"])
    if seg.empty:
        raise ValueError(f"{combo_dir}: segments.parquet is empty")
    wl = json.loads(seg["run_config_json"].iloc[0])["workload"]
    return Run(session, combo_dir, str(wl["model_name"]), wl.get("revision"), int(wl["batch_size"]),
               int(wl["sequence_length"]), int(wl["max_new_tokens"]), str(wl["precision"]))


def discover_runs(data_dir: Path, sessions: Sequence[str] | None = None) -> list[Run]:
    """
    Runs under ``data_dir`` sorted by ``(model_id, config_tag)``.

    When a ``(model_id, config_tag)`` appears in several sessions, the lexicographically
    latest session is kept and the others are logged as dropped.
    """
    allowed = set(sessions) if sessions is not None else None
    by_key: dict[tuple[str, str], list[Run]] = {}
    for session_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        traces = session_dir / TRACES_SUBDIR
        if (allowed is not None and session_dir.name not in allowed) or not traces.is_dir():
            continue
        for combo in sorted(p for p in traces.iterdir() if (p / "segments.parquet").is_file()):
            run = _read_run(session_dir.name, combo)
            by_key.setdefault((run.model_id, run.config_tag), []).append(run)
    runs: list[Run] = []
    for key, group in sorted(by_key.items()):
        group.sort(key=lambda r: r.session)
        if len(group) > 1:
            _LOG.warning("Duplicate %s %s: keeping session %s, dropping %s",
                         *key, group[-1].session, [r.session for r in group[:-1]])
        runs.append(group[-1])
    return runs


def assemble_run(run: Run, structure: ModelStructure) -> pa.Table:
    """
    One run as an Arrow table matching :data:`FINE_SCHEMA`.

    Raises
    ------
    ValueError
        Not exactly one segment, power source other than ``samples``, empty power trace,
        or a join that would duplicate firings.
    """
    layers, segments = load_combo(run.combo_dir)
    if len(segments) != 1:
        raise ValueError(f"{run.combo_dir}: expected 1 segment, found {len(segments)}")
    seg = segments[0]
    source = seg.meta.get("sampler_power_source")
    if source != "samples":
        raise ValueError(f"{run.combo_dir}: aligned_end needs nvmlDeviceGetSamples traces, got {source!r}")
    if seg.time_ns.size == 0 or seg.firings.empty:
        raise ValueError(f"{run.combo_dir}: no power samples or no firings in the measured window")
    fwd = max(1, int(seg.meta["forward_count"]))
    df = attribute_firings(seg.firings, seg.time_ns, seg.power_w)
    registry = layers[["layer_idx", "layer_name", "layer_type"]].drop_duplicates()
    df = df.merge(registry, on="layer_idx", how="left", validate="m:1")
    feats = fine_features(structure, run.batch_size, run.prompt_len, run.max_new_tokens, run.precision)
    n = len(df)
    df = df.merge(feats, on=["layer_name", "decode_step"], how="left")
    if len(df) != n:
        raise ValueError(f"{run.combo_dir}: analytical features are not unique per (layer_name, decode_step)")
    df = df.assign(
        model_id=run.model_id, config_tag=run.config_tag, session=run.session,
        phase=np.where(df["decode_step"].eq(0), "prefill", "decode"),
        batch_size=run.batch_size, prompt_len=run.prompt_len, max_new_tokens=run.max_new_tokens,
        precision=run.precision, dtype_bytes=dtype_bytes(run.precision), forward_passes=fwd,
        n_params=structure.total_params,
        energy_aligned_end_j=df["energy_j"], duration_s=df["duration_s"] * fwd,
        measured_full_forward_j=trapezoid_energy_j(seg.time_ns, seg.power_w, float(seg.meta["wall_time_s"])) / fwd,
    )
    return pa.Table.from_pandas(df[FINE_SCHEMA.names], schema=FINE_SCHEMA, preserve_index=False, safe=False)


_Job = tuple[Run, ModelStructure]


def _assemble_safe(job: _Job) -> pa.Table | str:
    """Worker entry point: the run table, or the error message."""
    try:
        return assemble_run(*job)
    except (OSError, ValueError, KeyError) as exc:
        return repr(exc)


def _iter_results(jobs: Sequence[_Job], workers: int) -> Iterator[tuple[Run, pa.Table | str]]:
    """Results in job order; at most ``2 * workers`` runs in flight."""
    if workers <= 1:
        yield from ((job[0], _assemble_safe(job)) for job in jobs)
        return
    it = iter(jobs)
    # spawn: fork() of a parent holding torch / arrow thread pools can deadlock
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        pending = deque((job, ex.submit(_assemble_safe, job)) for job in itertools.islice(it, 2 * workers))
        while pending:
            job, fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, ex.submit(_assemble_safe, nxt)))
            yield job[0], fut.result()


def _failure(run: Run, error: str) -> dict[str, str]:
    return {"session": run.session, "model_id": run.model_id, "config_tag": run.config_tag, "error": error}


def _structures(runs: Sequence[Run], failed: list[dict[str, str]]) -> list[_Job]:
    """Pair each run with its model structure (one trace per model/revision)."""
    cache: dict[tuple[str, str | None], ModelStructure | str] = {}
    jobs: list[_Job] = []
    for run in runs:
        key = (run.model_id, run.revision)
        if key not in cache:
            try:
                cache[key] = extract_structure(run.model_id, run.revision)
            except StructuralExtractionError as exc:
                _LOG.warning("Structure extraction failed: %s", exc)
                cache[key] = str(exc)
        structure = cache[key]
        if isinstance(structure, str):
            failed.append(_failure(run, f"structure extraction failed: {structure}"))
        else:
            jobs.append((run, structure))
    return jobs


def _write_runs(tmp: Path, jobs: Sequence[_Job], workers: int, failed: list[dict[str, str]]) -> list[Run]:
    """Stream every run into ``tmp`` (one row group per run); return the written runs."""
    written: list[Run] = []
    with pq.ParquetWriter(tmp, FINE_SCHEMA, compression="zstd") as writer:
        for i, (run, result) in enumerate(_iter_results(jobs, workers), 1):
            if isinstance(result, str):
                _LOG.warning("Skipping %s %s: %s", run.model_id, run.config_tag, result)
                failed.append(_failure(run, result))
            else:
                writer.write_table(result, row_group_size=max(1, result.num_rows))
                written.append(run)
            if i % 50 == 0:
                _LOG.info("%d/%d runs assembled", i, len(jobs))
    return written


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sidecar_path(parquet_path: Path) -> Path:
    """``<stem>.meta.json`` next to the Parquet file."""
    return parquet_path.with_name(f"{parquet_path.stem}.meta.json")


def build(
    data_dir: Path,
    *,
    sessions: Sequence[str] | None = None,
    out_path: Path | None = None,
    max_runs: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> Path:
    """
    Build the fine-grained ``aligned_end`` Parquet file from the traces under ``data_dir``.

    Parameters
    ----------
    data_dir :
        Collection root holding one folder per session.
    sessions :
        Session folder names to include (default: all).
    out_path :
        Destination (default ``<data_dir>/fine_firings_aligned_end.parquet``).
    max_runs :
        Keep only the first runs (sorted by model, config), for smoke builds.
    workers :
        Processes assembling runs in parallel; the main process writes.
    overwrite :
        Replace an existing file.

    Returns
    -------
    Path
        The Parquet file; its sidecar is :func:`sidecar_path`.

    Raises
    ------
    FileExistsError
        Destination exists and ``overwrite`` is false.
    FileNotFoundError
        No run found.
    RuntimeError
        No run could be assembled.
    """
    data_dir = Path(data_dir)
    out = Path(out_path) if out_path is not None else data_dir / DEFAULT_NAME
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild")
    runs = discover_runs(data_dir, sessions)[:max_runs]
    if not runs:
        raise FileNotFoundError(f"no traces under {data_dir}/<session>/{TRACES_SUBDIR} (sessions={sessions})")
    t0 = time.monotonic()
    failed: list[dict[str, str]] = []
    jobs = _structures(runs, failed)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        written = _write_runs(tmp, jobs, workers, failed)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not written:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"no run could be assembled ({len(failed)} failures, first: {failed[:1]})")
    os.replace(tmp, out)
    n_rows = pq.ParquetFile(out).metadata.num_rows
    _write_sidecar(sidecar_path(out), {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attribution": "aligned_end",
        "units": UNITS,
        "sessions": sorted({r.session for r in written}),
        "n_runs": len(written),
        "n_rows": n_rows,
        "max_runs": max_runs,
        "build_seconds": round(time.monotonic() - t0, 1),
        "runs": [{"session": r.session, "model_id": r.model_id, "config_tag": r.config_tag} for r in written],
        "failed": failed,
    })
    _LOG.info("Wrote %s: %d runs, %d rows, %d failures", out, len(written), n_rows, len(failed))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build fine_firings_aligned_end.parquet from collected traces.")
    parser.add_argument("data_dir", type=Path, help="collection root (holds <session>/layer_energy/traces/)")
    parser.add_argument("--sessions", nargs="+", default=None, help="session folder names (default: all)")
    parser.add_argument("--out", type=Path, default=None, help=f"output Parquet (default: <data_dir>/{DEFAULT_NAME})")
    parser.add_argument("--max-runs", type=int, default=None, help="keep only the first N runs (smoke build)")
    parser.add_argument("--workers", type=int, default=1, help="parallel assembly processes")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(build(args.data_dir, sessions=args.sessions, out_path=args.out, max_runs=args.max_runs,
                workers=args.workers, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
