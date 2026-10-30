"""``aligned_end`` attribution: per-firing energy from a raw power trace and leaf hook timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def power_knots(time_ns: np.ndarray, power_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw knots: samples sorted by time; duplicate timestamps keep the **last** reading.

    Returns ``(knot_t_ns int64, knot_p_w float64)``.
    """
    order = np.argsort(time_ns, kind="stable")
    t = np.asarray(time_ns, dtype=np.int64)[order]
    p = np.asarray(power_w, dtype=np.float64)[order]
    if t.size < 2:
        return t, p
    keep = np.append(t[1:] != t[:-1], True)
    return t[keep], p[keep]


def backward_hold_energy_j(
    knot_t_ns: np.ndarray, knot_p_w: np.ndarray, start_ns: np.ndarray, end_ns: np.ndarray
) -> np.ndarray:
    """
    ``aligned_end`` energy (J) of each window ``[start_ns, end_ns]``.

    Reading ``P_j`` (timestamped at the end of its averaging period) is held over
    ``(t_{j-1}, t_j]``; the step function is integrated over every window. Parts of a
    window outside ``[t_0, t_last]`` contribute nothing; reversed windows give 0.
    """
    out = np.zeros(np.size(start_ns), dtype=np.float64)
    if knot_t_ns.size < 2 or out.size == 0:
        return out
    t0 = int(knot_t_ns[0])
    rel = (knot_t_ns - t0).astype(np.float64)
    cum = np.concatenate(([0.0], np.cumsum(knot_p_w[1:] * np.diff(rel))))

    def integral(x_ns: np.ndarray) -> np.ndarray:
        x = np.clip((np.asarray(x_ns, dtype=np.int64) - t0).astype(np.float64), 0.0, rel[-1])
        j = np.clip(np.searchsorted(rel, x, side="right"), 1, rel.size - 1)
        return cum[j - 1] + knot_p_w[j] * (x - rel[j - 1])

    return np.maximum(integral(end_ns) - integral(start_ns), 0.0) * 1e-9


def next_start_windows(firings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Next-start windows per forward block: firing ``i`` spans ``[start_i, start_{i+1}]``.

    The last firing of a block ends at its own ``end_ns``. ``firings`` must be ordered
    by ``forward_idx`` then execution order.
    """
    start = firings["start_ns"].to_numpy(np.int64)
    nxt = firings.groupby("forward_idx", sort=False)["start_ns"].shift(-1)
    end = np.where(nxt.isna(), firings["end_ns"].to_numpy(np.int64), nxt.fillna(0).to_numpy(np.int64))
    return start, end.astype(np.int64)


def attribute_firings(firings: pd.DataFrame, time_ns: np.ndarray, power_w: np.ndarray) -> pd.DataFrame:
    """
    Mean ``aligned_end`` energy and duration per firing position over the forward blocks.

    Parameters
    ----------
    firings :
        Hook records with ``forward_idx, layer_idx, start_ns, end_ns`` in execution order.
    time_ns, power_w :
        Power trace of the measured window.

    Returns
    -------
    pandas.DataFrame
        One row per position in a forward block: ``firing_idx`` (1-based), ``layer_idx``,
        ``decode_step`` (occurrence index of the leaf in the block, 0 = prefill),
        ``energy_j`` and raw ``duration_s`` (``end - start``), both averaged over blocks.
        Identity columns come from the first block holding the position.
    """
    fir = firings.sort_values("forward_idx", kind="stable").reset_index(drop=True)
    knot_t, knot_p = power_knots(time_ns, power_w)
    start, end = next_start_windows(fir)
    per = pd.DataFrame({
        "forward_idx": fir["forward_idx"].to_numpy(),
        "layer_idx": fir["layer_idx"].to_numpy(np.int64),
        "position": fir.groupby("forward_idx", sort=False).cumcount().to_numpy(),
        "decode_step": fir.groupby(["forward_idx", "layer_idx"], sort=False).cumcount().to_numpy(),
        "energy_j": backward_hold_energy_j(knot_t, knot_p, start, end),
        "duration_s": np.maximum(fir["end_ns"].to_numpy(np.int64) - start, 0) * 1e-9,
    })
    grouped = per.groupby("position", sort=True)
    out = grouped[["layer_idx", "decode_step"]].first()
    out[["energy_j", "duration_s"]] = grouped[["energy_j", "duration_s"]].mean()
    out = out.reset_index()
    out.insert(0, "firing_idx", out.pop("position") + 1)
    return out
