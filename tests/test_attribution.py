"""CPU tests for Pipeline 2b attribution: knots, backward-hold integration, per-position means."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pearl.dataset.attribution import attribute_firings, backward_hold_energy_j, next_start_windows, power_knots


def _loop_backward_hold(kt: np.ndarray, kp: np.ndarray, cs: np.ndarray, ce: np.ndarray) -> np.ndarray:
    out = np.zeros(cs.size)
    for j in range(1, kt.size):
        out += kp[j] * np.maximum(0.0, np.minimum(ce, kt[j]) - np.maximum(cs, kt[j - 1])) * 1e-9
    return out


def test_power_knots_sorts_and_keeps_last_duplicate() -> None:
    t, p = power_knots(np.array([30, 10, 20, 20]), np.array([3.0, 1.0, 2.0, 2.5]))
    assert t.tolist() == [10, 20, 30]
    assert p.tolist() == [1.0, 2.5, 3.0]


def test_backward_hold_holds_reading_over_previous_interval() -> None:
    kt = np.array([0, 10, 20], dtype=np.int64)
    kp = np.array([999.0, 50.0, 100.0])
    e = backward_hold_energy_j(kt, kp, np.array([5, 0, -10, 25, 15]), np.array([15, 20, 5, 40, 10]))
    assert e == pytest.approx([(50 * 5 + 100 * 5) * 1e-9, (500 + 1000) * 1e-9, 50 * 5 * 1e-9, 0.0, 0.0])


def test_backward_hold_matches_per_knot_loop() -> None:
    rng = np.random.default_rng(0)
    t0 = 3_520_189_375_406_782
    kt = t0 + np.cumsum(rng.integers(5_000_000, 25_000_000, 200))
    kp = rng.uniform(60.0, 300.0, kt.size)
    cs = np.sort(rng.integers(kt[0] - 10**7, kt[-1] + 10**7, 5000))
    ce = cs + rng.integers(0, 2_000_000, cs.size)
    np.testing.assert_allclose(backward_hold_energy_j(kt, kp, cs, ce), _loop_backward_hold(kt, kp, cs, ce),
                               rtol=1e-9, atol=1e-15)


def test_backward_hold_needs_two_knots() -> None:
    assert backward_hold_energy_j(np.array([5]), np.array([1.0]), np.array([0]), np.array([10])).tolist() == [0.0]


def _firings() -> pd.DataFrame:
    rows = [  # forward 0: a, b, a(step 1); forward 1 identical layout shifted by 1000 ns
        (0, 1, 0, 100), (0, 2, 150, 300), (0, 1, 400, 450),
        (1, 1, 1000, 1200), (1, 2, 1200, 1300), (1, 1, 1500, 1600),
    ]
    return pd.DataFrame(rows, columns=["forward_idx", "layer_idx", "start_ns", "end_ns"])


def test_next_start_windows_per_block() -> None:
    cs, ce = next_start_windows(_firings())
    assert cs.tolist() == [0, 150, 400, 1000, 1200, 1500]
    assert ce.tolist() == [150, 400, 450, 1200, 1500, 1600]


def test_attribute_firings_means_over_blocks() -> None:
    t = np.array([0, 2000], dtype=np.int64)
    out = attribute_firings(_firings(), t, np.array([0.0, 1e9]))  # 1e9 W held -> 1 J per ns
    assert out["firing_idx"].tolist() == [1, 2, 3]
    assert out["layer_idx"].tolist() == [1, 2, 1]
    assert out["decode_step"].tolist() == [0, 0, 1]
    assert out["energy_j"].tolist() == pytest.approx([(150 + 200) / 2, (250 + 300) / 2, (50 + 100) / 2])
    assert out["duration_s"].tolist() == pytest.approx([150e-9, 125e-9, 75e-9])


def test_attribute_firings_conserves_block_energy_under_constant_power() -> None:
    fir = _firings()
    out = attribute_firings(fir, np.array([-10, 5000]), np.array([80.0, 80.0]))
    span_ns = [450 - 0, 1600 - 1000]
    assert out["energy_j"].sum() == pytest.approx(80.0 * np.mean(span_ns) * 1e-9)
