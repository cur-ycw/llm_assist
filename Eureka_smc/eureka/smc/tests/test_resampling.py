"""Unit tests for systematic resampling (plan §9)."""

from __future__ import annotations

import numpy as np
import pytest

from eureka.smc.resampling import systematic_resample


def test_reproducible_under_fixed_seed():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    a = systematic_resample(w, np.random.default_rng(123))
    b = systematic_resample(w, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_indices_in_range_and_length_n():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    idx = systematic_resample(w, np.random.default_rng(0))
    assert idx.shape == (4,)
    assert idx.min() >= 0 and idx.max() < 4


def test_uniform_weights_pick_each_once():
    # With equal weights the comb lands exactly one position per cell.
    w = np.full(8, 1.0 / 8)
    idx = systematic_resample(w, np.random.default_rng(7))
    assert sorted(idx.tolist()) == list(range(8))


def test_zero_weight_particle_never_selected():
    w = np.array([0.0, 0.5, 0.5, 0.0])
    for seed in range(20):
        idx = systematic_resample(w, np.random.default_rng(seed))
        assert 0 not in idx.tolist()
        assert 3 not in idx.tolist()


def test_dominant_weight_dominates_counts():
    w = np.array([0.9, 0.05, 0.05])
    # Expected copies of particle 0 ~ 0.9 * N.
    counts = np.zeros(3)
    trials = 2000
    for seed in range(trials):
        idx = systematic_resample(w, np.random.default_rng(seed))
        counts += np.bincount(idx, minlength=3)
    frac0 = counts[0] / counts.sum()
    assert frac0 == pytest.approx(0.9, abs=0.02)


def test_unnormalised_weights_accepted():
    w = np.array([1.0, 2.0, 3.0, 4.0])  # sum = 10, not normalised
    idx = systematic_resample(w, np.random.default_rng(0))
    assert idx.shape == (4,)
    assert idx.min() >= 0 and idx.max() < 4


def test_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        systematic_resample(np.zeros(4), np.random.default_rng(0))


def test_empty_input_returns_empty():
    idx = systematic_resample(np.empty(0), np.random.default_rng(0))
    assert idx.shape == (0,)
