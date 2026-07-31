"""Unit tests for resampling (systematic + multinomial, plan §9 / new method §2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from eureka.smc.resampling import multinomial_resample, systematic_resample


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


# ---- multinomial_resample（预算 KL 主路径，新方法 §2.5）----

def test_multinomial_length_m_and_in_range():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    idx = multinomial_resample(q, m=6, rng=np.random.default_rng(0))
    assert idx.shape == (6,)                     # 抽 M 个（可 > N）
    assert idx.min() >= 0 and idx.max() < 4


def test_multinomial_reproducible():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    a = multinomial_resample(q, 8, np.random.default_rng(42))
    b = multinomial_resample(q, 8, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_multinomial_long_run_frequency_matches_q():
    # 检查表 §6-9：重采样长期频率与 q 一致。
    q = np.array([0.5, 0.3, 0.15, 0.05])
    rng = np.random.default_rng(0)
    idx = multinomial_resample(q, m=200_000, rng=rng)
    freq = np.bincount(idx, minlength=4) / idx.size
    assert np.allclose(freq, q, atol=0.01)


def test_multinomial_zero_prob_never_selected():
    q = np.array([0.0, 0.6, 0.4, 0.0])
    idx = multinomial_resample(q, m=500, rng=np.random.default_rng(1))
    assert 0 not in idx.tolist() and 3 not in idx.tolist()


def test_multinomial_unnormalised_accepted():
    q = np.array([1.0, 3.0])                     # sum=4
    idx = multinomial_resample(q, m=10_000, rng=np.random.default_rng(2))
    frac1 = (idx == 1).mean()
    assert frac1 == pytest.approx(0.75, abs=0.02)


def test_multinomial_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        multinomial_resample(np.zeros(4), m=4, rng=np.random.default_rng(0))


def test_multinomial_empty_or_zero_m():
    assert multinomial_resample(np.array([0.5, 0.5]), 0, np.random.default_rng(0)).shape == (0,)
    assert multinomial_resample(np.empty(0), 4, np.random.default_rng(0)).shape == (0,)
