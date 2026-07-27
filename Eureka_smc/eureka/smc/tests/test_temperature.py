"""Unit tests for the SMC temperature schedule (plan §9)."""

from __future__ import annotations

import numpy as np
import pytest

from eureka.smc.temperature import (
    ess,
    ess_from_log_weights,
    find_next_lambda,
    log_incremental_weights,
)


# --------------------------------------------------------------------------- ESS


def test_ess_uniform_weights_equals_n():
    # delta_lambda = 0 -> all log-weights zero -> uniform -> ESS == N.
    for n in (1, 2, 4, 8):
        rewards = np.random.default_rng(0).normal(size=n)
        assert ess(rewards, 0.0, beta_target=10.0) == pytest.approx(n)


def test_ess_equal_rewards_equals_n():
    # Equal rewards -> equal weights for ANY delta_lambda -> ESS == N.
    rewards = np.full(6, 0.42)
    for delta in (0.01, 0.3, 1.0):
        assert ess(rewards, delta, beta_target=20.0) == pytest.approx(6)


def test_ess_one_hot_limit_equals_one():
    # One reward dominates hugely -> weight concentrates on one particle -> ESS -> 1.
    logw = np.array([0.0, -1e3, -1e3, -1e3])
    assert ess_from_log_weights(logw) == pytest.approx(1.0)


def test_ess_hand_computed_two_particles():
    # logw = [0, log(3)] -> w = [1, 3]; ESS = (4)^2 / (1 + 9) = 16/10 = 1.6.
    logw = np.array([0.0, np.log(3.0)])
    assert ess_from_log_weights(logw) == pytest.approx(1.6)


def test_ess_no_overflow_large_beta():
    rewards = np.array([0.0, 1.0])
    val = ess(rewards, delta_lambda=1.0, beta_target=10_000.0)
    assert np.isfinite(val)
    assert val == pytest.approx(1.0)  # collapses to one effective particle


def test_ess_between_one_and_n():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(2, 12))
        rewards = rng.normal(size=n)
        val = ess(rewards, delta_lambda=float(rng.uniform(0, 1)), beta_target=5.0)
        assert 1.0 - 1e-9 <= val <= n + 1e-9


def test_log_incremental_weights_formula():
    rewards = np.array([0.0, 0.5, 1.0])
    got = log_incremental_weights(rewards, delta_lambda=0.25, beta_target=8.0)
    assert got == pytest.approx(0.25 * 8.0 * rewards)


# ---------------------------------------------------------------- find_next_lambda


def test_find_next_lambda_equal_rewards_takes_full_step():
    # ESS == N always, so the largest capped step is taken.
    rewards = np.full(8, 0.3)
    lam = find_next_lambda(rewards, 0.0, beta_target=5.0, kappa=0.9, max_delta=1 / 3)
    assert lam == pytest.approx(1 / 3)


def test_find_next_lambda_respects_max_delta_cap():
    rewards = np.full(4, 0.0)  # ESS == N for any step
    lam = find_next_lambda(rewards, 0.0, beta_target=5.0, kappa=0.5, max_delta=0.1)
    assert lam == pytest.approx(0.1)


def test_find_next_lambda_never_exceeds_one():
    rewards = np.full(4, 0.1)
    lam = find_next_lambda(rewards, 0.95, beta_target=5.0, kappa=0.5, max_delta=1 / 3)
    assert lam == pytest.approx(1.0)


def test_find_next_lambda_terminal_is_noop():
    rewards = np.array([0.0, 1.0, 0.5])
    assert find_next_lambda(rewards, 1.0, 5.0, 0.5, 1 / 3) == pytest.approx(1.0)
    # lam_prev slightly above 1 (float drift) still no-ops.
    assert find_next_lambda(rewards, 1.0 + 1e-9, 5.0, 0.5, 1 / 3) == pytest.approx(
        1.0 + 1e-9
    )


def test_find_next_lambda_ess_floor_is_satisfied():
    # For a spread reward vector, the returned step must keep ESS >= kappa*N.
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = int(rng.integers(4, 16))
        rewards = rng.normal(size=n)
        kappa = float(rng.uniform(0.5, 0.9))
        lam = find_next_lambda(rewards, 0.0, beta_target=20.0, kappa=kappa, max_delta=1 / 3)
        delta = lam
        # Allow the degenerate tiny-step escape hatch (delta == tol) to dip below.
        if delta > 1e-4 + 1e-9:
            assert ess(rewards, delta, 20.0) >= kappa * n - 1e-6


def test_find_next_lambda_monotone_in_kappa():
    # Higher ESS floor (larger kappa) => smaller or equal step.
    rewards = np.linspace(0.0, 1.0, 8)
    prev = None
    for kappa in (0.5, 0.7, 0.9):
        lam = find_next_lambda(rewards, 0.0, beta_target=20.0, kappa=kappa, max_delta=1 / 3)
        if prev is not None:
            assert lam <= prev + 1e-9
        prev = lam


def test_find_next_lambda_single_particle():
    # N == 1: ESS == 1 == kappa*1 boundary (kappa <= 1) -> full step allowed.
    rewards = np.array([0.7])
    lam = find_next_lambda(rewards, 0.0, beta_target=20.0, kappa=0.9, max_delta=1 / 3)
    assert lam == pytest.approx(1 / 3)


def test_find_next_lambda_negative_rewards_ok():
    rewards = np.array([-3.0, -1.0, -2.5, -0.2])
    lam = find_next_lambda(rewards, 0.0, beta_target=10.0, kappa=0.5, max_delta=1 / 3)
    assert 0.0 < lam <= 1 / 3 + 1e-9


def test_find_next_lambda_extreme_spread_still_progresses():
    # Even when every positive step over-spreads the weights, must move forward.
    rewards = np.array([0.0, 1000.0])
    lam = find_next_lambda(rewards, 0.0, beta_target=20.0, kappa=0.9, max_delta=1 / 3)
    assert lam > 0.0
