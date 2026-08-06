"""KL/rESS Full controller tests."""

from __future__ import annotations

import numpy as np
import pytest

from eureka.smc.kl_controller import (
    boltzmann_gibbs,
    boltzmann_gibbs_dimensionless,
    effective_parents,
    entropy_effective_parents,
    kl_to_uniform,
    normalize_potential,
    relative_ess,
    resolve,
    solve_alpha_for_ress,
    solve_lambda_for_ress,
    target_effective_parents,
    target_relative_ess,
    tau_budget,
)


def test_second_moment_effective_size_and_relative_ess():
    q = np.array([0.5, 0.25, 0.25])
    assert effective_parents(q) == pytest.approx(1.0 / (0.5**2 + 2 * 0.25**2))
    assert relative_ess(q) == pytest.approx(effective_parents(q) / 3)
    assert entropy_effective_parents(q) != pytest.approx(effective_parents(q))


def test_probability_and_extreme_scale_are_finite():
    scores = np.array([100.0] * 7 + [20000.0])
    for lam in (0.0, 1.0, 1e3, 1e6):
        q = boltzmann_gibbs(scores, lam)
        assert np.all(np.isfinite(q))
        assert q.sum() == pytest.approx(1.0)
    step = resolve(scores, b=5, B=64, n=8, k_min=2, eta=1.0)
    assert np.all(np.isfinite(step.q))
    assert np.isfinite(step.lam)


def test_budget_target_uses_eta_and_delayed_progress():
    n, k_min, b, B = 16, 2, 64, 80
    assert tau_budget(b / B, k_min, n) == pytest.approx(0.825)
    assert target_relative_ess(b, B, n, k_min, eta=1.0, progress_prev=0.0) == pytest.approx(0.825)
    assert target_relative_ess(b, B, n, k_min, eta=1.0, progress_prev=1.0) < 0.825
    assert target_relative_ess(b, B, n, k_min, eta=1.0, progress_prev=-1.0) > 0.825
    first = resolve(scores=np.linspace(0, 1, 8), b=b, B=B, n=8, k_min=2,
                    gamma=0.01, eta=1.0, progress_prev=0.0)
    second = resolve(scores=np.linspace(0, 1, 8), b=b, B=B, n=8, k_min=2,
                     gamma=100.0, eta=1.0, progress_prev=0.0)
    assert first.tau_target == pytest.approx(second.tau_target)


def test_resolve_targets_ress_not_kl():
    scores = np.linspace(0.0, 1.0, 8)
    step = resolve(scores, b=32, B=64, n=8, k_min=2, eta=1.0, progress_prev=0.0)
    assert step.relative_ess == pytest.approx(step.tau_feasible, abs=2e-5)
    assert step.k_eff == pytest.approx(8.0 * step.relative_ess)
    assert step.kl_actual == pytest.approx(kl_to_uniform(step.q))


def test_first_round_is_not_uniform_when_init_is_in_budget():
    scores = np.linspace(0.0, 1.0, 16)
    step = resolve(scores, b=64, B=80, n=16, k_min=2, eta=1.0, progress_prev=0.0)
    assert step.h == pytest.approx(0.8)
    assert step.tau_target == pytest.approx(0.825)
    assert step.lam > 0.0
    assert step.relative_ess == pytest.approx(step.tau_target, abs=2e-5)


def test_all_equal_scores_are_uniform():
    scores = np.full(8, 3.14)
    step = resolve(scores, b=64, B=80, n=8, k_min=2, eta=1.0)
    assert step.lam == 0.0
    assert step.relative_ess == pytest.approx(1.0)
    assert step.kl_actual == pytest.approx(0.0)
    assert np.allclose(step.q, 1.0 / 8)


def test_lambda_solver_precision_and_monotonicity():
    scores = np.linspace(0.0, 1.0, 8)
    previous = 1.0
    for target in (0.9, 0.7, 0.4, 0.2):
        lam, actual, saturated = solve_lambda_for_ress(scores, target, tol=1e-9)
        assert not saturated
        assert actual == pytest.approx(target, abs=2e-5)
        assert lam >= 0.0
        assert actual <= previous + 1e-9
        previous = actual


def test_tied_max_limits_ress():
    scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    step = resolve(scores, b=0, B=64, n=8, k_min=1, eta=1.0)
    assert step.m_ties == 4
    assert step.tau_feasible >= 4 / 8 - 1e-12
    assert step.relative_ess >= 4 / 8 - 1e-5


def test_positive_affine_invariance():
    scores = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 2.0])
    scaled = 137.0 * scores - 42.0
    a, _, _ = solve_lambda_for_ress(scores, 0.4, tol=1e-9)
    b, _, _ = solve_lambda_for_ress(scaled, 0.4, tol=1e-9)
    assert np.allclose(boltzmann_gibbs(scores, a), boltzmann_gibbs(scaled, b), atol=1e-4)
    assert b == pytest.approx(a / 137.0, rel=2e-2)


def test_dimensionless_controller_is_affine_invariant():
    scores = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 2.0])
    scaled = 137.0 * scores - 42.0
    normalized, span = normalize_potential(scores)
    normalized_scaled, scaled_span = normalize_potential(scaled)
    assert span == pytest.approx(2.0)
    assert scaled_span == pytest.approx(137.0 * span)
    assert np.allclose(normalized, normalized_scaled)

    step = resolve(scores, b=32, B=64, n=scores.size, k_min=2, eta=1.0)
    scaled_step = resolve(scaled, b=32, B=64, n=scores.size, k_min=2, eta=1.0)
    assert scaled_step.alpha == pytest.approx(step.alpha, rel=1e-6, abs=1e-6)
    assert np.allclose(scaled_step.q, step.q, atol=2e-6)
    assert scaled_step.lambda_equivalent == pytest.approx(
        step.lambda_equivalent / 137.0, rel=2e-4
    )


def test_dimensionless_controller_zero_span_is_uniform():
    scores = np.full(8, 3.14)
    normalized, span = normalize_potential(scores)
    assert span == 0.0
    assert np.all(normalized == 0.0)
    step = resolve(scores, b=64, B=80, n=8, k_min=2, eta=1.0)
    assert step.alpha == 0.0
    assert step.lambda_equivalent == 0.0
    assert np.allclose(step.q, 1.0 / 8.0)
