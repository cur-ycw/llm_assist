"""Adaptive-temperature (ESS-driven) annealing schedule for the SMC controller.

Migrated from the SMCEvolve framework (paper §Algorithm 1; source
``smcevolve/temperature.py``) onto Eureka reward-function search. See the migration
plan ``Eureka_SMCEvolve_框架迁移实施计划.md`` §6.2.

A reward-function candidate ``x`` is a particle with search score ``R(x)``. The
bridge distribution is ``pi_lambda(x) ∝ p0(x) * exp(lambda * beta_target * R(x))``
with the inverse temperature reparameterised as ``beta_t = lambda_t * beta_target``,
``lambda`` climbing from 0 to 1. Between two consecutive stages the incremental
importance weight of a particle carried over from a *resampled* (hence uniform)
population is

    w_i ∝ exp[(lambda_t - lambda_{t-1}) * beta_target * R_i].

Everything here is pure numpy — no Isaac Gym, no LLM — so it is unit-testable and
runs in any environment.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "log_incremental_weights",
    "ess_from_log_weights",
    "ess",
    "find_next_lambda",
]


def log_incremental_weights(
    rewards: np.ndarray, delta_lambda: float, beta_target: float
) -> np.ndarray:
    """Un-normalised log incremental weights ``delta_lambda * beta_target * R_i``.

    Only the *increment* over the previous stage matters: immediately after a
    resample every particle weight is reset to uniform, so the log-weight for the
    next stage is exactly the tilt applied to the reward vector (plan §6.2).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    return delta_lambda * beta_target * rewards


def ess_from_log_weights(logw: np.ndarray) -> float:
    """Effective sample size from un-normalised log-weights, via log-sum-exp.

    ``ESS = (sum_i w_i)^2 / sum_i w_i^2`` computed after subtracting ``max(logw)``
    so it never overflows even for very large ``beta`` (plan §6.2 / §9). Ranges in
    ``[1, N]``: uniform weights give ``N``, a single dominant weight gives ``1``.
    """
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    w = np.exp(logw - logw.max())  # in (0, 1], stable
    s = w.sum()
    return float(s * s / np.square(w).sum())


def ess(rewards: np.ndarray, delta_lambda: float, beta_target: float) -> float:
    """ESS of the incremental-weight population for a candidate ``delta_lambda``."""
    return ess_from_log_weights(
        log_incremental_weights(rewards, delta_lambda, beta_target)
    )


def find_next_lambda(
    rewards: np.ndarray,
    lam_prev: float,
    beta_target: float,
    kappa: float,
    max_delta: float,
    tol: float = 1e-4,
) -> float:
    """Largest ``lambda_t`` keeping ESS at or above ``kappa * N``.

    Bisects the step ``delta in (0, min(max_delta, 1 - lam_prev)]`` for the largest
    value whose incremental-weight ESS is ``>= kappa * N``. ESS is monotonically
    non-increasing in ``delta`` (a larger temperature jump spreads the weights),
    which makes the bisection well defined.

    ``max_delta`` is normally set by the caller to ``1 / min_smc_iterations`` so at
    least that many annealing stages are taken (plan §6.2). Returns a value in
    ``[lam_prev, 1.0]``.

    Boundary behaviour:
      * ``1 - lam_prev <= 0`` (already terminal): returns ``lam_prev``.
      * all-equal rewards / ``N == 1``: ESS == N for every ``delta`` → returns the
        full capped step ``lam_prev + hi``.
      * even the smallest positive step already violates the floor (degenerate,
        highly-spread scores): returns ``lam_prev + tol`` to still make forward
        progress rather than stalling at ``lam_prev`` forever.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    n = rewards.size
    target = kappa * n

    hi = min(max_delta, 1.0 - lam_prev)
    if hi <= 0.0:
        return float(lam_prev)

    # Fast path: the largest allowed step still satisfies the ESS floor.
    if ess(rewards, hi, beta_target) >= target:
        return float(lam_prev + hi)

    # Degenerate path: even a minimal step over-spreads the weights. Take a tiny
    # forward step so the schedule still progresses toward lambda = 1.
    if ess(rewards, tol, beta_target) < target:
        return float(lam_prev + min(tol, hi))

    # Bisect for the boundary delta where ESS crosses the floor.
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ess(rewards, mid, beta_target) >= target:
            lo = mid
        else:
            hi = mid
    return float(lam_prev + lo)
