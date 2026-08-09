"""预算—进展自适应的 KL 父代选择控制器。

Full 控制器以二阶矩相对有效样本数
``rESS(q) = 1 / (N * sum(q**2))`` 为目标。预算给出基础目标，上一轮的
``Gamma`` 只经由延迟反馈修正下一轮目标：

``tau = clip(tau_budget(h) * exp(-eta * h * Gamma_prev), k_min/N, 1)``。

选择分布仍为原始任务分数 ``J`` 上的 Boltzmann--Gibbs 分布；默认
``W=uniform, U=J``。小写 ``gamma`` 是旧预算幂日程的兼容参数，不参与
Full 控制器计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "ControllerStep",
    "target_effective_parents",
    "tau_budget",
    "target_relative_ess",
    "boltzmann_gibbs",
    "boltzmann_gibbs_dimensionless",
    "normalize_potential",
    "shannon_entropy",
    "entropy_effective_parents",
    "effective_parents",
    "relative_ess",
    "kl_to_uniform",
    "feasible_target",
    "feasible_relative_ess",
    "solve_lambda",
    "solve_lambda_for_ress",
    "solve_alpha_for_ress",
    "resolve",
]

_EPS = 1e-12


def _probabilities(q: Sequence[float], name: str = "q") -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError(f"{name} must be a non-empty finite non-negative vector")
    total = float(q.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return q / total


def _scores(scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a non-empty finite vector")
    return scores


def tau_budget(h: float, k_min: float, n: int, k_cap: Optional[float] = None) -> float:
    """Budget-only rESS target, linearly interpolated from ``k_min/n`` to ``k_cap/n``.

    ``k_cap`` upper-bounds the target effective-parent count.  It defaults to ``n``
    (legacy: interpolate up to the whole resident population).  Pass
    ``k_cap=children_per_round`` so the target is expressed against the number of
    offspring actually drawn each round (M), not the population (N): with N=16 draws
    of M=8, a target above 8 is unrealizable and leaves the first rounds' selection
    a no-op.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    h = float(np.clip(h, 0.0, 1.0))
    tau_floor = float(np.clip(k_min, 1.0, n)) / float(n)
    cap = float(n) if k_cap is None else float(np.clip(k_cap, float(np.clip(k_min, 1.0, n)), n))
    tau_ceil = cap / float(n)
    return float(tau_floor + (tau_ceil - tau_floor) * h)


def target_relative_ess(
    b: float,
    B: float,
    n: int,
    k_min: float,
    eta: float = 1.0,
    progress_prev: float = 0.0,
    k_cap: Optional[float] = None,
) -> float:
    """Return the delayed-progress target rESS.

    Positive preceding progress makes the next-round parent allocation more
    concentrated; negative progress preserves more diversity.  ``progress_prev``
    is clipped to the natural ``[-1, 1]`` range of Gamma.  ``k_cap`` upper-bounds
    the target effective-parent count (defaults to ``n``); with ``k_cap`` set, even
    the diversity-preserving branch relaxes only up to ``k_cap/n``, not to one.
    """
    if B <= 0.0 or n <= 0:
        raise ValueError("B and n must be positive")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    if not np.isfinite(progress_prev):
        raise ValueError("progress_prev must be finite")
    h = float(np.clip(b / B, 0.0, 1.0))
    floor = float(np.clip(k_min, 1.0, n)) / float(n)
    cap = float(n) if k_cap is None else float(np.clip(k_cap, float(np.clip(k_min, 1.0, n)), n))
    ceil = cap / float(n)
    base = tau_budget(h, k_min, n, k_cap)
    gamma_prev = float(np.clip(progress_prev, -1.0, 1.0))
    return float(np.clip(base * np.exp(-eta * h * gamma_prev), floor, ceil))


def target_effective_parents(
    b: float,
    B: float,
    n: int,
    k_min: float,
    gamma: Optional[float] = None,
    *,
    eta: float = 1.0,
    progress_prev: float = 0.0,
) -> float:
    """Legacy budget-power helper; Full ``resolve`` does not call it.

    The argument remains available for old analysis code.  Full control must use
    :func:`target_relative_ess`, where ``eta`` and delayed ``Gamma`` are active.
    """
    if gamma is not None:
        if B <= 0 or n <= 0 or not np.isfinite(gamma) or gamma < 0:
            raise ValueError("B and n must be positive; gamma must be non-negative")
        k_floor = float(np.clip(k_min, 1.0, n))
        frac = float(np.clip(b / B, 0.0, 1.0))
        return float(np.clip(k_floor + (n - k_floor) * frac ** gamma, k_floor, n))
    return float(n * target_relative_ess(b, B, n, k_min, eta, progress_prev))


def boltzmann_gibbs(
    scores: Sequence[float], lam: float, weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Return ``q(i) ∝ W_i exp(lambda * J_i)`` with stable log-sum-exp.

    Omitting ``weights`` gives the Full default ``W=uniform``.  Zero prior weights
    are supported and remain zero.
    """
    scores = _scores(scores)
    n = scores.size
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and non-negative")
    prior = np.full(n, 1.0 / n) if weights is None else _probabilities(weights, "weights")
    if prior.size != n:
        raise ValueError("weights must match scores")
    positive = prior > 0.0
    logits = np.full(n, -np.inf, dtype=np.float64)
    logits[positive] = np.log(prior[positive]) + float(lam) * scores[positive]
    shift = float(np.max(logits))
    masses = np.zeros(n, dtype=np.float64)
    masses[positive] = np.exp(logits[positive] - shift)
    return masses / masses.sum()


def normalize_potential(potential: Sequence[float]) -> tuple[np.ndarray, float]:
    """Map a finite potential to ``[-1, 0]`` and return its range.

    The exact zero-range case is handled without introducing an arbitrary
    numerical threshold: a constant potential carries no ranking signal.
    """
    values = _scores(potential)
    maximum = float(values.max())
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros_like(values), 0.0
    normalized = (values - maximum) / span
    return normalized, span


def boltzmann_gibbs_dimensionless(
    normalized_potential: Sequence[float],
    alpha: float,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return ``q(i) ∝ W_i exp(alpha * U_tilde_i)`` stably."""
    values = _scores(normalized_potential)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    prior = np.full(values.size, 1.0 / values.size) if weights is None else _probabilities(weights, "weights")
    if prior.size != values.size:
        raise ValueError("weights must match normalized_potential")
    positive = prior > 0.0
    logits = np.full(values.size, -np.inf, dtype=np.float64)
    logits[positive] = np.log(prior[positive]) + float(alpha) * values[positive]
    shift = float(np.max(logits))
    masses = np.zeros(values.size, dtype=np.float64)
    masses[positive] = np.exp(logits[positive] - shift)
    return masses / masses.sum()


def shannon_entropy(q: Sequence[float]) -> float:
    """Shannon entropy ``H(q)`` (nats), retained as a diagnostic."""
    q = _probabilities(q)
    nz = q > _EPS
    return float(-(q[nz] * np.log(q[nz])).sum())


def effective_parents(q: Sequence[float]) -> float:
    """Second-moment effective parent count ``1 / sum(q**2)``."""
    q = _probabilities(q)
    return float(1.0 / np.square(q).sum())


def entropy_effective_parents(q: Sequence[float]) -> float:
    """Shannon-entropy effective count, retained as a diagnostic only."""
    return float(np.exp(shannon_entropy(q)))


def relative_ess(q: Sequence[float]) -> float:
    """Second-moment relative ESS ``1 / (N * sum(q**2))``."""
    q = _probabilities(q)
    return float(1.0 / (q.size * np.square(q).sum()))


def kl_to_uniform(q: Sequence[float]) -> float:
    """``D_KL(q || U_N)`` retained as a diagnostic field."""
    q = _probabilities(q)
    return float(np.log(q.size) - shannon_entropy(q)) if q.size > 1 else 0.0


def feasible_target(
    scores: Sequence[float], k_star: float, n: Optional[int] = None
) -> tuple[float, float, int]:
    """Legacy KL-target feasibility tuple ``(K_feas, delta_feas, m_ties)``."""
    scores = _scores(scores)
    n = scores.size if n is None else int(n)
    if n != scores.size:
        raise ValueError("n must equal the number of scores")
    m_ties = int(np.count_nonzero(np.abs(scores - scores.max()) <= _EPS))
    k_feas = float(np.clip(max(float(k_star), float(m_ties)), 1.0, n))
    return k_feas, float(np.log(n / k_feas)), m_ties


def feasible_relative_ess(
    scores: Sequence[float], target_ress: float, n: Optional[int] = None
) -> tuple[float, float, int]:
    """Tie-aware rESS feasibility tuple ``(tau_feasible, K_feasible, m_ties)``."""
    scores = _scores(scores)
    n = scores.size if n is None else int(n)
    if n != scores.size:
        raise ValueError("n must equal the number of scores")
    m_ties = int(np.count_nonzero(np.abs(scores - scores.max()) <= _EPS))
    tau_feasible = max(float(target_ress), float(m_ties) / n)
    return tau_feasible, float(n * tau_feasible), m_ties


def solve_lambda_for_ress(
    scores: Sequence[float],
    target_ress: float,
    weights: Optional[Sequence[float]] = None,
    lam_hi: float = 1.0,
    max_expand: int = 60,
    tol: float = 1e-6,
) -> tuple[float, float, bool]:
    """Solve ``rESS(softmax(log W + lambda*J)) = target_ress`` by bisection."""
    scores = _scores(scores)
    n = scores.size
    if not np.isfinite(target_ress) or not 0.0 < target_ress <= 1.0:
        raise ValueError("target_ress must lie in (0, 1]")
    q0 = boltzmann_gibbs(scores, 0.0, weights)
    ress0 = relative_ess(q0)
    if target_ress >= ress0 - tol or n == 1:
        return 0.0, ress0, False
    support = q0 > 0.0
    if np.ptp(scores[support]) <= _EPS:
        return 0.0, ress0, True

    def ress_at(lam: float) -> float:
        return relative_ess(boltzmann_gibbs(scores, lam, weights))

    hi = float(lam_hi)
    if not np.isfinite(hi) or hi <= 0.0:
        raise ValueError("lam_hi must be finite and positive")
    prev = ress_at(hi)
    for _ in range(max_expand):
        if prev <= target_ress:
            break
        nxt = ress_at(hi * 2.0)
        hi *= 2.0
        if abs(nxt - prev) < tol:
            prev = nxt
            break
        prev = nxt
    if prev > target_ress + tol:
        return hi, prev, True

    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if ress_at(mid) > target_ress:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    return float(lam), float(ress_at(lam)), False


def solve_alpha_for_ress(
    normalized_potential: Sequence[float],
    target_ress: float,
    weights: Optional[Sequence[float]] = None,
    alpha_hi: float = 1.0,
    max_expand: int = 60,
    tol: float = 1e-6,
) -> tuple[float, float, bool]:
    """Solve rESS in the dimensionless ``alpha`` coordinate."""
    values = _scores(normalized_potential)
    n = values.size
    if not np.isfinite(target_ress) or not 0.0 < target_ress <= 1.0:
        raise ValueError("target_ress must lie in (0, 1]")
    q0 = boltzmann_gibbs_dimensionless(values, 0.0, weights)
    ress0 = relative_ess(q0)
    if target_ress >= ress0 - tol or n == 1:
        return 0.0, ress0, False
    support = q0 > 0.0
    if np.ptp(values[support]) <= _EPS:
        return 0.0, ress0, True

    def ress_at(alpha: float) -> float:
        return relative_ess(boltzmann_gibbs_dimensionless(values, alpha, weights))

    hi = float(alpha_hi)
    if not np.isfinite(hi) or hi <= 0.0:
        raise ValueError("alpha_hi must be finite and positive")
    prev = ress_at(hi)
    for _ in range(max_expand):
        if prev <= target_ress:
            break
        nxt = ress_at(hi * 2.0)
        hi *= 2.0
        if abs(nxt - prev) < tol:
            prev = nxt
            break
        prev = nxt
    if prev > target_ress + tol:
        return hi, prev, True

    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if ress_at(mid) > target_ress:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0
    return float(alpha), float(ress_at(alpha)), False


def solve_lambda(
    scores: Sequence[float], delta: float, lam_hi: float = 1.0,
    max_expand: int = 60, tol: float = 1e-6,
) -> tuple[float, float, bool]:
    """Legacy KL solver retained for source compatibility.

    Full calls :func:`solve_lambda_for_ress`; this function does not control Full.
    """
    scores = _scores(scores)
    if scores.size <= 1:
        return 0.0, 0.0, False
    if delta <= tol or np.ptp(scores) <= _EPS:
        return 0.0, 0.0, False
    target_kl = float(np.clip(delta, 0.0, np.log(scores.size)))
    def kl_at(lam: float) -> float:
        return kl_to_uniform(boltzmann_gibbs(scores, lam))
    hi = float(lam_hi)
    prev = kl_at(hi)
    for _ in range(max_expand):
        if prev >= target_kl:
            break
        nxt = kl_at(hi * 2.0)
        hi *= 2.0
        if abs(nxt - prev) < tol:
            prev = nxt
            break
        prev = nxt
    if prev < target_kl - tol:
        return hi, prev, True
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if kl_at(mid) < target_kl:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    return float(lam), float(kl_at(lam)), False


@dataclass
class ControllerStep:
    """Complete, auditable solution of one Full controller round."""

    h: float
    tau_budget: float
    progress_prev: float
    tau_target: float
    tau_feasible: float
    relative_ess: float
    k_eff: float
    kl_actual: float
    alpha: float
    potential_span: float
    normalized_potential: np.ndarray
    lambda_equivalent: float
    q: np.ndarray
    max_q: float
    saturated: bool
    # Legacy diagnostics kept because island's existing event fields read them.
    k_star: float
    k_feas: float
    m_ties: int
    delta_req: float
    delta_feas: float

    @property
    def lam(self) -> float:
        """Deprecated raw-scale diagnostic; equivalent to ``alpha / span``."""
        return self.lambda_equivalent


def resolve(
    scores: Sequence[float],
    b: float,
    B: float,
    n: int,
    k_min: float,
    gamma: Optional[float] = None,
    lam_hi: float = 1.0,
    tol: float = 1e-6,
    *,
    eta: float = 1.0,
    progress_prev: float = 0.0,
    weights: Optional[Sequence[float]] = None,
    U: Optional[Sequence[float]] = None,
    k_cap: Optional[float] = None,
) -> ControllerStep:
    """Resolve one round from budget plus the *previous* round's Gamma.

    ``gamma`` is accepted for old callers (including the untouched island), but is
    ignored.  ``U`` is the optional explicit Full utility and defaults to ``scores``.
    ``k_cap`` upper-bounds the target effective-parent count (defaults to ``n``); set
    it to ``children_per_round`` so the schedule is anchored to the M offspring drawn
    per round instead of the resident population N.
    """
    del gamma
    raw_scores = _scores(scores)
    utility = raw_scores if U is None else _scores(U)
    if utility.size != raw_scores.size or n != raw_scores.size:
        raise ValueError("n, scores, and U must have the same length")
    h = float(np.clip(b / B, 0.0, 1.0)) if B > 0.0 else (_ for _ in ()).throw(ValueError("B must be positive"))
    tau_b = tau_budget(h, k_min, n, k_cap)
    tau_target = target_relative_ess(b, B, n, k_min, eta, progress_prev, k_cap)
    normalized, span = normalize_potential(utility)
    if span == 0.0:
        tau_feas, k_feas, m_ties = 1.0, float(n), n
        alpha, ress, saturated = 0.0, 1.0, False
        q = boltzmann_gibbs_dimensionless(normalized, alpha, weights)
    else:
        tau_feas, k_feas, m_ties = feasible_relative_ess(utility, tau_target, n)
        alpha, ress, saturated = solve_alpha_for_ress(
            normalized, tau_feas, weights, alpha_hi=lam_hi, tol=tol
        )
        q = boltzmann_gibbs_dimensionless(normalized, alpha, weights)
    lambda_equivalent = alpha / span if span > 0.0 else 0.0
    k_star = float(n * tau_target)
    return ControllerStep(
        h=h, tau_budget=tau_b, progress_prev=float(np.clip(progress_prev, -1.0, 1.0)),
        tau_target=tau_target, tau_feasible=tau_feas, relative_ess=relative_ess(q),
        k_eff=effective_parents(q), kl_actual=kl_to_uniform(q), alpha=alpha,
        potential_span=span, normalized_potential=normalized,
        lambda_equivalent=lambda_equivalent, q=q, max_q=float(q.max()),
        saturated=saturated, k_star=k_star, k_feas=k_feas, m_ties=m_ties,
        delta_req=float(np.log(n / k_star)),
        delta_feas=float(np.log(n / k_feas)),
    )
