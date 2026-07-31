"""Resampling for the SMC controller.

Two schemes, both pure numpy with a caller-owned seeded ``np.random.Generator``:

  * ``multinomial_resample`` — the **main path** of the budget-conditioned KL
    controller (new method §2.5): draw ``M`` parent copies i.i.d. from the
    Boltzmann–Gibbs selection distribution ``q_t`` (``A^j ~ Categorical(q_t)``), so
    the expected copy count of parent ``i`` is ``M·q_t(i)``. ``M = min(N, b_t)`` lets
    the final partial round spend exactly the remaining budget.
  * ``systematic_resample`` — low-variance comb resampling migrated from SMCEvolve.
    Retained for the resampling-scheme ablation (experiment plan §9.4:
    multinomial / systematic / residual); NOT on the current main path.
"""

from __future__ import annotations

import numpy as np

__all__ = ["multinomial_resample", "systematic_resample"]


def multinomial_resample(
    q: np.ndarray, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``m`` ancestor indices i.i.d. from ``Categorical(q)`` (new method §2.5).

    Parameters
    ----------
    q:
        Non-negative selection probabilities over the ``N`` current particles. They
        are renormalised internally (guards tiny floating-point drift), so they need
        not sum to exactly 1, but must contain at least one positive entry.
    m:
        Number of parent copies to draw = number of LLM modification calls this
        round (``M_t = min(N, b_t)``).
    rng:
        Seeded ``numpy.random.Generator`` owned by the island. A single ``rng.choice``
        call keeps runs reproducible.

    Returns
    -------
    np.ndarray
        Integer array of length ``m`` with ancestor indices in ``[0, N)``. Higher
        ``q(i)`` ⇒ more expected copies (``E[n_i] = m·q(i)``). The caller clones the
        corresponding parents into fresh particles (new ids, ``clone_ancestor_id``
        set) — this function only picks who reproduces.
    """
    q = np.asarray(q, dtype=np.float64)
    n = q.size
    if n == 0 or m <= 0:
        return np.empty(0, dtype=np.intp)
    total = q.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("multinomial_resample requires finite weights with sum > 0")
    p = q / total
    return rng.choice(n, size=int(m), replace=True, p=p).astype(np.intp)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return ancestor indices drawn by systematic resampling.

    Parameters
    ----------
    weights:
        Non-negative importance weights. They are normalised internally, so they
        need not sum to 1, but must contain at least one positive entry.
    rng:
        A seeded ``numpy.random.Generator`` owned by the caller (per island). Only a
        single ``rng.random()`` draw is consumed, keeping runs reproducible.

    Returns
    -------
    np.ndarray
        Integer array of length ``N`` with ancestor indices in ``[0, N)``. Higher
        weight ⇒ more expected copies. The caller clones the corresponding parents
        into fresh particles (new ids, ``clone_ancestor_id`` set) — this function
        only picks who survives.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n = weights.size
    if n == 0:
        return np.empty(0, dtype=np.intp)

    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("systematic_resample requires finite weights with sum > 0")

    # Regular comb of N positions offset by one shared uniform draw.
    positions = (rng.random() + np.arange(n)) / n

    cumsum = np.cumsum(weights / total)
    cumsum[-1] = 1.0  # guard against floating-point drift below 1.0

    return np.searchsorted(cumsum, positions, side="left").astype(np.intp)
