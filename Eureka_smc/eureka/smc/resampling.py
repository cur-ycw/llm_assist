"""Systematic (low-variance) resampling for the SMC controller.

Migrated from SMCEvolve (``smcevolve/island.py``) per migration plan §6.3. Uses a
single uniform draw and a regular comb over the CDF, which has strictly lower
variance than ``np.random.choice`` multinomial resampling and reproduces the
SMCEvolve source behaviour. Pure numpy; the RNG is caller-owned so each island can
carry its own seeded ``np.random.Generator`` for reproducible runs (plan §6.3, §9).
"""

from __future__ import annotations

import numpy as np

__all__ = ["systematic_resample"]


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
