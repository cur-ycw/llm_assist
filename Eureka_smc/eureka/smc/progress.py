"""纯 Python 的父子群体进展度量。

``Gamma`` 只描述上一轮实际获得修改资源的父代槽位 ``Z_t`` 与原始子代
集合 ``Y_t`` 的相对排序：

* ``K_eff = 1 / sum(q**2)``；
* ``k = clip(ceil(K_eff), 1, M)``；
* 两组都按原始任务分数 ``J`` 取 Top-k；
* pairwise 比较中平局记半分；
* ``Gamma = 2*A - 1``。

本模块不改变候选或分数，只返回可审计的纯数值快照。Gamma 在控制器中按
一轮延迟使用；首轮应由调用方传入 ``progress_prev=0``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "ProgressStep",
    "effective_parents",
    "top_k",
    "pairwise_tie_auc",
    "gamma_from_sets",
    "compute_gamma",
    "compute_progress",
]

_EPS = 1e-12


def _scores(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                        dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite scores")
    return result


def effective_parents(q: Sequence[float]) -> float:
    """Return the second-moment effective population size ``1 / sum(q**2)``."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("q must be a non-empty finite probability vector")
    total = float(q.sum())
    if total <= 0:
        raise ValueError("q must have positive mass")
    q = q / total
    return float(1.0 / np.square(q).sum())


def top_k(values: Sequence[float], k: int) -> np.ndarray:
    """Return stable indices of the ``k`` largest values, descending by score."""
    values = _scores(values, "values")
    k = int(np.clip(k, 0, values.size))
    # mergesort makes ties deterministic without perturbing raw J.
    return np.argsort(-values, kind="mergesort")[:k]


def pairwise_tie_auc(parent_scores: Sequence[float], child_scores: Sequence[float]) -> float:
    """P(child > parent) with ties worth one half."""
    parents = _scores(parent_scores, "parent_scores")
    children = _scores(child_scores, "child_scores")
    if parents.size == 0 or children.size == 0:
        raise ValueError("pairwise comparison requires two non-empty sets")
    diff = children[:, None] - parents[None, :]
    wins = (diff > 0.0).astype(np.float64)
    wins += 0.5 * (diff == 0.0)
    return float(wins.mean())


def gamma_from_sets(parent_scores: Sequence[float], child_scores: Sequence[float]) -> float:
    """Compute ``Gamma = 2*A - 1`` for two already selected Top-k sets."""
    return float(2.0 * pairwise_tie_auc(parent_scores, child_scores) - 1.0)


def _select_parent_scores(
    q: np.ndarray,
    parent_scores: Sequence[float],
    parent_indices: Optional[Sequence[int]],
) -> np.ndarray:
    scores = _scores(parent_scores, "parent_scores")
    if parent_indices is None:
        if scores.size != q.size:
            raise ValueError("parent_scores must match q when parent_indices is omitted")
        return scores
    indices = np.asarray(parent_indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= scores.size):
        raise ValueError("parent_indices out of range")
    return scores[indices]


@dataclass(frozen=True)
class ProgressStep:
    """Auditable progress calculation for one mutation round."""

    k_eff: float
    k: int
    p_eff: np.ndarray
    y_eff: np.ndarray
    auc: float
    gamma: float


def compute_progress(
    q: Sequence[float],
    parent_scores: Sequence[float],
    child_scores: Sequence[float],
    parent_indices: Optional[Sequence[int]] = None,
) -> ProgressStep:
    """Compute the frozen ``Z_t``/``Y_t`` progress definition.

    ``parent_scores`` is the population's raw ``J``.  ``parent_indices`` identifies
    the slots in ``Z_t`` that actually received mutation resources.  When omitted,
    all supplied parent scores are treated as ``Z_t`` (a convenient direct API for
    tests and callers that already materialized the resource-bearing set).
    """
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    z_scores = _select_parent_scores(q_arr, parent_scores, parent_indices)
    y_scores = _scores(child_scores, "child_scores")
    k_eff = effective_parents(q_arr)
    m = y_scores.size
    if m == 0:
        return ProgressStep(k_eff, 0, np.empty(0, dtype=np.int64),
                            np.empty(0, dtype=np.int64), 0.5, 0.0)
    k = int(np.clip(np.ceil(k_eff), 1, m))
    p_local = top_k(z_scores, k)
    y_local = top_k(y_scores, k)
    auc = pairwise_tie_auc(z_scores[p_local], y_scores[y_local])
    return ProgressStep(k_eff, k, p_local, y_local, auc, float(2.0 * auc - 1.0))


def compute_gamma(
    q: Sequence[float],
    parent_scores: Sequence[float],
    child_scores: Sequence[float],
    parent_indices: Optional[Sequence[int]] = None,
) -> float:
    """Convenience wrapper returning only ``Gamma``."""
    return compute_progress(q, parent_scores, child_scores, parent_indices).gamma
