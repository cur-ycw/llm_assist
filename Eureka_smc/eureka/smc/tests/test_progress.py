"""Tests for the frozen effective-region Gamma definition."""

import numpy as np
import pytest

from eureka.smc.progress import (
    compute_gamma,
    compute_progress,
    effective_parents,
    gamma_from_sets,
    pairwise_tie_auc,
    top_k,
)


def test_effective_parents_is_second_moment():
    q = np.array([0.5, 0.25, 0.25])
    assert effective_parents(q) == pytest.approx(1.0 / np.square(q).sum())


def test_k_is_ceil_and_clipped_to_child_count():
    q = np.array([0.6, 0.2, 0.1, 0.1])
    result = compute_progress(q, [1, 2, 3, 4], [10, 20])
    assert result.k_eff == pytest.approx(1.0 / np.square(q).sum())
    assert result.k == 2
    assert result.p_eff.tolist() == [3, 2]
    assert result.y_eff.tolist() == [1, 0]


def test_top_k_uses_raw_scores_and_stable_ties():
    assert top_k(np.array([1.0, 3.0, 3.0, 2.0]), 3).tolist() == [1, 2, 3]


def test_pairwise_ties_count_half():
    assert pairwise_tie_auc([1.0], [1.0]) == pytest.approx(0.5)
    assert pairwise_tie_auc([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.875)
    assert gamma_from_sets([1.0], [2.0]) == pytest.approx(1.0)
    assert gamma_from_sets([2.0], [1.0]) == pytest.approx(-1.0)


def test_gamma_uses_actual_resource_parent_slots():
    q = np.full(4, 0.25)
    # Z_t is slots 1 and 3, not the first two population entries.
    result = compute_progress(q, [100.0, 1.0, 90.0, 2.0], [3.0, 4.0], [1, 3])
    assert result.k == 2
    assert result.p_eff.tolist() == [1, 0]
    assert result.y_eff.tolist() == [1, 0]
    assert result.gamma == pytest.approx(1.0)
    assert compute_gamma(q, [100.0, 1.0, 90.0, 2.0], [3.0, 4.0], [1, 3]) == pytest.approx(1.0)


def test_gamma_negative_and_zero_for_worse_or_tied_regions():
    q = np.full(3, 1 / 3)
    assert compute_gamma(q, [3, 4, 5], [1, 2, 3]) == pytest.approx(-8 / 9)
    assert compute_gamma(q, [1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_invalid_or_empty_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        compute_progress([0.0, 0.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        pairwise_tie_auc([], [1.0])
