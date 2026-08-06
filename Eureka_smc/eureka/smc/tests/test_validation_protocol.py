"""Validation/test 复评协议的纯逻辑测试。"""

from __future__ import annotations

import pytest

from eureka.smc.particle import EvalRecord, RewardParticle
from eureka.smc.validate_test import (
    check_seed_panels,
    run_validation_test,
    select_archive_top_k,
)


class ScriptedEvaluator:
    """按候选 id 返回预设结果的无 GPU 评估替身。"""

    def __init__(self, seeds, scores):
        self.seeds = tuple(seeds)
        self.scores = scores
        self.calls = []

    def evaluate(self, code, candidate_id, artifact_dir=None):
        self.calls.append((candidate_id, code, artifact_dir))
        score = self.scores[candidate_id]
        return EvalRecord(
            search_score=score,
            valid=score is not None,
            executable=score is not None,
            feedback=f"score={score}",
        )


def _particle(candidate_id: str, score: float | None) -> RewardParticle:
    return RewardParticle(
        id=candidate_id,
        reward_code=f"def reward_{candidate_id}(): pass",
        eval=EvalRecord(search_score=score, valid=score is not None, executable=True),
    )


def test_select_archive_top_k_sorts_score_then_id_and_excludes_invalid():
    archive = [
        _particle("z", 0.8),
        _particle("b", 0.9),
        _particle("a", 0.9),
        _particle("invalid", None),
        _particle("c", 0.7),
    ]

    selected = select_archive_top_k(archive, 3)

    assert [particle.id for particle in selected] == ["a", "b", "z"]


def test_validation_selects_unique_winner_then_tests_only_that_candidate(tmp_path):
    archive = [_particle("search-best", 0.95), _particle("generalizes", 0.90), _particle("third", 0.80)]
    validation = ScriptedEvaluator((10, 11), {
        "search-best": 0.2,
        "generalizes": 0.8,
        "third": 0.1,
    })
    test = ScriptedEvaluator((20, 21), {"generalizes": 0.75})

    result = run_validation_test(
        archive, validation, test, top_k=2, artifact_root=tmp_path)

    assert result.top_k_candidate_ids == ("search-best", "generalizes")
    assert result.selected_candidate_id == "generalizes"
    assert result.selected_validation_record.search_score == pytest.approx(0.8)
    assert result.test_record.search_score == pytest.approx(0.75)
    assert [call[0] for call in validation.calls] == ["search-best", "generalizes"]
    assert [call[0] for call in test.calls] == ["generalizes"]
    assert validation.calls[0][2] == tmp_path / "validation" / "000_search-best"
    assert test.calls[0][2] == tmp_path / "test" / "generalizes"
    assert result.validation_seeds == (10, 11)
    assert result.test_seeds == (20, 21)


def test_validation_tie_uses_stable_archive_top_k_order():
    archive = [_particle("later", 0.8), _particle("first", 0.9)]
    validation = ScriptedEvaluator((10,), {"first": 0.5, "later": 0.5})
    test = ScriptedEvaluator((20,), {"first": 0.6})

    result = run_validation_test(archive, validation, test, top_k=2)

    # search score 先决定 top-k 顺序，因此 first 在 validation 平分时唯一胜出。
    assert result.top_k_candidate_ids == ("first", "later")
    assert result.selected_candidate_id == "first"
    assert [call[0] for call in test.calls] == ["first"]


@pytest.mark.parametrize(
    ("validation", "test", "message"),
    [
        ((1, 2), (2, 3), "overlap"),
        ((1, 1), (2, 3), "duplicate"),
        ((1, 2), (3, 3), "duplicate"),
    ],
)
def test_seed_panels_must_be_disjoint_and_unique(validation, test, message):
    with pytest.raises(ValueError, match=message):
        check_seed_panels(validation, test)


def test_search_validation_test_seed_panels_must_be_pairwise_disjoint():
    with pytest.raises(ValueError, match="search/validation"):
        check_seed_panels((10,), (20,), search_seeds=(10,))
    with pytest.raises(ValueError, match="search/test"):
        check_seed_panels((10,), (20,), search_seeds=(20,))


    archive = [_particle("a", 0.9), _particle("b", 0.8)]
    validation = ScriptedEvaluator((10,), {"a": None, "b": None})
    test = ScriptedEvaluator((20,), {})

    result = run_validation_test(archive, validation, test, top_k=2)

    assert result.selected is False
    assert result.selected_candidate_id is None
    assert result.selected_validation_record is None
    assert result.test_record is None
    assert test.calls == []


def test_result_to_json_serializes_records():
    archive = [_particle("a", 0.9)]
    validation = ScriptedEvaluator((10,), {"a": 0.8})
    test = ScriptedEvaluator((20,), {"a": 0.7})

    payload = run_validation_test(archive, validation, test).to_json()

    assert payload["top_k_candidate_ids"] == ["a"]
    assert payload["validation_records"]["a"]["search_score"] == pytest.approx(0.8)
    assert payload["test_record"]["search_score"] == pytest.approx(0.7)
