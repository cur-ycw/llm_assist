from __future__ import annotations

from pathlib import Path

import pytest

from eureka.smc.archive import CandidateArchive, code_hash
from eureka.smc.particle import EvalRecord, RewardParticle


def particle(pid: str, code: str, score: float, *, valid: bool = True) -> RewardParticle:
    return RewardParticle(
        id=pid,
        reward_code=code,
        eval=EvalRecord(search_score=score if valid else None, valid=valid, executable=valid),
    )


def test_archive_keeps_valid_candidates_and_deduplicates_code():
    archive = CandidateArchive()
    assert archive.add(particle("bad", "bad", 100.0, valid=False)) is False
    assert archive.add(particle("a-low", "same", 1.0)) is True
    assert archive.add(particle("a-high", "same", 2.0)) is True
    assert len(archive) == 1
    assert archive.top_k(1)[0].id == "a-high"


def test_top_k_is_stable_for_equal_scores():
    archive = CandidateArchive([
        particle("z", "z-code", 1.0),
        particle("a", "a-code", 1.0),
        particle("m", "m-code", 0.5),
    ])
    expected = [p.id for p in archive.top_k(3)]
    assert expected == [p.id for p in archive.top_k(3)]
    assert [p.id for p in archive.top_k(2)] == expected[:2]


def test_archive_snapshot_restore_round_trip():
    original = CandidateArchive([particle("p1", "code-1", 1.0), particle("p2", "code-2", 3.0)])
    restored = CandidateArchive.from_snapshot(original.snapshot())
    assert len(restored) == 2
    assert [(p.reward_code, p.search_score) for p in restored.top_k(2)] == [
        (p.reward_code, p.search_score) for p in original.top_k(2)
    ]
    assert restored.snapshot() == original.snapshot()


def test_archive_rejects_tampered_hash():
    snapshot = CandidateArchive([particle("p", "code", 1.0)]).snapshot()
    snapshot["entries"][0]["code_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        CandidateArchive.from_snapshot(snapshot)


def test_code_hash_is_stable_across_line_endings():
    assert code_hash("a\r\nb") == code_hash("a\nb")


def test_snapshot_does_not_require_filesystem_artifact(tmp_path: Path):
    p = particle("p", "code", 1.0)
    p.artifact_dir = tmp_path / "does-not-exist"
    assert CandidateArchive([p]).snapshot()["entries"]
