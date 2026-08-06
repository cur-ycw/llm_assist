"""去重的有效奖励候选档案。

档案是 ``SMCIsland`` 主循环之外的基础设施：所有完成有效评估的候选都可写入，
被拒绝的候选也不会因为没有进入当前粒子群而丢失。档案按奖励代码的 SHA-256
去重，并以确定性的顺序返回精英候选。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

from .particle import RewardParticle

__all__ = [
    "ArchiveEntry",
    "CandidateArchive",
    "RewardArchive",
    "Archive",
    "code_hash",
    "hash_code",
]


SCHEMA_VERSION = 1


def code_hash(code: str) -> str:
    """返回奖励代码的稳定 SHA-256 哈希。

    仅统一换行符，不做空白或 AST 规范化；这样档案不会把可能有意义的源码
    文本改写成另一份代码。
    """
    if not isinstance(code, str):
        raise TypeError("reward code must be a string")
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _valid_score(candidate: Any) -> bool:
    try:
        valid = bool(candidate.valid)
        score = candidate.search_score
    except AttributeError:
        return False
    return valid and score is not None and math.isfinite(float(score))


def _particle_json(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "to_json"):
        value = candidate.to_json()
        if isinstance(value, dict):
            return dict(value)
    if isinstance(candidate, dict):
        return dict(candidate)
    raise TypeError("archive candidates must provide to_json() or be mappings")


def _restore_particle(data: dict[str, Any]) -> RewardParticle:
    """Restore through the canonical particle codec.

    Keeping archive restoration on ``RewardParticle.from_json`` prevents this
    persistence boundary from drifting when particle fields evolve.
    """
    return RewardParticle.from_json(data)


@dataclass(frozen=True)
class ArchiveEntry:
    """档案中的候选及其代码哈希。"""

    code_hash: str
    candidate: Any

    @property
    def score(self) -> float:
        return float(self.candidate.search_score)


class CandidateArchive:
    """保存所有有效候选，并按代码哈希保留每份代码的最佳评估。"""

    schema_version = SCHEMA_VERSION

    def __init__(self, candidates: Optional[Iterable[Any]] = None) -> None:
        self._entries: dict[str, ArchiveEntry] = {}
        if candidates is not None:
            self.extend(candidates)

    def add(self, candidate: Any) -> bool:
        """加入候选；返回是否新增代码或替换了已有代表。

        无效候选被忽略，且不会占用档案容量。相同代码只保留分数更高者；分数
        相同时保留 id 字典序较小者，避免输入顺序影响结果。
        """
        if not _valid_score(candidate):
            return False
        digest = code_hash(candidate.reward_code)
        current = self._entries.get(digest)
        incoming = ArchiveEntry(digest, candidate)
        if current is None:
            self._entries[digest] = incoming
            return True
        current_key = (current.score, str(getattr(current.candidate, "id", "")))
        incoming_key = (incoming.score, str(getattr(candidate, "id", "")))
        if incoming.score > current.score or (
            incoming.score == current.score and incoming_key[1] < current_key[1]
        ):
            self._entries[digest] = incoming
            return True
        return False

    add_candidate = add

    def extend(self, candidates: Iterable[Any]) -> int:
        return sum(self.add(candidate) for candidate in candidates)

    update = extend

    def top_k(self, k: int) -> list[Any]:
        """返回分数降序的前 ``k`` 个候选，排序完全确定。"""
        if k < 0:
            raise ValueError("k must be non-negative")
        entries = sorted(
            self._entries.values(),
            key=lambda entry: (-entry.score, str(getattr(entry.candidate, "id", "")), entry.code_hash),
        )
        return [entry.candidate for entry in entries[:k]]

    @property
    def candidates(self) -> list[Any]:
        return self.top_k(len(self))

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(sorted(self._entries.values(), key=lambda e: e.code_hash))

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.candidates)

    def snapshot(self) -> dict[str, Any]:
        """生成可 JSON 序列化的快照，不包含运行时对象引用。"""
        return {
            "schema_version": self.schema_version,
            "entries": [
                {"code_hash": entry.code_hash, "candidate": _particle_json(entry.candidate)}
                for entry in self.entries
            ],
        }

    def restore(self, snapshot: dict[str, Any]) -> "CandidateArchive":
        """从快照原地恢复并返回自身。"""
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != self.schema_version:
            raise ValueError("unsupported archive snapshot version")
        entries = snapshot.get("entries")
        if not isinstance(entries, list):
            raise ValueError("archive snapshot entries must be a list")
        restored: dict[str, ArchiveEntry] = {}
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
                raise ValueError("invalid archive snapshot entry")
            candidate = _restore_particle(item["candidate"])
            digest = code_hash(candidate.reward_code)
            supplied = item.get("code_hash")
            if supplied != digest:
                raise ValueError("archive entry code hash does not match reward code")
            restored[digest] = ArchiveEntry(digest, candidate)
        self._entries = restored
        return self

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "CandidateArchive":
        return cls().restore(snapshot)


RewardArchive = CandidateArchive
Archive = CandidateArchive
hash_code = code_hash
