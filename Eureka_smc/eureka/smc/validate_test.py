"""Validation/test 复评协议的独立纯逻辑接口。

搜索阶段的分数不能直接作为最终选择依据：本模块先从 archive 的 search score
中稳定取出 top-k，再在不重叠的 validation seed panel 上复评，并只把唯一的
validation 冠军送入 test panel。这里不依赖 ``island``、``archive`` 或 Isaac
Gym，真实运行时只需传入现有 ``Evaluator`` 实例即可。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .particle import EvalRecord


@dataclass(frozen=True)
class ValidationTestResult:
    """一次 validation -> test 协议的结构化结果。

    ``validation_records`` 只包含 archive top-k 候选，键为 candidate id，顺序按稳定的
    top-k 顺序保留（Python dict insertion order）。没有 validation 有效候选时不会运行
    ``test_record`` 均为 ``None``。
    """

    top_k_candidate_ids: tuple[str, ...]
    validation_records: Mapping[str, EvalRecord]
    selected_candidate_id: Optional[str]
    selected_validation_record: Optional[EvalRecord]
    test_record: Optional[EvalRecord]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]

    @property
    def selected(self) -> bool:
        return self.selected_candidate_id is not None

    def to_json(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的摘要，不改变 ``EvalRecord``。"""
        return {
            "top_k_candidate_ids": list(self.top_k_candidate_ids),
            "validation_records": {
                cid: record.to_json() for cid, record in self.validation_records.items()
            },
            "selected_candidate_id": self.selected_candidate_id,
            "selected_validation_record": (
                self.selected_validation_record.to_json()
                if self.selected_validation_record is not None else None
            ),
            "test_record": self.test_record.to_json() if self.test_record is not None else None,
            "validation_seeds": list(self.validation_seeds),
            "test_seeds": list(self.test_seeds),
        }


def _unique_seeds(seeds: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(seed) for seed in seeds)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate seeds: {result}")
    return result


def check_seed_panels(
    validation_seeds: Sequence[int], test_seeds: Sequence[int],
    search_seeds: Optional[Sequence[int]] = None,
) -> None:
    """检查 search/validation/test panel 各自无重复且两两不重叠。"""
    panels = {
        "validation": _unique_seeds(validation_seeds, "validation seed panel"),
        "test": _unique_seeds(test_seeds, "test seed panel"),
    }
    if search_seeds is not None:
        panels["search"] = _unique_seeds(search_seeds, "search seed panel")
    names = sorted(panels)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = sorted(set(panels[left_name]).intersection(panels[right_name]))
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} seed panels overlap: {overlap}"
                )


def _archive_values(archive: Any) -> list[Any]:
    """读取常见 archive 容器，保持容器给出的顺序作为最终 tie-break。"""
    if isinstance(archive, Mapping):
        return list(archive.values())
    for attr in ("particles", "items"):
        value = getattr(archive, attr, None)
        if value is not None and not callable(value):
            return list(value.values()) if isinstance(value, Mapping) else list(value)
    return list(archive)


def _field(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _candidate_id(candidate: Any) -> str:
    value = _field(candidate, "id")
    if value is None:
        raise ValueError("archive candidate is missing id")
    return str(value)


def _candidate_score(candidate: Any) -> Optional[float]:
    score = _field(candidate, "search_score")
    if score is None:
        evaluation = _field(candidate, "eval")
        score = _field(evaluation, "search_score") if evaluation is not None else None
    if score is None:
        return None
    score = float(score)
    return score if math.isfinite(score) else None


def _candidate_code(candidate: Any) -> str:
    code = _field(candidate, "reward_code")
    if code is None:
        raise ValueError(f"candidate {_candidate_id(candidate)!r} is missing reward_code")
    return str(code)


def _candidate_valid(candidate: Any) -> bool:
    value = _field(candidate, "valid")
    if value is None:
        evaluation = _field(candidate, "eval")
        value = _field(evaluation, "valid") if evaluation is not None else True
    return bool(value)


def select_archive_top_k(archive: Any, k: int) -> list[Any]:
    """按 archive 的 canonical top-k 或 score/id 稳定选出前 k 个候选。

    无效候选（没有有效且有限 ``search_score``）不会进入 top-k。重复 id 会被
    拒绝，防止 validation 结果无法映射回唯一候选。
    """
    if k <= 0:
        raise ValueError("top_k must be positive")
    if hasattr(archive, "top_k") and callable(archive.top_k):
        candidates = list(archive.top_k(k))
    else:
        candidates = _archive_values(archive)
        ranked = []
        for candidate in candidates:
            score = _candidate_score(candidate)
            if _candidate_valid(candidate) and score is not None:
                ranked.append((candidate, score))
        ranked.sort(key=lambda item: (-item[1], _candidate_id(item[0])))
        candidates = [candidate for candidate, _ in ranked[:k]]
    ids = [_candidate_id(candidate) for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("archive candidates must have unique ids")
    if any(not _candidate_valid(candidate) or _candidate_score(candidate) is None
           for candidate in candidates):
        raise ValueError("archive top_k returned an invalid candidate")
    return candidates


def _evaluator_seeds(evaluator: Any, explicit: Optional[Sequence[int]], name: str) -> tuple[int, ...]:
    seeds = explicit if explicit is not None else getattr(evaluator, "seeds", ())
    return _unique_seeds(seeds, name)


def _evaluate(evaluator: Any, code: str, candidate_id: str, artifact_dir: Optional[Path]) -> EvalRecord:
    if hasattr(evaluator, "evaluate"):
        record = evaluator.evaluate(code, candidate_id, artifact_dir)
    elif callable(evaluator):
        record = evaluator(code, candidate_id, artifact_dir)
    else:
        raise TypeError("evaluator must provide evaluate() or be callable")
    if not isinstance(record, EvalRecord):
        raise TypeError("evaluator must return EvalRecord")
    return record


def run_validation_test(
    archive: Any,
    validation_evaluator: Any,
    test_evaluator: Any,
    *,
    top_k: int = 3,
    validation_seeds: Optional[Sequence[int]] = None,
    test_seeds: Optional[Sequence[int]] = None,
    artifact_root: Optional[Path] = None,
) -> ValidationTestResult:
    """执行固定的 archive top-k -> validation winner -> test 协议。

    evaluator 通常是已经用对应 seed panel 构造好的 ``Evaluator``。可用显式
    ``validation_seeds``/``test_seeds``覆盖结果记录中的 panel（用于轻量替身）；
    seed panel 必须不重叠。validation 平分时按 search top-k 顺序和 candidate id
    稳定决胜，因而每次只会选择一个候选。
    """
    validation_panel = _evaluator_seeds(
        validation_evaluator, validation_seeds, "validation seed panel")
    test_panel = _evaluator_seeds(test_evaluator, test_seeds, "test seed panel")
    check_seed_panels(validation_panel, test_panel)

    top_candidates = select_archive_top_k(archive, top_k)
    validation_records: dict[str, EvalRecord] = {}
    for index, candidate in enumerate(top_candidates):
        cid = _candidate_id(candidate)
        directory = None
        if artifact_root is not None:
            directory = Path(artifact_root) / "validation" / f"{index:03d}_{cid}"
        record = _evaluate(validation_evaluator, _candidate_code(candidate), cid, directory)
        validation_records[cid] = record

    valid_records = [
        (index, cid, validation_records[cid])
        for index, cid in enumerate(_candidate_id(candidate) for candidate in top_candidates)
        if validation_records[cid].valid
        and validation_records[cid].search_score is not None
        and math.isfinite(float(validation_records[cid].search_score))
    ]
    selected_id: Optional[str] = None
    selected_record: Optional[EvalRecord] = None
    if valid_records:
        # index 是 archive top-k 的稳定次级键；id 作为防御性最终次级键。
        _, selected_id, selected_record = max(
            valid_records, key=lambda item: (item[2].search_score, -item[0], item[1]))

    test_record = None
    if selected_id is not None:
        selected_candidate = next(c for c in top_candidates if _candidate_id(c) == selected_id)
        directory = None
        if artifact_root is not None:
            directory = Path(artifact_root) / "test" / selected_id
        test_record = _evaluate(
            test_evaluator, _candidate_code(selected_candidate), selected_id, directory)

    return ValidationTestResult(
        top_k_candidate_ids=tuple(_candidate_id(c) for c in top_candidates),
        validation_records=validation_records,
        selected_candidate_id=selected_id,
        selected_validation_record=selected_record,
        test_record=test_record,
        validation_seeds=validation_panel,
        test_seeds=test_panel,
    )


__all__ = [
    "ValidationTestResult",
    "check_seed_panels",
    "select_archive_top_k",
    "run_validation_test",
]
