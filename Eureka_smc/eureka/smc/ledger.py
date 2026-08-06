"""SMC 搜索的可序列化预算账本。

逻辑搜索预算与实际执行成本必须分开：前者定义方法公平性，后者用于
LLM/GPU 资源审计。账本只保存 JSON 标量，供 checkpoint 与 summary 复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["BudgetLedger"]


@dataclass
class BudgetLedger:
    """Accumulate logical slots, candidate outcomes, and observed real costs."""

    logical_init_slots: int = 0
    logical_mutation_slots: int = 0
    proposals: int = 0
    noop_proposals: int = 0
    valid_candidates: int = 0
    invalid_candidates: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    init_repairs: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    physical_rl_evals: int = 0
    cache_hits: int = 0
    validation_rl_evals: int = 0
    test_rl_evals: int = 0
    _seen_evaluations: int = field(default=0, repr=False)

    @property
    def logical_search_total(self) -> int:
        return self.logical_init_slots + self.logical_mutation_slots

    def record_evaluations(self, particles: list[Any]) -> None:
        """Record evaluated particles after the evaluator has returned."""
        self._seen_evaluations += len(particles)
        for particle in particles:
            if particle.valid and particle.search_score is not None:
                self.valid_candidates += 1
            else:
                self.invalid_candidates += 1

    def record_mutation_slots(self, slots: int) -> None:
        if slots < 0:
            raise ValueError("mutation slots must be non-negative")
        self.logical_mutation_slots += int(slots)

    def record_noop(self) -> None:
        self.noop_proposals += 1

    def record_acceptance(self, accepted: bool) -> None:
        if accepted:
            self.accepted_candidates += 1
        else:
            self.rejected_candidates += 1

    def sync_actual_costs(self, proposer: Any, evaluator: Any) -> None:
        """Snapshot counters exposed by existing proposer/evaluator implementations."""
        self.llm_calls = int(getattr(proposer, "n_calls", self.llm_calls))
        self.prompt_tokens = int(getattr(proposer, "total_prompt_tokens", self.prompt_tokens))
        self.completion_tokens = int(getattr(proposer, "total_completion_tokens", self.completion_tokens))
        self.physical_rl_evals = int(getattr(evaluator, "n_evals", self.physical_rl_evals))
        self.cache_hits = max(0, self._seen_evaluations - self.physical_rl_evals)

    def snapshot(self) -> dict[str, int]:
        return {
            "logical_init_slots": self.logical_init_slots,
            "logical_mutation_slots": self.logical_mutation_slots,
            "logical_search_total": self.logical_search_total,
            "proposals": self.proposals,
            "noop_proposals": self.noop_proposals,
            "valid_candidates": self.valid_candidates,
            "invalid_candidates": self.invalid_candidates,
            "accepted_candidates": self.accepted_candidates,
            "rejected_candidates": self.rejected_candidates,
            "init_repairs": self.init_repairs,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "physical_rl_evals": self.physical_rl_evals,
            "cache_hits": self.cache_hits,
            "validation_rl_evals": self.validation_rl_evals,
            "test_rl_evals": self.test_rl_evals,
            "seen_evaluations": self._seen_evaluations,
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> "BudgetLedger":
        fields = cls().__dict__.keys()
        restored = cls()
        for name in fields:
            key = "seen_evaluations" if name == "_seen_evaluations" else name
            if key in data:
                setattr(restored, name, int(data[key]))
        return restored
