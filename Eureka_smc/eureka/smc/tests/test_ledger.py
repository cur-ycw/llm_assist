from types import SimpleNamespace
from typing import Optional

from eureka.smc.ledger import BudgetLedger
from eureka.smc.particle import EvalRecord, RewardParticle


def _particle(pid: str, score: Optional[float]) -> RewardParticle:
    return RewardParticle(
        id=pid,
        reward_code="def reward(): pass",
        eval=EvalRecord(
            search_score=score,
            valid=score is not None,
            executable=score is not None,
        ),
    )


def test_ledger_separates_logical_slots_from_actual_costs():
    ledger = BudgetLedger(logical_init_slots=16)
    ledger.record_mutation_slots(8)
    ledger.record_evaluations([_particle("ok", 1.0), _particle("bad", None)])
    ledger.record_noop()
    ledger.record_acceptance(True)
    ledger.record_acceptance(False)
    ledger.sync_actual_costs(
        SimpleNamespace(n_calls=25, total_prompt_tokens=100, total_completion_tokens=40),
        SimpleNamespace(n_evals=1),
    )

    snapshot = ledger.snapshot()
    assert snapshot["logical_search_total"] == 24
    assert snapshot["valid_candidates"] == 1
    assert snapshot["invalid_candidates"] == 1
    assert snapshot["noop_proposals"] == 1
    assert snapshot["accepted_candidates"] == 1
    assert snapshot["rejected_candidates"] == 1
    assert snapshot["llm_calls"] == 25
    assert snapshot["physical_rl_evals"] == 1
    assert snapshot["cache_hits"] == 1


def test_ledger_snapshot_round_trip():
    ledger = BudgetLedger(logical_init_slots=16, logical_mutation_slots=64)
    ledger.record_evaluations([_particle("p0", 0.5)])
    ledger.record_acceptance(True)

    restored = BudgetLedger.from_snapshot(ledger.snapshot())

    assert restored.snapshot() == ledger.snapshot()
