"""SMC search controller for Eureka reward-function evolution.

Batch 1 shipped the pure numeric core (ESS temperature schedule + systematic
resampling) and a feasibility simulation. Batch 2 adds the data model, score
normalisation, proposer, evaluator (real + fake), event log and the single-island
live loop (``eureka_reflection`` proposal, reward-only MH acceptance).

See ``Eureka_SMCEvolve_框架迁移实施计划.md``.
"""

from __future__ import annotations

from .actions import RF_ACTIONS, ActionConfig, assign_actions
from .contracts import (
    RewardSignature,
    check_mutation_parameter,
    check_mutation_structure,
    reward_signature,
)
from .event_logger import EventLogger
from .evaluator import Evaluator, FakeEvaluator, IsaacGymEvalConfig, IsaacGymEvaluator
from .island import IslandResult, SMCIsland, SMCIslandConfig
from .particle import EvalRecord, RewardParticle
from .proposer import (
    EurekaReflectionProposer,
    FakeProposer,
    TaskContext,
    extract_design_thought,
    extract_reward_code,
)
from .resampling import systematic_resample
from .score import ScoreConfig, clipped_linear, compute_search_score
from .temperature import (
    ess,
    ess_from_log_weights,
    find_next_lambda,
    log_incremental_weights,
)

__all__ = [
    # numeric core
    "ess",
    "ess_from_log_weights",
    "find_next_lambda",
    "log_incremental_weights",
    "systematic_resample",
    # score
    "ScoreConfig",
    "clipped_linear",
    "compute_search_score",
    # data model
    "EvalRecord",
    "RewardParticle",
    # RF-Agent actions / contracts
    "RF_ACTIONS",
    "ActionConfig",
    "assign_actions",
    "RewardSignature",
    "reward_signature",
    "check_mutation_structure",
    "check_mutation_parameter",
    # proposer / evaluator
    "TaskContext",
    "extract_reward_code",
    "extract_design_thought",
    "EurekaReflectionProposer",
    "FakeProposer",
    "Evaluator",
    "FakeEvaluator",
    "IsaacGymEvaluator",
    "IsaacGymEvalConfig",
    # loop
    "EventLogger",
    "SMCIsland",
    "SMCIslandConfig",
    "IslandResult",
]
