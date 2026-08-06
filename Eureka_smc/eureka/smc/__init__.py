"""SMC search controller for Eureka reward-function evolution.

The parent-resource-allocation layer is the **budget-conditioned KL–Feynman–Kac
controller** (``kl_controller``): selection concentration is driven purely by the
remaining LLM-modification budget (target effective parents K* → KL radius → solved
selection strength λ → Boltzmann–Gibbs parent distribution over **raw** J), parents
are resampled multinomially into LLM modification calls, and children enter via an
asymmetric sigmoid transition kernel. Runs until the modification budget is spent.

See ``预算与进展自适应_SMC奖励搜索_实验计划.md``.
"""

from __future__ import annotations

from .actions import RF_ACTIONS, ActionConfig, assign_actions
from .archive import Archive, ArchiveEntry, CandidateArchive, RewardArchive, code_hash, hash_code
from .checkpoint import CHECKPOINT_VERSION, CheckpointError, load_checkpoint, save_checkpoint
from .contracts import (
    RewardSignature,
    check_mutation_parameter,
    check_mutation_structure,
    reward_signature,
)
from .context import format_crossover, format_different, format_path
from .event_logger import EventLogger
from .evaluator import Evaluator, FakeEvaluator, IsaacGymEvalConfig, IsaacGymEvaluator
from .island import IslandResult, SMCIsland, SMCIslandConfig
from .kl_controller import (
    ControllerStep,
    boltzmann_gibbs,
    effective_parents,
    entropy_effective_parents,
    feasible_target,
    kl_to_uniform,
    relative_ess,
    resolve,
    solve_lambda,
    solve_lambda_for_ress,
    target_effective_parents,
    target_relative_ess,
    tau_budget,
)
from .progress import (
    ProgressStep,
    compute_gamma,
    compute_progress,
    gamma_from_sets,
    pairwise_tie_auc,
)
from .particle import EvalRecord, RewardParticle
from .proposer import (
    EurekaReflectionProposer,
    FakeProposer,
    TaskContext,
    extract_design_thought,
    extract_reward_code,
)
from .resampling import multinomial_resample, systematic_resample
from .score import ScoreConfig, clipped_linear, compute_search_score
from .validate_test import (
    ValidationTestResult,
    check_seed_panels,
    run_validation_test,
    select_archive_top_k,
)

__all__ = [
    # budget-conditioned KL controller
    "ControllerStep",
    "target_effective_parents",
    "target_relative_ess",
    "tau_budget",
    "boltzmann_gibbs",
    "kl_to_uniform",
    "effective_parents",
    "entropy_effective_parents",
    "relative_ess",
    "feasible_target",
    "solve_lambda",
    "solve_lambda_for_ress",
    "resolve",
    # progress
    "ProgressStep",
    "compute_gamma",
    "compute_progress",
    "gamma_from_sets",
    "pairwise_tie_auc",
    "multinomial_resample",
    "systematic_resample",
    # score
    "ScoreConfig",
    "clipped_linear",
    "compute_search_score",
    # data model
    "EvalRecord",
    "RewardParticle",
    # archive / checkpoint
    "ArchiveEntry",
    "CandidateArchive",
    "RewardArchive",
    "code_hash",
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "save_checkpoint",
    "load_checkpoint",
    # RF-Agent actions / contracts
    "RF_ACTIONS",
    "ActionConfig",
    "assign_actions",
    "RewardSignature",
    "reward_signature",
    "check_mutation_structure",
    "check_mutation_parameter",
    "format_crossover",
    "format_path",
    "format_different",
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
