"""单岛 SMC 搜索循环（迁移计划 §6.1、§6.3、§6.6、§6.8）。

流程：
  initialize() 生成 N 个 valid 初始粒子（λ=0，不做分数选择）
  → 循环退火阶段，直到 λ≥1（annealing_complete）或触及 max_smc_iterations：
        1. 由当前奖励向量和 ESS 阈值求 λ_next（find_next_lambda）
        2. 按增量权重 systematic_resample 父代 → clone 成新粒子群
        3. 对每个 clone 做 K 次（首版 K=1）eureka_reflection 提议 + 评估 + reward-only
           MH 接受；接受则 child 成为下一步提议源，拒绝则保留当前状态
  → 返回与粒子群分离保存的 best_so_far 及终止原因。

接受规则 α = min(1, exp(β_t·ΔR))（β_t = λ_next·β_target）忠实于 SMCEvolve 论文的
reward-only 近似，不恢复未知的 LLM 提议概率比（计划 §6.6，属 SMC-inspired 优化启发式）。

provenance（计划 §4.1 不变量）：
  * clone 只写 ``clone_ancestor_id``，继承父的 accepted_transition，不伪造代码优化；
  * 被接受的 child 的 ``accepted_transition_parent_id`` 指向被替换代码的真实来源
    （``state_origin_id``），因此沿它回溯得到纯 accepted-edit 链，自动跳过 clone/reject。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .event_logger import EventLogger
from .particle import EvalRecord, RewardParticle
from .resampling import systematic_resample
from .temperature import ess_from_log_weights, find_next_lambda, log_incremental_weights

logger = logging.getLogger(__name__)

__all__ = ["SMCIslandConfig", "SMCIsland", "IslandResult"]


@dataclass
class SMCIslandConfig:
    island_id: int = 0
    n_particles: int = 8
    beta_target: float = 2.0        # Batch-1 仿真推荐（小 N 下避免谱系坍缩）
    kappa: float = 0.5
    min_iters: int = 3              # max_delta = 1/min_iters
    max_iters: int = 15             # max_smc_iterations
    n_proposals: int = 1            # 每个重采样粒子的 MH 链长 K
    max_init_retries: int = 3       # 初始无效粒子的补采样上限（相对 N）
    seed: int = 0


@dataclass
class IslandResult:
    best: Optional[RewardParticle]
    termination_reason: str         # annealing_complete | budget_exhausted | insufficient_valid_particles
    n_stages: int
    final_lambda: float
    min_ess: float


class SMCIsland:
    def __init__(self, config: SMCIslandConfig, proposer, evaluator,
                 event_logger: Optional[EventLogger] = None,
                 artifact_root: Optional[Path] = None):
        self.cfg = config
        self.proposer = proposer
        self.evaluator = evaluator
        self.log = event_logger or EventLogger(Path("events.jsonl"), clock=False)
        self.artifact_root = Path(artifact_root) if artifact_root else Path("candidates")
        self.rng = np.random.default_rng(config.seed)
        self._id_counter = 0
        self.best: Optional[RewardParticle] = None  # 与粒子群分离保存的历史最优

    # ---- 工具 ----
    def _next_id(self) -> str:
        self._id_counter += 1
        return f"i{self.cfg.island_id}-p{self._id_counter}"

    def _artifact(self, pid: str) -> Path:
        d = self.artifact_root / pid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _consider_best(self, p: RewardParticle) -> None:
        if p.valid and p.search_score is not None:
            if self.best is None or p.search_score > (self.best.search_score or -np.inf):
                self.best = p

    def _make_particle(self, code: str, generation: int, *, proposal_parent_id,
                       accepted_transition_parent_id, clone_ancestor_id,
                       state_origin_id: Optional[str] = None) -> RewardParticle:
        pid = self._next_id()
        rec: EvalRecord = self.evaluator.evaluate(code, pid, self._artifact(pid))
        p = RewardParticle(
            id=pid, reward_code=code, eval=rec, island_id=self.cfg.island_id,
            generation=generation, proposal_parent_id=proposal_parent_id,
            accepted_transition_parent_id=accepted_transition_parent_id,
            clone_ancestor_id=clone_ancestor_id, artifact_dir=self._artifact(pid),
        )
        p.metadata["state_origin_id"] = state_origin_id or pid
        self.log.log("eval", id=pid, valid=rec.valid, search_score=rec.search_score,
                     generation=generation, proposal_parent_id=proposal_parent_id,
                     accepted_transition_parent_id=accepted_transition_parent_id,
                     clone_ancestor_id=clone_ancestor_id)
        self._consider_best(p)
        return p

    # ---- 初始化（计划 §6.1）----
    def initialize(self) -> list[RewardParticle]:
        n = self.cfg.n_particles
        particles: list[RewardParticle] = []
        budget = n + self.cfg.max_init_retries * n
        codes = list(self.proposer.initial_batch(n))
        attempts = 0
        while len(particles) < n and attempts < budget:
            if not codes:
                codes = list(self.proposer.initial_batch(n - len(particles)))
            code = codes.pop()
            attempts += 1
            if code is None:
                continue
            p = self._make_particle(code, generation=0, proposal_parent_id=None,
                                    accepted_transition_parent_id=None, clone_ancestor_id=None)
            if p.valid:
                particles.append(p)
            else:
                self.log.log("init_invalid", id=p.id)
        self.log.log("init_done", n_valid=len(particles), attempts=attempts)
        return particles

    # ---- 一个退火阶段（计划 §6.3、§6.6）----
    def _stage(self, particles: list[RewardParticle], lam_prev: float,
               stage_idx: int) -> tuple[list[RewardParticle], float, float]:
        rewards = np.array([p.search_score for p in particles], dtype=np.float64)
        lam_next = find_next_lambda(rewards, lam_prev, self.cfg.beta_target,
                                    self.cfg.kappa, 1.0 / self.cfg.min_iters)
        delta = lam_next - lam_prev
        beta_t = lam_next * self.cfg.beta_target

        logw = log_incremental_weights(rewards, delta, self.cfg.beta_target)
        ess = ess_from_log_weights(logw)
        w = np.exp(logw - logw.max())
        idx = systematic_resample(w, self.rng)
        self.log.log("stage_start", stage=stage_idx, lam_prev=lam_prev, lam_next=lam_next,
                     beta_t=beta_t, ess=ess, ancestors=idx.tolist(),
                     rewards=rewards.tolist())

        # clone 重采样父代 → 新粒子群（clone 不重新评估，复制父的 eval）
        chain: list[RewardParticle] = []
        for a in idx:
            parent = particles[int(a)]
            clone = RewardParticle(
                id=self._next_id(), reward_code=parent.reward_code, eval=parent.eval,
                island_id=self.cfg.island_id, generation=parent.generation + 1,
                proposal_parent_id=parent.proposal_parent_id,
                accepted_transition_parent_id=parent.accepted_transition_parent_id,
                clone_ancestor_id=parent.id,
            )
            clone.metadata["state_origin_id"] = parent.metadata.get("state_origin_id", parent.id)
            clone.artifact_dir = parent.artifact_dir
            self.log.log("resample_clone", id=clone.id, ancestor=parent.id)
            chain.append(self._mutate_chain(clone, beta_t, stage_idx))
        return chain, lam_next, ess

    def _mutate_chain(self, current: RewardParticle, beta_t: float, stage_idx: int) -> RewardParticle:
        """对一个重采样粒子做 K 次 reward-only MH 提议（计划 §6.6）。"""
        for step in range(self.cfg.n_proposals):
            child_code = self.proposer.reflect(current.reward_code, current.eval.feedback)
            if child_code is None or child_code == current.reward_code:
                self.log.log("proposal_noop", id=current.id, stage=stage_idx, step=step)
                continue
            child = self._make_particle(
                child_code, generation=current.generation,
                proposal_parent_id=current.id,
                accepted_transition_parent_id=current.metadata.get("state_origin_id", current.id),
                clone_ancestor_id=None,
            )
            if not child.valid or child.search_score is None:
                # mutation 无效直接拒绝，保留父代（计划 §4.2）
                self.log.log("accept_decision", parent=current.id, child=child.id,
                             stage=stage_idx, step=step, accepted=False, reason="invalid_child")
                continue
            delta_r = child.search_score - (current.search_score or 0.0)
            if delta_r >= 0.0:
                accept = True
            else:
                accept = self.rng.random() < float(np.exp(beta_t * delta_r))
            self.log.log("accept_decision", parent=current.id, child=child.id, stage=stage_idx,
                         step=step, delta_r=delta_r, beta_t=beta_t, accepted=accept)
            if accept:
                current = child  # child 成为下一步提议源
        return current

    # ---- 主循环（计划 §6.8）----
    def run(self) -> IslandResult:
        particles = self.initialize()
        if len(particles) < 1:
            return IslandResult(self.best, "insufficient_valid_particles", 0, 0.0, 0.0)

        lam = 0.0
        stage = 0
        min_ess = float(len(particles))
        while lam < 1.0 - 1e-9 and stage < self.cfg.max_iters:
            particles, lam, ess = self._stage(particles, lam, stage)
            min_ess = min(min_ess, ess)
            stage += 1
            self.log.log("stage_end", stage=stage, lam=lam,
                         best_score=self.best.search_score if self.best else None,
                         unique_lineages=len({p.metadata.get("state_origin_id") for p in particles}))

        reason = "annealing_complete" if lam >= 1.0 - 1e-9 else "budget_exhausted"
        self.log.log("island_done", termination_reason=reason, n_stages=stage,
                     final_lambda=lam, best_id=self.best.id if self.best else None,
                     best_score=self.best.search_score if self.best else None)
        return IslandResult(self.best, reason, stage, lam, min_ess)
