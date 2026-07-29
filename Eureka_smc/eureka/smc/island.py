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

from .actions import ActionConfig, assign_actions
from .contracts import check_mutation_parameter, check_mutation_structure
from .context import format_crossover, format_different, format_path
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
    action_cfg: Optional[ActionConfig] = None  # 五操作路由（None/generic=单一 eureka_reflection）


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
        # RF-Agent 历史型 action（crossover/path/different）的料源：注册每个**被评估过**的
        # 粒子（id→粒子），供体/谱系/异谱系意图都从这里确定性选出。generic/变异路径不读它，
        # 只写不读，故不影响 Phase-2/3a 行为与 rng 消费。
        self._registry: dict[str, RewardParticle] = {}
        ac = config.action_cfg
        self._crossover_k = getattr(ac, "crossover_k", 2) if ac is not None else 2
        self._history_k = getattr(ac, "history_k", 4) if ac is not None else 4

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

    def _reflect_many(self, items: list[tuple[str, str]]) -> list[Optional[str]]:
        """并发反思（真实 proposer 有 reflect_batch）；否则串行兜底（Fake/测试保持确定性）。"""
        if hasattr(self.proposer, "reflect_batch"):
            return self.proposer.reflect_batch(items)
        return [self.proposer.reflect(code, fb) for code, fb in items]

    def _propose_many(
        self, current: list[RewardParticle]
    ) -> list[tuple[Optional[str], Optional[str], Optional[str]]]:
        """返回与 ``current`` 对齐的 ``(child_code, action, design_thought)``。

        ``generic``（或无 action_cfg）→ 完全走旧的单一 eureka_reflection 路径，action/thought
        为 None，**rng 消费与 Phase-2 逐字节一致**。``rf`` → 确定性 ``assign_actions`` 派 action
        （不抽 rng）+ ``propose_batch``。proposer 无 propose_batch 时退回 reflect（不带 action 指令）。
        """
        ac = self.cfg.action_cfg
        if ac is None or ac.mode == "generic":
            codes = self._reflect_many([(c.reward_code, c.eval.feedback) for c in current])
            return [(code, None, None) for code in codes]
        acts = assign_actions(len(current), ac.enabled, ac.enabled_weights())
        if hasattr(self.proposer, "propose_batch"):
            items = [(c.reward_code, c.eval.feedback, a, self._action_context(c, a))
                     for c, a in zip(current, acts)]
            pairs = self.proposer.propose_batch(items)
        else:  # 兜底：老式 proposer 无 propose_batch → reflect（丢失 action 指令，仍可跑）
            pairs = [(self.proposer.reflect(c.reward_code, c.eval.feedback), None)
                     for c in current]
        return [(code, act, thought) for (code, thought), act in zip(pairs, acts)]

    # ---- 历史型 action 的确定性料源（crossover/path/different；无 RNG）----
    def _action_context(self, c: RewardParticle, action: str) -> Optional[str]:
        """为历史型 action 从注册表确定性组装上下文块；无可用料 → None（退化为普通反思）。

        变异 action（m1/m2）与 generic 恒返回 None，不读注册表，故不影响默认路径。
        """
        if action == "crossover":
            donors = self._elite_donors(c.reward_code, self._crossover_k)
            return format_crossover(donors) if donors else None
        if action == "path_reasoning":
            chain = self._accepted_chain(c.metadata.get("state_origin_id"))
            return format_path(chain) if len(chain) >= 2 else None
        if action == "different_thought":
            chain = self._accepted_chain(c.metadata.get("state_origin_id"))
            own_ids = {n.id for n in chain} | {c.id}
            others = self._other_thoughts(own_ids, self._history_k)
            return format_different(others) if others else None
        return None

    def _elite_donors(self, exclude_code: str, k: int) -> list[RewardParticle]:
        """注册表里分数最高的 k 个 valid 供体（去重代码、排除粒子自身当前代码）。

        确定性：按 (分数降序, id 升序) 排序；同代码只保留其最佳评估实例。
        """
        best_by_code: dict[str, RewardParticle] = {}
        for p in self._registry.values():
            if not (p.valid and p.search_score is not None) or p.reward_code == exclude_code:
                continue
            q = best_by_code.get(p.reward_code)
            if q is None or p.search_score > q.search_score or (
                    p.search_score == q.search_score and p.id < q.id):
                best_by_code[p.reward_code] = p
        uniq = sorted(best_by_code.values(),
                      key=lambda p: (-(p.search_score or -np.inf), p.id))
        return uniq[:k]

    def _accepted_chain(self, start_id: Optional[str]) -> list[RewardParticle]:
        """沿 accepted_transition_parent_id 回溯注册表，返回时间正序（最早→最新）的接受链。

        起点为当前代码状态的 state_origin_id（真实被评估、被接受的代码）；clone 本身不入注册表，
        故从其 state_origin_id 起步。链上每个都是真实评估状态，天然跳过 clone/reject。
        """
        chain: list[RewardParticle] = []
        seen: set = set()
        nid = start_id
        while nid and nid in self._registry and nid not in seen:
            seen.add(nid)
            node = self._registry[nid]
            chain.append(node)
            nid = node.accepted_transition_parent_id
        chain.reverse()
        return chain

    def _other_thoughts(self, exclude_ids: set, k: int) -> list[RewardParticle]:
        """注册表里带 design_thought、且不在 ``exclude_ids``（本谱系）中的异谱系粒子。

        去重设计意图（同一句意图只留最佳分实例），按 (分数降序, id 升序) 取前 k。
        """
        best_by_thought: dict[str, RewardParticle] = {}
        for p in self._registry.values():
            if not p.valid or not p.design_thought or p.id in exclude_ids:
                continue
            key = " ".join(p.design_thought.strip().split())
            if not key:
                continue
            q = best_by_thought.get(key)
            pscore = p.search_score if p.search_score is not None else -np.inf
            qscore = (q.search_score if q and q.search_score is not None else -np.inf)
            if q is None or pscore > qscore or (pscore == qscore and p.id < q.id):
                best_by_thought[key] = p
        uniq = sorted(best_by_thought.values(),
                      key=lambda p: (-(p.search_score or -np.inf), p.id))
        return uniq[:k]

    def _audit_contract(self, particle: RewardParticle, parent_code: str, action: str) -> None:
        """按 action 对 child 做 AST 契约审计（只记 metadata + 事件，不 gate）。"""
        if action == "mutation_structure":
            hit = check_mutation_structure(parent_code, particle.reward_code)
        elif action == "mutation_parameter":
            hit = check_mutation_parameter(parent_code, particle.reward_code)
        else:
            return  # 其它 action（crossover/path/different）Phase-3a 无契约
        particle.metadata["contract_action"] = action
        particle.metadata["contract_hit"] = hit
        self.log.log("contract_audit", id=particle.id, action=action, contract_hit=hit)

    def _make_particles_batch(self, specs: list[dict]) -> list[RewardParticle]:
        """给一批代码分配 id → 一次 ``evaluate_batch`` 并发评估 → 构造粒子（顺序对齐）。

        ``specs`` 每项含 code + provenance 字段；id 按 specs 顺序分配（``_next_id`` 仅自增
        计数器、不消费 rng），evaluate_batch 对齐返回，因此每个 evaluator 的 rng 仍按
        0..N-1 顺序被消费——与旧逐粒子路径等价，Fake 集成测试确定性不变。
        """
        if not specs:
            return []
        ids = [self._next_id() for _ in specs]
        dirs = [self._artifact(pid) for pid in ids]
        items = [(s["code"], pid, d) for s, pid, d in zip(specs, ids, dirs)]
        recs = self.evaluator.evaluate_batch(items)
        particles: list[RewardParticle] = []
        for s, pid, d, rec in zip(specs, ids, dirs, recs):
            p = RewardParticle(
                id=pid, reward_code=s["code"], eval=rec, island_id=self.cfg.island_id,
                generation=s["generation"], proposal_parent_id=s.get("proposal_parent_id"),
                accepted_transition_parent_id=s.get("accepted_transition_parent_id"),
                clone_ancestor_id=s.get("clone_ancestor_id"), artifact_dir=d,
                proposal_action=s.get("proposal_action"),
                design_thought=s.get("design_thought"),
            )
            p.metadata["state_origin_id"] = s.get("state_origin_id") or pid
            # RF-Agent 契约审计（仅对 mutation_* 且有父代码时；只记录不 gate）
            if s.get("proposal_action") and s.get("parent_code"):
                self._audit_contract(p, s["parent_code"], s["proposal_action"])
            self.log.log("eval", id=pid, valid=rec.valid, search_score=rec.search_score,
                         generation=s["generation"], proposal_parent_id=s.get("proposal_parent_id"),
                         accepted_transition_parent_id=s.get("accepted_transition_parent_id"),
                         clone_ancestor_id=s.get("clone_ancestor_id"),
                         proposal_action=s.get("proposal_action"))
            self._consider_best(p)
            self._registry[p.id] = p  # 登记料源（历史型 action 用；只写不读默认路径）
            particles.append(p)
        return particles

    # ---- 初始化（计划 §6.1）----
    def initialize(self) -> list[RewardParticle]:
        n = self.cfg.n_particles
        particles: list[RewardParticle] = []
        budget = n + self.cfg.max_init_retries * n
        codes = list(self.proposer.initial_batch(n))  # 单次 n= 调用，服务端并行
        attempts = 0
        while len(particles) < n and attempts < budget:
            need = n - len(particles)
            if not codes:
                codes = list(self.proposer.initial_batch(need))
            batch_codes: list[str] = []
            while codes and len(batch_codes) < need and attempts < budget:
                code = codes.pop()
                attempts += 1
                if code is None:
                    continue
                batch_codes.append(code)
            if not batch_codes:
                continue
            new_particles = self._make_particles_batch(
                [dict(code=c, generation=0) for c in batch_codes])  # 一波并发评估
            for p in new_particles:
                if p.valid:
                    particles.append(p)
                else:
                    self.log.log("init_invalid", id=p.id)
        self.log.log("init_done", n_valid=len(particles), attempts=attempts)
        return particles[:n]

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
        current: list[RewardParticle] = []
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
            current.append(clone)

        # K 轮（首版 K=1）批量 MH：每轮对所有粒子当前状态**并发** reflect + evaluate，
        # 再逐粒子接受。轮内 in-place 更新 current[i]，故 K>1 时每粒子链依赖仍正确保留。
        for step in range(self.cfg.n_proposals):
            proposals = self._propose_many(current)
            specs, slots = [], []
            for i, (c, (child_code, action, thought)) in enumerate(zip(current, proposals)):
                if child_code is None or child_code == c.reward_code:
                    self.log.log("proposal_noop", id=c.id, stage=stage_idx, step=step,
                                 action=action)
                    continue
                specs.append(dict(
                    code=child_code, generation=c.generation, proposal_parent_id=c.id,
                    accepted_transition_parent_id=c.metadata.get("state_origin_id", c.id),
                    clone_ancestor_id=None, proposal_action=action, design_thought=thought,
                    parent_code=c.reward_code))
                slots.append(i)
            children = self._make_particles_batch(specs)
            for child, i in zip(children, slots):
                current[i] = self._mh_accept(current[i], child, beta_t, stage_idx, step)
        return current, lam_next, ess

    def _mh_accept(self, current: RewardParticle, child: RewardParticle,
                   beta_t: float, stage_idx: int, step: int) -> RewardParticle:
        """reward-only MH 接受（计划 §6.6）：ΔR≥0 必接受；否则以 exp(β_t·ΔR) 概率接受。"""
        if not child.valid or child.search_score is None:
            self.log.log("accept_decision", parent=current.id, child=child.id, stage=stage_idx,
                         step=step, accepted=False, reason="invalid_child")
            return current
        delta_r = child.search_score - (current.search_score or 0.0)
        if delta_r >= 0.0:
            accept = True
        else:
            accept = self.rng.random() < float(np.exp(beta_t * delta_r))
        self.log.log("accept_decision", parent=current.id, child=child.id, stage=stage_idx,
                     step=step, delta_r=delta_r, beta_t=beta_t, accepted=accept)
        return child if accept else current

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
