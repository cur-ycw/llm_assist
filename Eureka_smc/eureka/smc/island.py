"""单岛预算条件化 KL 奖励搜索循环（新方法：预算与进展自适应 SMC 奖励搜索 §2）。

流程：
  initialize() 生成 N 个 valid 初始粒子（不做分数选择；init 的 N 次 LLM 调用单独计入 B_total=N+B）
  → 循环修改轮，直到修改预算 b_t 耗尽（budget_exhausted）：
        1. 由**剩余预算** b_t 解算目标有效父代数 K*_t、KL 半径 δ_t、选择强度 λ_t 与父代分布
           q_t = softmax(λ_t·J)（kl_controller.resolve；J 为**原始**任务性能，不归一化）
        2. 抽 M_t = min(N, b_t) 个父代副本 A^j ~ Categorical(q_t)（multinomial_resample）→ clone
        3. 每个副本一次 LLM 修改（RF 五操作路由）+ 评估 + **sigmoid 非对称接受**：改进确定进入，
           非改进以 σ(λ_t·Δ) 概率进入；拒绝则保留父代代码
        4. b_t -= M_t（每次 LLM 修改调用都计预算，含无效候选，§7.2）
  → 返回与粒子群分离保存的 best_so_far（按原始 J）及终止原因。

选择集中度**只由剩余预算决定**（不看 ESS / 近期成功率 / EMA，§2.3）：预算充足→K*≈N（广探、
首轮 λ=0 均匀），预算耗尽→K*→K_min（集中）。这取代了旧的 ESS 自适应退火桥（temperature.py 已删）。

provenance（不变量）：
  * clone 只写 ``clone_ancestor_id``，继承父的 accepted_transition，不伪造代码优化；
  * 被接受的 child 的 ``accepted_transition_parent_id`` 指向被替换代码的真实来源
    （``state_origin_id``），因此沿它回溯得到纯 accepted-edit 链，自动跳过 clone/reject。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import kl_controller
from .actions import ActionConfig, assign_actions
from .contracts import check_mutation_parameter, check_mutation_structure
from .context import format_crossover, format_different, format_path
from .event_logger import EventLogger
from .particle import EvalRecord, RewardParticle
from .resampling import multinomial_resample

logger = logging.getLogger(__name__)

__all__ = ["SMCIslandConfig", "SMCIsland", "IslandResult"]


@dataclass
class SMCIslandConfig:
    island_id: int = 0
    n_particles: int = 16           # 初始父代种群规模（init 生成并修复到这么多有效种子）
    children_per_round: int = 8     # 每轮子代数 M（=重采样-变异次数；Option A：init 后种群稳定为此值）
    budget: int = 64               # 修改预算 B（初始种群后允许的 LLM 修改次数）
    k_min: int = 2                 # 有效父代数下界 K_min（预算耗尽时的集中度）
    gamma: float = 1.0             # K*_t = K_min + (N-K_min)(b_t/B)^gamma 的曲率
    max_init_repair: int = 8        # 初始种子 traceback-repair 的最大波数（RF-Agent 式，≈其 max_try_num=9）
    max_same_repair: int = 3        # 同一种子连续修复失败多少次后丢弃、改抽全新（RF-Agent max_same_try_cnt）
    seed: int = 0
    action_cfg: Optional[ActionConfig] = None  # 五操作路由（None/generic=单一 eureka_reflection）


@dataclass
class IslandResult:
    best: Optional[RewardParticle]
    termination_reason: str         # budget_exhausted | insufficient_valid_particles
    n_rounds: int
    budget_used: int                # 已消耗的 LLM 修改调用数（应等于 B）
    last_lambda: float              # 最后一轮的选择强度 λ_t（诊断）


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

    # ---- 初始化（计划 §6.1 + RF-Agent 初始 traceback-repair）----
    def initialize(self) -> list[RewardParticle]:
        """生成并**修复**到 N 个有效初始种子（RF-Agent 式：报错喂回让 LLM 自 debug）。

        逐波进行：生成一波候选 → 并发评估 → 有效的收下；无效的把其 traceback 喂回 proposer
        的 ``repair_batch`` 修复（同一种子连续失败 ``max_same_repair`` 次则丢弃、改抽全新），
        直到集满 N 个有效或达到 ``max_init_repair`` 波数上限。冷启动可执行率低的任务（如 Ant）
        靠这个自修复把种群顶到 N，而不是像旧版那样冷抽到上限就返回残缺种群。
        """
        n = self.cfg.n_particles
        can_repair = (hasattr(self.proposer, "repair_batch") and self.cfg.max_init_repair > 0)
        valid: list[RewardParticle] = []
        wave = [dict(code=c, tries=0) for c in self.proposer.initial_batch(n) if c]
        waves = 0
        max_waves = self.cfg.max_init_repair + 1
        while len(valid) < n and waves < max_waves:
            need = n - len(valid)
            if len(wave) < need:  # None/不足 → 用全新初始码补足本波
                extra = [c for c in self.proposer.initial_batch(need - len(wave)) if c]
                wave.extend(dict(code=c, tries=0) for c in extra)
            wave = wave[:need]
            if not wave:
                break
            particles = self._make_particles_batch(
                [dict(code=w["code"], generation=0) for w in wave])
            waves += 1
            repairs: list[tuple[str, str, int]] = []  # (failed_code, traceback, next_tries)
            fresh_need = 0
            for w, p in zip(wave, particles):
                if p.valid:
                    valid.append(p)
                    if len(valid) >= n:
                        break
                else:
                    self.log.log("init_invalid", id=p.id, tries=w["tries"])
                    if not can_repair or w["tries"] + 1 >= self.cfg.max_same_repair:
                        fresh_need += 1                          # 连修失败 / 不支持修复 → 换全新
                    else:
                        repairs.append((p.reward_code, p.eval.error or "execution error",
                                        w["tries"] + 1))
            if len(valid) >= n:
                break
            next_wave: list[dict] = []
            if repairs:  # 带 traceback 的自修复
                fixed = self.proposer.repair_batch([(c, tb) for c, tb, _ in repairs])
                next_wave.extend(dict(code=rc, tries=tries)
                                 for (_, _, tries), rc in zip(repairs, fixed) if rc)
            if fresh_need:  # 换全新初始码
                fresh = [c for c in self.proposer.initial_batch(fresh_need) if c]
                next_wave.extend(dict(code=c, tries=0) for c in fresh)
            wave = next_wave
        self.log.log("init_done", n_valid=len(valid), waves=waves)
        return valid[:n]

    # ---- 一个修改轮（新方法 §2.3–2.6）----
    @staticmethod
    def _gini(counts) -> float:
        """修改次数分配的 Gini 系数（§8.4；0=均匀，→1=集中到单父代）。"""
        x = np.sort(np.asarray(counts, dtype=np.float64))
        n = x.size
        s = x.sum()
        if n == 0 or s <= 0.0:
            return 0.0
        cum = np.cumsum(x)
        return float((n + 1 - 2.0 * cum.sum() / cum[-1]) / n)

    def _round(self, particles: list[RewardParticle], b_t: int,
               round_idx: int) -> tuple[list[RewardParticle], "kl_controller.ControllerStep", int]:
        n = len(particles)
        # 原始 J（population 均为 valid；None 兜底为当轮有限最小值，控制器要求有限输入）
        raw = [p.search_score for p in particles]
        finite = [x for x in raw if x is not None]
        floor = min(finite) if finite else 0.0
        J = np.array([x if x is not None else floor for x in raw], dtype=np.float64)

        step = kl_controller.resolve(J, float(b_t), float(self.cfg.budget), n,
                                     float(self.cfg.k_min), float(self.cfg.gamma))
        m_t = min(self.cfg.children_per_round, int(b_t))   # 每轮子代数 M（Option A：种群随之稳定为 M）
        idx = multinomial_resample(step.q, m_t, self.rng)

        counts = np.bincount(idx, minlength=n)
        self.log.log(
            "stage_start", stage=round_idx, budget_remaining=int(b_t),
            budget_frac=float(b_t) / float(self.cfg.budget), m_t=m_t,
            k_star=step.k_star, k_feas=step.k_feas, m_ties=step.m_ties,
            kl_requested=step.delta_req, kl_feasible=step.delta_feas,
            kl_actual=step.kl_actual, lam=step.lam, k_eff=step.k_eff,
            max_parent_prob=step.max_q, kl_saturated=step.saturated,
            unique_parents=int(np.count_nonzero(counts)),
            resample_counts=counts.tolist(), alloc_gini=self._gini(counts),
            pop_score_min=float(J.min()), pop_score_median=float(np.median(J)),
            pop_score_max=float(J.max()), pop_score_range=float(J.max() - J.min()),
            archive_best=self.best.search_score if self.best else None,
            ancestors=idx.tolist(),
        )

        # clone 重采样父代 → 本轮活动集（clone 不重新评估，复制父的 eval）
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

        # 每个副本一次 LLM 修改（RF 五操作路由）→ 并发评估 → 逐粒子 sigmoid 接受。
        proposals = self._propose_many(current)
        specs, slots = [], []
        for i, (c, (child_code, action, thought)) in enumerate(zip(current, proposals)):
            if child_code is None or child_code == c.reward_code:
                self.log.log("proposal_noop", id=c.id, stage=round_idx, action=action)
                continue
            specs.append(dict(
                code=child_code, generation=c.generation, proposal_parent_id=c.id,
                accepted_transition_parent_id=c.metadata.get("state_origin_id", c.id),
                clone_ancestor_id=None, proposal_action=action, design_thought=thought,
                parent_code=c.reward_code))
            slots.append(i)
        children = self._make_particles_batch(specs)
        for child, i in zip(children, slots):
            current[i] = self._sigmoid_accept(current[i], child, step.lam, round_idx)
        return current, step, m_t

    def _sigmoid_accept(self, current: RewardParticle, child: RewardParticle,
                        lam_t: float, round_idx: int) -> RewardParticle:
        """非对称 sigmoid 接受（新方法 §2.6）：Δ>0 确定进入；Δ≤0 以 σ(λ_t·Δ) 概率进入。"""
        if not child.valid or child.search_score is None:
            self.log.log("accept_decision", parent=current.id, child=child.id, stage=round_idx,
                         accepted=False, reason="invalid_child")
            return current
        delta = child.search_score - (current.search_score or 0.0)
        improved = delta > 0.0
        if improved:
            accept = True
            p_accept = 1.0
        else:
            p_accept = float(1.0 / (1.0 + np.exp(-lam_t * delta)))  # σ(λ_t·Δ), Δ≤0 → ≤0.5
            accept = self.rng.random() < p_accept
        self.log.log("accept_decision", parent=current.id, child=child.id, stage=round_idx,
                     delta=delta, lam=lam_t, p_accept=p_accept, improved=improved,
                     accepted=accept)
        return child if accept else current

    # ---- 主循环（新方法 §2.5：预算耗尽即停）----
    def run(self) -> IslandResult:
        particles = self.initialize()
        if len(particles) < 1:
            return IslandResult(self.best, "insufficient_valid_particles", 0, 0, 0.0)

        b_t = int(self.cfg.budget)
        round_idx = 0
        last_lambda = 0.0
        # 每轮花 min(children_per_round, b_t)（≥1）→ 至多 ceil(B/M) 轮；+1 兜底
        max_rounds = math.ceil(self.cfg.budget / max(1, self.cfg.children_per_round)) + 1
        while b_t > 0 and round_idx < max_rounds:
            particles, step, m_t = self._round(particles, b_t, round_idx)
            b_t -= m_t                      # 每次 LLM 修改调用计预算（含无效候选，§7.2）
            last_lambda = step.lam
            round_idx += 1
            self.log.log("stage_end", stage=round_idx, budget_remaining=b_t,
                         best_score=self.best.search_score if self.best else None,
                         unique_lineages=len({p.metadata.get("state_origin_id") for p in particles}))

        budget_used = int(self.cfg.budget) - max(0, b_t)
        self.log.log("island_done", termination_reason="budget_exhausted", n_rounds=round_idx,
                     budget_used=budget_used, last_lambda=last_lambda,
                     best_id=self.best.id if self.best else None,
                     best_score=self.best.search_score if self.best else None)
        return IslandResult(self.best, "budget_exhausted", round_idx, budget_used, last_lambda)
