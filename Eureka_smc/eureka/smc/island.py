"""单岛预算条件化 KL 奖励搜索循环（新方法：预算与进展自适应 SMC 奖励搜索 §2）。

流程：
  initialize() 生成 N 个 valid 初始粒子（不做分数选择）。**预算记账**：总预算 budget=B_total
  即含这 N 个初始生成（B_total = N + 修改次数）；init 一建立就"花掉" N 个预算，故首个修改轮
  的剩余预算已是 b_t = B_total − N（而非 B_total），集中度从一开始就非均匀——这 N 次初始生成
  已经消耗了预算、也已经承载了初始奖励的强度（用户明示：init 不是边界特例，它就在 B_total 里）。
  → 循环修改轮，直到剩余预算 b_t 耗尽（budget_exhausted）：
        1. 由**剩余预算** b_t 解算目标有效父代数 K*_t、KL 半径 δ_t、选择强度 λ_t 与父代分布
           q_t = softmax(λ_t·J)（kl_controller.resolve；J 为**原始**任务性能，不归一化）
        2. 从 N 个父代槽位抽取 M_t 个待修改 slot（multinomial，可重复）→ clone
        3. 每个副本一次 LLM 修改（RF 五操作路由）+ 评估 + **sigmoid 非对称接受**：改进确定进入，
           非改进以 σ(λ_t·Δ) 概率进入；拒绝则保留对应槽位的父代代码
        4. b_t -= M_t（每次 LLM 修改调用都计预算，含无效候选，§7.2）
  → 返回与粒子群分离保存的 best_so_far（按原始 J）及终止原因。

选择集中度**只由剩余预算决定**（不看 ESS / 近期成功率 / EMA，§2.3）：分母恒为 B_total，故首个
修改轮 b_t/B_total = (B_total−N)/B_total < 1 → 已有集中度（不是 λ=0 均匀）；预算耗尽→K*→K_min
（最集中）。这取代了旧的 ESS 自适应退火桥（temperature.py 已删）。

provenance（不变量）：
  * clone 只写 ``clone_ancestor_id``，继承父的 accepted_transition，不伪造代码优化；
  * 被接受的 child 的 ``accepted_transition_parent_id`` 指向被替换代码的真实来源
    （``state_origin_id``），因此沿它回溯得到纯 accepted-edit 链，自动跳过 clone/reject。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import kl_controller, progress
from .actions import ActionConfig, assign_actions
from .archive import CandidateArchive
from .contracts import check_mutation_parameter, check_mutation_structure
from .context import format_crossover, format_different, format_path
from .event_logger import EventLogger
from .ledger import BudgetLedger
from .particle import EvalRecord, RewardParticle
from .resampling import multinomial_resample

logger = logging.getLogger(__name__)

__all__ = ["SMCIslandConfig", "SMCIsland", "IslandResult"]


@dataclass
class SMCIslandConfig:
    island_id: int = 0
    n_particles: int = 16           # 初始化生成并保持的父代槽位数 N
    children_per_round: int = 8     # 每轮修改子代数 M；Isaac Gym 标准为 8
    mutation_rounds: Optional[int] = None # init 后修改轮数；None 时由修改预算推导
    budget: int = 80                 # 兼容字段：总逻辑候选预算 = init_budget + mutation_budget
    init_budget: Optional[int] = None     # 初始生成逻辑预算；默认等于 N
    mutation_budget: Optional[int] = None # 后续修改逻辑预算；默认等于 rounds×M
    k_min: int = 2                 # 有效父代数下界 K_min（预算耗尽时的集中度）
    gamma: float = 1.0             # 兼容保留（旧预算幂日程曲率；Full 控制器忽略，仅诊断用）
    eta: float = 1.0               # 进展延迟反馈增益：τ_t = τ_budget(h)·exp(-η·h·Γ_{t-1})
    max_init_repair: int = 8        # 初始种子 traceback-repair 的最大波数（RF-Agent 式，≈其 max_try_num=9）
    max_same_repair: int = 3        # 同一种子连续修复失败多少次后丢弃、改抽全新（RF-Agent max_same_try_cnt）
    seed: int = 0
    action_cfg: Optional[ActionConfig] = None  # 五操作路由（None/generic=单一 eureka_reflection）

    def __post_init__(self) -> None:
        """校验初始化池、每轮修改资源和总候选预算的关系。"""
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.children_per_round <= 0:
            raise ValueError("children_per_round must be positive")
        init_budget = self.n_particles if self.init_budget is None else int(self.init_budget)
        if init_budget != self.n_particles:
            raise ValueError("init_budget must equal n_particles")
        explicit_rounds = self.mutation_rounds is not None
        explicit_mutation_budget = self.mutation_budget is not None
        if explicit_rounds and self.mutation_rounds < 0:
            raise ValueError("mutation_rounds must be non-negative")
        if explicit_mutation_budget and self.mutation_budget < 0:
            raise ValueError("mutation_budget must be non-negative")
        if explicit_rounds and explicit_mutation_budget:
            if self.mutation_budget != self.mutation_rounds * self.children_per_round:
                raise ValueError(
                    "mutation_budget must equal mutation_rounds * children_per_round"
                )
        elif explicit_rounds:
            self.mutation_budget = self.mutation_rounds * self.children_per_round
        elif explicit_mutation_budget:
            self.mutation_rounds = math.ceil(
                self.mutation_budget / self.children_per_round
            ) if self.mutation_budget else 0
        else:
            mutation_budget = self.budget - init_budget
            if mutation_budget < 0:
                raise ValueError("budget must cover the initial particle batch")
            self.mutation_budget = mutation_budget
            self.mutation_rounds = math.ceil(
                mutation_budget / self.children_per_round
            ) if mutation_budget else 0
        self.init_budget = init_budget
        expected_budget = self.init_budget + self.mutation_budget
        if self.budget != expected_budget:
            raise ValueError("budget must equal init_budget + mutation_budget")
        if self.k_min < 1 or self.k_min > self.n_particles:
            raise ValueError("k_min must lie in [1, n_particles]")
        if self.eta < 0 or not np.isfinite(self.eta):
            raise ValueError("eta must be finite and non-negative")


@dataclass
class IslandResult:
    best: Optional[RewardParticle]
    termination_reason: str         # budget_exhausted | insufficient_valid_particles
    n_rounds: int
    budget_used: int                # 已消耗的总预算（init N + LLM 修改调用），应等于 B_total
    last_lambda: float              # 派生诊断 λ_equivalent=alpha/span（兼容字段）
    last_alpha: float = 0.0         # 最后一轮的无量纲选择强度 alpha_t


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
        self.best: Optional[RewardParticle] = None  # all valid evaluated candidates' search-best
        self.archive = CandidateArchive()
        self.ledger = BudgetLedger()
        # RF-Agent 历史型 action（crossover/path/different）的料源：注册每个**被评估过**的
        # 粒子（id→粒子），供体/谱系/异谱系意图都从这里确定性选出。generic/变异路径不读它，
        # 只写不读，故不影响 Phase-2/3a 行为与 rng 消费。
        self._registry: dict[str, RewardParticle] = {}
        ac = config.action_cfg
        self._crossover_k = getattr(ac, "crossover_k", 2) if ac is not None else 2
        self._history_k = getattr(ac, "history_k", 4) if ac is not None else 4

    def _config_fingerprint(self) -> str:
        payload = asdict(self.cfg)
        payload["action_cfg"] = asdict(self.cfg.action_cfg) if self.cfg.action_cfg else None
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _component_runtime_state(component: Any) -> dict[str, Any]:
        state = {
            name: int(getattr(component, name))
            for name in ("n_calls", "total_prompt_tokens", "total_completion_tokens", "n_evals")
            if hasattr(component, name)
        }
        rng = getattr(component, "rng", None)
        if isinstance(rng, np.random.Generator):
            state["rng_state"] = rng.bit_generator.state
        return state

    @staticmethod
    def _restore_component_runtime_state(component: Any, state: dict[str, Any]) -> None:
        for name in ("n_calls", "total_prompt_tokens", "total_completion_tokens", "n_evals"):
            if name in state and hasattr(component, name):
                setattr(component, name, int(state[name]))
        rng = getattr(component, "rng", None)
        if "rng_state" in state and isinstance(rng, np.random.Generator):
            rng.bit_generator.state = state["rng_state"]

    def snapshot_runtime(self, *, phase: str, population: list[RewardParticle],
                         budget_remaining: int, round_idx: int,
                         progress_prev: float, last_alpha: float,
                         last_lambda: float) -> dict[str, Any]:
        """Return a stage-boundary-only JSON state for deterministic resume."""
        self.log.flush()
        return {
            "schema_version": 1,
            "phase": phase,
            "config_fingerprint": self._config_fingerprint(),
            "population": [p.to_json() for p in population],
            "best_id": self.best.id if self.best else None,
            "registry": [p.to_json() for p in self._registry.values()],
            "archive": self.archive.snapshot(),
            "ledger": self.ledger.snapshot(),
            "budget_remaining": int(budget_remaining),
            "round_idx": int(round_idx),
            "progress_prev": float(progress_prev),
            "last_alpha": float(last_alpha),
            "last_lambda": float(last_lambda),
            "id_counter": int(self._id_counter),
            "rng_state": self.rng.bit_generator.state,
            "proposer_runtime": self._component_runtime_state(self.proposer),
            "evaluator_runtime": self._component_runtime_state(self.evaluator),
            "event_next_seq": self.log.next_seq,
        }

    def restore_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        """Restore a stage-boundary snapshot after validating its method contract."""
        if state.get("schema_version") != 1:
            raise ValueError("unsupported island runtime checkpoint schema")
        if state.get("config_fingerprint") != self._config_fingerprint():
            raise ValueError("checkpoint configuration fingerprint does not match")
        population = [RewardParticle.from_json(item) for item in state["population"]]
        if len(population) != self.cfg.n_particles:
            raise ValueError("checkpoint population does not match n_particles")
        registry = [RewardParticle.from_json(item) for item in state["registry"]]
        self._registry = {p.id: p for p in registry}
        if len(self._registry) != len(registry):
            raise ValueError("checkpoint registry contains duplicate particle IDs")
        self.archive = CandidateArchive.from_snapshot(state["archive"])
        self.ledger = BudgetLedger.from_snapshot(state["ledger"])
        best_id = state.get("best_id")
        self.best = self._registry.get(best_id) if best_id else None
        if best_id and self.best is None:
            raise ValueError("checkpoint best_id is absent from registry")
        self._id_counter = int(state["id_counter"])
        self.rng.bit_generator.state = state["rng_state"]
        self._restore_component_runtime_state(self.proposer, state.get("proposer_runtime", {}))
        self._restore_component_runtime_state(self.evaluator, state.get("evaluator_runtime", {}))
        return {
            "phase": state["phase"],
            "population": population,
            "budget_remaining": int(state["budget_remaining"]),
            "round_idx": int(state["round_idx"]),
            "progress_prev": float(state["progress_prev"]),
            "last_alpha": float(state["last_alpha"]),
            "last_lambda": float(state["last_lambda"]),
            "event_next_seq": int(state["event_next_seq"]),
        }

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
            self._registry[p.id] = p  # 全部已评估节点，供谱系审计与历史型 action 使用。
            self.archive.add(p)        # 所有 valid/finite 候选（含后续被拒子代）进入选择 archive。
            self._consider_best(p)
            particles.append(p)
        self.ledger.record_evaluations(particles)
        self.ledger.sync_actual_costs(self.proposer, self.evaluator)
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

    def _round(self, particles: list[RewardParticle], b_t: int, round_idx: int,
               progress_prev: float = 0.0
               ) -> tuple[list[RewardParticle], "kl_controller.ControllerStep", int, float]:
        n = len(particles)
        # 原始 J（population 均为 valid；None 兜底为当轮有限最小值，控制器要求有限输入）
        raw = [p.search_score for p in particles]
        finite = [x for x in raw if x is not None]
        floor = min(finite) if finite else 0.0
        J = np.array([x if x is not None else floor for x in raw], dtype=np.float64)

        # 首版冻结：势函数仅为原始任务搜索分数，所有固定槽位具有相等先验权重。
        # q 仍由 rESS 控制器根据 J 形成，uniform W 不等于均匀父代抽样。
        U_t = J
        W_t = np.full(n, 1.0 / float(n), dtype=np.float64)
        # 预算给出基础 τ；上一轮进展 Γ_{t-1} 经延迟反馈修正本轮目标（Full 控制器 §2.3）。
        step = kl_controller.resolve(J, float(b_t), float(self.cfg.budget), n,
                                     float(self.cfg.k_min),
                                     eta=float(self.cfg.eta),
                                     progress_prev=float(progress_prev),
                                     U=U_t, weights=W_t)
        m_t = min(self.cfg.children_per_round, int(b_t))   # 最后一轮允许不足 M，正常情况下固定为 M
        idx = multinomial_resample(step.q, m_t, self.rng)

        counts = np.bincount(idx, minlength=n)
        self.log.log(
            "stage_start", stage=round_idx, budget_remaining=int(b_t),
            budget_frac=float(b_t) / float(self.cfg.budget), m_t=m_t,
            eta=float(self.cfg.eta), progress_prev=step.progress_prev,
            tau_budget=step.tau_budget, tau_target=step.tau_target,
            tau_feasible=step.tau_feasible, relative_ess=step.relative_ess,
            k_star=step.k_star, k_feas=step.k_feas, m_ties=step.m_ties,
            kl_requested=step.delta_req, kl_feasible=step.delta_feas,
            kl_actual=step.kl_actual, lam=step.lam, k_eff=step.k_eff,
            alpha=step.alpha, potential_span=step.potential_span,
            normalized_potential_min=float(step.normalized_potential.min()),
            normalized_potential_max=float(step.normalized_potential.max()),
            lambda_equivalent=step.lambda_equivalent,
            potential_definition="raw_search_score", prior_weight_definition="uniform_slots",
            max_parent_prob=step.max_q, kl_saturated=step.saturated,
            unique_parents=int(np.count_nonzero(counts)),
            resample_counts=counts.tolist(), alloc_gini=self._gini(counts),
            pop_score_min=float(J.min()), pop_score_median=float(np.median(J)),
            pop_score_max=float(J.max()), pop_score_range=float(J.max() - J.min()),
            archive_best=self.best.search_score if self.best else None,
            ancestors=idx.tolist(),
        )

        # clone 重采样父代仅作为每个修改 slot 的 proposal 上下文；接受后写回原始 N 槽位。
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
                self.ledger.record_noop()
                continue
            specs.append(dict(
                code=child_code, generation=c.generation, proposal_parent_id=c.id,
                clone_ancestor_id=None, proposal_action=action, design_thought=thought,
                parent_code=c.reward_code))
            slots.append(i)
        children = self._make_particles_batch(specs)
        # Y_t：本轮**原始**子代分数（接受前，用于进展度量；无效子代按当轮 floor 兜底）。
        child_scores = [c.search_score if (c.valid and c.search_score is not None) else floor
                        for c in children]
        # N 槽位不变式：每个资源 slot 仍回写到其被抽中的原始父代槽位；
        # 同一父代被抽中多次时，按 proposal 顺序处理，最后一个被接受的 child 留在该槽位。
        next_population = list(particles)
        for child, i in zip(children, slots):
            accepted = self._sigmoid_accept(current[i], child, step.alpha,
                                            step.potential_span, round_idx)
            self.ledger.record_acceptance(accepted is child)
            if accepted is child:
                child.accepted_transition_parent_id = current[i].metadata.get(
                    "state_origin_id", current[i].id
                )
                child.metadata["state_origin_id"] = child.id
                next_population[int(idx[i])] = child

        # Γ_t: retain repeated resampling indices because each occurrence received
        # an independent LLM-modification resource in this stage.
        resource_parent_indices = idx.tolist()
        if child_scores and resource_parent_indices:
            pstep = progress.compute_progress(step.q, J, child_scores,
                                              parent_indices=resource_parent_indices)
            gamma_t = pstep.gamma
            self.log.log("progress", stage=round_idx, k_eff=pstep.k_eff, k=pstep.k,
                         auc=pstep.auc, gamma=gamma_t, n_children=len(child_scores),
                         n_resource_slots=len(resource_parent_indices),
                         n_unique_resource_parents=int(np.count_nonzero(counts)))
        else:
            gamma_t = 0.0  # 无子代（全 noop）→ 无进展信号，下一轮退回纯预算目标
            self.log.log("progress", stage=round_idx, gamma=gamma_t,
                         n_children=len(child_scores), n_resource_slots=len(resource_parent_indices),
                         n_unique_resource_parents=int(np.count_nonzero(counts)))
        return next_population, step, m_t, gamma_t

    def _sigmoid_accept(self, current: RewardParticle, child: RewardParticle,
                        alpha_t: float, potential_span: float = 1.0,
                        round_idx: Optional[int] = None) -> RewardParticle:
        """Use the dimensionless acceptance coordinate ``alpha * delta / span``.

        The four-argument form is retained for existing callers: its final
        positional value is the stage index and uses the historical unit span.
        The mutation path always supplies the explicit span and stage.
        """
        if round_idx is None:
            round_idx = int(potential_span)
            potential_span = 1.0
        if not child.valid or child.search_score is None:
            self.log.log("accept_decision", parent=current.id, child=child.id, stage=round_idx,
                         accepted=False, reason="invalid_child")
            return current
        delta = child.search_score - (current.search_score or 0.0)
        improved = delta > 0.0
        normalized_delta = delta / potential_span if potential_span > 0.0 else 0.0
        if improved:
            accept = True
            p_accept = 1.0
        elif potential_span == 0.0:
            p_accept = 0.5
            accept = self.rng.random() < p_accept
        else:
            z = float(alpha_t * normalized_delta)
            # Stable sigmoid; the non-improvement branch has z <= 0.
            p_accept = float(np.exp(z) / (1.0 + np.exp(z))) if z >= -40.0 else 0.0
            accept = self.rng.random() < p_accept
        self.log.log("accept_decision", parent=current.id, child=child.id, stage=round_idx,
                     delta=delta, normalized_delta=normalized_delta, alpha=alpha_t,
                     potential_span=potential_span,
                     lambda_equivalent=(alpha_t / potential_span if potential_span > 0.0 else 0.0),
                     p_accept=p_accept, improved=improved, accepted=accept)
        return child if accept else current

    # ---- 主循环（新方法 §2.5：预算耗尽即停）----
    def run(
        self,
        *,
        resume_state: Optional[dict[str, Any]] = None,
        checkpoint_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> IslandResult:
        """Run from a fresh init or a completed-stage runtime snapshot.

        ``checkpoint_callback`` is invoked only after init, a completed stage,
        and terminal completion. It must persist the supplied JSON state before
        returning or raise; no batch-in-flight state is checkpointed.
        """
        if resume_state is None:
            particles = self.initialize()
            if len(particles) != int(self.cfg.n_particles):
                return IslandResult(self.best, "insufficient_valid_particles", 0, 0, 0.0)

            # 预算记账：总预算 B_total 含 init 的 N 个初始生成 → init 一建立就扣掉 N，
            # 故首个修改轮的剩余预算已是 B_total − N（首轮集中度非均匀，用户明示 init 在预算内）。
            b_total = int(self.cfg.budget)
            b_t = int(self.cfg.mutation_budget)
            self.ledger.logical_init_slots = int(self.cfg.init_budget)
            self.ledger.sync_actual_costs(self.proposer, self.evaluator)
            self.log.log("init_budget_spent", n_init=int(self.cfg.n_particles),
                         init_budget=int(self.cfg.init_budget), mutation_budget=b_t,
                         budget_total=b_total, budget_remaining=b_t,
                         ledger=self.ledger.snapshot())
            round_idx = 0
            last_lambda = 0.0
            last_alpha = 0.0
            progress_prev = 0.0             # 首轮无历史进展 → 纯预算目标（Γ_0 = 0，计划 §2.3）
            if checkpoint_callback is not None:
                checkpoint_callback(self.snapshot_runtime(
                    phase="initialized", population=particles, budget_remaining=b_t,
                    round_idx=round_idx, progress_prev=progress_prev,
                    last_alpha=last_alpha, last_lambda=last_lambda,
                ))
        else:
            restored = self.restore_runtime(resume_state)
            if restored["phase"] not in {"initialized", "stage_end"}:
                raise ValueError("checkpoint phase is not resumable")
            particles = restored["population"]
            b_t = restored["budget_remaining"]
            round_idx = restored["round_idx"]
            progress_prev = restored["progress_prev"]
            last_alpha = restored["last_alpha"]
            last_lambda = restored["last_lambda"]
            self.log.restore_next_seq(restored["event_next_seq"])

        b_total = int(self.cfg.budget)
        # 修改批次由显式 mutation_rounds 定义；budget 校验保证每个标准轮都有 children_per_round 个 slot。
        max_rounds = int(self.cfg.mutation_rounds)
        while b_t > 0 and round_idx < max_rounds:
            particles, step, m_t, progress_prev = self._round(
                particles, b_t, round_idx, progress_prev)   # Γ 延迟一轮：本轮 Γ_t 喂下一轮
            b_t -= m_t                      # 每次 LLM 修改调用计预算（含无效候选，§7.2）
            self.ledger.record_mutation_slots(m_t)
            self.ledger.sync_actual_costs(self.proposer, self.evaluator)
            last_lambda = step.lambda_equivalent
            last_alpha = step.alpha
            round_idx += 1
            self.log.log("stage_end", stage=round_idx, budget_remaining=b_t,
                         ledger=self.ledger.snapshot(),
                         best_score=self.best.search_score if self.best else None,
                         last_alpha=last_alpha, last_lambda=last_lambda,
                         unique_lineages=len({p.metadata.get("state_origin_id") for p in particles}))
            if checkpoint_callback is not None:
                checkpoint_callback(self.snapshot_runtime(
                    phase="stage_end", population=particles, budget_remaining=b_t,
                    round_idx=round_idx, progress_prev=progress_prev,
                    last_alpha=last_alpha, last_lambda=last_lambda,
                ))

        budget_used = int(self.cfg.init_budget) + (int(self.cfg.mutation_budget) - max(0, b_t))
        self.ledger.sync_actual_costs(self.proposer, self.evaluator)
        self.log.log("island_done", termination_reason="budget_exhausted", n_rounds=round_idx,
                     budget_used=budget_used, ledger=self.ledger.snapshot(),
                     last_lambda=last_lambda,
                     best_id=self.best.id if self.best else None,
                     best_score=self.best.search_score if self.best else None)
        terminal_state = self.snapshot_runtime(
            phase="completed", population=particles, budget_remaining=b_t,
            round_idx=round_idx, progress_prev=progress_prev,
            last_alpha=last_alpha, last_lambda=last_lambda,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(terminal_state)
        return IslandResult(self.best, "budget_exhausted", round_idx, budget_used, last_lambda, last_alpha)
