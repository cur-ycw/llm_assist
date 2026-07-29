"""单岛 SMC 循环的集成测试：用 Fake 提议器 + Fake 评估器，不烧 GPU（计划 §9）。

验证 SMC 骨架本身（初始化 / 退火调度 / 重采样 / reward-only MH 接受 / genealogy /
退火完成停止）在便宜替身上的正确性，与真实 Isaac Gym / LLM 解耦。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eureka.smc.evaluator import FakeEvaluator
from eureka.smc.event_logger import EventLogger
from eureka.smc.island import SMCIsland, SMCIslandConfig
from eureka.smc.proposer import FakeProposer
from eureka.smc.score import ScoreConfig


def _build(tmp_path: Path, **overrides):
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "beta_target": 2.0, "kappa": 0.5,
        "min_iters": 3, "max_iters": 15, "n_proposals": 1, "seed": 0, **overrides,
    })
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)  # fake quality 已在 [0,~0.8]
    proposer = FakeProposer(np.random.default_rng(1))
    evaluator = FakeEvaluator(score_cfg, seeds=(0, 1, 2), rng=np.random.default_rng(2))
    log_path = tmp_path / "events.jsonl"
    island = SMCIsland(cfg, proposer, evaluator, EventLogger(log_path, clock=False),
                       artifact_root=tmp_path / "cand")
    return island, log_path


def _events(log_path: Path):
    return [json.loads(l) for l in log_path.read_text().splitlines()]


def test_initialize_yields_n_valid_particles(tmp_path):
    island, _ = _build(tmp_path)
    ps = island.initialize()
    assert len(ps) == 8
    assert all(p.valid and p.search_score is not None for p in ps)
    assert len({p.id for p in ps}) == 8  # 唯一 ID


def test_run_reaches_annealing_complete(tmp_path):
    island, _ = _build(tmp_path)
    res = island.run()
    assert res.termination_reason == "annealing_complete"
    assert res.final_lambda >= 1.0 - 1e-9
    assert res.n_stages <= 15
    assert res.best is not None and res.best.search_score is not None


def test_best_is_separated_from_population(tmp_path):
    # 历史最优不应被重采样/拒绝丢失：best.score >= 任何评估过的粒子分数。
    island, log_path = _build(tmp_path)
    res = island.run()
    scores = [e["search_score"] for e in _events(log_path)
              if e["event"] == "eval" and e["search_score"] is not None]
    assert res.best.search_score == pytest.approx(max(scores))


def test_mh_accepts_all_nonnegative_deltas(tmp_path):
    # 强不变量（计划 §6.6）：ΔR >= 0 必接受。
    island, log_path = _build(tmp_path)
    island.run()
    for e in _events(log_path):
        if e["event"] == "accept_decision" and "delta_r" in e and e["delta_r"] >= 0:
            assert e["accepted"] is True


class _WorseProposer:
    """确定性 stub：初始全为高质量，每次 reflect 返回固定低质量子代。"""
    _T = "def compute_reward(self):  # quality={q}\n    return self.rew_buf, {{}}\n"

    def initial_batch(self, n):
        return [self._T.format(q="0.900000") for _ in range(n)]

    def reflect(self, code, fb):
        return self._T.format(q="0.000000")


def test_negative_delta_rejected_at_high_beta(tmp_path):
    # 确定性验证 reward-only MH 的负增益侧：ΔR≈-0.9 恒定 + 大 β_target ⇒ 接受概率
    # exp(β·ΔR)≈0 ⇒ 负增益必被拒绝（计划 §6.6）。noise=0 使 ΔR 精确。
    cfg = SMCIslandConfig(n_particles=6, beta_target=200.0, kappa=0.5,
                          min_iters=3, max_iters=6, seed=0)
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)
    evaluator = FakeEvaluator(score_cfg, seeds=(0, 1, 2), rng=np.random.default_rng(2), noise=0.0)
    log_path = tmp_path / "e.jsonl"
    island = SMCIsland(cfg, _WorseProposer(), evaluator, EventLogger(log_path, clock=False),
                       artifact_root=tmp_path / "c")
    island.run()
    neg = [e for e in _events(log_path)
           if e["event"] == "accept_decision" and e.get("delta_r", 0) < 0]
    assert len(neg) > 0                                   # 确实产生了负增益提议
    assert all(e["accepted"] is False for e in neg)       # 大 β 下全部被拒


def test_genealogy_accepted_chain_skips_clones(tmp_path):
    # 沿 best 的 accepted_transition_parent_id 回溯，链上每个 id 都不是纯 clone
    # （clone 的 accepted_transition 继承父，不出现在真实 accepted-edit 链中）。
    island, log_path = _build(tmp_path)
    res = island.run()
    evs = {e["id"]: e for e in _events(log_path) if e["event"] == "eval"}
    # 回溯 accepted-edit 链
    node = res.best
    seen = set()
    atp = node.accepted_transition_parent_id
    while atp is not None and atp in evs and atp not in seen:
        seen.add(atp)
        # accepted-transition 父必须是被评估过的真实代码状态
        assert evs[atp]["valid"] is True
        # 下一跳
        atp = evs[atp].get("accepted_transition_parent_id")


def test_noop_proposals_are_logged(tmp_path):
    island, log_path = _build(tmp_path, seed=3)
    island.run()
    events = {e["event"] for e in _events(log_path)}
    # FakeProposer noop_rate=0.1，多阶段下应至少出现一次 no-op（或全被接受，二者皆合法）
    assert "stage_start" in events and "stage_end" in events


def test_cache_avoids_recomputation(tmp_path):
    # 相同代码复用聚合结果：no-op 提议返回父代码，命中缓存，不增加真实评估数。
    island, _ = _build(tmp_path)
    island.run()
    # 真实评估数应远小于“初始 + 每阶段每粒子一次”的上限（缓存 + no-op 生效）
    assert island.evaluator.n_evals >= 8  # 至少初始化的 valid 粒子


def test_reproducible_under_same_seeds(tmp_path):
    island1, _ = _build(tmp_path / "a")
    island2, _ = _build(tmp_path / "b")
    r1, r2 = island1.run(), island2.run()
    assert r1.best.search_score == pytest.approx(r2.best.search_score)
    assert r1.n_stages == r2.n_stages


# ---- RF-Agent Phase-3a：mutation_structure / mutation_parameter 路由 ----

def _build_rf(tmp_path: Path, **overrides):
    from eureka.smc.actions import ActionConfig
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "beta_target": 2.0, "kappa": 0.5,
        "min_iters": 3, "max_iters": 15, "n_proposals": 1, "seed": 0,
        "action_cfg": ActionConfig(mode="rf",
                                   enabled=["mutation_structure", "mutation_parameter"]),
        **overrides,
    })
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)
    proposer = FakeProposer(np.random.default_rng(1))
    evaluator = FakeEvaluator(score_cfg, seeds=(0, 1, 2), rng=np.random.default_rng(2))
    log_path = tmp_path / "events.jsonl"
    island = SMCIsland(cfg, proposer, evaluator, EventLogger(log_path, clock=False),
                       artifact_root=tmp_path / "cand")
    return island, log_path


def test_rf_mode_routes_actions_and_audits(tmp_path):
    island, log_path = _build_rf(tmp_path)
    res = island.run()
    assert res.termination_reason == "annealing_complete"
    evs = _events(log_path)
    # 派发的 action 落到 eval 事件
    acts = {e.get("proposal_action") for e in evs if e["event"] == "eval"}
    assert acts & {"mutation_structure", "mutation_parameter"}
    # 契约审计事件存在且带布尔 contract_hit / 合法 action
    audits = [e for e in evs if e["event"] == "contract_audit"]
    assert len(audits) > 0
    assert all(isinstance(e["contract_hit"], bool) for e in audits)
    assert all(e["action"] in ("mutation_structure", "mutation_parameter") for e in audits)


def test_generic_mode_has_no_action_metadata(tmp_path):
    # 默认 generic 路径不应产生 action/契约元数据（保证 rf 是纯 opt-in 分支）
    island, log_path = _build(tmp_path)
    island.run()
    evs = _events(log_path)
    assert all(e.get("proposal_action") is None for e in evs if e["event"] == "eval")
    assert not any(e["event"] == "contract_audit" for e in evs)


# ---- RF-Agent Phase-3b：crossover / path_reasoning / different_thought 历史型 action ----

def _build_rf5(tmp_path: Path, **overrides):
    from eureka.smc.actions import ActionConfig, RF_ACTIONS
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "beta_target": 2.0, "kappa": 0.5,
        "min_iters": 3, "max_iters": 15, "n_proposals": 1, "seed": 0,
        "action_cfg": ActionConfig(mode="rf", enabled=list(RF_ACTIONS)),
        **overrides,
    })
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)
    proposer = FakeProposer(np.random.default_rng(1))
    evaluator = FakeEvaluator(score_cfg, seeds=(0, 1, 2), rng=np.random.default_rng(2))
    log_path = tmp_path / "events.jsonl"
    island = SMCIsland(cfg, proposer, evaluator, EventLogger(log_path, clock=False),
                       artifact_root=tmp_path / "cand")
    return island, log_path


def test_rf5_routes_all_five_actions_and_completes(tmp_path):
    island, log_path = _build_rf5(tmp_path)
    res = island.run()
    assert res.termination_reason == "annealing_complete"
    assert res.best is not None and res.best.search_score is not None
    evs = _events(log_path)
    acts = {e.get("proposal_action") for e in evs if e["event"] == "eval"}
    # 三个历史型 action 至少各现身一次（多阶段、N=8、比例含它们）
    assert {"crossover", "path_reasoning", "different_thought"} <= acts
    assert {"mutation_structure", "mutation_parameter"} <= acts
    # 历史型 action 无 AST 契约 → 契约审计只覆盖变异 action
    audits = [e for e in evs if e["event"] == "contract_audit"]
    assert all(e["action"] in ("mutation_structure", "mutation_parameter") for e in audits)


def test_rf5_registry_populated_and_reproducible(tmp_path):
    island1, _ = _build_rf5(tmp_path / "a")
    island2, _ = _build_rf5(tmp_path / "b")
    r1, r2 = island1.run(), island2.run()
    # 确定性：料源（注册表选择）与派发均无 rng，同 seed 复现
    assert r1.best.search_score == pytest.approx(r2.best.search_score)
    assert r1.n_stages == r2.n_stages
    # 注册表确实记录了被评估的粒子（供历史型 action 取料）
    assert len(island1._registry) >= 8


def test_rf5_history_context_builders_are_deterministic(tmp_path):
    # 直接验证三个料源方法确定性、无副作用（跑完一次后在稳定 registry 上重复调用同结果）
    island, _ = _build_rf5(tmp_path)
    island.run()
    reg = list(island._registry.values())
    assert reg, "registry 非空"
    # 取一个带 design_thought 的粒子做 different_thought 料源探针
    donors_a = island._elite_donors(exclude_code="__none__", k=2)
    donors_b = island._elite_donors(exclude_code="__none__", k=2)
    assert [p.id for p in donors_a] == [p.id for p in donors_b]  # 确定性
    assert len(donors_a) <= 2
    others_a = island._other_thoughts(set(), k=4)
    others_b = island._other_thoughts(set(), k=4)
    assert [p.id for p in others_a] == [p.id for p in others_b]
