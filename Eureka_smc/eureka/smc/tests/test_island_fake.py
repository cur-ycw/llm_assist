"""单岛预算条件化 KL 循环的集成测试：用 Fake 提议器 + Fake 评估器，不烧 GPU。

验证控制器骨架本身（初始化 / 预算条件化 KL 父代选择 / multinomial 重采样 / sigmoid 非对称
接受 / genealogy / 预算耗尽停止 / RF 五操作路由）在便宜替身上的正确性，与真实 Isaac Gym /
LLM 解耦。父代选择/接受用**原始 J**（新方法 §2.2），不归一化。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eureka.smc.evaluator import FakeEvaluator
from eureka.smc.event_logger import EventLogger
from eureka.smc.island import SMCIsland, SMCIslandConfig
from eureka.smc.particle import EvalRecord, RewardParticle
from eureka.smc.proposer import FakeProposer
from eureka.smc.score import ScoreConfig


def _build(tmp_path: Path, **overrides):
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "budget": 16, "k_min": 2, "gamma": 1.0, "seed": 0, **overrides,
    })
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)  # fake quality 已在 [0,~0.8]（原始 J）
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


def test_run_reaches_budget_exhausted(tmp_path):
    island, _ = _build(tmp_path)
    res = island.run()
    assert res.termination_reason == "budget_exhausted"
    assert res.budget_used == 16                 # B 全部消耗
    assert res.n_rounds == 2                      # ceil(16 / 8) = 2 轮
    assert res.best is not None and res.best.search_score is not None


def test_partial_final_round_spends_remaining_budget(tmp_path):
    # B 非 N 整数倍：末轮 M_t = min(N, b_t) 只花剩余预算（新方法 §2.5 / 阶段A-15）。
    island, log_path = _build(tmp_path, budget=20)   # 8 + 8 + 4
    res = island.run()
    assert res.budget_used == 20
    assert res.n_rounds == 3
    starts = [e for e in _events(log_path) if e["event"] == "stage_start"]
    assert [e["m_t"] for e in starts] == [8, 8, 4]


def test_first_round_is_uniform_selection(tmp_path):
    # 预算满 b_0=B → K*=N → δ=0 → λ=0 → q=U_N（首轮广探，新方法 §2.3）。
    island, log_path = _build(tmp_path)
    island.run()
    first = next(e for e in _events(log_path) if e["event"] == "stage_start")
    assert first["lam"] == pytest.approx(0.0, abs=1e-6)
    assert first["max_parent_prob"] == pytest.approx(1.0 / 8, abs=1e-6)
    assert first["k_star"] == pytest.approx(8.0, abs=0.1)


def test_concentration_increases_as_budget_drains(tmp_path):
    # 随预算耗尽 K* 单调不增、max 父代概率不减（除非全同分回退）。
    island, log_path = _build(tmp_path, budget=32)
    island.run()
    starts = [e for e in _events(log_path) if e["event"] == "stage_start"]
    ks = [e["k_star"] for e in starts]
    assert ks == sorted(ks, reverse=True)         # K* 单调不增


def test_best_is_separated_from_population(tmp_path):
    # 历史最优不应被重采样/拒绝丢失：best.score >= 任何评估过的粒子分数。
    island, log_path = _build(tmp_path)
    res = island.run()
    scores = [e["search_score"] for e in _events(log_path)
              if e["event"] == "eval" and e["search_score"] is not None]
    assert res.best.search_score == pytest.approx(max(scores))


def test_archive_best_monotonic_nondecreasing(tmp_path):
    # 检查表 §6-12：archive 历史最佳单调不下降。
    island, log_path = _build(tmp_path, budget=32)
    island.run()
    ends = [e for e in _events(log_path) if e["event"] == "stage_end"]
    bests = [e["best_score"] for e in ends if e["best_score"] is not None]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(bests, bests[1:]))


# ---- sigmoid 非对称接受核（新方法 §2.6）----

def _particle(pid: str, score: float) -> RewardParticle:
    rec = EvalRecord(search_score=score, valid=True, executable=True)
    return RewardParticle(id=pid, reward_code=f"# quality={score}\ndef f(): pass", eval=rec)


def test_sigmoid_accept_improvement_always_enters(tmp_path):
    # Δ>0 → 确定进入（p=1），与 λ 无关。
    island, _ = _build(tmp_path)
    parent, child = _particle("p", 0.3), _particle("c", 0.9)
    for lam in (0.0, 1.0, 100.0):
        assert island._sigmoid_accept(parent, child, lam, 0) is child


def test_sigmoid_accept_lambda_zero_is_half(tmp_path):
    # 检查表 §6-10：λ=0 时非改进接受概率 = σ(0) = 0.5。
    island, log_path = _build(tmp_path)
    parent, child = _particle("p", 0.9), _particle("c", 0.3)  # Δ<0
    island._sigmoid_accept(parent, child, 0.0, 0)
    dec = next(e for e in _events(log_path) if e["event"] == "accept_decision")
    assert dec["p_accept"] == pytest.approx(0.5)


def test_sigmoid_accept_large_lambda_rejects_worse(tmp_path):
    # 大 λ + 明显负 Δ → σ(λΔ)≈0 → 必拒（确定性：p_accept 极小，任何 rng 抽样都拒）。
    island, log_path = _build(tmp_path)
    parent, child = _particle("p", 0.9), _particle("c", 0.1)  # Δ=-0.8
    kept = island._sigmoid_accept(parent, child, 100.0, 0)
    assert kept is parent
    dec = next(e for e in _events(log_path) if e["event"] == "accept_decision")
    assert dec["p_accept"] < 1e-6 and dec["accepted"] is False


def test_invalid_child_rejected(tmp_path):
    island, _ = _build(tmp_path)
    parent = _particle("p", 0.5)
    bad = RewardParticle(id="c", reward_code="x",
                         eval=EvalRecord(search_score=None, valid=False, executable=False))
    assert island._sigmoid_accept(parent, bad, 1.0, 0) is parent


def test_all_positive_delta_accepts_over_full_run(tmp_path):
    # 强不变量：run 全程 Δ>0（improved）的接受决策必为 True。
    island, log_path = _build(tmp_path)
    island.run()
    for e in _events(log_path):
        if e["event"] == "accept_decision" and e.get("improved") is True:
            assert e["accepted"] is True


# ---- genealogy / registry ----

def test_genealogy_accepted_chain_skips_clones(tmp_path):
    island, log_path = _build(tmp_path)
    res = island.run()
    evs = {e["id"]: e for e in _events(log_path) if e["event"] == "eval"}
    node = res.best
    seen = set()
    atp = node.accepted_transition_parent_id
    while atp is not None and atp in evs and atp not in seen:
        seen.add(atp)
        assert evs[atp]["valid"] is True
        atp = evs[atp].get("accepted_transition_parent_id")


def test_registry_populated(tmp_path):
    island, _ = _build(tmp_path)
    island.run()
    assert len(island._registry) >= 8  # 至少初始化的粒子被登记


def test_noop_and_stage_events_logged(tmp_path):
    island, log_path = _build(tmp_path, seed=3)
    island.run()
    events = {e["event"] for e in _events(log_path)}
    assert "stage_start" in events and "stage_end" in events


def test_reproducible_under_same_seeds(tmp_path):
    # 三条独立 seeded RNG（island seed=0 / FakeProposer(1) / FakeEvaluator(2)）；multinomial
    # 重采样与派发均确定性 → 同 seed 复现。
    island1, _ = _build(tmp_path / "a")
    island2, _ = _build(tmp_path / "b")
    r1, r2 = island1.run(), island2.run()
    assert r1.best.search_score == pytest.approx(r2.best.search_score)
    assert r1.n_rounds == r2.n_rounds
    assert r1.budget_used == r2.budget_used


# ---- RF-Agent Phase-3a：mutation_structure / mutation_parameter 路由 ----

def _build_rf(tmp_path: Path, **overrides):
    from eureka.smc.actions import ActionConfig
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "budget": 16, "k_min": 2, "gamma": 1.0, "seed": 0,
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
    assert res.termination_reason == "budget_exhausted"
    evs = _events(log_path)
    acts = {e.get("proposal_action") for e in evs if e["event"] == "eval"}
    assert acts & {"mutation_structure", "mutation_parameter"}
    audits = [e for e in evs if e["event"] == "contract_audit"]
    assert len(audits) > 0
    assert all(isinstance(e["contract_hit"], bool) for e in audits)
    assert all(e["action"] in ("mutation_structure", "mutation_parameter") for e in audits)


def test_generic_mode_has_no_action_metadata(tmp_path):
    island, log_path = _build(tmp_path)
    island.run()
    evs = _events(log_path)
    assert all(e.get("proposal_action") is None for e in evs if e["event"] == "eval")
    assert not any(e["event"] == "contract_audit" for e in evs)


# ---- RF-Agent Phase-3b：crossover / path_reasoning / different_thought 历史型 action ----

def _build_rf5(tmp_path: Path, **overrides):
    from eureka.smc.actions import ActionConfig, RF_ACTIONS
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "budget": 24, "k_min": 2, "gamma": 1.0, "seed": 0,
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
    assert res.termination_reason == "budget_exhausted"
    assert res.best is not None and res.best.search_score is not None
    evs = _events(log_path)
    acts = {e.get("proposal_action") for e in evs if e["event"] == "eval"}
    assert {"crossover", "path_reasoning", "different_thought"} <= acts
    assert {"mutation_structure", "mutation_parameter"} <= acts
    audits = [e for e in evs if e["event"] == "contract_audit"]
    assert all(e["action"] in ("mutation_structure", "mutation_parameter") for e in audits)


def test_rf5_registry_populated_and_reproducible(tmp_path):
    island1, _ = _build_rf5(tmp_path / "a")
    island2, _ = _build_rf5(tmp_path / "b")
    r1, r2 = island1.run(), island2.run()
    assert r1.best.search_score == pytest.approx(r2.best.search_score)
    assert r1.n_rounds == r2.n_rounds
    assert len(island1._registry) >= 8


def test_rf5_history_context_builders_are_deterministic(tmp_path):
    island, _ = _build_rf5(tmp_path)
    island.run()
    reg = list(island._registry.values())
    assert reg, "registry 非空"
    donors_a = island._elite_donors(exclude_code="__none__", k=2)
    donors_b = island._elite_donors(exclude_code="__none__", k=2)
    assert [p.id for p in donors_a] == [p.id for p in donors_b]
    assert len(donors_a) <= 2
    others_a = island._other_thoughts(set(), k=4)
    others_b = island._other_thoughts(set(), k=4)
    assert [p.id for p in others_a] == [p.id for p in others_b]
