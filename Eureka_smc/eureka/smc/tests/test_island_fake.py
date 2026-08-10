"""单岛预算条件化 KL 循环的集成测试：用 Fake 提议器 + Fake 评估器，不烧 GPU。

验证控制器骨架本身（初始化 + RF-Agent 式 traceback-repair / 预算条件化 KL 父代选择 /
multinomial 重采样 / sigmoid 非对称接受 / genealogy / 预算耗尽停止 / RF 五操作路由）在便宜
替身上的正确性。父代选择/接受用**原始 J**（新方法 §2.2）。

数量约定：初始 N 个父代槽位始终保留；每轮仅修改 M 个经重采样选出的槽位。
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


def _island(proposer, tmp_path: Path, **overrides):
    cfg = SMCIslandConfig(**{
        "n_particles": 8, "children_per_round": 4, "budget": 16,
        "k_min": 2, "gamma": 1.0, "seed": 0, **overrides,
    })
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)  # fake quality 已在 [0,~0.8]（原始 J）
    evaluator = FakeEvaluator(score_cfg, seeds=(0, 1, 2), rng=np.random.default_rng(2))
    log_path = tmp_path / "events.jsonl"
    island = SMCIsland(cfg, proposer, evaluator, EventLogger(log_path, clock=False),
                       artifact_root=tmp_path / "cand")
    return island, log_path


def _build(tmp_path: Path, **overrides):
    return _island(FakeProposer(np.random.default_rng(1)), tmp_path, **overrides)


def _events(log_path: Path):
    return [json.loads(l) for l in log_path.read_text().splitlines()]


def test_initialize_yields_n_valid_particles(tmp_path):
    island, _ = _build(tmp_path)
    ps = island.initialize()
    assert len(ps) == 8
    assert all(p.valid and p.search_score is not None for p in ps)
    assert len({p.id for p in ps}) == 8  # 唯一 ID


def test_init_repair_fills_to_n_valid(tmp_path):
    # 初始 60% 无效，repair_batch 修成有效 → RF-Agent 式修复应集满 N（不再冷抽到上限就残缺）。
    proposer = FakeProposer(np.random.default_rng(1), init_invalid_frac=0.6)
    island, log_path = _island(proposer, tmp_path)
    ps = island.initialize()
    assert len(ps) == 8                       # 修复到满 N
    assert all(p.valid for p in ps)
    evs = _events(log_path)
    assert any(e["event"] == "init_invalid" for e in evs)   # 确实发生过无效
    done = next(e for e in evs if e["event"] == "init_done")
    assert done["n_valid"] == 8 and done["waves"] >= 2      # 多波修复


def test_init_repair_disabled_falls_back(tmp_path):
    # max_init_repair=0 时不修复，只靠"换全新"补足；仍应尽力集满 N（FakeProposer 全新总有效）。
    proposer = FakeProposer(np.random.default_rng(1), init_invalid_frac=0.5)
    island, _ = _island(proposer, tmp_path, max_init_repair=0)
    ps = island.initialize()
    assert all(p.valid for p in ps)


class _MutInvalidOnceProposer(FakeProposer):
    """测试替身：首个变异子代（reflect 首调）强制无效，其余正常；repair_batch 修成有效。

    用于验证 mutation 轮的 traceback-repair：初始化只走 initial_batch/repair_batch，不触碰
    reflect，故 ``_injected`` 到第一次变异 reflect 才翻转，确保恰好注入一个无效变异子代。
    """

    def __init__(self, rng, **kw):
        super().__init__(rng, **kw)
        self._injected = False

    def reflect(self, parent_code: str, parent_feedback: str):
        if not self._injected:
            self._injected = True
            return self._INVALID          # 一个无效变异子代
        return super().reflect(parent_code, parent_feedback)


def test_mutation_repair_fixes_invalid_child(tmp_path):
    # 变异轮出现无效子代时，RF-Agent 式 repair 应把它修成有效（不再白白少一个 valid 算子）。
    proposer = _MutInvalidOnceProposer(np.random.default_rng(1))
    island, log_path = _island(proposer, tmp_path, max_mutation_repair=8)
    particles = island.initialize()
    island._round(particles, b_t=island.cfg.mutation_budget, round_idx=0)
    reps = [e for e in _events(log_path) if e["event"] == "mutation_repair"]
    assert reps, "应触发至少一次 mutation 修复"
    assert any(e["valid"] for e in reps), "修复后应产出有效子代"


def test_mutation_repair_disabled_leaves_invalid(tmp_path):
    # max_mutation_repair=0（默认）时不修复：无效变异子代进接受阶段被判 invalid_child、槽位退回父代。
    proposer = _MutInvalidOnceProposer(np.random.default_rng(1))
    island, log_path = _island(proposer, tmp_path, max_mutation_repair=0)
    particles = island.initialize()
    island._round(particles, b_t=island.cfg.mutation_budget, round_idx=0)
    evs = _events(log_path)
    assert not [e for e in evs if e["event"] == "mutation_repair"]
    assert any(e["event"] == "accept_decision" and e.get("reason") == "invalid_child"
               for e in evs), "关闭修复时无效子代应被判 invalid_child"



def test_explicit_budget_split_requires_consistent_rounds(tmp_path):
    island, _ = _build(tmp_path, init_budget=8, mutation_budget=8, mutation_rounds=2)
    assert island.cfg.budget == 16
    assert island.cfg.init_budget == 8
    assert island.cfg.mutation_budget == 8
    assert island.cfg.mutation_rounds == 2

    with pytest.raises(ValueError, match="mutation_budget must equal"):
        _build(tmp_path / "bad", init_budget=8, mutation_budget=8, mutation_rounds=3)


def test_explicit_budget_split_requires_consistent_rounds(tmp_path):
    island, _ = _build(tmp_path, init_budget=8, mutation_budget=8, mutation_rounds=2)
    assert island.cfg.budget == 16
    assert island.cfg.init_budget == 8
    assert island.cfg.mutation_budget == 8
    assert island.cfg.mutation_rounds == 2

    with pytest.raises(ValueError, match="mutation_budget must equal"):
        _build(tmp_path / "bad", init_budget=8, mutation_budget=8, mutation_rounds=3)



def test_run_stops_when_initialize_does_not_fill_all_slots(tmp_path, monkeypatch):
    island, _ = _build(tmp_path)
    partial = [_particle("p0", 0.5)]
    monkeypatch.setattr(island, "initialize", lambda: partial)

    result = island.run()

    assert result.termination_reason == "insufficient_valid_particles"
    assert result.budget_used == 0
    assert result.n_rounds == 0


    island, _ = _build(tmp_path)
    res = island.run()
    assert res.termination_reason == "budget_exhausted"
    assert res.budget_used == 16                 # B_total 全部消耗（init 8 + 修改 8）
    # 修改预算 = B_total − N = 16 − 8 = 8；8 / children_per_round(4) = 2 轮
    assert res.n_rounds == 2
    assert res.best is not None and res.best.search_score is not None


def test_population_keeps_n_slots(tmp_path):
    # N 槽位不变式：每轮只更新被分配资源的槽位，活动 population 始终为 N。
    island, log_path = _build(tmp_path)
    island.run()
    starts = [e for e in _events(log_path) if e["event"] == "stage_start"]
    assert len(starts[0]["resample_counts"]) == 8
    assert all(len(s["resample_counts"]) == 8 for s in starts)
    assert all(len(e["resample_counts"]) == 8 for e in starts)


def test_partial_final_round_spends_remaining_budget(tmp_path):
    # B_total 非 M 整数倍：修改预算 = 18 − 8 = 10 → 末轮 m_t = min(M, b_t) 只花剩余（阶段A-15）。
    island, log_path = _build(tmp_path, budget=18)   # init8 → 修改 4+4+2 = 10
    res = island.run()
    assert res.budget_used == 18
    assert res.n_rounds == 3
    starts = [e for e in _events(log_path) if e["event"] == "stage_start"]
    assert [e["m_t"] for e in starts] == [4, 4, 2]


def test_first_round_reflects_init_budget_consumption(tmp_path):
    # 预算记账：B_total 含 init 的 N → init 一建立即扣 N，首个修改轮 b_t = B_total − N。
    # 故首轮 budget_frac = (16−8)/16 = 0.5 < 1 → K* < N、λ > 0（非均匀，init 不是边界特例）。
    island, log_path = _build(tmp_path)   # n_particles=8, budget=16, children_per_round=4
    island.run()
    first = next(e for e in _events(log_path) if e["event"] == "stage_start")
    assert first["budget_remaining"] == 8                       # B_total − N = 16 − 8
    assert first["budget_frac"] == pytest.approx(0.5, abs=1e-9)
    assert first["k_star"] == pytest.approx(5.0, abs=0.1)       # 2 + (8−2)·0.5
    assert first["lam"] > 0.0                                   # 非 λ=0 均匀
    assert first["max_parent_prob"] > 1.0 / 8                   # 已有集中度
    assert first["potential_definition"] == "raw_search_score"
    assert first["prior_weight_definition"] == "uniform_slots"


def test_concentration_increases_as_budget_drains(tmp_path):
    island, log_path = _build(tmp_path, budget=32)
    island.run()
    starts = [e for e in _events(log_path) if e["event"] == "stage_start"]
    ks = [e["k_star"] for e in starts]
    assert ks == sorted(ks, reverse=True)         # K* 单调不增


def test_best_is_separated_from_population(tmp_path):
    island, log_path = _build(tmp_path)
    res = island.run()
    scores = [e["search_score"] for e in _events(log_path)
              if e["event"] == "eval" and e["search_score"] is not None]
    assert res.best.search_score == pytest.approx(max(scores))


def test_archive_best_monotonic_nondecreasing(tmp_path):
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
    island, _ = _build(tmp_path)
    parent, child = _particle("p", 0.3), _particle("c", 0.9)
    for lam in (0.0, 1.0, 100.0):
        assert island._sigmoid_accept(parent, child, lam, 0) is child


def test_sigmoid_accept_lambda_zero_is_half(tmp_path):
    island, log_path = _build(tmp_path)
    parent, child = _particle("p", 0.9), _particle("c", 0.3)  # Δ<0
    island._sigmoid_accept(parent, child, 0.0, 0)
    dec = next(e for e in _events(log_path) if e["event"] == "accept_decision")
    assert dec["p_accept"] == pytest.approx(0.5)


def test_sigmoid_accept_large_lambda_rejects_worse(tmp_path):
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
    island, log_path = _build(tmp_path)
    island.run()
    for e in _events(log_path):
        if e["event"] == "accept_decision" and e.get("improved") is True:
            assert e["accepted"] is True


def test_valid_rejected_children_stay_in_archive_without_accepted_edge(tmp_path, monkeypatch):
    island, _ = _build(tmp_path)
    monkeypatch.setattr(
        island,
        "_sigmoid_accept",
        lambda current, child, alpha, span, stage: current,
    )
    island.run()

    tentative = [
        node for node in island._registry.values()
        if node.proposal_parent_id is not None
    ]
    assert tentative
    assert all(node.valid for node in tentative)
    assert all(node.accepted_transition_parent_id is None for node in tentative)
    archive_ids = {node.id for node in island.archive}
    assert {node.id for node in tentative} <= archive_ids


def test_accepted_child_writes_transition_edge_after_acceptance(tmp_path, monkeypatch):
    island, _ = _build(tmp_path)
    monkeypatch.setattr(
        island,
        "_sigmoid_accept",
        lambda current, child, alpha, span, stage: child,
    )
    island.run()

    accepted = [
        node for node in island._registry.values()
        if node.proposal_parent_id is not None
    ]
    assert accepted
    assert all(node.accepted_transition_parent_id is not None for node in accepted)
    assert all(node.metadata["state_origin_id"] == node.id for node in accepted)


def test_stage_boundary_resume_matches_uninterrupted_run(tmp_path):
    uninterrupted, _ = _build(tmp_path / "uninterrupted")
    expected = uninterrupted.run()

    interrupted, _ = _build(tmp_path / "interrupted")
    snapshots = []

    def stop_after_first_stage(state):
        snapshots.append(state)
        if state["phase"] == "stage_end":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.run(checkpoint_callback=stop_after_first_stage)
    state = snapshots[-1]
    assert state["phase"] == "stage_end"

    resumed, resumed_log = _build(tmp_path / "interrupted")
    actual = resumed.run(resume_state=state)

    assert actual.termination_reason == expected.termination_reason
    assert actual.budget_used == expected.budget_used
    assert actual.n_rounds == expected.n_rounds
    assert actual.best.id == expected.best.id
    assert actual.best.search_score == pytest.approx(expected.best.search_score)
    def logical_particle_json(particle: RewardParticle):
        data = particle.to_json()
        data.pop("artifact_dir", None)
        return data

    assert [logical_particle_json(p) for p in resumed.archive] == [
        logical_particle_json(p) for p in uninterrupted.archive
    ]
    assert set(resumed._registry) == set(uninterrupted._registry)
    assert resumed.ledger.snapshot() == uninterrupted.ledger.snapshot()
    resumed_events = _events(resumed_log)
    assert [event["seq"] for event in resumed_events] == list(range(len(resumed_events)))


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
    assert len(island._registry) >= 8


def test_noop_and_stage_events_logged(tmp_path):
    island, log_path = _build(tmp_path, seed=3)
    island.run()
    events = {e["event"] for e in _events(log_path)}
    assert "stage_start" in events and "stage_end" in events


def test_reproducible_under_same_seeds(tmp_path):
    island1, _ = _build(tmp_path / "a")
    island2, _ = _build(tmp_path / "b")
    r1, r2 = island1.run(), island2.run()
    assert r1.best.search_score == pytest.approx(r2.best.search_score)
    assert r1.n_rounds == r2.n_rounds
    assert r1.budget_used == r2.budget_used


# ---- RF-Agent Phase-3a：mutation_structure / mutation_parameter 路由 ----

def _build_rf(tmp_path: Path, **overrides):
    from eureka.smc.actions import ActionConfig
    return _island(FakeProposer(np.random.default_rng(1)), tmp_path, **{
        "n_particles": 8, "children_per_round": 8, "budget": 16,
        "action_cfg": ActionConfig(mode="rf",
                                   enabled=["mutation_structure", "mutation_parameter"]),
        **overrides,
    })


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
    # children_per_round=8 才能让 rf_ratio [2,2,2,1,1]=8 每轮路由到全部 5 个 action。
    return _island(FakeProposer(np.random.default_rng(1)), tmp_path, **{
        "n_particles": 8, "children_per_round": 8, "budget": 24,
        "action_cfg": ActionConfig(mode="rf", enabled=list(RF_ACTIONS)),
        **overrides,
    })


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
