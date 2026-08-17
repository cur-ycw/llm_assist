"""纯贪婪接受消融（greedy_accept）单测：默认关=走原 sigmoid 接受核；开启=delta>0 硬阈值、不消耗 rng。

直接对 ``SMCIsland._sigmoid_accept`` 打桩测：
- 开启：改差子代恒被拒、改好子代恒接受、平局(delta==0)被拒；且**不消耗 rng**（确定性）。
- 关闭（默认）：改差子代在 span 上按 sigmoid 概率接受（会消耗 rng）→ 证明是精确 no-op、原核不变。
"""

from __future__ import annotations

import numpy as np

from eureka.smc.evaluator import FakeEvaluator
from eureka.smc.event_logger import EventLogger
from eureka.smc.island import SMCIsland, SMCIslandConfig
from eureka.smc.particle import EvalRecord, RewardParticle
from eureka.smc.proposer import FakeProposer
from eureka.smc.score import ScoreConfig


def _particle(pid: str, score: float) -> RewardParticle:
    return RewardParticle(
        id=pid,
        reward_code="def compute_reward(self):\n    return self.rew_buf, {}\n",
        eval=EvalRecord(search_score=score, valid=True, executable=True))


def _island(greedy: bool, tmp_path, *, accept_sharpness: float = 5.0, seed: int = 0) -> SMCIsland:
    cfg = SMCIslandConfig(
        n_particles=4, children_per_round=4, budget=20, k_min=1, seed=seed,
        accept_sharpness=accept_sharpness, greedy_accept=greedy)
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)
    return SMCIsland(
        cfg,
        FakeProposer(np.random.default_rng(1)),
        FakeEvaluator(score_cfg, seeds=(0,), rng=np.random.default_rng(2)),
        EventLogger(tmp_path / "ev.jsonl", clock=False),
        artifact_root=tmp_path / "cand")


def test_greedy_accepts_only_improvement(tmp_path):
    isl = _island(greedy=True, tmp_path=tmp_path)
    cur = _particle("cur", 1.0)
    better = _particle("better", 1.5)
    worse = _particle("worse", 0.5)
    tie = _particle("tie", 1.0)
    # 改好 → 接受（返回 child）；改差 / 平局 → 拒绝（返回 current）
    assert isl._sigmoid_accept(cur, better, alpha_t=1.0, potential_span=1.0, round_idx=3).id == "better"
    assert isl._sigmoid_accept(cur, worse, alpha_t=1.0, potential_span=1.0, round_idx=3).id == "cur"
    assert isl._sigmoid_accept(cur, tie, alpha_t=1.0, potential_span=1.0, round_idx=3).id == "cur"


def test_greedy_does_not_consume_rng(tmp_path):
    """贪婪分支确定性、不掷硬币：对改差子代做决定前后 rng 状态逐字节不变。"""
    isl = _island(greedy=True, tmp_path=tmp_path)
    cur = _particle("cur", 1.0)
    worse = _particle("worse", 0.5)
    before = isl.rng.bit_generator.state
    for _ in range(20):
        isl._sigmoid_accept(cur, worse, alpha_t=1.0, potential_span=1.0, round_idx=3)
    after = isl.rng.bit_generator.state
    assert before == after


def test_greedy_off_uses_sigmoid_and_consumes_rng(tmp_path):
    """默认关：改差子代按 sigmoid 概率接受（掷硬币）→ rng 被消耗，原接受核逐字节保留。"""
    isl = _island(greedy=False, tmp_path=tmp_path, accept_sharpness=1.0)
    cur = _particle("cur", 1.0)
    worse = _particle("worse", 0.9)   # 小幅改差 → sigmoid 概率非 0/1，必掷硬币
    before = isl.rng.bit_generator.state
    isl._sigmoid_accept(cur, worse, alpha_t=1.0, potential_span=1.0, round_idx=3)
    after = isl.rng.bit_generator.state
    assert before != after


def test_greedy_off_can_accept_worse_child_over_trials(tmp_path):
    """默认关且 β/alpha 温和时，多次试验里改差子代**能**被接受 → 证明 accept-worse 特性仍在（未被误关）。"""
    isl = _island(greedy=False, tmp_path=tmp_path, accept_sharpness=1.0, seed=7)
    cur = _particle("cur", 1.0)
    worse = _particle("worse", 0.9)
    accepted = sum(
        isl._sigmoid_accept(cur, worse, alpha_t=1.0, potential_span=1.0, round_idx=3).id == "worse"
        for _ in range(200))
    assert accepted > 0
