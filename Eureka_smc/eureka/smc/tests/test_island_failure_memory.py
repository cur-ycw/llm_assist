"""失败探索记忆（failure_memory）集成测试：默认关=精确 no-op；开启=被拒子代记 b+c 摘要并注入。

设计要点：FakeProposer 系测试替身**不读 parent_feedback 内容**（只据 quality 标记产码），因此
开启失败记忆只会「附加」failure_memory_record 事件、并不改变子代/接受/rng 轨迹。据此可断言：
同 seed 下「开 vs 关」两轮，除 record 事件外事件流逐字节一致 → 证明注入不扰动搜索 determinism。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eureka.smc.evaluator import FakeEvaluator
from eureka.smc.event_logger import EventLogger
from eureka.smc.island import SMCIsland, SMCIslandConfig
from eureka.smc.proposer import FakeProposer
from eureka.smc.score import ScoreConfig


class _RejectingProposer(FakeProposer):
    """初始码带命名分量 dict；变异恒返回**更差**子代（带不同分量）→ 必被拒 → 触发失败记忆。

    同时记录每次 reflect 收到的 ``parent_feedback``，供断言「Previously tried」注入块。
    输出只依赖 parent 的 quality 标记、与 feedback 文本无关，故开/关失败记忆产码完全一致。
    """

    def __init__(self, rng, **kw):
        super().__init__(rng, **kw)
        self.seen_feedback: list[str] = []

    def initial_batch(self, n):
        self.n_calls += n
        out = []
        for i in range(n):
            q = 0.50 + 0.02 * i           # 递增 quality，制造非均匀父代分布
            out.append(f"def compute_reward(self):  # quality={q:.6f}\n"
                       f"    d = {{'base_r': 1.0, 'p{i}': 2.0}}\n"
                       f"    return self.rew_buf, d\n")
        return out

    def _worse_child(self, parent_code: str) -> str:
        q = max(0.0, self._quality_of(parent_code) - 0.20)   # 明确更差 → 被拒
        return (f"def compute_reward(self):  # quality={q:.6f}\n"
                f"    d = {{'base_r': 1.0, 'risky_term': 3.0}}\n"
                f"    return self.rew_buf, d\n")

    def reflect(self, parent_code, parent_feedback):
        self.seen_feedback.append(parent_feedback)
        self.n_calls += 1
        return self._worse_child(parent_code)

    def reflect_batch(self, items):
        return [self.reflect(c, fb) for c, fb in items]


def _run(tmp_path: Path, enabled: bool, tag: str):
    proposer = _RejectingProposer(np.random.default_rng(1))
    cfg = SMCIslandConfig(
        n_particles=4, children_per_round=4, budget=20,   # init4 + 4×4 = 4 轮
        k_min=1, gamma=1.0, seed=0,
        accept_sharpness=50.0,                             # 大 β：更差子代确定性被拒
        failure_memory_enabled=enabled, failure_memory_k=3)
    score_cfg = ScoreConfig(lower=0.0, upper=1.0)
    evaluator = FakeEvaluator(score_cfg, seeds=(0, 1, 2), rng=np.random.default_rng(2))
    log_path = tmp_path / f"events_{tag}.jsonl"
    island = SMCIsland(cfg, proposer, evaluator,
                       EventLogger(log_path, clock=False), artifact_root=tmp_path / f"cand_{tag}")
    island.run()
    events = [json.loads(l) for l in log_path.read_text().splitlines()]
    return island, proposer, events


def _strip(events):
    """去掉 seq/t（record 事件插入会移动后续 seq）与 record 事件，留纯搜索轨迹用于比对。"""
    out = []
    for e in events:
        if e.get("event") == "failure_memory_record":
            continue
        out.append({k: v for k, v in e.items() if k not in ("seq", "t")})
    return out


def test_failure_memory_disabled_is_exact_noop(tmp_path):
    island, proposer, events = _run(tmp_path, enabled=False, tag="off")
    # 关闭：不记录、不注入
    assert island._failure_memory == {}
    assert not any(e["event"] == "failure_memory_record" for e in events)
    assert all("Previously tried" not in fb for fb in proposer.seen_feedback)


def test_failure_memory_enabled_records_and_injects(tmp_path):
    island, proposer, events = _run(tmp_path, enabled=True, tag="on")
    recs = [e for e in events if e["event"] == "failure_memory_record"]
    # 1) 被拒的 valid 子代被记录
    assert recs, "开启后应至少有一条 failure_memory_record"
    assert island._failure_memory, "应按父代累积失败记忆"
    # 2) b+c 摘要：分量增减 + ΔJ
    s = recs[0]["summary"]
    assert "added components {risky_term}" in s
    assert "removed components" in s and "delta -" in s
    # 3) capped 到 k
    assert all(len(v) <= island.cfg.failure_memory_k for v in island._failure_memory.values())
    # 4) 后续轮从同一失败父代变异时，prompt 里出现「Previously tried」注入块
    assert any("Previously tried" in fb for fb in proposer.seen_feedback), \
        "重复变异失败父代时应注入失败记忆块"


def test_failure_memory_does_not_perturb_trajectory(tmp_path):
    # Fake 提议器不读 feedback → 开/关仅差 record 事件，其余搜索轨迹（eval/accept/resample）逐字节一致。
    _, _, ev_off = _run(tmp_path, enabled=False, tag="traj_off")
    _, _, ev_on = _run(tmp_path, enabled=True, tag="traj_on")
    assert _strip(ev_off) == _strip(ev_on)
