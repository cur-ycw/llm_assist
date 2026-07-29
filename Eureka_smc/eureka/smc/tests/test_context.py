"""历史型 action 上下文格式化的单元测试（计划 §8：crossover/path/different 供料拼装）。

这些函数是纯逻辑，用鸭子类型记录（SimpleNamespace）即可测，无需真实粒子/GPU/LLM。
"""

from __future__ import annotations

from types import SimpleNamespace

from eureka.smc.context import (
    MAX_DONOR_CHARS,
    first_sentence,
    format_crossover,
    format_different,
    format_path,
    truncate_code,
)


def _rec(id="a", code="def compute_reward():\n    return 0, {}", thought=None, score=None):
    return SimpleNamespace(id=id, reward_code=code, design_thought=thought, search_score=score)


def test_truncate_code_caps_long_code():
    long = "x = 1\n" * 2000
    out = truncate_code(long, max_chars=100)
    assert len(out) <= 100 + 40
    assert "truncated" in out


def test_truncate_code_keeps_short_code():
    short = "def compute_reward():\n    return 0, {}"
    assert truncate_code(short) == short.strip()


def test_first_sentence_stops_at_period():
    assert first_sentence("Add a distance term. Then more text.") == "Add a distance term."


def test_first_sentence_empty_on_none():
    assert first_sentence(None) == ""
    assert first_sentence("") == ""


def test_format_crossover_lists_variants_with_scores_and_code():
    donors = [_rec(id="d1", code="def compute_reward():\n    return 1, {}", score=0.8),
              _rec(id="d2", code="def compute_reward():\n    return 2, {}", score=0.6)]
    out = format_crossover(donors)
    assert "Variant 1" in out and "Variant 2" in out
    assert "0.8000" in out and "0.6000" in out
    assert "return 1" in out and "return 2" in out
    assert "```python" in out


def test_format_path_orders_earliest_to_latest_with_intents():
    chain = [_rec(id="c1", thought="start simple.", score=0.1),
             _rec(id="c2", thought="add grasp term.", score=0.5),
             _rec(id="c3", thought="tune temperatures.", score=0.7)]
    out = format_path(chain)
    lines = [l for l in out.splitlines() if l.startswith("Step")]
    assert len(lines) == 3
    assert lines[0].startswith("Step 1") and "start simple" in lines[0]
    assert lines[2].startswith("Step 3") and "tune temperatures" in lines[2]
    # 分数按轨迹顺序出现
    assert out.index("0.1000") < out.index("0.5000") < out.index("0.7000")


def test_format_path_initial_reward_when_no_thought():
    chain = [_rec(id="c1", thought=None, score=0.1),
             _rec(id="c2", thought="add term.", score=0.5)]
    out = format_path(chain)
    assert "(initial reward)" in out


def test_format_different_lists_distinct_intents():
    others = [_rec(id="o1", thought="focus on distance shaping.", score=0.4),
              _rec(id="o2", thought="focus on velocity penalty.", score=0.3)]
    out = format_different(others)
    assert "distance shaping" in out and "velocity penalty" in out
    assert out.count("\n- ") == 2  # 两条要点


def test_max_donor_chars_is_reasonable():
    assert MAX_DONOR_CHARS >= 1000
