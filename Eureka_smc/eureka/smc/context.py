"""三个历史型 RF-Agent action（crossover / path_reasoning / different_thought）的
**上下文材料格式化**（迁移计划 §6.4.1、§6.5）。

与 ``actions.py``（派发）/``contracts.py``（审计）一样，本模块是**纯逻辑**：把一批
历史奖励记录格式化成喂给 LLM 的自然语言/代码块，不调用 LLM、不接触 Isaac Gym、不消费
任何 RNG，可完全离线单测。

三个历史型 action 的「料」由 island 从其粒子注册表里确定性地选出（top-k elite / accepted-edit
谱系 / 异谱系设计意图），再交给这里的三个 ``format_*`` 拼成上下文块。上下文块会被
proposer 追加在 action 指令之后、``code_output_tip`` 之前。island 负责在没有可用料时传
``None``（此时该 action 退化为普通反思），故这里的三个函数总在**非空输入**上被调用。

记录对象采用鸭子类型：只要求有 ``.id`` / ``.reward_code`` / ``.design_thought`` /
``.search_score`` 属性（真实为 ``RewardParticle``，测试可用 ``SimpleNamespace``）。
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "MAX_DONOR_CHARS",
    "truncate_code",
    "first_sentence",
    "format_crossover",
    "format_path",
    "format_different",
]

# 单个供体奖励代码在 crossover 上下文里的字符上限（避免多供体撑爆上下文窗口）。
# 常规 jit 奖励函数 < 2000 字符，极少触发；触发时保留头部并显式标注截断。
MAX_DONOR_CHARS = 2400


def _fmt_score(s: Optional[float]) -> str:
    return "n/a" if s is None else f"{s:.4f}"


def truncate_code(code: str, max_chars: int = MAX_DONOR_CHARS) -> str:
    """把过长的奖励代码截到 ``max_chars``，显式标注截断（保头，函数签名+核心塑形在前）。"""
    code = code.strip()
    if len(code) <= max_chars:
        return code
    return code[:max_chars].rstrip() + "\n    # ... (truncated for context length)"


def first_sentence(text: Optional[str], max_chars: int = 240) -> str:
    """取一句设计意图：首个句号/换行截断，再兜底字符上限，供 different_thought 列表用。"""
    if not text:
        return ""
    t = " ".join(text.strip().split())
    for stop in (". ", "。", "\n"):
        i = t.find(stop)
        if 0 < i <= max_chars:
            return t[: i + (0 if stop == "\n" else len(stop))].strip()
    return t[:max_chars].strip()


def format_crossover(donors: list) -> str:
    """crossover（a_c3）：列出若干高分供体奖励的完整代码 + 分数，供 LLM 融合其长处。

    ``donors`` 已由 island 按分数降序选出且去重（非空）。返回追加进提示的上下文块。
    """
    blocks = []
    for i, d in enumerate(donors, 1):
        code = truncate_code(d.reward_code)
        blocks.append(
            f"[Variant {i}] (search_score={_fmt_score(d.search_score)})\n"
            f"```python\n{code}\n```")
    return ("High-performing reward variants for you to combine:\n\n"
            + "\n\n".join(blocks))


def format_path(chain: list) -> str:
    """path_reasoning（a_r4）：把 accepted-edit 谱系（最早→最新）连同分数与每步意图列成轨迹。

    ``chain`` 已由 island 沿 ``accepted_transition_parent_id`` 回溯并反转成时间正序（长度≥2）。
    """
    lines = []
    for i, n in enumerate(chain, 1):
        intent = first_sentence(getattr(n, "design_thought", None)) or "(initial reward)"
        lines.append(f"Step {i}: search_score={_fmt_score(n.search_score)} -- {intent}")
    return ("Trajectory of reward edits that were each accepted for improving training "
            "(earliest to latest):\n" + "\n".join(lines))


def format_different(others: list) -> str:
    """different_thought（a_d5）：列出异谱系已探索过的设计意图，供 LLM 主动远离。

    ``others`` 已由 island 从注册表里挑出（去重设计意图、排除本粒子谱系，非空）。
    """
    lines = [f"- {first_sentence(getattr(o, 'design_thought', None))}"
             for o in others
             if first_sentence(getattr(o, "design_thought", None))]
    return ("Design directions that have already been explored on this task "
            "(propose something meaningfully different from all of these):\n"
            + "\n".join(lines))
