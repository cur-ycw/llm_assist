"""RF-Agent 奖励专用 proposal action 的注册与路由（迁移计划 §6.4.1、§6.5）。

本模块是**纯逻辑**：只定义五个 action 的名字、Phase-3a 的启用/比例配置，以及一个
**确定性、不消费任何 RNG** 的 slot→action 派发函数。它不调用 LLM、不接触 Isaac Gym，
可完全离线单测。

设计约束（见计划 §6.5 与测试确定性要求）：
  * ``assign_actions`` 必须无 rng —— island / proposer / evaluator 各自有独立 seeded RNG，
    若在此额外抽样会改变默认路径的 rng 消费顺序，令 ``test_reproducible_under_same_seeds``
    漂移。故派发用确定性 round-robin（按比例权重铺 slot）。
  * ``mode="generic"``（默认）时上层完全走旧的单一 ``eureka_reflection`` 路径，本模块不介入；
    ``mode="rf"`` 时才按 ``enabled``/``rf_ratio`` 路由。
  * ``RF_ACTIONS`` 固定五名、``rf_ratio`` 默认 ``[2,2,2,1,1]``（RF-Agent 论文固定动作配比），
    Phase-3a 只把前两名放进 ``enabled``，crossover/path/different 留 Phase-3b 前向兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RF_ACTIONS", "ActionConfig", "assign_actions"]

# 五个 action 的固定顺序（与 rf_ratio 一一对应）。
MUTATION_STRUCTURE = "mutation_structure"
MUTATION_PARAMETER = "mutation_parameter"
CROSSOVER = "crossover"
PATH_REASONING = "path_reasoning"
DIFFERENT_THOUGHT = "different_thought"

RF_ACTIONS = [
    MUTATION_STRUCTURE,   # a_m1：改奖励结构（增删组件 / 改组合、门控、耦合）
    MUTATION_PARAMETER,   # a_m2：固定结构内调参（权重、阈值、温度、clip、scale）
    CROSSOVER,            # a_c3：横向利用高分 elite（Phase-3b）
    PATH_REASONING,       # a_r4：纵向总结 accepted-edit 谱系（Phase-3b）
    DIFFERENT_THOUGHT,    # a_d5：从异谱系历史主动远离（Phase-3b）
]


@dataclass
class ActionConfig:
    """五操作路由配置（来自 ``cfg.algo.actions``）。

    ``mode="generic"``：不路由，上层走单一 eureka_reflection（默认，行为=Phase-2）。
    ``mode="rf"``：按 ``enabled`` 中各 action 对应的 ``rf_ratio`` 权重路由。
    ``gate_contracts``：Phase-3a 恒 False —— 契约仅**审计命中率**，不据此丢弃候选
      （MH 是 reward-only，不应因契约拒绝改变搜索分布）。
    """

    mode: str = "generic"
    enabled: list[str] = field(default_factory=lambda: [MUTATION_STRUCTURE, MUTATION_PARAMETER])
    rf_ratio: list[int] = field(default_factory=lambda: [2, 2, 2, 1, 1])
    gate_contracts: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("generic", "rf"):
            raise ValueError(f"未知 actions.mode={self.mode!r}（应为 generic|rf）")
        unknown = [a for a in self.enabled if a not in RF_ACTIONS]
        if unknown:
            raise ValueError(f"enabled 含未知 action：{unknown}（合法={RF_ACTIONS}）")
        if len(self.rf_ratio) != len(RF_ACTIONS):
            raise ValueError(
                f"rf_ratio 长度须为 {len(RF_ACTIONS)}（对应 {RF_ACTIONS}），实际 {len(self.rf_ratio)}")

    def enabled_weights(self) -> list[int]:
        """``enabled`` 中各 action 在 ``RF_ACTIONS`` 里对应的比例权重。"""
        pos = {a: i for i, a in enumerate(RF_ACTIONS)}
        return [self.rf_ratio[pos[a]] for a in self.enabled]


def assign_actions(n: int, enabled: list[str], ratio: list[int]) -> list[str]:
    """把 ``n`` 个提议 slot **确定性**地铺成 action 序列（无 RNG）。

    按 ``enabled`` 中各 action 的 ``ratio`` 权重做加权 round-robin：先展开成一个长度为
    ``sum(weights)`` 的基本轮次（如 enabled=[m1,m2], ratio=[2,2] → [m1,m1,m2,m2]），再循环
    平铺到 ``n`` 个 slot。权重为 0 的 action 不出现；全 0 或空 enabled 视作配置错误。

    返回长度恰为 ``n`` 的 action 名列表。确定性：相同入参恒得相同输出，且不触碰任何 rng。
    """
    if n <= 0:
        return []
    if not enabled:
        raise ValueError("assign_actions: enabled 为空")
    if len(ratio) != len(enabled):
        raise ValueError(f"assign_actions: ratio 长度 {len(ratio)} != enabled 长度 {len(enabled)}")
    cycle: list[str] = []
    for action, w in zip(enabled, ratio):
        if w < 0:
            raise ValueError(f"assign_actions: 负权重 {w} for {action}")
        cycle.extend([action] * w)
    if not cycle:
        raise ValueError("assign_actions: 所有权重为 0，无可派发 action")
    return [cycle[i % len(cycle)] for i in range(n)]
