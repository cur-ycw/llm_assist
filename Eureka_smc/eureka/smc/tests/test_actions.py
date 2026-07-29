"""RF-Agent action 路由的单元测试（计划 §8：assign_actions 确定性与比例）。"""

from __future__ import annotations

import pytest

from eureka.smc.actions import RF_ACTIONS, ActionConfig, assign_actions


def test_rf_actions_fixed_five():
    assert RF_ACTIONS == ["mutation_structure", "mutation_parameter", "crossover",
                          "path_reasoning", "different_thought"]


def test_assign_deterministic_and_length():
    a = assign_actions(8, ["mutation_structure", "mutation_parameter"], [1, 1])
    b = assign_actions(8, ["mutation_structure", "mutation_parameter"], [1, 1])
    assert a == b                       # 确定性（无 rng）
    assert len(a) == 8
    assert set(a) == {"mutation_structure", "mutation_parameter"}


def test_assign_ratio_weighting():
    # cycle=[m1,m1,m2]，铺 6 slot → m1×4, m2×2
    a = assign_actions(6, ["mutation_structure", "mutation_parameter"], [2, 1])
    assert a.count("mutation_structure") == 4
    assert a.count("mutation_parameter") == 2


def test_assign_only_enabled_appear():
    a = assign_actions(20, ["mutation_parameter"], [1])
    assert set(a) == {"mutation_parameter"}


def test_assign_empty_n():
    assert assign_actions(0, ["mutation_structure"], [1]) == []


def test_assign_guards():
    with pytest.raises(ValueError):
        assign_actions(4, [], [])                            # 空 enabled
    with pytest.raises(ValueError):
        assign_actions(4, ["mutation_structure"], [1, 2])    # ratio 长度不匹配
    with pytest.raises(ValueError):
        assign_actions(4, ["mutation_structure"], [0])       # 全 0 权重


def test_actionconfig_validation():
    with pytest.raises(ValueError):
        ActionConfig(mode="nope")
    with pytest.raises(ValueError):
        ActionConfig(mode="rf", enabled=["bogus"])
    with pytest.raises(ValueError):
        ActionConfig(rf_ratio=[1, 1])                        # 长度须为 5
    ac = ActionConfig(mode="rf", enabled=["mutation_structure", "mutation_parameter"])
    assert ac.enabled_weights() == [2, 2]                    # 默认 rf_ratio 前两位


def test_enabled_weights_reflects_ratio_positions():
    # crossover 在 RF_ACTIONS 第 2 位、different_thought 第 4 位
    ac = ActionConfig(mode="rf", enabled=["crossover", "different_thought"],
                      rf_ratio=[2, 2, 3, 1, 5])
    assert ac.enabled_weights() == [3, 5]


def test_generic_is_default():
    assert ActionConfig().mode == "generic"
