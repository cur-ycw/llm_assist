"""AST 组件契约的单元测试（计划 §8：m1 结构变化 / m2 仅参数变化 / 解析失败不抛）。"""

from __future__ import annotations

from eureka.smc.contracts import (
    check_mutation_parameter,
    check_mutation_structure,
    reward_signature,
)

PARENT = '''@torch.jit.script
def compute_reward(franka_grasp_pos, drawer_grasp_pos, cabinet_dof_pos):
    d = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    t_d: float = 0.1
    distance_reward = torch.exp(-d / t_d)
    pos = cabinet_dof_pos[:, 3]
    position_reward = torch.exp(pos / 0.5)
    total = distance_reward + position_reward
    rc = {"distance_reward": distance_reward, "position_reward": position_reward}
    return total, rc
'''

INLINE = PARENT.replace(
    '    rc = {"distance_reward": distance_reward, "position_reward": position_reward}\n'
    "    return total, rc",
    '    return total, {"distance_reward": distance_reward, "position_reward": position_reward}')

# 只改数值（温度 0.1→0.2、0.5→0.7），结构原样
PARAM_CHILD = PARENT.replace("0.1", "0.2").replace("0.5", "0.7")

# 加一个新组件 velocity_reward（结构/依赖/组件集合都变）
STRUCT_CHILD = (PARENT
                .replace("total = distance_reward + position_reward",
                         "velocity_reward = torch.exp(cabinet_dof_pos[:, 3] / 0.2)\n"
                         "    total = distance_reward + position_reward + velocity_reward")
                .replace('"position_reward": position_reward}',
                         '"position_reward": position_reward, "velocity_reward": velocity_reward}'))


def test_signature_components_named_dict():
    s = reward_signature(PARENT)
    assert s.parse_ok
    assert s.components == frozenset({"distance_reward", "position_reward"})


def test_signature_components_inline_dict():
    s = reward_signature(INLINE)
    assert s.parse_ok
    assert s.components == frozenset({"distance_reward", "position_reward"})


def test_named_and_inline_same_structural_hash():
    # 具名 dict 变量 vs 内联 dict 是不同结构写法，但组件集合一致；此处只要求都能解析组件。
    assert reward_signature(PARENT).components == reward_signature(INLINE).components


def test_param_change_hits_parameter_not_structure():
    assert check_mutation_parameter(PARENT, PARAM_CHILD) is True
    assert check_mutation_structure(PARENT, PARAM_CHILD) is False


def test_structure_change_hits_structure_not_parameter():
    assert check_mutation_structure(PARENT, STRUCT_CHILD) is True
    assert check_mutation_parameter(PARENT, STRUCT_CHILD) is False


def test_identical_code_neither_hits():
    assert check_mutation_structure(PARENT, PARENT) is False
    assert check_mutation_parameter(PARENT, PARENT) is False  # 数值也相同 → 非参数变异


def test_parse_failure_returns_false_not_raise():
    assert reward_signature("def (").parse_ok is False
    assert check_mutation_structure(PARENT, "def (") is False
    assert check_mutation_parameter("@@bad@@", PARENT) is False


def test_numeric_literals_captured():
    s = reward_signature(PARENT)
    # 0.1 与 0.5 应出现在数值多重集里
    assert 0.1 in s.numeric_literals and 0.5 in s.numeric_literals
