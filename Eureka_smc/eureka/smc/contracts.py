"""奖励代码的 AST/组件契约审计（迁移计划 §6.4.1 局部性由 AST 验证，非 LLM 自称）。

纯函数、无副作用、可离线单测。核心思想：把奖励代码解析成 AST，抽取

  * ``components``：奖励分量集合 = ``return total, <dict>`` 里 dict 的字符串 keys
    （同时支持内联 dict 字面量与「return 具名变量、上文赋值该 dict」两种写法）；
  * ``self_deps``：奖励读取的环境信号 = ``self.<attr>`` 属性 ∪ 被使用的函数入参名
    （free-function jit 风格无 self.*，此时入参即依赖）；
  * ``numeric_literals``：全部数值常量的有序多重集（排除 bool）；
  * ``structural_hash``：**把数值常量归零后**的 ``ast.dump`` —— 只保留结构/组件/耦合/
    依赖，抹掉具体数值。两段代码若仅数值不同 → structural_hash 相同。

由此干净区分两个变异 action：
  * ``mutation_structure``（a_m1）：structural_hash **变化**（增删组件、改组合/门控/耦合、换算子、改依赖）。
  * ``mutation_parameter``（a_m2）：structural_hash **不变** 且 numeric_literals **变化**（只动数值/温度/scale）。

解析失败一律返回 ``parse_ok=False``，两个 check 均返回 ``False``，绝不抛异常（保护主循环）。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["RewardSignature", "reward_signature", "check_mutation_structure",
           "check_mutation_parameter"]

_SENTINEL = 0  # 数值常量归一化占位（仅用于 structural_hash）


def _is_number(v) -> bool:
    return isinstance(v, (int, float, complex)) and not isinstance(v, bool)


class _NumStrip(ast.NodeTransformer):
    """把数值常量的值归一到 sentinel（保留字符串/bool/None，它们是结构的一部分）。"""

    def visit_Constant(self, node: ast.Constant):  # py3.8+
        if _is_number(node.value):
            return ast.copy_location(ast.Constant(value=_SENTINEL, kind=None), node)
        return node

    # py<3.8 兼容（本仓运行在 3.8/3.9，Constant 已统一，但保留兜底）
    def visit_Num(self, node):  # pragma: no cover
        return ast.copy_location(ast.Constant(value=_SENTINEL, kind=None), node)


@dataclass
class RewardSignature:
    parse_ok: bool
    components: frozenset = field(default_factory=frozenset)
    self_deps: frozenset = field(default_factory=frozenset)
    numeric_literals: tuple = ()
    structural_hash: Optional[str] = None


def _find_reward_func(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """取名为 compute_reward 的函数；退而取最后一个函数定义。"""
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    for f in funcs:
        if f.name == "compute_reward":
            return f
    return funcs[-1]


def _dict_keys(node: ast.Dict) -> set:
    keys = set()
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.add(k.value)
    return keys


def _resolve_components(func: ast.FunctionDef) -> set:
    """从 return 的第二个元素解析组件集合（内联 dict 或具名 dict 变量）。"""
    returns = [n for n in ast.walk(func) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return set()
    val = returns[-1].value
    comp_node = None
    if isinstance(val, ast.Tuple) and len(val.elts) >= 2:
        comp_node = val.elts[1]
    elif isinstance(val, ast.Dict):
        comp_node = val
    if comp_node is None:
        return set()
    if isinstance(comp_node, ast.Dict):
        return _dict_keys(comp_node)
    if isinstance(comp_node, ast.Name):
        # 找 <name> = {...} 的最近赋值
        target = comp_node.id
        found: set = set()
        for n in ast.walk(func):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict):
                if any(isinstance(t, ast.Name) and t.id == target for t in n.targets):
                    found = _dict_keys(n.value)
        return found
    return set()


def _self_deps(func: ast.FunctionDef) -> set:
    deps = set()
    # self.<attr>
    for n in ast.walk(func):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            deps.add(f"self.{n.attr}")
    # 被使用的入参名（free-function jit 风格：入参即环境信号）
    params = {a.arg for a in func.args.args if a.arg != "self"}
    used = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    deps |= {p for p in params if p in used}
    return deps


def _numeric_literals(tree: ast.AST) -> tuple:
    nums = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and _is_number(n.value)]
    return tuple(sorted(nums, key=lambda x: (str(type(x)), repr(x))))


def reward_signature(code: str) -> RewardSignature:
    """解析奖励代码，返回结构签名。任何解析异常 → parse_ok=False。"""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return RewardSignature(parse_ok=False)
    func = _find_reward_func(tree)
    if func is None:
        return RewardSignature(parse_ok=False)
    try:
        components = frozenset(_resolve_components(func))
        deps = frozenset(_self_deps(func))
        numerics = _numeric_literals(func)
        stripped = _NumStrip().visit(ast.parse(code))
        ast.fix_missing_locations(stripped)
        structural_hash = ast.dump(stripped, annotate_fields=True)
    except Exception:  # noqa: BLE001 —— 审计不得因罕见 AST 形态破坏主循环
        return RewardSignature(parse_ok=False)
    return RewardSignature(parse_ok=True, components=components, self_deps=deps,
                           numeric_literals=numerics, structural_hash=structural_hash)


def check_mutation_structure(parent_code: str, child_code: str) -> bool:
    """结构变异命中：structural_hash 变化（组件/依赖/耦合/算子任一改变）。

    两侧任一解析失败 → False（无法判定即不算命中，只影响命中率统计）。
    """
    p, c = reward_signature(parent_code), reward_signature(child_code)
    if not (p.parse_ok and c.parse_ok):
        return False
    return p.structural_hash != c.structural_hash


def check_mutation_parameter(parent_code: str, child_code: str) -> bool:
    """参数变异命中：structural_hash 不变 且 数值常量变化（结构原样，只动数值/温度/scale）。"""
    p, c = reward_signature(parent_code), reward_signature(child_code)
    if not (p.parse_ok and c.parse_ok):
        return False
    return p.structural_hash == c.structural_hash and p.numeric_literals != c.numeric_literals
