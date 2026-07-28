#!/usr/bin/env python
"""幂等注册 bidex(+单手) GPT 变体到共享 isaacgymenvs 训练包。

Eureka 运行时会把 envs/bidex/<env>.py 模板 rename+注入生成 tasks/<env>gpt.py，
但 isaacgym_task_map 里没有 <Class>GPT 条目、且首次导入前需要一个可导入的桩文件。
本脚本补齐这两件事，且可反复重跑：

  1. 桩文件：从模板渲染 tasks/<env>gpt.py（.replace(Class, Class+"GPT")），仅当缺失时写
     （运行时 eureka 会覆盖同一文件；已存在的手写/上轮生成文件不动）。
  2. 注册：在 tasks/__init__.py 末尾用 marker 区块**追加** import + isaacgym_task_map[...]=... 。
     不碰已有 dict 字面量（零手术）；每个 import 包 try/except —— 某个 LLM 生成的坏奖励
     文件只会让该任务不可用，绝不连累 FrankaCabinet 等其它任务的包导入（保护并行基线）。

用法：
  conda activate eureka && python register_bidex_gpt.py            # 默认 4+1
  python register_bidex_gpt.py --all                              # 全部 bidex
不依赖 GPU（除末尾可选的 import 验证）。
"""
import argparse
import importlib.util
import os
import re
import sys

# (env_name, native_class_name) —— module = env_name+"gpt"，gpt_class = native+"GPT"
DEFAULT_TASKS = [
    ("shadow_hand_over",             "ShadowHandOver"),
    ("shadow_hand_catch_underarm",   "ShadowHandCatchUnderarm"),
    ("shadow_hand_door_open_outward","ShadowHandDoorOpenOutward"),
    ("shadow_hand_kettle",           "ShadowHandKettle"),
    ("shadow_hand",                  "ShadowHand"),   # 单手：桩 shadow_handgpt.py 已存在，仅注册
]

# bidex 全家桶（--all 用）：env_name -> native class
ALL_BIDEX = {
    "shadow_hand_block_stack": "ShadowHandBlockStack",
    "shadow_hand_bottle_cap": "ShadowHandBottleCap",
    "shadow_hand_catch_abreast": "ShadowHandCatchAbreast",
    "shadow_hand_catch_over2underarm": "ShadowHandCatchOver2Underarm",
    "shadow_hand_catch_underarm": "ShadowHandCatchUnderarm",
    "shadow_hand_door_close_inward": "ShadowHandDoorCloseInward",
    "shadow_hand_door_close_outward": "ShadowHandDoorCloseOutward",
    "shadow_hand_door_open_inward": "ShadowHandDoorOpenInward",
    "shadow_hand_door_open_outward": "ShadowHandDoorOpenOutward",
    "shadow_hand_grasp_and_place": "ShadowHandGraspAndPlace",
    "shadow_hand_kettle": "ShadowHandKettle",
    "shadow_hand_lift_underarm": "ShadowHandLiftUnderarm",
    "shadow_hand_over": "ShadowHandOver",
    "shadow_hand_pen": "ShadowHandPen",
    "shadow_hand_push_block": "ShadowHandPushBlock",
    "shadow_hand_re_orientation": "ShadowHandReOrientation",
    "shadow_hand_scissors": "ShadowHandScissors",
    "shadow_hand_swing_cup": "ShadowHandSwingCup",
    "shadow_hand_switch": "ShadowHandSwitch",
    "shadow_hand_two_catch_underarm": "ShadowHandTwoCatchUnderarm",
}

# 模板搜索顺序（Eureka_smc 优先，回退 stock Eureka）
TEMPLATE_DIRS = [
    "/root/ycw/Eureka_smc/eureka/envs/bidex",
    "/root/ycw/Eureka_smc/eureka/envs/isaac",
    "/root/ycw/Eureka/eureka/envs/bidex",
    "/root/ycw/Eureka/eureka/envs/isaac",
]

MARK_START = "# === AUTO GPT REGISTRATION (bidex + single-hand) START ==="
MARK_END = "# === AUTO GPT REGISTRATION (bidex + single-hand) END ==="


def tasks_dir():
    spec = importlib.util.find_spec("isaacgymenvs")
    if spec is None:
        sys.exit("找不到 isaacgymenvs（需 conda activate eureka）")
    return os.path.join(os.path.dirname(spec.origin), "tasks")


def find_template(env_name):
    for d in TEMPLATE_DIRS:
        p = os.path.join(d, f"{env_name}.py")
        if os.path.isfile(p):
            return p
    return None


def render_stub(tdir, env_name, native_cls):
    """仅当 tasks/<env_name>gpt.py 缺失时，从模板渲染桩。返回 (module_name, action)。"""
    module = f"{env_name}gpt"
    dst = os.path.join(tdir, f"{module}.py")
    if os.path.exists(dst):
        return module, "exists"
    tmpl = find_template(env_name)
    if tmpl is None:
        return module, "NO_TEMPLATE"
    code = open(tmpl).read()
    gpt_cls = f"{native_cls}GPT"
    n = code.count(native_cls)
    rendered = code.replace(native_cls, gpt_cls)
    # sanity：类定义必须成功改名
    if f"class {gpt_cls}(" not in rendered:
        return module, f"RENAME_FAILED(class {native_cls} 未出现)"
    with open(dst, "w") as f:
        f.write(rendered)
    return module, f"rendered({n} refs)"


def build_block(tasks):
    lines = [MARK_START,
             "# 由 smc_run_logs/register_bidex_gpt.py 生成，可安全重跑覆盖。",
             "# 每个 import 独立 try/except：坏的 LLM 生成文件只影响该任务，不破坏整包导入。"]
    for env_name, native_cls in tasks:
        module = f"{env_name}gpt"
        gpt_cls = f"{native_cls}GPT"
        lines += [
            "try:",
            f"    from .{module} import {gpt_cls}",
            f'    isaacgym_task_map["{gpt_cls}"] = {gpt_cls}',
            "except Exception as _gpt_reg_err:",
            f'    print(f"[gpt-register] skip {gpt_cls}: {{_gpt_reg_err}}")',
        ]
    lines.append(MARK_END)
    return "\n".join(lines) + "\n"


def inject_init(tdir, tasks):
    init_path = os.path.join(tdir, "__init__.py")
    src = open(init_path).read()
    if "isaacgym_task_map" not in src:
        sys.exit("__init__.py 里没有 isaacgym_task_map，结构异常，中止")
    # 删旧块（幂等）
    pat = re.compile(re.escape(MARK_START) + r".*?" + re.escape(MARK_END) + r"\n?", re.S)
    src_clean = pat.sub("", src).rstrip("\n") + "\n\n"
    new_src = src_clean + build_block(tasks)
    # 写前先 py_compile 校验，坏了不落盘（保护基线共用文件）
    import py_compile, tempfile
    tf = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tf.write(new_src); tf.close()
    try:
        py_compile.compile(tf.name, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tf.name)
        sys.exit(f"生成的 __init__.py 语法错误，未写入：{e}")
    os.unlink(tf.name)
    with open(init_path, "w") as f:
        f.write(new_src)
    return init_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="注册全部 bidex 任务")
    ap.add_argument("--verify", action="store_true", help="末尾 import isaacgymenvs.tasks 验证（需 GPU 环境可导入 isaacgym）")
    args = ap.parse_args()

    if args.all:
        tasks = sorted(ALL_BIDEX.items())
    else:
        tasks = DEFAULT_TASKS

    tdir = tasks_dir()
    print(f"[tasks_dir] {tdir}\n")

    print("== 1. 渲染桩文件 ==")
    for env_name, native_cls in tasks:
        module, action = render_stub(tdir, env_name, native_cls)
        print(f"  {module:34s} {action}")

    print("\n== 2. 注入 __init__.py 注册块 ==")
    init_path = inject_init(tdir, tasks)
    print(f"  已更新 {init_path}（{len(tasks)} 个 <Class>GPT，marker 区块幂等）")

    print("\n== 3. 计划注册的 task key ==")
    for env_name, native_cls in tasks:
        print(f"  {native_cls}GPT   <- tasks/{env_name}gpt.py")

    if args.verify:
        print("\n== 4. import 验证 ==")
        import importlib
        import isaacgymenvs.tasks as T
        importlib.reload(T)
        got = sorted(k for k in T.isaacgym_task_map if k.endswith("GPT"))
        print(f"  map 中的 *GPT keys: {got}")


if __name__ == "__main__":
    main()
