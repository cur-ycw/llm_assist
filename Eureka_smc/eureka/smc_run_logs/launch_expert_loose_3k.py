#!/usr/bin/env python
"""专家 @原判定(松) @3000 iter 对照组 —— 补齐与 strict 组一致的 iter，消除之前 1500/3000
对照不一致。native 当前为松判定(已被 calibrate_strict 还原)，直接跑 task=FrankaCabinet 即松。
跑起来(导入完成)后，把 native 永久改成 strict —— 即"改回现在的判定标准"，使搜索模板与
baseline 任务统一用严判定。三个 seed = strict 组同款 {100,102,104}。
"""
import importlib.util, os, subprocess, time
from pathlib import Path

EUREKA = Path("/root/ycw/Eureka_smc/eureka")
ISAAC = Path(os.path.dirname(importlib.util.find_spec("isaacgymenvs").origin))
OUT = EUREKA / "smc_run_logs" / "calib_strict"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [100, 102, 104]
MAX_ITERS = 3000

LOOSE = """    successes = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)"""

STRICT = """    opened = cabinet_dof_pos[:, 3] > 0.39
    grasp_ok = (franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2]) \\
        & (franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2]) \\
        & (franka_lfinger_pos[:, 0] >= drawer_grasp_pos[:, 0] - distX_offset) \\
        & (franka_rfinger_pos[:, 0] >= drawer_grasp_pos[:, 0] - distX_offset)
    success_now = opened & grasp_ok
    successes = torch.where(success_now, torch.ones_like(successes), successes)
    reset_buf = torch.where(success_now, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)"""


def started(log):
    try:
        return "fps step:" in log.read_text(errors="ignore")
    except Exception:
        return False


def main():
    native = ISAAC / "tasks" / "franka_cabinet.py"
    txt = native.read_text()
    assert txt.count(LOOSE) == 1 and "grasp_ok" not in txt, "native 不是预期的松判定状态，中止"

    procs = []
    for i, s in enumerate(SEEDS):
        log = OUT / f"expertloose_seed{s}.txt"
        f = open(log, "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 3))
        p = subprocess.Popen(
            ["python", "-u", str(ISAAC / "train.py"), "hydra/output=subprocess",
             "task=FrankaCabinet", f"experiment=expertloose_seed{s}",
             "wandb_activate=False", "wandb_entity=", "wandb_project=",
             "headless=True", "capture_video=False", "force_render=False",
             f"max_iterations={MAX_ITERS}", f"seed={s}"],
            stdout=f, stderr=f, env=env)
        procs.append((p, log, s))
        print(f"launched expertloose seed{s} on GPU {i%3} pid={p.pid}")

    print("等 3 个松判定专家训练导入完成后，把 native 永久改成 strict…")
    t0 = time.time()
    while time.time() - t0 < 300:
        if all(started(log) for _, log, _ in procs):
            break
        time.sleep(5)
    native.write_text(txt.replace(LOOSE, STRICT))
    print(f"[native] 已永久改为 strict（当前判定标准）；grasp_ok in native = {'grasp_ok' in native.read_text()}")
    print("松判定专家对照组在后台继续跑（不阻塞）。")


if __name__ == "__main__":
    main()
