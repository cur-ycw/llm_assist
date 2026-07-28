#!/usr/bin/env python
"""严苛判定校准：在收紧的 compute_success（必须抓着把手拉开才算成功）下重训两个奖励，
看 (a) p73 的双峰满分是"真抓开"(保持高)还是"甩开"(掉到 0)，(b) 专家奖励信号是否被判定
收紧误杀(应仍 ≳ 松判定的 ~0.01)。种子取 p73 松 heldout 的两个"抓到"seed(100,102)+一个
"没抓到"seed(104)。判定收紧只改 successes/reset 两处，奖励本身不动。

不改动搜索代码；对 p73 直接改写它已训好的 env_code.py 的 compute_success；对专家临时补丁
安装版 franka_cabinet.py(备份→改→启动导入后即还原)。
"""
import importlib.util, os, shutil, subprocess, sys, time
from pathlib import Path

EUREKA = Path("/root/ycw/Eureka_smc/eureka")
ISAAC = Path(os.path.dirname(importlib.util.find_spec("isaacgymenvs").origin))
OUT = EUREKA / "smc_run_logs" / "calib_strict"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [100, 102, 104]
MAX_ITERS = 3000   # 跑满(2× 官方默认)让策略有时间真学会抓握，严判定测得公平

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


def patch(text, where):
    n = text.count(LOOSE)
    assert n == 1, f"{where}: 期望恰好 1 处松判定，实际 {n} 处"
    return text.replace(LOOSE, STRICT)


def launch(task, seed, gpu, tag):
    log = OUT / f"{tag}_seed{seed}.txt"
    f = open(log, "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.Popen(
        ["python", "-u", str(ISAAC / "train.py"), "hydra/output=subprocess",
         f"task={task}", f"experiment={tag}_seed{seed}",
         "wandb_activate=False", "wandb_entity=", "wandb_project=",
         "headless=True", "capture_video=False", "force_render=False",
         f"max_iterations={MAX_ITERS}", f"seed={seed}"],
        stdout=f, stderr=f, env=env)
    return p, log


def started(log):
    try:
        return "fps step:" in log.read_text(errors="ignore")
    except Exception:
        return False


def main():
    native = ISAAC / "tasks" / "franka_cabinet.py"
    gpt = ISAAC / "tasks" / "frankacabinetgpt.py"
    backup = ISAAC / "tasks" / "franka_cabinet.py.cal_bak"

    # ---- p73 strict：改写已训好的候选文件的 compute_success ----
    p73_src = (EUREKA / "outputs/eureka_smc/2026-07-27_11-58-01/candidates/i0-p73/env_code.py").read_text()
    gpt.write_text(patch(p73_src, "p73"))
    print(f"[p73] 写入 strict 版 → {gpt}")

    # ---- expert strict：备份并补丁安装版 native ----
    shutil.copy2(native, backup)
    native.write_text(patch(native.read_text(), "expert"))
    print(f"[expert] native 已打 strict 补丁（备份 {backup.name}）")

    procs = []
    try:
        for i, s in enumerate(SEEDS):
            procs.append(launch("FrankaCabinetGPT", s, i % 3, "p73strict") + ("p73", s))
        for i, s in enumerate(SEEDS):
            procs.append(launch("FrankaCabinet", s, i % 3, "expertstrict") + ("expert", s))
        print(f"已启动 {len(procs)} 个训练；等所有进程导入完成后还原 native…")
        # 等全部 import 到 "fps step"（或 300s 超时）再还原 native，确保 expert 进程读到的是 strict
        t0 = time.time()
        while time.time() - t0 < 300:
            if all(started(log) for _, log, _, _ in procs):
                break
            time.sleep(5)
    finally:
        shutil.copy2(backup, native)
        os.remove(backup)
        print("[expert] native 已还原为原始松判定版")

    print("等所有训练跑完（约 20-25 min）…")
    for p, log, tag, s in procs:
        p.wait()
        print(f"  done {tag} seed{s} -> {log.name}")
    print("ALL CALIB DONE")


if __name__ == "__main__":
    main()
