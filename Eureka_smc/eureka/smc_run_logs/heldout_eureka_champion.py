#!/usr/bin/env python
"""原版 Eureka 冠军奖励的 heldout 评估，与 SMC 冠军 p73 的 0.419 逐项对齐：
  seeds=[100..104], max_iterations=1500, 松判定(冠军 env 本就松), 评分=consecutive_successes 末10%窗均值。
冠军 = outputs/eureka/2026-07-28_07-20-29 的 iter0_response12（seed42 训练分 0.99999）。
其完整 env 文件 env_iter0_response12.py 已含注入奖励+松 compute_success，直接拷成 frankacabinetgpt.py 训练。
不碰安装版 native / 搜索代码。用法：conda activate eureka && source Eureka/env.sh && python heldout_eureka_champion.py
"""
import importlib.util, os, shutil, subprocess, sys, time, glob, re
from pathlib import Path
import numpy as np

EUREKA = Path("/root/ycw/Eureka_smc/eureka")
BASELINE = Path("/root/ycw/Eureka/eureka/outputs/eureka/2026-07-28_07-20-29")
CHAMP_ENV = BASELINE / "env_iter0_response12.py"
ISAAC = Path(os.path.dirname(importlib.util.find_spec("isaacgymenvs").origin))
OUT = EUREKA / "smc_run_logs" / "heldout_eureka_champ"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [100, 101, 102, 103, 104]
MAX_ITERS = 1500
WINDOW_FRAC = 0.1


def window_mean(summ_dir, metric="consecutive_successes", frac=WINDOW_FRAC):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(summ_dir); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = next((t for t in tags if t.split("/")[-1] == metric or t == metric), None)
    if tag is None:
        return None
    vals = np.asarray([e.value for e in ea.Scalars(tag)], float)
    if vals.size == 0:
        return None
    k = max(1, int(round(vals.size * frac)))
    return float(vals[-k:].mean())


def launch(seed, gpu, tag):
    log = OUT / f"{tag}_seed{seed}.txt"
    f = open(log, "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.Popen(
        ["python", "-u", str(ISAAC / "train.py"), "hydra/output=subprocess",
         "task=FrankaCabinetGPT", f"experiment={tag}_seed{seed}",
         "wandb_activate=False", "wandb_entity=", "wandb_project=",
         "headless=True", "capture_video=False", "force_render=False",
         f"max_iterations={MAX_ITERS}", f"seed={seed}"],
        stdout=f, stderr=f, env=env)
    return p, log, seed


def main():
    gpt = ISAAC / "tasks" / "frankacabinetgpt.py"
    assert CHAMP_ENV.exists(), f"找不到冠军 env: {CHAMP_ENV}"
    # 松判定 sanity：冠军 env 必须是松（无 grasp_ok），否则与 p73 0.419 口径不一致
    src = CHAMP_ENV.read_text()
    assert "grasp_ok" not in src, "冠军 env 含 strict 判定，与 loose 协议不符，中止"
    assert "successes = torch.where(cabinet_dof_pos[:, 3] > 0.39" in src, "未找到松判定行"
    shutil.copy2(CHAMP_ENV, gpt)
    print(f"[setup] 冠军 env → {gpt}  (loose, 松判定已核)")

    procs = [launch(s, i % 3, "champ") for i, s in enumerate(SEEDS)]
    print(f"[launch] {len(procs)} 个训练 (seeds={SEEDS}, {MAX_ITERS} iter, 3 卡轮转)")
    for p, log, s in procs:
        p.wait()
        print(f"  done seed{s} -> {log.name}")

    print("\n== 收集末10%窗均值 ==")
    results = {}
    for _, log, s in procs:
        txt = log.read_text(errors="ignore")
        m = re.search(r"Tensorboard Directory:\s*(.+)", txt)
        wm = None
        if m and os.path.isdir(m.group(1).strip()):
            wm = window_mean(m.group(1).strip())
        else:  # 回退：从 experiment 名找 summaries
            cand = sorted(glob.glob(str(ISAAC.parent / "runs" / f"champ_seed{s}*" / "summaries")))
            if cand:
                wm = window_mean(cand[-1])
        results[s] = wm
        print(f"  seed {s}: {'%.4f' % wm if wm is not None else '无TB'}")

    vals = [v for v in results.values() if v is not None]
    per = np.array([results[s] if results[s] is not None else np.nan for s in SEEDS])
    print("\n" + "=" * 60)
    print(f"原版 Eureka 冠军 heldout per_seed = {per}")
    if vals:
        print(f"原版 Eureka 冠军 heldout mean     = {np.mean(vals):.4f}  (n={len(vals)})")
    print("对比: 专家 ~0.010  |  Eureka+SMC(p73) 0.419")
    print("=" * 60)
    np.savez(OUT / "heldout_result.npz", seeds=SEEDS, per_seed=per,
             mean=np.mean(vals) if vals else np.nan)


if __name__ == "__main__":
    main()
