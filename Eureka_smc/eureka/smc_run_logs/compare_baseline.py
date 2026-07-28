#!/usr/bin/env python
"""对比 SMC 迁移框架 vs 原生专家奖励 PPO baseline。
评分口径两边一致：consecutive_successes 末 window_frac 窗均值（upper=1 归一化）。
- baseline：读 smc_run_logs/baseline_ppo/baseline_seed{S}.txt 里的 "Tensorboard Directory:"，
  再从该 summaries 目录读 consecutive_successes 曲线，取末 10% 窗均值。
- SMC：读最新 run 的 smc_summary.npz（冠军 search_score 与 heldout per_seed / mean）。
用法：python smc_run_logs/compare_baseline.py
"""
import glob, os, re, sys
import numpy as np

EUREKA = "/root/ycw/Eureka_smc/eureka"
WINDOW_FRAC = 0.1


def load_tb_metric(summ_dir, metric="consecutive_successes"):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(summ_dir); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = next((t for t in tags if t.split("/")[-1] == metric or t == metric), None)
    if tag is None:
        return None
    vals = [e.value for e in ea.Scalars(tag)]
    return vals


def window_mean(vals, frac=WINDOW_FRAC):
    a = np.asarray(vals, float)
    k = max(1, int(round(a.size * frac)))
    return float(a[-k:].mean())


def baseline_scores():
    out = {}
    for f in sorted(glob.glob(f"{EUREKA}/smc_run_logs/baseline_ppo/baseline_seed*.txt")):
        seed = int(re.search(r"seed(\d+)", f).group(1))
        txt = open(f, errors="ignore").read()
        m = re.search(r"Tensorboard Directory:\s*(.+)", txt)
        if not m:
            out[seed] = None; continue
        summ = m.group(1).strip()
        vals = load_tb_metric(summ) if os.path.isdir(summ) else None
        out[seed] = window_mean(vals) if vals else None
    return out


def smc_summary():
    runs = sorted(glob.glob(f"{EUREKA}/outputs/eureka_smc/*/smc_summary.npz"))
    if not runs:
        return None
    d = np.load(runs[-1], allow_pickle=True)
    return {k: d[k] for k in d.files}, runs[-1]


def main():
    print("=" * 60)
    print("BASELINE (原生专家奖励 PPO, consecutive_successes 末10%均值)")
    bs = baseline_scores()
    vals = [v for v in bs.values() if v is not None]
    for s, v in bs.items():
        print(f"  seed {s}: {'跑完 '+format(v, '.4f') if v is not None else '未完成/无TB'}")
    if vals:
        print(f"  >>> baseline mean={np.mean(vals):.4f}  (n={len(vals)})")
    print("=" * 60)
    smc = smc_summary()
    if smc is None:
        print("SMC 尚无 smc_summary.npz（实验未跑完）")
        return
    d, path = smc
    print(f"SMC 迁移框架 (冠军)  [{os.path.dirname(path).split('/')[-1]}]")
    print(f"  冠军 search_score(seed42) = {float(d['best_search_score']):.4f}")
    ho = d.get("heldout_per_seed")
    print(f"  冠军 heldout per_seed     = {np.asarray(ho, float)}")
    print(f"  冠军 heldout mean         = {float(d['heldout_search_score']):.4f}")
    print("=" * 60)
    if vals and not np.isnan(float(d['heldout_search_score'])):
        b, h = np.mean(vals), float(d['heldout_search_score'])
        print(f"结论: SMC冠军 heldout {h:.4f}  vs  专家baseline {b:.4f}  "
              f"→ {'SMC 胜' if h > b else '专家胜'} (Δ={h-b:+.4f})")


if __name__ == "__main__":
    main()
