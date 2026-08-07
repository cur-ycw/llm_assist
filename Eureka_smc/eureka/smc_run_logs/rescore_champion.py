#!/usr/bin/env python
"""Canonical champion re-scorer — ONE evaluation yardstick for ALL experiments / methods.

报告标准(2026-08-07 修订):与 Eureka/RF-Agent 原生口径统一。
  * 主指标 (primary)  = max-over-curve of ``consecutive_successes`` (fallback ``gt_reward``)。
                        喂主表、做方法间比较(三方原生都按 max 选冠军)。
  * 次要 (secondary) = final_window_mean(last 10%)。作对照/审计,每个实验都一致记录。

对任意方法(SMC / RF-Agent / Eureka / Full)的冠军 test tensorboard 一律走这里,
保证 last-10% 与 max 两列在所有实验中口径完全一致、同源同 window。

用法:
  # 显式给每个 test seed 的 summaries 目录
  python rescore_champion.py --label RF-Agent-Ant --tb DIR0 DIR1 ... DIRk
  # 或自动从一个 run workspace 抓最后 K 个 policy-* 的 summaries(test 是最后阶段)
  python rescore_champion.py --label SMC-Ant --run /path/to/workspace --last 5
在对应 conda 环境下运行(SMC->eureka,RF-Agent->rfagent);两 env 都能 import load_tensorboard_logs。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# 复用冻结的聚合定义(纯 numpy,无 tb 依赖);两 env 的 eureka 目录都在此相对位置。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from smc.score import aggregate_metric  # noqa: E402


def _get_loader():
    """两 env 都提供 utils.file_utils.load_tensorboard_logs;探测可用者。"""
    try:
        from utils.file_utils import load_tensorboard_logs
        return load_tensorboard_logs
    except Exception:
        for p in ("/root/ycw/Eureka/eureka", "/root/ycw/RF-Agent/RF_Agent"):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        from utils.file_utils import load_tensorboard_logs
        return load_tensorboard_logs


def _summaries_dir(policy_dir):
    """run workspace 下 policy-*/runs/<run>/summaries。"""
    hit = glob.glob(os.path.join(policy_dir, "runs", "*", "summaries"))
    return hit[0] if hit else policy_dir


def score_dirs(tb_dirs, metric="consecutive_successes", fallback="gt_reward", window_frac=0.1):
    load = _get_loader()
    rows = []
    for d in tb_dirs:
        logs = load(d)
        key = metric if (metric in logs and len(logs[metric])) else fallback
        series = logs[key]
        rows.append({
            "dir": d,
            "metric": key,
            "n": len(series),
            "last10": aggregate_metric(series, "final_window_mean", window_frac),  # 主
            "max": aggregate_metric(series, "max"),                                # 次
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--tb", nargs="*", default=[], help="每个 test seed 的 summaries 目录")
    ap.add_argument("--run", default=None, help="run workspace,自动抓最后 --last 个 policy-*")
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--window-frac", type=float, default=0.1)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tb_dirs = list(args.tb)
    if args.run:
        pol = sorted(glob.glob(os.path.join(args.run, "policy-*")), key=os.path.getmtime)
        tb_dirs = [_summaries_dir(p) for p in pol[-args.last:]]
    if not tb_dirs:
        ap.error("need --tb or --run")

    rows = score_dirs(tb_dirs, window_frac=args.window_frac)
    last10 = [r["last10"] for r in rows]
    mx = [r["max"] for r in rows]

    print(f"# {args.label}  (metric={rows[0]['metric']}, window={args.window_frac})")
    print(f"{'seed':>4} {'max(primary)':>16} {'last10%(secondary)':>20} {'n':>5}")
    for i, r in enumerate(rows):
        print(f"{i:>4} {r['max']:>16.6f} {r['last10']:>20.6f} {r['n']:>5}")
    print("-" * 48)
    print(f"PRIMARY   max    : mean={np.mean(mx):.6f}  std={np.std(mx):.6f}  raw={[round(x,4) for x in mx]}")
    print(f"SECONDARY last10%: mean={np.mean(last10):.6f}  std={np.std(last10):.6f}  raw={[round(x,4) for x in last10]}")

    out = {
        "label": args.label, "window_frac": args.window_frac, "per_seed": rows,
        "primary_max_mean": float(np.mean(mx)), "primary_max_std": float(np.std(mx)),
        "secondary_last10_mean": float(np.mean(last10)), "secondary_last10_std": float(np.std(last10)),
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[json] {args.json_out}")


if __name__ == "__main__":
    main()
