"""单 seed 噪声下的 β_target 选择评估（零 GPU 纯数值）。

背景（迁移决策）：搜索阶段每个奖励只用固定 seed 42 评估一次（与 Eureka 协议一致、预算
公平），该单 seed 分数即定义为奖励表现，不再多 seed 降噪。问题：这个固定 seed 的运气会不会
经由 SMC 重采样被**不可逆地锁进种群**，把"seed42 运气好"当成"真质量高"？这由 β_target
控制（权重比 = exp(Δλ·β·ΔR)，β 同时放大信号与噪声）。本脚本量化：给定单 seed 噪声，
不同 β_target / N 下种群最终**真实质量**被富集还是被噪声带跑。

噪声模型（据 FrankaCabinet 冒烟实测的双峰"撞上/撞不上"）：
  单 seed 观测 R ~ Bernoulli(q)，q 为奖励真实质量（=多 seed 期望成功率）。
  即真质量 q 的奖励，在某个随机 seed 上"撞上"(R≈1) 的概率为 q，否则 R≈0。
  （另提供 gaussian 连续噪声模型作对照，噪声更温和。）

流程复刻单岛 SMC 的**选择部分**：find_next_lambda 退火 → systematic_resample →
每粒子 1 次变异(reward-only MH)。变异建模为 q' = clip(q + N(improve, sd))，并对新代码
重新抽一次单 seed 观测 R'（新 seed42 draw）。变异是否改进 q 的具体数值不影响**跨 β 的
相对结论**（我们比较的是同一噪声下 β 的效果）。

用法：cd eureka && PYTHONPATH=. python -m smc.simulate_single_seed_noise
"""

from __future__ import annotations

import argparse
import numpy as np

from smc.temperature import find_next_lambda, log_incremental_weights
from smc.resampling import systematic_resample


def observe(q: np.ndarray, rng: np.random.Generator, model: str) -> np.ndarray:
    """单固定 seed 观测。binary: R~Bernoulli(q)（双峰）；gaussian: clip(q+N(0,0.3))。"""
    if model == "binary":
        return (rng.random(len(q)) < q).astype(float)
    return np.clip(q + rng.normal(0.0, 0.3, len(q)), 0.0, 1.0)


def run_smc(N, beta_target, kappa, q_init, rng, model,
            max_stages=15, min_iters=3, improve=0.05, improve_sd=0.12):
    """跑一次单岛 SMC 选择过程，返回 (最终真实质量数组, 存活的原始 lineage 集合)。"""
    q = q_init.copy()
    R = observe(q, rng, model)
    lineage = np.arange(N)
    lam, max_delta = 0.0, 1.0 / min_iters
    for _ in range(max_stages):
        if lam >= 1.0:
            break
        lam_next = find_next_lambda(R, lam, beta_target, kappa, max_delta)
        delta = lam_next - lam
        if delta <= 1e-9:
            break  # 无法前进
        logw = log_incremental_weights(R, delta, beta_target)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        anc = systematic_resample(w, rng)
        q, R, lineage = q[anc], R[anc], lineage[anc]
        beta_t = beta_target * lam_next
        for i in range(N):  # 每粒子 1 次 reward-only MH 变异
            q_child = float(np.clip(q[i] + rng.normal(improve, improve_sd), 0.0, 1.0))
            R_child = observe(np.array([q_child]), rng, model)[0]
            if rng.random() < min(1.0, np.exp(beta_t * (R_child - R[i]))):
                q[i], R[i] = q_child, R_child  # 接受（同 lineage 内部演化）
        lam = lam_next
    return q, set(lineage.tolist())


def eval_cell(N, beta_target, kappa, model, trials, rng, q_dist="uniform"):
    finals, uniques, best_survive, init_means = [], [], [], []
    for _ in range(trials):
        if q_dist == "uniform":
            q_init = rng.random(N)
        else:  # 少数高质量 + 多数平庸，更接近真实初始池
            q_init = np.clip(rng.beta(1.5, 4.0, N), 0, 1)
        best_idx = int(np.argmax(q_init))
        q_final, surviving = run_smc(N, beta_target, kappa, q_init, rng, model)
        finals.append(q_final.mean())
        uniques.append(len(surviving))
        best_survive.append(1.0 if best_idx in surviving else 0.0)
        init_means.append(q_init.mean())
    finals, init_means = np.array(finals), np.array(init_means)
    return dict(
        final_q=finals.mean(),
        init_q=init_means.mean(),
        lift=(finals - init_means).mean(),          # 相对初始的真实质量提升（含变异漂移）
        unique=np.mean(uniques),                     # 存活原始 lineage 数（越高越不坍缩）
        best_survive=np.mean(best_survive),          # 初始最优 lineage 存活率（越高越robust）
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--model", choices=["binary", "gaussian"], default="binary")
    ap.add_argument("--qdist", choices=["uniform", "beta"], default="beta")
    args = ap.parse_args()

    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    Ns = [4, 8, 16]
    kappa = 0.5

    print(f"\n单 seed 噪声下 β_target 评估 | 噪声模型={args.model} | 初始 q 分布={args.qdist} "
          f"| trials={args.trials} | κ={kappa}")
    print("指标: lift=真实质量相对初始的提升 | unique=存活原始lineage数(抗坍缩) | "
          "best_surv=初始最优lineage存活率(抗噪杀)")
    print("=" * 92)
    for N in Ns:
        print(f"\nN={N}")
        print(f"  {'β_target':>9} | {'final_q':>8} | {'lift':>7} | {'unique':>7}/{N} | {'best_surv':>9}")
        print("  " + "-" * 62)
        for b in betas:
            rng = np.random.default_rng(args.seed + N)  # 同 N 下各 β 用同随机流，可比
            r = eval_cell(N, b, kappa, args.model, args.trials, rng, args.qdist)
            flag = ""
            if b > 0 and r["best_survive"] < 0.5:
                flag += " ⚠抗噪弱"
            if r["unique"] <= 1.05:
                flag += " ⚠坍缩"
            print(f"  {b:>9.1f} | {r['final_q']:>8.3f} | {r['lift']:>+7.3f} | "
                  f"{r['unique']:>7.2f}/{N} | {r['best_survive']:>9.2f}{flag}")

    print("\n" + "=" * 92)
    print("读法：β=0 是无选择基线(只有变异漂移)。lift 越高=越能靠单seed分数选出真质量；")
    print("      但 β 太大 → unique 掉到 1(坍缩)、best_surv 掉(真好但seed运气差的被杀)。")
    print("      在双峰(binary)噪声下找 lift 尚可、同时 unique/best_surv 不崩的最大 β 即推荐值。")


if __name__ == "__main__":
    main()
