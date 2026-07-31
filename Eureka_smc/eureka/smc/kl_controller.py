"""预算条件化的 KL 父代选择控制器（新方法：预算与进展自适应 SMC 奖励搜索 §2.3–2.4）。

替换原 ESS 自适应退火桥（``temperature.py``）。核心思想：**父代选择的集中度只由剩余
LLM 修改预算决定**，不看 ESS、近期成功率或 EMA。

一轮的解算链（``resolve``）：
  1. 目标有效父代数        K*_t = K_min + (N-K_min)·(b_t/B)^γ      —— 预算足→N，预算尽→K_min
  2. 目标 KL 半径          δ_t  = log(N / K*_t)
  3. 并列可行性修正        m 个并列最高分 → K_feas = max(K*, m)，δ_feas = log(N/K_feas)
  4. 一维二分反解 λ_t       使   KL( softmax(λ_t·J) ‖ U_N ) = δ_feas
  5. 父代分布              q_t  = softmax(λ_t·J)         （J 为**原始**任务性能，不归一化）

可达时 K_eff = exp(H(q_t)) = N·exp(-δ_t) = K*_t。全体同分退化为 q=U_N, λ=0。

一切纯 numpy、无 Isaac Gym / LLM，因此可完整单测（对应实验计划 §6 阶段 A 检查表 1–13）。
数值稳定：softmax 统一减去当轮最大 logit（log-sum-exp），该平移不改变分布、不构成奖励
标准化（§2.2）。``q=softmax(λ(aJ+b))`` 关于正仿射 (a>0) 不变：λ 随 1/a 缩放、λ·Δ 不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ControllerStep",
    "target_effective_parents",
    "boltzmann_gibbs",
    "shannon_entropy",
    "effective_parents",
    "kl_to_uniform",
    "feasible_target",
    "solve_lambda",
    "resolve",
]

_EPS = 1e-12


def target_effective_parents(
    b: float, B: float, n: int, k_min: float, gamma: float
) -> float:
    """目标有效父代数 ``K*_t = K_min + (N-K_min)·(b/B)^γ``，clip 到 ``[K_min, N]``。

    ``b`` 为**本轮开始时**的剩余修改预算，``B`` 为总修改预算。预算充足（b≈B）时 K*≈N（广探），
    预算耗尽（b≈0）时 K*→K_min（集中）。近期成功率/EMA/目标成功率均不进入（§2.3）。
    """
    if B <= 0 or n <= 0:
        raise ValueError("B and n must be positive")
    k_min = float(np.clip(k_min, 1.0, n))
    frac = float(np.clip(b / B, 0.0, 1.0))
    k_star = k_min + (n - k_min) * (frac ** gamma)
    return float(np.clip(k_star, k_min, float(n)))


def boltzmann_gibbs(scores: np.ndarray, lam: float) -> np.ndarray:
    """父代选择分布 ``q(i) = softmax(λ·J_i)``，log-sum-exp 稳定（减最大 logit）。

    ``λ=0`` → 均匀；``λ→∞`` → 质量集中到最高分（有并列则均分给并列者）。全体同分对任意
    λ 都返回均匀分布。
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    logits = lam * scores
    logits = logits - logits.max()          # 平移不改变 softmax（§2.2）
    w = np.exp(logits)
    s = w.sum()
    if not np.isfinite(s) or s <= 0.0:       # 极端 λ 下的兜底
        return np.full(n, 1.0 / n)
    return w / s


def shannon_entropy(q: np.ndarray) -> float:
    """香农熵 ``H(q) = -Σ q_i log q_i``（nat）。"""
    q = np.asarray(q, dtype=np.float64)
    nz = q > _EPS
    return float(-(q[nz] * np.log(q[nz])).sum())


def effective_parents(q: np.ndarray) -> float:
    """有效父代数 ``K_eff = exp(H(q))``（∈ [1, N]）。"""
    return float(np.exp(shannon_entropy(q)))


def kl_to_uniform(q: np.ndarray) -> float:
    """``D_KL(q ‖ U_N) = log N - H(q) = Σ q_i log(q_i·N)``（∈ [0, log N]）。"""
    q = np.asarray(q, dtype=np.float64)
    n = q.size
    if n <= 1:
        return 0.0
    return float(np.log(n) - shannon_entropy(q))


def feasible_target(
    scores: np.ndarray, k_star: float, n: int
) -> tuple[float, float, int]:
    """并列可行性修正：返回 ``(K_feas, δ_feas, m_ties)``。

    若有 ``m`` 个并列最高分粒子，则可达的最集中分布把质量均分给这 m 个 → 有效父代数不可能
    低于 m。故 ``K_feas = max(K*, m)``、``δ_feas = log(N/K_feas)``（§2.4）。不人为打破并列。
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return float(k_star), 0.0, 0
    mx = scores.max()
    m_ties = int(np.count_nonzero(scores >= mx - _EPS))
    k_feas = max(float(k_star), float(m_ties))
    k_feas = min(k_feas, float(n))
    delta_feas = float(np.log(n / k_feas)) if k_feas > 0 else 0.0
    return k_feas, delta_feas, m_ties


def solve_lambda(
    scores: np.ndarray,
    delta: float,
    lam_hi: float = 1.0,
    max_expand: int = 60,
    tol: float = 1e-6,
) -> tuple[float, float, bool]:
    """一维二分反解 ``λ`` 使 ``D_KL(softmax(λ·J) ‖ U_N) = δ``。

    KL(λ) 关于 λ 单调不减（λ 越大分布越集中），故二分良定义。返回 ``(λ, kl_actual, saturated)``：
      * 全体同分 或 ``δ ≤ tol``：``(0.0, 0.0, False)``（§2.4 全同分退化）；
      * ``δ`` 超过可达上界（受并列限制，即使很大的 λ 也够不到）：取能达到的最大 λ，
        置 ``saturated=True``（``kl_saturated`` 诊断，§8.5）；
      * 否则二分到容差内。``lam_hi`` 不足时按需倍增扩张（至多 ``max_expand`` 次）。
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n <= 1:
        return 0.0, 0.0, False
    spread = float(scores.max() - scores.min())
    if spread <= _EPS or delta <= tol:       # 全同分 或 目标≈均匀
        return 0.0, 0.0, False

    def kl_at(lam: float) -> float:
        return kl_to_uniform(boltzmann_gibbs(scores, lam))

    # 扩张上界，直到 KL(lam_hi) 追过 δ 或饱和（并列封顶使 KL 无法再升）。
    hi = float(lam_hi)
    expand = 0
    while kl_at(hi) < delta and expand < max_expand:
        prev = kl_at(hi)
        hi *= 2.0
        expand += 1
        if kl_at(hi) - prev < tol:           # KL 已封顶（并列限制），再升 λ 无用
            break
    kl_hi = kl_at(hi)
    if kl_hi < delta - tol:                   # 请求的 δ 不可达 → 饱和
        return float(hi), float(kl_hi), True

    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if kl_at(mid) < delta:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return float(lam), float(kl_at(lam)), False


@dataclass
class ControllerStep:
    """一轮控制器解算的完整快照（供 island 派发 + §8.4 诊断日志）。"""

    k_star: float           # 目标有效父代数
    k_feas: float           # 并列修正后的可行目标
    m_ties: int             # 并列最高分粒子数
    delta_req: float        # 请求 KL 半径 log(N/K*)
    delta_feas: float       # 可行 KL 半径 log(N/K_feas)
    lam: float              # 反解出的选择强度 λ_t
    kl_actual: float        # q_t 实际达到的 KL
    k_eff: float            # exp(H(q_t))，可达时≈K*
    q: np.ndarray           # 父代选择分布
    max_q: float            # max_i q_t(i)
    saturated: bool         # KL 是否饱和（够不到请求 δ）


def resolve(
    scores: np.ndarray,
    b: float,
    B: float,
    n: int,
    k_min: float,
    gamma: float,
    lam_hi: float = 1.0,
    tol: float = 1e-6,
) -> ControllerStep:
    """预算 → K* → δ → λ → q 的一次完整解算（§2.3–2.5 选择部分）。

    ``scores`` 为当前 N 个粒子的**原始** J（无效粒子的分数应在调用前替换为有限占位或被过滤；
    本控制器假定输入均为有限实数）。返回 ``ControllerStep``。
    """
    scores = np.asarray(scores, dtype=np.float64)
    k_star = target_effective_parents(b, B, n, k_min, gamma)
    k_feas, delta_feas, m_ties = feasible_target(scores, k_star, n)
    delta_req = float(np.log(n / k_star)) if k_star > 0 else 0.0
    lam, kl_actual, saturated = solve_lambda(scores, delta_feas, lam_hi=lam_hi, tol=tol)
    q = boltzmann_gibbs(scores, lam)
    return ControllerStep(
        k_star=k_star, k_feas=k_feas, m_ties=m_ties,
        delta_req=delta_req, delta_feas=delta_feas,
        lam=lam, kl_actual=kl_actual, k_eff=effective_parents(q),
        q=q, max_q=float(q.max()) if q.size else 0.0, saturated=saturated,
    )
