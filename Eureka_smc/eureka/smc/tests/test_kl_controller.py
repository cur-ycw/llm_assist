"""KL 父代选择控制器的单元测试（实验计划 §6 阶段 A 检查表 1–13）。

纯数值、无 GPU/LLM，验证预算条件化 KL 控制器的核心正确性：概率合法、KL 二分精度、可达时
K_eff≈K*、δ 单调、全同分/并列退化、极端尺度稳定、正仿射不变、饱和检测。
"""

from __future__ import annotations

import numpy as np
import pytest

from eureka.smc.kl_controller import (
    boltzmann_gibbs,
    effective_parents,
    feasible_target,
    kl_to_uniform,
    resolve,
    solve_lambda,
    target_effective_parents,
)


# ---- 1. 概率合法性 ----
def test_q_is_valid_distribution():
    scores = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    for lam in (0.0, 1.0, 5.0, 50.0):
        q = boltzmann_gibbs(scores, lam)
        assert np.all(q >= 0.0)
        assert q.sum() == pytest.approx(1.0)


# ---- 2. KL 二分求解误差 ----
def test_kl_bisection_precision():
    scores = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    n = scores.size
    for delta in (0.1, 0.5, 1.0):
        lam, kl_actual, sat = solve_lambda(scores, delta, tol=1e-8)
        if not sat:
            assert abs(kl_actual - delta) < 1e-4


# ---- 3. 可达时 K_eff ≈ K* ----
def test_k_eff_matches_target_when_reachable():
    scores = np.linspace(0.0, 1.0, 8)          # 无并列、分布平滑 → 目标可达
    n = 8
    for k_star in (6.0, 4.0, 3.0):
        delta = float(np.log(n / k_star))
        lam, _, sat = solve_lambda(scores, delta, tol=1e-9)
        assert not sat
        q = boltzmann_gibbs(scores, lam)
        assert effective_parents(q) == pytest.approx(k_star, abs=1e-2)


# ---- 4. δ 增大 → 分布更集中（K_eff 更小），非更均匀 ----
def test_larger_delta_more_concentrated():
    scores = np.linspace(0.0, 1.0, 8)
    n = 8
    prev_keff = float(n)
    for k_star in (7.0, 5.0, 3.0, 2.0):
        delta = float(np.log(n / k_star))
        lam, _, _ = solve_lambda(scores, delta)
        keff = effective_parents(boltzmann_gibbs(scores, lam))
        assert keff <= prev_keff + 1e-6      # 单调不增
        prev_keff = keff


# ---- 5. 全体同分 → q=U_N, λ=0 ----
def test_all_equal_scores_falls_back_uniform():
    scores = np.full(8, 3.14)
    lam, kl, sat = solve_lambda(scores, delta=1.0)
    assert lam == 0.0 and kl == 0.0 and not sat
    q = boltzmann_gibbs(scores, lam)
    assert np.allclose(q, 1.0 / 8)
    step = resolve(scores, b=10, B=64, n=8, k_min=2, gamma=1.0)
    assert step.lam == 0.0
    assert np.allclose(step.q, 1.0 / 8)


# ---- 6. 并列最高分：不打破并列，限制可达 KL ----
def test_tied_max_limits_reachable_kl():
    scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])  # 4 并列最高
    n = 8
    k_feas, delta_feas, m = feasible_target(scores, k_star=2.0, n=n)
    assert m == 4
    assert k_feas == pytest.approx(4.0)      # max(K*=2, m=4)
    assert delta_feas == pytest.approx(np.log(n / 4.0))
    # 请求 δ 对应 K*=2（比可达更集中）→ 极大 λ 下质量均分给 4 个并列者，K_eff→4 不到 2
    lam, _, _ = solve_lambda(scores, delta_feas)
    q = boltzmann_gibbs(scores, lam)
    top = np.argsort(scores)[-4:]
    assert q[top].sum() == pytest.approx(1.0, abs=1e-3)  # 质量全在并列最高者
    assert effective_parents(q) == pytest.approx(4.0, abs=1e-2)


# ---- 7. 稀疏零分 [0,...,0,0.1]：无 NaN/Inf/overflow ----
def test_sparse_nonzero_numerically_stable():
    scores = np.array([0.0] * 7 + [0.1])
    for lam in (0.0, 10.0, 1e3, 1e6):
        q = boltzmann_gibbs(scores, lam)
        assert np.all(np.isfinite(q))
        assert q.sum() == pytest.approx(1.0)
    step = resolve(scores, b=1, B=64, n=8, k_min=2, gamma=1.0)
    assert np.all(np.isfinite(step.q))
    assert np.isfinite(step.lam) and np.isfinite(step.kl_actual)


# ---- 8. 正仿射不变：J 与 aJ+b 同 KL 目标下 q 与 λ·Δ 一致 ----
def test_positive_affine_invariance():
    scores = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 2.0])
    n = 8
    a, b = 137.0, -42.0
    scaled = a * scores + b
    delta = float(np.log(n / 3.0))
    lam1, _, _ = solve_lambda(scores, delta, tol=1e-9)
    lam2, _, _ = solve_lambda(scaled, delta, tol=1e-9)
    q1 = boltzmann_gibbs(scores, lam1)
    q2 = boltzmann_gibbs(scaled, lam2)
    assert np.allclose(q1, q2, atol=1e-4)                 # 分布一致
    assert lam2 == pytest.approx(lam1 / a, rel=1e-2)      # λ 随 1/a 缩放
    # λ·Δ 不变：任取一对粒子
    d_raw = scores[3] - scores[1]
    d_scaled = scaled[3] - scaled[1]
    assert lam1 * d_raw == pytest.approx(lam2 * d_scaled, rel=1e-2)


# ---- 8b. 极端尺度 [100,...,100,20000] 稳定 ----
def test_extreme_scale_stable():
    scores = np.array([100.0] * 7 + [20000.0])
    step = resolve(scores, b=5, B=64, n=8, k_min=2, gamma=1.0)
    assert np.all(np.isfinite(step.q))
    assert step.q.sum() == pytest.approx(1.0)
    assert np.isfinite(step.lam)


# ---- 目标有效父代数日程 ----
def test_target_effective_parents_schedule():
    n, k_min = 8, 2
    # 预算满 → K*≈N
    assert target_effective_parents(64, 64, n, k_min, 1.0) == pytest.approx(8.0)
    # 预算尽 → K*→K_min
    assert target_effective_parents(0, 64, n, k_min, 1.0) == pytest.approx(2.0)
    # 单调不减 于 b
    prev = -1.0
    for b in range(0, 65, 8):
        k = target_effective_parents(b, 64, n, k_min, 1.0)
        assert k >= prev - 1e-9
        prev = k
    # γ 改变曲率但端点不变
    assert target_effective_parents(32, 64, n, k_min, 2.0) < target_effective_parents(32, 64, n, k_min, 0.5)


# ---- KL/熵 恒等式 ----
def test_kl_entropy_identity():
    q = np.array([0.4, 0.3, 0.2, 0.1])
    n = q.size
    assert kl_to_uniform(q) == pytest.approx(np.log(n) - (-(q * np.log(q)).sum()))
    u = np.full(n, 1.0 / n)
    assert kl_to_uniform(u) == pytest.approx(0.0)


# ---- resolve 首轮均匀、末轮集中 ----
def test_resolve_budget_endpoints():
    scores = np.linspace(0.0, 1.0, 8)
    full = resolve(scores, b=64, B=64, n=8, k_min=2, gamma=1.0)   # 预算满
    assert full.lam == pytest.approx(0.0, abs=1e-6)               # K*=N → δ=0 → λ=0
    assert full.max_q == pytest.approx(1.0 / 8, abs=1e-6)
    tail = resolve(scores, b=1, B=64, n=8, k_min=2, gamma=1.0)    # 预算尽
    assert tail.k_star == pytest.approx(2.0, abs=0.2)
    assert tail.max_q > full.max_q                                # 更集中
