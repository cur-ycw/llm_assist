"""搜索分数的提取与固定尺度归一化（迁移计划 §5）。

这是从通用 SMCEvolve 迁移到 Eureka 最关键的适配层：把每任务尺度各异的原始 RL 指标
（如 ``consecutive_successes``）映射到一个跨阶段一致、可比较的 ``[0,1]`` 能量 ``R(x)``，
使 ``exp(beta * ΔR)`` 的含义稳定。

设计要点：
  * **固定尺度**：用任务级冻结的 ``lower/upper`` 做 clipped-linear，不做每代 min-max
    重归一化（否则同一代码的目标能量会随种群变化，破坏跨阶段一致性，计划 §5.1）。
  * **聚合量可配**：默认 ``final_window_mean``（曲线末段均值），而非原 Eureka 的
    ``max``——max-over-curve 是乐观、高方差的选择统计量，会放大 RL 噪声（审查 #4）。
  * 指标缺失时返回 ``None``（无效候选不进入温度二分，计划 §4.2）。

纯 numpy，无 Isaac Gym 依赖，可单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = ["ScoreConfig", "aggregate_metric", "clipped_linear", "compute_search_score"]


@dataclass(frozen=True)
class ScoreConfig:
    """search_score 的提取与归一化配置（每任务在 pilot 后冻结 lower/upper）。"""

    metric: str = "consecutive_successes"   # 主指标；缺失时回退到 fallback_metric
    fallback_metric: str = "gt_reward"      # 无 success 概念的任务（如 Cartpole 早期）
    aggregate: str = "final_window_mean"    # final_window_mean | mean | max
    window_frac: float = 0.1                # final_window_mean 取末段比例
    lower: float = 0.0
    upper: float = 500.0                    # Cartpole 占位：max_episode_length；pilot 后冻结


def aggregate_metric(values: Sequence[float], how: str, window_frac: float = 0.1) -> float:
    """把一条指标时间序列聚合成单个标量。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty metric series")
    if how == "max":
        return float(arr.max())
    if how == "mean":
        return float(arr.mean())
    if how == "final_window_mean":
        k = max(1, int(round(arr.size * window_frac)))
        return float(arr[-k:].mean())
    raise ValueError(f"unknown aggregate: {how}")


def clipped_linear(raw: float, lower: float, upper: float) -> float:
    """固定尺度 clipped-linear 归一化到 ``[0,1]``（计划 §5.1）。"""
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    return float(np.clip((raw - lower) / (upper - lower), 0.0, 1.0))


def compute_search_score(
    tensorboard_logs: Mapping[str, Sequence[float]], cfg: ScoreConfig
) -> Optional[tuple[float, float]]:
    """返回 ``(raw, normalized)``；主指标与回退指标都缺失时返回 ``None``。

    ``normalized`` 是喂给 SMC 权重/接受的唯一标量 ``R(x)``。
    """
    metric = cfg.metric
    if metric not in tensorboard_logs or len(tensorboard_logs[metric]) == 0:
        metric = cfg.fallback_metric
    if metric not in tensorboard_logs or len(tensorboard_logs[metric]) == 0:
        return None
    raw = aggregate_metric(tensorboard_logs[metric], cfg.aggregate, cfg.window_frac)
    return raw, clipped_linear(raw, cfg.lower, cfg.upper)
