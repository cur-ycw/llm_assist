"""搜索分数的提取（新方法 §0/§2.2：直接使用原始任务性能 J，**不做归一化**）。

新方法冻结「奖励标准化」为延期项：父代选择、sigmoid 接受、best/archive 排序、held-out 复评
一律使用环境求解器返回的**原始** J（``compute_search_score`` 返回原始聚合标量）。KL 父代分布
``q=softmax(λ·J)`` 关于正仿射变换不变，尺度由 λ 吸收，故无需跨任务定标（详见 kl_controller）。

设计要点：
  * **原始 J**：``compute_search_score`` 对主指标（或回退指标）时间序列做窗口聚合后**直接返回**，
    不再套 clipped-linear。``clipped_linear`` 函数保留（未来 reporting/消融可能用），但不喂控制器。
  * 聚合量可配：默认 ``final_window_mean``（曲线末段均值），避免 max-over-curve 的乐观高方差。
  * 指标缺失时返回 ``None``（无效候选不进入控制器）。

纯 numpy，无 Isaac Gym 依赖，可单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = ["ScoreConfig", "aggregate_metric", "clipped_linear", "compute_search_score"]


@dataclass(frozen=True)
class ScoreConfig:
    """search_score 的提取配置。

    ``lower/upper`` 在新方法中**已退出热路径**（不再归一化），仅为配置向后兼容保留；如未来
    重启归一化研究再启用（须建新方法版本，§0）。
    """

    metric: str = "consecutive_successes"   # 主指标；缺失时回退到 fallback_metric
    fallback_metric: str = "gt_reward"      # 无 success 概念的任务（如 Cartpole 早期）
    aggregate: str = "final_window_mean"    # final_window_mean | mean | max
    window_frac: float = 0.1                # final_window_mean 取末段比例
    lower: float = 0.0                      # 已退出热路径（保留兼容）
    upper: float = 500.0                    # 已退出热路径（保留兼容）


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
    """固定尺度 clipped-linear 归一化到 ``[0,1]``（保留供未来 reporting/消融；不在控制器热路径）。"""
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    return float(np.clip((raw - lower) / (upper - lower), 0.0, 1.0))


def compute_search_score(
    tensorboard_logs: Mapping[str, Sequence[float]], cfg: ScoreConfig
) -> Optional[float]:
    """返回**原始** J（聚合标量）；主指标与回退指标都缺失时返回 ``None``。

    这是喂给 KL 控制器权重 / sigmoid 接受 / best 排序的唯一标量（新方法 §2.2，不归一化）。
    """
    metric = cfg.metric
    if metric not in tensorboard_logs or len(tensorboard_logs[metric]) == 0:
        metric = cfg.fallback_metric
    if metric not in tensorboard_logs or len(tensorboard_logs[metric]) == 0:
        return None
    return aggregate_metric(tensorboard_logs[metric], cfg.aggregate, cfg.window_frac)
