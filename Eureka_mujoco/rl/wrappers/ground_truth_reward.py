"""Per-task ground-truth (sparse / paper metric) reward functions.

These are the EUREKA "consecutive_successes" / "gt_reward" signals — the metric
the evolutionary loop uses to pick the best LLM reward. They are NOT what the
RL policy optimizes; the policy optimizes the LLM-generated `compute_reward`.

Convention: gt_reward_fn(env, obs, action, info, env_reward) -> float
- env is the unwrapped gym env
- info is the dict the env returned from step (pre-wrapper modification)
- env_reward is the env's native reward (useful for locomotion: forward velocity)
"""
import numpy as np


def ant_gt_reward(env, obs, action, info, env_reward):
    # Paper metric for Ant locomotion: forward velocity (x direction).
    # AntEnv-v4 puts x_velocity in info.
    return float(info.get("x_velocity", 0.0))


def walker2d_gt_reward(env, obs, action, info, env_reward):
    return float(info.get("x_velocity", 0.0))


def reacher_gt_reward(env, obs, action, info, env_reward):
    # Reacher metric: negative distance from fingertip to target.
    try:
        vec = env.get_body_com("fingertip") - env.get_body_com("target")
        return float(-np.linalg.norm(vec))
    except Exception:
        return float(info.get("reward_dist", 0.0))


def metaworld_gt_reward(env, obs, action, info, env_reward):
    # MetaWorld tasks return info['success'] (0/1) — paper metric.
    return float(info.get("success", 0.0))


GT_REWARD_REGISTRY = {
    "AntGPT": ant_gt_reward,
    "WalkerGPT": walker2d_gt_reward,
    "ReacherGPT": metaworld_gt_reward,
    "DoorGPT": metaworld_gt_reward,
    "PickGPT": metaworld_gt_reward,
}

GROUND_TRUTH_REWARD_FNS = GT_REWARD_REGISTRY
