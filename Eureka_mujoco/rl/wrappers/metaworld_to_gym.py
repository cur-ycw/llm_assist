"""Helper to instantiate Farama MetaWorld V3 tasks as gymnasium envs.

`gym.make("Meta-World/MT1", env_name="door-open-v3", seed=...)` returns
a gym.Env-compatible env. This module just centralizes the call and
fixes the per-episode task sampling (MT1 returns a task pool — we sample
one new task per reset for variety, matching MetaWorld single-task RL).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import metaworld  # noqa: F401  — registers the `Meta-World/MT1` namespace


def _relax_target_bounds_if_needed(env: gym.Env) -> gym.Env:
    chain = []
    cur = env
    while True:
        chain.append(cur)
        if not hasattr(cur, "env"):
            break
        cur = cur.env

    template_space = None
    for wrapped_env in chain:
        space = getattr(wrapped_env, "observation_space", None)
        if not isinstance(space, gym.spaces.Box):
            continue
        if space.shape != (39,):
            continue
        low = np.array(space.low, copy=True)
        high = np.array(space.high, copy=True)
        if np.all(low[-3:] == 0.0) and np.all(high[-3:] == 0.0):
            low[-3:] = -np.inf
            high[-3:] = np.inf
            template_space = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
            break

    if template_space is None:
        return env

    for wrapped_env in chain:
        space = getattr(wrapped_env, "observation_space", None)
        if isinstance(space, gym.spaces.Box) and space.shape == (39,):
            wrapped_env.observation_space = template_space
    return env


def make_metaworld_env(env_name: str, seed: int = 0):
    """Returns a gymnasium-compatible MetaWorld env."""
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
    return _relax_target_bounds_if_needed(env)
