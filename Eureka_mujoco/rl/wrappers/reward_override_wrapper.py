"""Wrap a gym env and replace its native reward with a user-supplied
`compute_reward(env, obs, action) -> (reward, component_dict)`.

Records ground-truth reward, GPT reward, task_score, and per-component values to
`info` so SB3 Monitor + a callback can log them to tensorboard.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl.wrappers.reward_view import (
    build_ant_reward_view,
    build_geometric_control_reward_view,
    build_manipulation_reward_view,
    build_reward_view,
    build_walker_reward_view,
)


class RewardOverrideWrapper(gym.Wrapper):
    def __init__(self, env, gpt_reward_fn, gt_reward_fn, env_family: str = "locomotion"):
        super().__init__(env)
        if env_family not in {"locomotion", "walker", "ant", "manipulation", "reach", "pick", "door", "geometric_control"}:
            raise ValueError(f"unknown env_family={env_family!r}")
        self._env_family = env_family
        self._gpt_reward_fn = gpt_reward_fn
        self._gt_reward_fn = gt_reward_fn
        self._last_obs = None
        self._last_x = None
        self._native_env_return = 0.0
        self._episode_forward_displacement = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        try:
            self._last_x = float(self.env.unwrapped.data.qpos[0])
        except Exception:
            self._last_x = None
        self._native_env_return = 0.0
        self._episode_forward_displacement = 0.0
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._native_env_return += float(env_reward)

        if self._env_family in {"locomotion", "walker", "ant"}:
            prev_x = self._last_x
        else:
            prev_x = None

        unwrapped = self.env.unwrapped
        if self._env_family == "manipulation":
            reward_view = build_manipulation_reward_view(unwrapped, obs, action, info, env_reward, terminated)
        elif self._env_family == "reach":
            # Pass the raw MetaWorld env to the LLM reward, mirroring the original
            # Eureka behaviour where the reward function sees the full env class.
            reward_view = unwrapped
        elif self._env_family == "pick":
            # Pass the raw MetaWorld env to the LLM reward, mirroring the original
            # Eureka behaviour where the reward function sees the full env class.
            reward_view = unwrapped
        elif self._env_family == "door":
            # Pass the raw MetaWorld env to the LLM reward, mirroring the original
            # Eureka behaviour where the reward function sees the full env class.
            reward_view = unwrapped
        elif self._env_family == "geometric_control":
            reward_view = build_geometric_control_reward_view(unwrapped, obs, action, info, env_reward, terminated)
        elif self._env_family == "walker":
            # Pass the raw MuJoCo env to the LLM reward, mirroring the original
            # Eureka behaviour where the reward function sees the full env class.
            reward_view = unwrapped
        elif self._env_family == "ant":
            # Pass the raw MuJoCo env to the LLM reward, mirroring the original
            # Eureka behaviour where the reward function sees the full Task class.
            reward_view = unwrapped
        else:
            reward_view = build_reward_view(unwrapped, obs, action, info, env_reward, terminated)

        if prev_x is None and self._env_family in {"locomotion", "walker", "ant"}:
            x_velocity = float(getattr(unwrapped, "x_velocity", info.get("x_velocity", 0.0)))
            dt = float(getattr(unwrapped, "dt", 1.0))
            self._episode_forward_displacement += x_velocity * dt
        elif prev_x is not None:
            cur_x = float(unwrapped.data.qpos[0])
            self._episode_forward_displacement += cur_x - prev_x
            self._last_x = cur_x

        try:
            gpt_reward, reward_dict = self._gpt_reward_fn(reward_view, obs, action)
        except Exception as e:
            raise RuntimeError(f"GPT reward function crashed: {e!r}") from e

        gt_reward = self._gt_reward_fn(unwrapped, obs, action, info, env_reward)

        info = dict(info) if info is not None else {}
        info["env_reward"] = float(env_reward)
        if terminated or truncated:
            info["env_episode_return"] = float(self._native_env_return)
            info["fitness"] = float(self._episode_forward_displacement)
        info["gpt_reward"] = float(gpt_reward)
        info["gt_reward"] = float(gt_reward)
        info["task_score"] = float(gt_reward)
        if self._env_family in {"manipulation", "reach", "pick", "door"}:
            info["consecutive_successes"] = float(gt_reward)

        if isinstance(reward_dict, dict):
            for k, v in reward_dict.items():
                try:
                    info[f"reward_{k}"] = float(np.asarray(v).item())
                except Exception:
                    pass

        self._last_obs = obs
        return obs, float(gpt_reward), terminated, truncated, info
