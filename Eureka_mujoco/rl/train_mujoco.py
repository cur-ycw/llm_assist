"""SB3 PPO training script — drop-in replacement for IsaacGym's train.py."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_RL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_RL_ROOT.parent))
from rl.wrappers.reward_override_wrapper import RewardOverrideWrapper
from rl.wrappers.ground_truth_reward import GT_REWARD_REGISTRY
from rl.wrappers.metaworld_to_gym import make_metaworld_env
from task_interfaces import get_task_interface


TASK_TO_ENV_ID = {
    "AntGPT": ("Ant-v4", False),
    "WalkerGPT": ("Walker2d-v4", False),
    "ReacherGPT": ("reach-v3", True),
    "DoorGPT": ("door-open-v3", True),
    "PickGPT": ("pick-place-v3", True),
}


def load_gpt_reward(module_path: str):
    spec = importlib.util.spec_from_file_location("_gpt_reward_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "compute_reward"):
        raise RuntimeError(f"GPT reward module {module_path} has no top-level `compute_reward` function")
    return module.compute_reward


def make_env_factory(task: str, env_id: str, is_metaworld: bool, gpt_reward_fn, gt_reward_fn, seed: int):
    env_family = get_task_interface(task).interface_type

    def _thunk():
        if is_metaworld:
            env = make_metaworld_env(env_id, seed=seed)
        else:
            env = gym.make(env_id)
        env = RewardOverrideWrapper(env, gpt_reward_fn, gt_reward_fn, env_family=env_family)
        env = Monitor(env, info_keywords=())
        env.reset(seed=seed)
        return env

    return _thunk


class PerComponentRewardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._component_keys: set[str] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        env_vals, env_ep_returns, gpt_vals, gt_vals, task_vals, cs_vals, fitness_vals = [], [], [], [], [], [], []
        comp_vals: dict[str, list[float]] = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "env_reward" in info:
                env_vals.append(info["env_reward"])
            if "env_episode_return" in info:
                env_ep_returns.append(info["env_episode_return"])
            if "gpt_reward" in info:
                gpt_vals.append(info["gpt_reward"])
            if "gt_reward" in info:
                gt_vals.append(info["gt_reward"])
            if "task_score" in info:
                task_vals.append(info["task_score"])
            if "consecutive_successes" in info:
                cs_vals.append(info["consecutive_successes"])
            if "fitness" in info:
                fitness_vals.append(info["fitness"])
            for k, v in info.items():
                if k.startswith("reward_") and isinstance(v, (int, float, np.floating)):
                    comp_vals.setdefault(k, []).append(float(v))
                    self._component_keys.add(k)
        if env_vals:
            self.logger.record("env_reward", float(np.mean(env_vals)))
        if env_ep_returns:
            self.logger.record("env_episode_return", float(np.mean(env_ep_returns)))
        if gpt_vals:
            self.logger.record("gpt_reward", float(np.mean(gpt_vals)))
        if gt_vals:
            self.logger.record("gt_reward", float(np.mean(gt_vals)))
        if task_vals:
            self.logger.record("task_score", float(np.mean(task_vals)))
        if cs_vals:
            self.logger.record("consecutive_successes", float(np.mean(cs_vals)))
        if fitness_vals:
            self.logger.record("fitness", float(np.mean(fitness_vals)))
        for k, vs in comp_vals.items():
            self.logger.record(k, float(np.mean(vs)))
        return True


def maybe_init_wandb(args, env_id: str):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "WandB logging was requested, but the `wandb` package could not be imported. Install it in the `eureka_mujoco` environment first."
        ) from exc

    project = args.wandb_project or os.getenv("WANDB_PROJECT")
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    if not project:
        raise RuntimeError("WandB logging was requested, but no project was provided.")

    tags = [tag for tag in args.wandb_tags.split(",") if tag]
    run = wandb.init(
        project=project,
        entity=entity or None,
        name=args.wandb_run_name or Path(args.logdir).name,
        group=args.wandb_group or None,
        job_type=args.wandb_job_type or "ppo",
        dir=args.logdir,
        sync_tensorboard=True,
        config={
            "task": args.task,
            "env_id": env_id,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "reward_module": args.gpt_reward_module,
        },
        tags=tags,
        reinit=True,
    )
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_TO_ENV_ID.keys()))
    parser.add_argument("--max_iterations", type=int, default=1_000_000, help="Total PPO timesteps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpt_reward_module", required=True, help="Path to a .py file with a top-level `compute_reward(env, obs, action)`.")
    parser.add_argument("--logdir", required=True, help="Tensorboard log directory.")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true", default=env_flag("USE_WANDB"))
    parser.add_argument("--wandb_entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_project", default=os.getenv("WANDB_PROJECT", ""))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_job_type", default="ppo")
    parser.add_argument("--wandb_tags", default="")
    args = parser.parse_args()

    env_id, is_metaworld = TASK_TO_ENV_ID[args.task]
    gpt_reward_fn = load_gpt_reward(args.gpt_reward_module)
    gt_reward_fn = GT_REWARD_REGISTRY[args.task]

    os.makedirs(args.logdir, exist_ok=True)
    tb_logdir_abs = str((Path(args.logdir) / "run_1").resolve())
    print(f"Tensorboard Directory: {tb_logdir_abs}", flush=True)

    factories = [
        make_env_factory(args.task, env_id, is_metaworld, gpt_reward_fn, gt_reward_fn, seed=args.seed + i)
        for i in range(args.n_envs)
    ]
    vec_cls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    vec_env = vec_cls(factories)

    wandb_run = maybe_init_wandb(args, env_id)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.logdir,
    )

    callback = PerComponentRewardCallback()
    t0 = time.time()
    try:
        print("fps step: 0", flush=True)
        model.learn(total_timesteps=args.max_iterations, callback=callback, tb_log_name="run")
        print(f"[train_mujoco] done. wall={time.time() - t0:.1f}s", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
