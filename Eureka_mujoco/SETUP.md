# Environment setup — Eureka MuJoCo/MetaWorld

This document lists the runtime environment used to develop and validate the
raw-env branch of this project. The five tasks (Ant, Walker, Reach, Pick, Door)
are exercised via `stable-baselines3` PPO on top of `gymnasium` MuJoCo and
Farama MetaWorld V3.

## 1. Conda environment

Python 3.10 is required. `mujoco` 3.x pulls in native GL libraries and needs a
Python that stays compatible with the wheels shipped for gymnasium 1.x and
metaworld 3.x.

```bash
conda create -n eureka_mujoco python=3.10 -y
conda activate eureka_mujoco
```

## 2. Python packages

Reference versions taken from the working environment on this host. Use these
as an "known good" pin; newer patch releases usually work but are not tested.

```bash
pip install \
    "gymnasium==1.2.3" \
    "mujoco==3.3.0" \
    "metaworld==3.0.0" \
    "Farama-Notifications==0.0.6" \
    "stable-baselines3==2.8.0" \
    "torch==2.6.0" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "matplotlib==3.10.9" \
    "hydra-core==1.3.3" \
    "omegaconf==2.3.1" \
    "openai==0.28.1" \
    "tensorboard==2.20.0" \
    "wandb==0.28.0" \
    "protobuf==7.35.1"
```

Notes:
- `openai==0.28.x` is required because `eureka/eureka_mujoco.py` uses the
  legacy `openai.ChatCompletion.create` API. Upgrading to the 1.x SDK requires
  code changes.
- `torch` is installed for the SB3 PPO backend. A GPU build such as
  `torch==2.6.0+cu124` from the PyTorch index works and matches the version
  used during development. CPU-only builds also work but are slow for 5M PPO
  timesteps per sample.
- `metaworld` 3.0 registers a `Meta-World/MT1` env id used by
  `rl/wrappers/metaworld_to_gym.py`.

## 3. Environment variables

Create a local override file (not checked in) that exports your credentials
BEFORE sourcing `env.sh`:

```bash
export OPENAI_API_KEY="sk-..."                      # required
export OPENAI_API_BASE="https://api.openai.com/v1"  # optional; env.sh defaults
                                                    # to chatanywhere
export USE_WANDB=1                                  # optional
export WANDB_ENTITY="your-entity"                   # optional
export WANDB_PROJECT="eureka-mujoco"                # optional
```

Then, from the project root:

```bash
source env.sh
```

`env.sh` will warn if `OPENAI_API_KEY` is unset. It does NOT ship any keys.

## 4. MuJoCo native runtime

`mujoco==3.3.0` bundles its own binaries; no separate MuJoCo installation is
required. On headless hosts, set an off-screen GL backend before running the
smoke tests so the passive env checker does not try to open a window:

```bash
export MUJOCO_GL=egl
```

## 5. Optional: IsaacGym (only if you also run the Isaac Gym version)

This repo does not require IsaacGym for the five MuJoCo/MetaWorld tasks. The
IsaacGym path is only used by the original Eureka codebase; `env.sh` exports
`ISAACGYM_PATH` and adjusts `LD_LIBRARY_PATH` only when that directory exists,
so the block is a no-op if IsaacGym is not installed.

If you do use IsaacGym:
- Isaac Gym Preview 4 (Python bindings are built for Python 3.8).
- Because MuJoCo and IsaacGym target different Python versions, do NOT install
  both into the `eureka_mujoco` env — keep them in separate conda envs.

## 6. Verify the install

Once dependencies are in place, run one smoke check per task to verify the
raw-env wrapper path is wired correctly. Place a smoke file such as the
following in your local `smoke/` directory (the folder is `.gitignore`'d):

```python
# smoke/ant_smoke_reward.py
import numpy as np
def compute_reward(env, obs, action):
    x_velocity = float(env.data.qvel[0])
    healthy_reward = float(env.healthy_reward)
    ctrl_cost = float(env.control_cost(action))
    reward = x_velocity + 0.5 * healthy_reward - ctrl_cost
    return float(reward), {
        "forward_reward": float(x_velocity),
        "healthy_reward": float(healthy_reward),
        "ctrl_cost": float(ctrl_cost),
    }
```

Then:

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('MUJOCO_GL', 'egl')
from smoke.ant_smoke_reward import compute_reward
from rl.wrappers.reward_override_wrapper import RewardOverrideWrapper
from rl.wrappers.ground_truth_reward import GT_REWARD_REGISTRY
import gymnasium as gym
env = gym.make('Ant-v4')
env = RewardOverrideWrapper(env, compute_reward, GT_REWARD_REGISTRY['AntGPT'], env_family='ant')
obs, info = env.reset(seed=0)
for _ in range(50):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc: env.reset()
print('ok')"
```

For MetaWorld tasks (`ReacherGPT`, `PickGPT`, `DoorGPT`), swap `gym.make(...)`
for `make_metaworld_env('reach-v3', seed=0)` from
`rl.wrappers.metaworld_to_gym` and use `env_family='reach' / 'pick' / 'door'`
accordingly.

## 7. Run a real Eureka experiment

Per-task launchers live under `scripts/`. Each pins the PPO budget expected
for that task family:

| Task   | Launcher                | Per-sample PPO steps |
|--------|-------------------------|----------------------|
| Ant    | `scripts/run_ant.sh`    | 5,000,000 (MuJoCo)   |
| Walker | `scripts/run_walker.sh` | 5,000,000 (MuJoCo)   |
| Reach  | `scripts/run_reach.sh`  | 10,000,000 (MetaWorld) |
| Pick   | `scripts/run_pick.sh`   | 10,000,000 (MetaWorld) |
| Door   | `scripts/run_door.sh`   | 10,000,000 (MetaWorld) |

```bash
# Single task with defaults (sample=16, iteration=5)
./scripts/run_walker.sh

# Override sample count and iteration count if you want a shorter probe
./scripts/run_walker.sh 4 2

# Full arbitrary override still works via the underlying driver
./run_eureka_mujoco.sh walker 16 5 5000000

# All five tasks in sequence using the per-task budgets above
./run_all_tasks.sh
```

The Hydra workspace lands under `eureka/outputs/eureka/<env>_<timestamp>/`
(git-ignored). Each PPO worker also writes tensorboard scalars under that
workspace's `tb_iter*_response*/` directories.

## 8. Hardware notes used during development

- Linux x86_64, kernel 5.15
- 1 NVIDIA GPU visible to PyTorch (SB3 PPO uses whichever GPU is selected by
  `set_freest_gpu()` in `eureka/utils/misc.py`)
- ~5 minutes wall time per PPO worker at `max_iterations=5_000_000`,
  `n_envs=8` on an RTX 3090. A full Eureka round with `sample=16` and
  `iteration=5` therefore takes on the order of 6-8 hours per task if
  workers run serially. Parallelism is bounded by GPU memory.
