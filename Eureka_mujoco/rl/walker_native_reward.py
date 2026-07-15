"""Walker2d-v4 native reward module for PPO baseline.

Matches the reward used by `eureka/envs/mujoco/walker2d_v4.py`:
    reward = x_velocity + healthy_reward - control_cost

The first argument is the raw MuJoCo Walker2dEnv (post-`unwrapped`), matching
the raw-env contract used by RewardOverrideWrapper.
"""


def compute_reward(env, obs, action):
    x_velocity = float(env.data.qvel[0])
    healthy_reward = float(env.healthy_reward)
    control_cost = float(env.control_cost(action))
    reward = x_velocity + healthy_reward - control_cost
    return reward, {
        "forward_reward": x_velocity,
        "healthy_reward": healthy_reward,
        "control_cost": -control_cost,
    }
