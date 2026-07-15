"""Ant-v4 native reward module for PPO baseline.

Matches the reward used by `eureka/envs/mujoco/ant_v4.py`:
    reward = x_velocity + healthy_reward - control_cost
"""


def compute_reward(env, obs, action):
    x_velocity = float(env.x_velocity)
    healthy_reward = float(env.healthy_reward)
    control_cost = float(env.control_cost)
    reward = x_velocity + healthy_reward - control_cost
    return reward, {
        "forward_reward": x_velocity,
        "healthy_reward": healthy_reward,
        "control_cost": -control_cost,
    }
