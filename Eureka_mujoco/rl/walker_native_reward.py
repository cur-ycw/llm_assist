"""Walker2d-v4 native reward module for PPO baseline.

Matches the reward used by `eureka/envs/mujoco/walker2d_v4.py`:
    reward = x_velocity + healthy_reward - control_cost
"""


def compute_reward(reward_view, obs, action):
    x_velocity = float(reward_view.x_velocity)
    healthy_reward = float(reward_view.healthy_reward)
    control_cost = float(reward_view.control_cost)
    reward = x_velocity + healthy_reward - control_cost
    return reward, {
        "forward_reward": x_velocity,
        "healthy_reward": healthy_reward,
        "control_cost": -control_cost,
    }
