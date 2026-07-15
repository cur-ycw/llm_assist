# Walker2d-v4 observation and reward-view context.
#
# The reward function is called as:
#   compute_reward(env, obs, action)
# where `env` is a WalkerRewardView object, not the raw Gym env.
#
# `obs` is the environment observation used by the policy.
# It is a 1-D NumPy array formed by concatenating:
# - generalized positions (qpos), excluding the absolute root x position
# - generalized velocities (qvel), clipped to a bounded range by the environment
#
# Treat `obs` as two semantic blocks:
# - position-related features: torso height / torso pitch / joint angles
# - velocity-related features: root forward velocity and joint angular velocities
#
# Important: do not guess reward logic from hard-coded obs indices unless the
# index meaning is obvious from the environment. Prefer the named fields on the
# reward view when possible.
#
# Named fields available on the WalkerRewardView (`env` argument):
# - env.x_velocity: forward velocity along the x axis
# - env.forward_reward: a forward-progress-related signal exposed by the environment
# - env.healthy_reward: a health / stability-related scalar exposed by the environment
# - env.control_cost: an action-magnitude-related scalar computed from the current action
# - env.control_cost_weight: coefficient used when computing the action-related scalar
# - env.env_reward: scalar environment reward from the current transition
# - env.terminated: whether the current transition ended the episode
# - env.is_healthy: whether the walker satisfies the healthy-state condition
# - env.x_position: absolute x position of the torso/root
# - env.z_position: torso height
# - env.torso_angle: torso pitch angle
# - env.dt: simulation timestep
# - env.healthy_z_min / env.healthy_z_max: torso-height healthy range
# - env.healthy_angle_min / env.healthy_angle_max: torso-angle healthy range
# - env.qpos: full generalized positions from the simulator state
# - env.qvel: full generalized velocities from the simulator state
#
# These named signals are provided as semantically meaningful building blocks.
# You may use them directly, combine them with observation-derived terms, or ignore
# them if a different design is more effective.
