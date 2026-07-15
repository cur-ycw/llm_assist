# Ant-v4 observation and reward-view context.
#
# The reward function is called as:
#   compute_reward(env, obs, action)
# where `env` is an AntRewardView object, not the raw Gym env.
#
# `obs` is the environment observation used by the policy.
# It is a 1-D NumPy array formed by concatenating:
# - generalized positions (qpos), excluding the absolute torso xy positions
# - generalized velocities (qvel)
# By default `obs.shape == (27,)` when contact forces are not included.
#
# Treat `obs` as two semantic blocks:
# - position-related features: torso height (z), torso orientation quaternion,
#   and joint angles for the 8 leg joints
# - velocity-related features: torso linear/angular velocity and joint angular velocities
#
# Important: do not guess reward logic from hard-coded obs indices unless the
# index meaning is obvious from the environment. Prefer the named fields on the
# reward view when possible.
#
# Named fields available on the AntRewardView (`env` argument):
# - env.x_velocity: forward velocity along the x axis
# - env.y_velocity: lateral velocity along the y axis
# - env.forward_reward: a forward-progress-related signal exposed by the environment
# - env.healthy_reward: a health / stability-related scalar exposed by the environment
# - env.control_cost: an action-magnitude-related scalar computed from the current action
# - env.control_cost_weight: coefficient used when computing the action-related scalar
# - env.contact_cost: a contact-force-related scalar (0 when contact forces are disabled)
# - env.contact_cost_weight: coefficient used when computing the contact-related scalar
# - env.env_reward: scalar environment reward from the current transition
# - env.terminated: whether the current transition ended the episode
# - env.is_healthy: whether the ant satisfies the healthy-state condition
# - env.x_position: absolute x position of the torso
# - env.y_position: absolute y position of the torso
# - env.z_position: torso height (z)
# - env.distance_from_origin: L2 norm of the torso xy position
# - env.dt: simulation timestep
# - env.healthy_z_min / env.healthy_z_max: torso-height healthy range
# - env.qpos: full generalized positions from the simulator state
#            (qpos[0:2] = torso xy, qpos[2] = torso z, qpos[3:7] = torso quaternion,
#             qpos[7:] = 8 leg joint angles)
# - env.qvel: full generalized velocities from the simulator state
#            (qvel[0:3] = torso linear velocity, qvel[3:6] = torso angular velocity,
#             qvel[6:] = 8 leg joint velocities)
#
# These named signals are provided as semantically meaningful building blocks.
# You may use them directly, combine them with observation-derived terms, or ignore
# them if a different design is more effective.
