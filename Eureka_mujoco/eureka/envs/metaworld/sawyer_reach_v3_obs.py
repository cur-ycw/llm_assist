# MetaWorld reach-v3 observation and reward-view context.
#
# The reward function is called as:
#   compute_reward(env, obs, action)
# where `env` is a ManipulationRewardView object, not the raw MetaWorld env.
#
# `obs` is a 1-D NumPy array. In this task, treat it as hand-centric reaching state.
# Observation vector length is typically 39.
#
# Practical semantic blocks:
# - obs[0:4]: end-effector / hand-state features, including gripper-related state
# - obs[4:7]: auxiliary state slots; depending on the wrapper/task variant these may
#   contain task-specific values or be less informative than the named fields on `env`
# - obs[18:22]: previous-observation features / padding-like slots depending on wrapper
# - obs[36:39]: target position
#
# Important: do not assume MuJoCo Reacher-v4 semantics such as a dedicated
# fingertip-to-target vector field inside `obs`. Prefer named fields on `env`
# when possible instead of guessing hard-coded obs indices.
#
# Named fields available on the ManipulationRewardView (`env` argument):
# - env.tcp_center: current tool-center-point / hand position
# - env.gripper_opening: current gripper opening scalar
# - env.left_pad_pos / env.right_pad_pos: gripper pad positions
# - env.init_tcp: initial tool-center-point position at reset
# - env.init_left_pad / env.init_right_pad: initial pad positions at reset
# - env.target_pos: target position for the reaching task
# - env.dt: simulation timestep
# - env.terminated: whether the current transition ended the episode
# - env.qpos / env.qvel: full simulator generalized position / velocity state
#
# Some generic manipulation fields may also exist on `env` but can be task-irrelevant
# for pure reaching. If a field does not improve the design, ignore it.
