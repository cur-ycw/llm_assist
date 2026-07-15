# Sawyer door-open V3 observation context.
#
# Observation layout:
# - obs[0:3]   = TCP / gripper xyz
# - obs[3]     = gripper opening scalar in [0, 1]
# - obs[4:7]   = handle / door-related object xyz
# - obs[7:11]  = object quaternion
# - obs[18:36] = previous-frame copy of obs[0:18]
# - obs[36:39] = target xyz
