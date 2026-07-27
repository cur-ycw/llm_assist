#!/bin/bash
# 真规模首跑：FrankaCabinet 单岛 SMC，固定 80 评估预算（init 16 + 4 轮×16）。
# N=16 全并行、1500 步、search seed=42、heldout=[100-104]。约 5 小时（80×1500÷3卡）。
# 正确激活顺序：先 conda activate（设 CONDA_PREFIX/PATH），再 source env.sh（用 $CONDA_PREFIX/lib
# 补 LD_LIBRARY_PATH + isaacgym + API keys）。否则 train.py 缺 libpython/gpustat 秒崩。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eureka
source /root/ycw/Eureka_smc/env.sh
cd /root/ycw/Eureka_smc/eureka
exec python eureka_smc.py \
  env=franka_cabinet \
  algo.n_particles=16 \
  algo.min_smc_iterations=4 \
  algo.max_smc_iterations=4 \
  algo.evaluation.max_concurrent_evals=16 \
  algo.evaluation.search_seed_panel=[42] \
  'algo.evaluation.heldout_seed_panel=[100,101,102,103,104]' \
  policy_train_iterations=1500
