#!/usr/bin/env bash
# bidex 512-budget baseline —— Eureka × kettle。
#   预算 512 = iteration 8 × sample 64; search 3000 / test 6000。
#   numEnvs=1024(资源受限下从默认 2048 下调,结果表注明);并发上限 12。
#   activate 先于 source env.sh(env.sh 的 LD_LIBRARY_PATH 依赖 CONDA_PREFIX);key pin index0。
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eureka
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export EUREKA_KEY_INDEX=0
source /root/ycw/Eureka/env.sh
cd /root/ycw/Eureka_smc/eureka
echo "[launch] cwd=$(pwd) key_set=$([ -n "${OPENAI_API_KEY:-}" ] && echo yes || echo NO) $(date -Is)"
exec python -u eureka.py \
  env=shadow_hand_kettle \
  iteration=8 \
  sample=64 \
  max_iterations=3000 \
  test_policy_train_iterations=6000 \
  num_envs=1024 \
  max_parallel_train=12 \
  num_eval=5 \
  use_wandb=False \
  capture_video=False
