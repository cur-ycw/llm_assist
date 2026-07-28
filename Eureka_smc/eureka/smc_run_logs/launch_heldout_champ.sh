#!/usr/bin/env bash
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eureka
export EUREKA_KEY_INDEX=1
source /root/ycw/Eureka/env.sh   # 仅为 LD_LIBRARY_PATH/isaacgym；本脚本不调 LLM
cd /root/ycw/Eureka_smc/eureka
echo "[launch] CONDA_PREFIX=$CONDA_PREFIX  $(date)"
exec python -u smc_run_logs/heldout_eureka_champion.py
