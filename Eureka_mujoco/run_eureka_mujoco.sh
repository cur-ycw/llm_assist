#!/usr/bin/env bash
# Production EUREKA run for MuJoCo/MetaWorld tasks.
# Usage: ./run_eureka_mujoco.sh <env> [sample] [iteration] [max_iterations]
#   env: ant | walker | reacher | door | pick
#   sample (default 16): LLM samples per iter
#   iteration (default 5): EUREKA outer iterations
#   max_iterations (default 1000000): PPO total_timesteps per sample
#
# Paper protocol: sample=16, iteration=5, max_iterations=1M per RL sample.

set -euo pipefail

ENV_NAME=${1:?usage: $0 <env> [sample] [iteration] [max_iterations]}
SAMPLE=${2:-16}
ITER=${3:-5}
MAX_ITER=${4:-1000000}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eureka_mujoco
source /root/ycw/Eureka_mujoco/env.sh

cd /root/ycw/Eureka_mujoco/eureka

LOG_DIR=/root/ycw/Eureka_mujoco/run_logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/root/ycw/Eureka_mujoco/eureka/outputs/eureka/${ENV_NAME}_${TS}
SUMMARY="$LOG_DIR/${ENV_NAME}_${TS}.log"

echo "[launcher] EUREKA-MuJoCo run: env=$ENV_NAME sample=$SAMPLE iter=$ITER max_iter=$MAX_ITER" | tee "$SUMMARY"
echo "[launcher] workspace: $RUN_DIR" | tee -a "$SUMMARY"

WANDB_ARGS=()
if [[ "${USE_WANDB:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  WANDB_ARGS+=(use_wandb=True)
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(wandb_username="${WANDB_ENTITY}")
  [[ -n "${WANDB_PROJECT:-}" ]] && WANDB_ARGS+=(wandb_project="${WANDB_PROJECT}")
  echo "[launcher] wandb: enabled entity=${WANDB_ENTITY:-<unset>} project=${WANDB_PROJECT:-<unset>}" | tee -a "$SUMMARY"
fi

python -u eureka_mujoco.py \
    env=${ENV_NAME} \
    sample=${SAMPLE} \
    iteration=${ITER} \
    max_iterations=${MAX_ITER} \
    n_envs=8 \
    num_eval=5 \
    hydra.run.dir="${RUN_DIR}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee -a "$SUMMARY"

echo "[launcher] done." | tee -a "$SUMMARY"
