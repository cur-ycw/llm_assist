#!/usr/bin/env bash
# Run all 5 MuJoCo/MetaWorld tasks SEQUENTIALLY at production scale.
# Each task gets one full EUREKA run (sample=16 x iter=5 x max_iter=1M).
# Estimated total: 50-75h wall on 3x RTX 3090.
#
# To run in parallel across tasks (faster but more GPU memory), launch each
# `./run_eureka_mujoco.sh <env>` in its own tmux/screen.
set -euo pipefail

cd /root/ycw/Eureka_mujoco

LOG_DIR=/root/ycw/Eureka_mujoco/run_logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/all_tasks_${TS}.log"
echo "[launcher] running all 5 tasks sequentially. summary: $SUMMARY" | tee "$SUMMARY"

for ENV_NAME in ant walker reacher door pick; do
    echo "[launcher] >>> starting $ENV_NAME at $(date)" | tee -a "$SUMMARY"
    ./run_eureka_mujoco.sh "$ENV_NAME" 16 5 1000000 2>&1 | tee -a "$SUMMARY"
    echo "[launcher] <<< finished $ENV_NAME at $(date)" | tee -a "$SUMMARY"
done

echo "[launcher] all 5 tasks done at $(date)" | tee -a "$SUMMARY"
