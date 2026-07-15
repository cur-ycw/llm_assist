#!/usr/bin/env bash
# Run all 5 MuJoCo/MetaWorld tasks SEQUENTIALLY at production scale using the
# per-task budgets defined in ./scripts/run_*.sh:
#   Ant / Walker  (MuJoCo)     : max_iterations = 5M PPO steps per sample
#   Reach / Pick / Door (MW)   : max_iterations = 10M PPO steps per sample
#
# One full EUREKA run per task (default: sample=16, iteration=5).
#
# To run tasks in parallel across GPUs, launch each ./scripts/run_<task>.sh in
# its own tmux/screen instead of using this driver.
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR=./run_logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/all_tasks_${TS}.log"
echo "[launcher] running all 5 tasks sequentially. summary: $SUMMARY" | tee "$SUMMARY"

for TASK in ant walker reach pick door; do
    echo "[launcher] >>> starting $TASK at $(date)" | tee -a "$SUMMARY"
    ./scripts/run_${TASK}.sh 2>&1 | tee -a "$SUMMARY"
    echo "[launcher] <<< finished $TASK at $(date)" | tee -a "$SUMMARY"
done

echo "[launcher] all 5 tasks done at $(date)" | tee -a "$SUMMARY"
