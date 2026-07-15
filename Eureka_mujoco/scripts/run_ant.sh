#!/usr/bin/env bash
# Launch a full Eureka Ant run.
# Ant is a MuJoCo locomotion task; per-sample PPO budget is 5M steps.
# Usage: ./scripts/run_ant.sh [sample] [iteration]
set -euo pipefail

SAMPLE=${1:-16}
ITER=${2:-5}
MAX_ITER=5000000

cd "$(dirname "$0")/.."
./run_eureka_mujoco.sh ant "$SAMPLE" "$ITER" "$MAX_ITER"
