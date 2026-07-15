#!/usr/bin/env bash
# Launch a full Eureka Walker2d run.
# Walker is a MuJoCo locomotion task; per-sample PPO budget is 5M steps.
# Usage: ./scripts/run_walker.sh [sample] [iteration]
set -euo pipefail

SAMPLE=${1:-16}
ITER=${2:-5}
MAX_ITER=5000000

cd "$(dirname "$0")/.."
./run_eureka_mujoco.sh walker "$SAMPLE" "$ITER" "$MAX_ITER"
