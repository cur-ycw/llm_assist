#!/usr/bin/env bash
# Launch a full Eureka Door run (MetaWorld door-open-v3).
# Door is a MetaWorld manipulation task; per-sample PPO budget is 10M steps.
# Usage: ./scripts/run_door.sh [sample] [iteration]
set -euo pipefail

SAMPLE=${1:-16}
ITER=${2:-5}
MAX_ITER=10000000

cd "$(dirname "$0")/.."
./run_eureka_mujoco.sh door "$SAMPLE" "$ITER" "$MAX_ITER"
