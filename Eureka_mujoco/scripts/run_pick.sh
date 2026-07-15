#!/usr/bin/env bash
# Launch a full Eureka Pick run (MetaWorld pick-place-v3).
# Pick is a MetaWorld manipulation task; per-sample PPO budget is 10M steps.
# Usage: ./scripts/run_pick.sh [sample] [iteration]
set -euo pipefail

SAMPLE=${1:-16}
ITER=${2:-5}
MAX_ITER=10000000

cd "$(dirname "$0")/.."
./run_eureka_mujoco.sh pick "$SAMPLE" "$ITER" "$MAX_ITER"
