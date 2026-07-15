#!/usr/bin/env bash
# Launch a full Eureka Reach run (MetaWorld reach-v3).
# Reach is a MetaWorld manipulation task; per-sample PPO budget is 10M steps.
# Usage: ./scripts/run_reach.sh [sample] [iteration]
set -euo pipefail

SAMPLE=${1:-16}
ITER=${2:-5}
MAX_ITER=10000000

cd "$(dirname "$0")/.."
./run_eureka_mujoco.sh reacher "$SAMPLE" "$ITER" "$MAX_ITER"
