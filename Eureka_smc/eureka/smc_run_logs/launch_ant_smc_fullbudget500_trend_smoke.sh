#!/usr/bin/env bash
# 一次完整 B=80 Ant SMC-Eureka 趋势验证运行；不与其他 AntGPT 写入者并发执行。
set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_ant_smc_n16_init16_r8_m8_b80_ppo500_seed0}"
ROOT=/root/ycw/Eureka_smc/eureka
OUTPUT_ROOT="$ROOT/outputs/ant_smc_fullbudget500_trend_smoke"
WORKSPACE="$OUTPUT_ROOT/$RUN_TAG"
LOG_DIR="$ROOT/smc_run_logs"
LAUNCH_LOG="$LOG_DIR/${RUN_TAG}.log"
CHECKPOINT="$WORKSPACE/smc_runtime_checkpoint.json"

mkdir -p "$WORKSPACE" "$LOG_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eureka
source /root/ycw/Eureka_smc/env.sh
cd "$ROOT"

if ! command -v gpustat >/dev/null; then
  echo "gpustat is required for IsaacGymEvaluator GPU selection." >&2
  exit 1
fi

ISAAC_PACKAGE=$(python - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("isaacgymenvs")
if spec is None or spec.origin is None:
    raise SystemExit("isaacgymenvs is not importable")
print(Path(spec.origin).resolve().parent)
PY
)
ANTGPT="$ISAAC_PACKAGE/tasks/antgpt.py"
if [[ ! -f "$ANTGPT" ]]; then
  echo "Expected generated Ant task target is missing: $ANTGPT" >&2
  exit 1
fi

if pgrep -af 'eureka_smc.py|train.py.*task=AntGPT|task=AntGPT.*train.py' >/dev/null; then
  echo "Refusing to start: an AntGPT trainer or SMC driver is already active." >&2
  pgrep -af 'eureka_smc.py|train.py.*task=AntGPT|task=AntGPT.*train.py' >&2 || true
  exit 1
fi

: "${EUREKA_KEY_INDEX:=0}"
export EUREKA_KEY_INDEX

{
  echo "run_tag=$RUN_TAG"
  echo "workspace=$WORKSPACE"
  echo "checkpoint=$CHECKPOINT"
  echo "python=$(command -v python)"
  echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
  echo "isaacgymenvs=$ISAAC_PACKAGE"
  echo "injection_target=$ANTGPT"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_status_begin"
  git status --porcelain
  echo "git_status_end"
  echo "launcher_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
  echo "gpu_inventory_begin"
  nvidia-smi || true
  gpustat --json || true
  echo "gpu_inventory_end"
  echo "budget_protocol=isaac_gym_smc: N=16 init=16 M=8 R=8 mutation=64 B=80; full batch parallelism follows the protocol"
} | tee "$LAUNCH_LOG"

exec python -u eureka_smc.py \
  env=ant \
  suffix=GPT \
  policy_train_iterations=500 \
  algo.evaluation.search_seed_panel='[42]' \
  algo.evaluation.cache=false \
  algo.reevaluation.enabled=true \
  algo.reevaluation.archive_top_k=3 \
  algo.reevaluation.validation_seed_panel='[200,201,202]' \
  algo.reevaluation.test_seed_panel='[300,301,302,303,304]' \
  algo.checkpoint.enabled=true \
  "algo.checkpoint.path=$CHECKPOINT" \
  'algo.checkpoint.resume_from=null' \
  "hydra.run.dir=$WORKSPACE" \
  2>&1 | tee -a "$LAUNCH_LOG"
