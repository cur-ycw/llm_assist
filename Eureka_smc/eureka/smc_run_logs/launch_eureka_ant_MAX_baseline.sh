#!/usr/bin/env bash
# Eureka 基线 —— 走 Eureka_smc/eureka/eureka.py(锁定决策 #2:与 SMC 共用 score.py/注入根)。
# 口径:原生 max 选择+报告;B=80(iteration=5 × sample=16);test seeds 0-4(num_eval=5);
# 迭代 search=500/test=500(RF-Agent Ant 标准);backbone gpt-4o-2024-08-06 temp1.0(config.yaml 默认)。
set -euo pipefail

ROOT=/root/ycw/Eureka_smc/eureka
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_eureka_ant_MAX_b80_ppo500}"
WORKSPACE="$ROOT/outputs/eureka_ant_baseline_MAX/$RUN_TAG"
LOG_DIR="$ROOT/smc_run_logs"
LAUNCH_LOG="$LOG_DIR/${RUN_TAG}.log"

mkdir -p "$WORKSPACE" "$LOG_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eureka
source /root/ycw/Eureka_smc/env.sh
cd "$ROOT"

if pgrep -af 'eureka_smc.py|eureka.py|train.py.*task=AntGPT|rfagent.py' | grep -v pgrep >/dev/null; then
  echo "Refusing to start: a trainer/driver is already active." >&2
  pgrep -af 'eureka_smc.py|eureka.py|train.py.*task=AntGPT|rfagent.py' >&2 || true
  exit 1
fi

: "${EUREKA_KEY_INDEX:=0}"
export EUREKA_KEY_INDEX

{
  echo "run_tag=$RUN_TAG"
  echo "workspace=$WORKSPACE"
  echo "python=$(command -v python)"
  echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "budget_protocol=eureka_native: iteration=5 sample=16 -> B=80; test num_eval=5 seeds0-4; iters search500/test500"
} | tee "$LAUNCH_LOG"

exec python -u eureka.py \
  env=ant \
  suffix=GPT \
  iteration=5 \
  sample=16 \
  max_iterations=500 \
  test_policy_train_iterations=500 \
  num_eval=5 \
  "hydra.run.dir=$WORKSPACE" \
  2>&1 | tee -a "$LAUNCH_LOG"
