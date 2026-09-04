#!/usr/bin/env bash
# bidex 512-budget baseline —— RF-Agent × kettle。
#   预算 512 = initial_size 32 + 15×(action_weight_num 求和 32); search 3000 / test 6000。
#   numEnvs=1024(与 Eureka 一致,资源受限,结果表注明非 2048 标准);并发上限 12。
#   conda rfagent(自带 isaacgymenvs,写=读);activate 后 source env.sh(LD_LIBRARY_PATH 按 rfagent 前缀 + chatanywhere key)。
set -uo pipefail
MODEL="${MODEL:-gpt-4o-2024-08-06}"
RF_DIR=/root/ycw/RF-Agent/RF_Agent
LOGDIR=/root/ycw/Eureka_smc/eureka/smc_run_logs
LOG="$LOGDIR/bidex512_rfagent_kettle.out"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rfagent
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
: "${EUREKA_KEY_INDEX:=0}"; export EUREKA_KEY_INDEX
source /root/ycw/Eureka/env.sh

# 单-driver 守卫:同一时刻只允许一个 rfagent.py(避免并发写同一注入文件)
if ps -eo pid,cmd | grep -E 'rfagent\.py' | grep -v grep >/dev/null; then
  echo "[rf] Refusing: an rfagent.py driver is already active." | tee -a "$LOG" >&2
  exit 1
fi

echo "[rf] $(date -Is) cwd=$RF_DIR key_set=$([ -n "${OPENAI_API_KEY:-}" ] && echo yes || echo NO) api_base=${OPENAI_API_BASE:-unset}" | tee "$LOG"
cd "$RF_DIR"
exec python -u rfagent.py \
  env=shadow_hand_kettle \
  model="$MODEL" \
  temperature=1.0 \
  simulations=512 \
  initial_size=32 \
  'action_weight_num=[0,8,8,8,4,4]' \
  num_envs=1024 \
  max_parallel_train=12 \
  max_iterations=3000 \
  test_max_iterations=6000 \
  num_eval=5 \
  train_seed=0 \
  use_wandb=False \
  capture_video=False \
  2>&1 | tee -a "$LOG"
