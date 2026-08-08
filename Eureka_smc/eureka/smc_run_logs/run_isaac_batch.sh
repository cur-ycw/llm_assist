#!/usr/bin/env bash
# SMC-Eureka §10 主表 —— 剩余 6 个 IsaacGym 任务 × 3 方 串行主编排器。
# 任务主序:QuadCopter→Humanoid→Anymal→FrankaCabinet→AllegroHand→ShadowHand。
# 每任务方法序:SMC(eureka env)→Eureka(eureka env)→RF-Agent(rfagent env)。
# 全局串行:同一时刻只有一个训练器占 GPU;SMC/Eureka 共用已装 isaacgymenvs,靠串行避免注入冲突。
# 口径:max 标准(SMC aggregate=max;Eureka/RF-Agent 原生 max);B=80;test seeds 0-4;gpt-4o-2024-08-06。
# 迭代:SMC/Eureka 从 cfg/env/<task>.yaml 读 per-task 标准;RF-Agent 用 CLI 覆盖(env yaml 不含 iters)。
# 稳健:set +e;每个 run 跑到底;DONE/FAIL 由日志 marker 判定,FAIL 不中断只记录;进度写 TSV。
set +e
set -u

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SMC_ROOT=/root/ycw/Eureka_smc/eureka
RF_ROOT=/root/ycw/RF-Agent/RF_Agent
LOGDIR=$SMC_ROOT/smc_run_logs/batch
PROGRESS=$SMC_ROOT/smc_run_logs/batch_progress.tsv
mkdir -p "$LOGDIR"
[ -f "$PROGRESS" ] || printf "ts\ttask\tmethod\tstatus\tlog\tresult\n" > "$PROGRESS"

# ymd 用于 tag（脚本内 date 可用）
stamp() { date -u +%Y%m%dT%H%M%SZ; }
row() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "$1" "$2" "$3" "$4" "$5" >> "$PROGRESS"; }

wait_gpu_free() {
  # 等到三卡显存基本释放（<500MiB）或超时 300s
  local t=0
  while [ $t -lt 300 ]; do
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500{c++} END{print c+0}')
    [ "$busy" -eq 0 ] && return 0
    sleep 10; t=$((t+10))
  done
  return 0
}

guard() {
  if pgrep -af 'eureka_smc.py|eureka.py env=|rfagent.py|train.py.*task=' | grep -v pgrep | grep -v run_isaac_batch >/dev/null; then
    echo "[guard] 已有训练器在跑,拒绝启动。" >&2
    pgrep -af 'eureka_smc.py|eureka.py env=|rfagent.py|train.py.*task=' | grep -v pgrep >&2
    exit 1
  fi
}

# 完成 marker（grep -E）
MARK_SMC='test raw J mean='
MARK_EUREKA='Final Success Mean:'
MARK_RF='Final Success Mean'

run_smc() {
  local task=$1 tag ws log
  tag="$(stamp)_smc_${task}_MAX_b80"
  ws="$SMC_ROOT/outputs/batch/${task}/smc/${tag}"
  log="$LOGDIR/${tag}.log"
  mkdir -p "$ws"
  row "$task" SMC START "$log" "-"
  ( source "$CONDA_SH"; conda activate eureka; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka_smc/env.sh; cd "$SMC_ROOT"
    python -u eureka_smc.py env="$task" suffix=GPT \
      algo.evaluation.search_seed_panel='[42]' algo.evaluation.cache=false \
      algo.reevaluation.enabled=true algo.reevaluation.test_seed_panel='[0,1,2,3,4]' \
      algo.checkpoint.enabled=true "algo.checkpoint.path=$ws/smc_runtime_checkpoint.json" 'algo.checkpoint.resume_from=null' \
      "hydra.run.dir=$ws"
  ) >"$log" 2>&1
  if grep -q "$MARK_SMC" "$log"; then
    row "$task" SMC DONE "$log" "$(grep "$MARK_SMC" "$log" | tail -1 | tr '\t' ' ')"
  else
    row "$task" SMC FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}

run_eureka() {
  local task=$1 tag ws log
  tag="$(stamp)_eureka_${task}_MAX_b80"
  ws="$SMC_ROOT/outputs/batch/${task}/eureka/${tag}"
  log="$LOGDIR/${tag}.log"
  mkdir -p "$ws"
  row "$task" Eureka START "$log" "-"
  ( source "$CONDA_SH"; conda activate eureka; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka_smc/env.sh; cd "$SMC_ROOT"
    python -u eureka.py env="$task" suffix=GPT iteration=5 sample=16 num_eval=5 "hydra.run.dir=$ws"
  ) >"$log" 2>&1
  if grep -q "$MARK_EUREKA" "$log"; then
    row "$task" Eureka DONE "$log" "$(grep "$MARK_EUREKA" "$log" | tail -1 | tr '\t' ' ')"
  else
    row "$task" Eureka FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}

run_rfagent() {
  local task=$1 s=$2 t=$3 tag log
  tag="$(stamp)_rfagent_${task}_MAX_b80"
  log="$LOGDIR/${tag}.log"
  row "$task" RF-Agent START "$log" "search=$s test=$t (hydra 默认输出目录，rescore 时从日志定位)"
  ( source "$CONDA_SH"; conda activate rfagent; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka/env.sh; cd "$RF_ROOT"
    python -u rfagent.py env="$task" model=gpt-4o-2024-08-06 \
      max_iterations="$s" test_max_iterations="$t"
  ) >"$log" 2>&1
  if grep -q "$MARK_RF" "$log"; then
    row "$task" RF-Agent DONE "$log" "$(grep "$MARK_RF" "$log" | tail -1 | tr '\t' ' ')"
  else
    row "$task" RF-Agent FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}

guard
echo "[batch] start $(stamp)  progress=$PROGRESS"

# task | rf_search | rf_test  （SMC/Eureka 的 iters 由 cfg/env 驱动，无需传）
run_task() { local task=$1 s=$2 t=$3; run_smc "$task"; run_eureka "$task"; run_rfagent "$task" "$s" "$t"; }

run_task quadcopter     500 500
run_task humanoid       500 1000
run_task anymal         500 1000
run_task franka_cabinet 750 1500
run_task allegro_hand   2500 5000
run_task shadow_hand    3000 6000

echo "[batch] all done $(stamp)"
row ALL BATCH DONE "$PROGRESS" "6 tasks x 3 methods finished"
