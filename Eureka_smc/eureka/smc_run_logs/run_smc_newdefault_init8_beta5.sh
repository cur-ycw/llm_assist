#!/usr/bin/env bash
# 新默认口径:init8 / 每轮8 / 9轮 / B=80 / β=5(2026-08-15 起写入 cfg/algo/smc.yaml 默认)。
# 本脚本不再 CLI 覆盖 population / accept_sharpness —— 全部走 yaml 新默认,验证默认已生效。
# 目标任务:humanoid、anymal、franka_cabinet(除 allegro/shadow 外;二者太耗时,单独另跑)。
# §10 口径不变:search seed=42;test seeds 0-4;cache=false;reeval on;无训练超时。串行,不抢 GPU。
set +e
set -u

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SMC_ROOT=/root/ycw/Eureka_smc/eureka
OUTROOT=$SMC_ROOT/outputs/smc_newdefault_init8_beta5
LOGDIR=$OUTROOT/logs
PROGRESS=$OUTROOT/newdefault_progress.tsv
mkdir -p "$LOGDIR"
[ -f "$PROGRESS" ] || printf "ts_cst\ttask\tmethod\tstatus\tlog\tresult\n" > "$PROGRESS"

stamp() { date -u -d '+8 hours' +%Y%m%dT%H%M%S+0800; }
row() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "$1" "$2" "$3" "$4" "$5" >> "$PROGRESS"; }

wait_gpu_free() {
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
  if pgrep -af 'eureka_smc.py|rfagent.py|train.py.*task=' | grep -v pgrep | grep -v run_ >/dev/null; then
    echo "[guard] 已有训练器在跑,拒绝启动。" >&2
    exit 1
  fi
}

MARK_SMC='test raw J mean='

run_smc() {
  local task=$1 tag ws log
  tag="${task}_smc_MAX_b80_init8_beta5_$(stamp)"
  ws="$OUTROOT/${tag}"
  log="$LOGDIR/${tag}.log"
  mkdir -p "$ws"
  row "$task" SMC-new START "$log" "init8/每轮8/9轮 B=80 β=5(走 yaml 新默认)"
  ( source "$CONDA_SH"; conda activate eureka; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka_smc/env.sh; cd "$SMC_ROOT"
    python -u eureka_smc.py env="$task" suffix=GPT \
      algo.evaluation.search_seed_panel='[42]' algo.evaluation.cache=false \
      algo.reevaluation.enabled=true algo.reevaluation.test_seed_panel='[0,1,2,3,4]' \
      algo.checkpoint.enabled=true "algo.checkpoint.path=$ws/smc_runtime_checkpoint.json" 'algo.checkpoint.resume_from=null' \
      "hydra.run.dir=$ws"
  ) >"$log" 2>&1
  if grep -q "$MARK_SMC" "$log"; then
    row "$task" SMC-new DONE "$log" "$(grep "$MARK_SMC" "$log" | tail -1 | tr '\t' ' ')"
  else
    row "$task" SMC-new FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}

guard
wait_gpu_free
echo "[newdefault] start $(stamp)  out=$OUTROOT"
row ALL NEWDEFAULT START "$PROGRESS" "init8/β=5 新默认 · humanoid+anymal+franka_cabinet"

run_smc humanoid
run_smc anymal
run_smc franka_cabinet

echo "[newdefault] all done $(stamp)"
row ALL NEWDEFAULT DONE "$PROGRESS" "init8/β=5 新默认 3 任务完成"
