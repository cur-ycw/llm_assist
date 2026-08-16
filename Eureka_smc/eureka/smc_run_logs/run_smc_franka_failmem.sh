#!/usr/bin/env bash
# franka_cabinet 单任务复测：failure_memory 默认开（2026-08-16 起写入 cfg/algo/smc.yaml 默认 enabled=true）。
# 其余口径与 newdefault 完全一致：init8/每轮8/9轮/B=80/β=5，走 yaml 默认，不 CLI 覆盖 population/β/failure_memory。
# §10 口径：search seed=42；test seeds 0-4；cache=false；reeval on；无训练超时。单任务，不抢 GPU。
# 对照基线：outputs/smc_newdefault_init8_beta5 里的 franka（failure_memory 关）search best=0.216 / test=0.057。
set +e
set -u

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SMC_ROOT=/root/ycw/Eureka_smc/eureka
OUTROOT=$SMC_ROOT/outputs/smc_franka_failmem
LOGDIR=$OUTROOT/logs
PROGRESS=$OUTROOT/failmem_progress.tsv
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
  tag="${task}_smc_MAX_b80_init8_beta5_failmem_$(stamp)"
  ws="$OUTROOT/${tag}"
  log="$LOGDIR/${tag}.log"
  mkdir -p "$ws"
  row "$task" SMC-failmem START "$log" "init8/每轮8/9轮 B=80 β=5 · failure_memory=ON(k=3)"
  ( source "$CONDA_SH"; conda activate eureka; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka_smc/env.sh; cd "$SMC_ROOT"
    python -u eureka_smc.py env="$task" suffix=GPT \
      algo.evaluation.search_seed_panel='[42]' algo.evaluation.cache=false \
      algo.reevaluation.enabled=true algo.reevaluation.test_seed_panel='[0,1,2,3,4]' \
      algo.checkpoint.enabled=true "algo.checkpoint.path=$ws/smc_runtime_checkpoint.json" 'algo.checkpoint.resume_from=null' \
      "hydra.run.dir=$ws"
  ) >"$log" 2>&1
  if grep -q "$MARK_SMC" "$log"; then
    row "$task" SMC-failmem DONE "$log" "$(grep "$MARK_SMC" "$log" | tail -1 | tr '\t' ' ')"
  else
    row "$task" SMC-failmem FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}

guard
wait_gpu_free
echo "[failmem] start $(stamp)  out=$OUTROOT"
row ALL FAILMEM START "$PROGRESS" "franka_cabinet · failure_memory=ON 复测"
run_smc franka_cabinet
echo "[failmem] done $(stamp)"
row ALL FAILMEM DONE "$PROGRESS" "franka_cabinet failure_memory 复测完成"
