#!/usr/bin/env bash
# humanoid 单任务复测：failure_memory 默认开（cfg/algo/smc.yaml 默认 enabled=true）。
# 换 humanoid 是因为 franka 信号太糊(test 0.057、跨 seed 方差大)；humanoid 对照清晰:
#   对照(OFF)= outputs/smc_newdefault_init8_beta5 里 humanoid，search best=5.637 / test=4.76。
#   诊断已知病症:全局最优 5.637 是 init 孤峰,stage0 就被 MH 接受的劣化子代顶替出种群 → 后续零增益。
#   看 failure_memory 能否把变异从"反复拐进同一坑"推离、让 running_best 突破 5.637。
# 口径与 newdefault 一致:init8/每轮8/9轮/B=80/β=5,走 yaml 默认。§10:search42/test0-4/cache off/reeval on。
set +e
set -u

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SMC_ROOT=/root/ycw/Eureka_smc/eureka
OUTROOT=$SMC_ROOT/outputs/smc_humanoid_failmem
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
row ALL FAILMEM START "$PROGRESS" "humanoid · failure_memory=ON 复测"
run_smc humanoid
echo "[failmem] done $(stamp)"
row ALL FAILMEM DONE "$PROGRESS" "humanoid failure_memory 复测完成"
