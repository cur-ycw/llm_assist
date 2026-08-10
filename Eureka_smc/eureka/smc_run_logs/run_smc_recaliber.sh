#!/usr/bin/env bash
# 新口径补跑:N→M 选择压分母修正 + k_min=1 生效后,重跑前 4 个旧口径 SMC 任务。
# 仅 SMC 变化 —— Eureka / RF-Agent 口径未动,不重跑。
# 统一输出到单一文件夹 outputs/recaliber_smc/;实验名后缀北京时间(CST, UTC+8)。
# 迭代数由 cfg/env/<task>.yaml 驱动(与主批一致);search seed 42;test seeds 0-4;B=80。
set +e
set -u

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SMC_ROOT=/root/ycw/Eureka_smc/eureka
OUTROOT=$SMC_ROOT/outputs/recaliber_smc
LOGDIR=$OUTROOT/logs
PROGRESS=$OUTROOT/recaliber_progress.tsv
mkdir -p "$LOGDIR"
[ -f "$PROGRESS" ] || printf "ts_cst\ttask\tmethod\tstatus\tlog\tresult\n" > "$PROGRESS"

# 北京时间戳(UTC+8);系统无 tzdata,用 UTC+8h 手算。格式 20260809T170841+0800
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
  if pgrep -af 'eureka_smc.py|eureka.py env=|rfagent.py|train.py.*task=' | grep -v pgrep | grep -v run_ >/dev/null; then
    echo "[guard] 已有训练器在跑,拒绝启动补跑。" >&2
    exit 1
  fi
}

MARK_SMC='test raw J mean='

run_smc() {
  local task=$1 tag ws log
  tag="${task}_smc_MAX_b80_$(stamp)"
  ws="$OUTROOT/${tag}"
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

MARK_RETEST='champion test raw J mean='

# allegro champion(i0-p38)test-only 补跑:搜索不受超时影响,只有 5000-iter test 被旧
# 7200s 削掉 seed0。用固定 best_reward_code.py + 新无超时评估器,只重跑 5 seed 的 test。
run_allegro_retest() {
  local champ tag ws log
  champ="/root/ycw/Eureka_smc/eureka/outputs/batch/allegro_hand/smc/20260809T122119Z_smc_allegro_hand_MAX_b80/best_reward_code.py"
  tag="allegro_hand_smc_RETEST5seed_$(stamp)"
  ws="$OUTROOT/${tag}"
  log="$LOGDIR/${tag}.log"
  mkdir -p "$ws"
  if [ ! -f "$champ" ]; then
    row allegro_hand SMC-RETEST FAIL "$log" "champion 代码不存在: $champ"
    return
  fi
  row allegro_hand SMC-RETEST START "$log" "test-only 5-seed 无超时"
  ( source "$CONDA_SH"; conda activate eureka; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka_smc/env.sh; cd "$SMC_ROOT"
    python -u eureka_smc.py env=allegro_hand suffix=GPT \
      "retest_champion_code=$champ" \
      algo.evaluation.cache=false \
      algo.reevaluation.enabled=true algo.reevaluation.test_seed_panel='[0,1,2,3,4]' \
      "hydra.run.dir=$ws"
  ) >"$log" 2>&1
  if grep -q "$MARK_RETEST" "$log"; then
    row allegro_hand SMC-RETEST DONE "$log" "$(grep "$MARK_RETEST" "$log" | tail -1 | tr '\t' ' ')"
  else
    row allegro_hand SMC-RETEST FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}


guard
wait_gpu_free
echo "[recaliber] start $(stamp)  out=$OUTROOT"
row ALL RECALIBER START "$PROGRESS" "新口径补跑 4 个 SMC(N->M, k_min=1)"

# 与主批相同的主序,仅 SMC。迭代由 cfg/env 驱动,无需传参。
run_smc quadcopter
run_smc humanoid
run_smc anymal
run_smc franka_cabinet

# allegro champion 5-seed 无超时 test 补跑(与 4 个 SMC 串行,不抢 GPU)。
run_allegro_retest

echo "[recaliber] all done $(stamp)"
row ALL RECALIBER DONE "$PROGRESS" "4 SMC 新口径补跑 + allegro 5-seed retest 完成"
