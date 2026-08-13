#!/usr/bin/env bash
# 消融:接受容忍度收紧 β=5(accept_sharpness)。与主批 SMC 逐字节相同,仅多一个
# algo.accept_sharpness=5 override。用于验证"收紧接受、砍掉改差子代长尾"是否能
# 避免冠军被更差子代覆盖/挤出活种群,并最终不劣于 β=1 基线。
# yaml 默认 accept_sharpness=1.0 不动 —— β=5 只在此处 CLI 覆盖,主表口径不受影响。
# 目标任务:
#   - allegro_hand:β=1 曾出现冠军谱系崩塌(22.6 被更差子代覆盖),最强诊断(对照旧基线)
#   - shadow_hand :β=1 基线为新口径干净 5-seed(主批产出),做最纯净 A/B(仅 β 不同)
# 迭代由 cfg/env 驱动;search seed 42;test seeds 0-4;B=80;无训练超时。
set +e
set -u

CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
SMC_ROOT=/root/ycw/Eureka_smc/eureka
OUTROOT=$SMC_ROOT/outputs/ablation_beta5
LOGDIR=$OUTROOT/logs
PROGRESS=$OUTROOT/ablation_progress.tsv
BETA=5
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
  if pgrep -af 'eureka_smc.py|eureka.py env=|rfagent.py|train.py.*task=' | grep -v pgrep | grep -v run_ >/dev/null; then
    echo "[guard] 已有训练器在跑,拒绝启动消融。" >&2
    exit 1
  fi
}

MARK_SMC='test raw J mean='

run_smc_beta() {
  local task=$1 tag ws log
  tag="${task}_smc_MAX_b80_beta${BETA}_$(stamp)"
  ws="$OUTROOT/${tag}"
  log="$LOGDIR/${tag}.log"
  mkdir -p "$ws"
  row "$task" SMC-b${BETA} START "$log" "accept_sharpness=$BETA"
  ( source "$CONDA_SH"; conda activate eureka; export EUREKA_KEY_INDEX=0; source /root/ycw/Eureka_smc/env.sh; cd "$SMC_ROOT"
    python -u eureka_smc.py env="$task" suffix=GPT \
      algo.accept_sharpness=$BETA \
      algo.evaluation.search_seed_panel='[42]' algo.evaluation.cache=false \
      algo.reevaluation.enabled=true algo.reevaluation.test_seed_panel='[0,1,2,3,4]' \
      algo.checkpoint.enabled=true "algo.checkpoint.path=$ws/smc_runtime_checkpoint.json" 'algo.checkpoint.resume_from=null' \
      "hydra.run.dir=$ws"
  ) >"$log" 2>&1
  if grep -q "$MARK_SMC" "$log"; then
    row "$task" SMC-b${BETA} DONE "$log" "$(grep "$MARK_SMC" "$log" | tail -1 | tr '\t' ' ')"
  else
    row "$task" SMC-b${BETA} FAIL "$log" "$(tail -1 "$log" | tr '\t' ' ')"
  fi
  wait_gpu_free
}

guard
wait_gpu_free
echo "[ablation-beta$BETA] start $(stamp)  out=$OUTROOT"
row ALL ABLATION START "$PROGRESS" "accept_sharpness=$BETA 消融(allegro + shadow)"

# allegro 先跑(崩塌诊断,最重要),shadow 后跑(最纯净 A/B)。串行,不抢 GPU。
run_smc_beta allegro_hand
run_smc_beta shadow_hand

echo "[ablation-beta$BETA] all done $(stamp)"
row ALL ABLATION DONE "$PROGRESS" "β=$BETA 消融完成(allegro + shadow)"
