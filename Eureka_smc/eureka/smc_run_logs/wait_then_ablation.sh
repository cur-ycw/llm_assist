#!/usr/bin/env bash
# recaliber 完成后自动接力跑 β=5 接受容忍度消融。
# 监听 recaliber_progress.tsv 的 "ALL RECALIBER DONE" 行(recaliber 结束标记)。
set -u
SMC_ROOT=/root/ycw/Eureka_smc/eureka
RECAL_PROGRESS=$SMC_ROOT/outputs/recaliber_smc/recaliber_progress.tsv
ABLATION=$SMC_ROOT/smc_run_logs/run_smc_ablation_beta5.sh

bjstamp() { date -u -d '+8 hours' +%Y%m%dT%H%M%S+0800; }
echo "[ablation-waiter] armed $(bjstamp); 等待 recaliber ALL RECALIBER DONE ..."
# 每 60s 轮询;recaliber 结束那行是 <ts>\tALL\tRECALIBER\tDONE\t...
while ! grep -qP '\tALL\tRECALIBER\tDONE\t' "$RECAL_PROGRESS" 2>/dev/null; do
  sleep 60
done
echo "[ablation-waiter] 检测到 ALL RECALIBER DONE,$(bjstamp) 启动 β=5 消融。"
sleep 30
bash "$ABLATION"
echo "[ablation-waiter] 消融脚本返回 $(bjstamp)。"
