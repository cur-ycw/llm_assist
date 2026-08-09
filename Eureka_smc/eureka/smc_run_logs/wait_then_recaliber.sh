#!/usr/bin/env bash
# 主批完成后自动接力新口径补跑。监听 batch_progress.tsv 的 "ALL BATCH DONE" 行。
set -u
SMC_ROOT=/root/ycw/Eureka_smc/eureka
PROGRESS=$SMC_ROOT/smc_run_logs/batch_progress.tsv
RECALIBER=$SMC_ROOT/smc_run_logs/run_smc_recaliber.sh

bjstamp() { date -u -d '+8 hours' +%Y%m%dT%H%M%S+0800; }
echo "[waiter] armed $(bjstamp); 等待主批 ALL BATCH DONE ..."
# 每 60s 轮询;主批那行是 <ts>\tALL\tBATCH\tDONE\t...
while ! grep -qP '\tALL\tBATCH\tDONE\t' "$PROGRESS" 2>/dev/null; do
  sleep 60
done
echo "[waiter] 检测到 ALL BATCH DONE,$(bjstamp) 启动补跑。"
# 再等 GPU 释放,给主批收尾余量
sleep 30
bash "$RECALIBER"
echo "[waiter] 补跑脚本返回 $(bjstamp)。"
