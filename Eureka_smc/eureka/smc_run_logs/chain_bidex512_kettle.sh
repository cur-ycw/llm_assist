#!/usr/bin/env bash
# 链式:先跑 Eureka×kettle,结束后自动串行启动 RF-Agent×kettle。
# 该脚本自身可由 nohup 放到后台,两阶段不会并行运行。
set -uo pipefail
BASE=/root/ycw/Eureka_smc/eureka/smc_run_logs
EUREKA_LOG="$BASE/bidex512_eureka_kettle.out"
CHAINLOG="$BASE/bidex512_chain_kettle.out"

printf '[chain] %s starting Eureka kettle\n' "$(date -Is)" | tee "$CHAINLOG"
nohup bash "$BASE/launch_bidex512_eureka_kettle.sh" > "$EUREKA_LOG" 2>&1 &
EUREKA_PID=$!
printf '[chain] %s Eureka kettle pid=%s\n' "$(date -Is)" "$EUREKA_PID" | tee -a "$CHAINLOG"

while kill -0 "$EUREKA_PID" 2>/dev/null; do
  sleep 300
done
wait "$EUREKA_PID" 2>/dev/null || true
printf '[chain] %s Eureka exited -> launching RF-Agent kettle\n' "$(date -Is)" | tee -a "$CHAINLOG"
bash "$BASE/launch_bidex512_rfagent_kettle.sh" >> "$CHAINLOG" 2>&1
RF_STATUS=$?
printf '[chain] %s RF-Agent kettle finished status=%s\n' "$(date -Is)" "$RF_STATUS" | tee -a "$CHAINLOG"
exit "$RF_STATUS"
