#!/usr/bin/env bash
# F3: train the no-GCN policy (seed 42), then evaluate it zero-shot.
# Memory-safe after the 2026-09-18 OOM reboot: a watchdog stops training if host
# MemAvailable drops below 6 GB. (No `ulimit -v`: PyTorch reserves far more
# virtual address space than it uses and would fail spuriously under it.)
set -u
cd "$(dirname "$0")"
L=../logs/f3_train.log
~/.venv/bin/python f3_train.py >> "$L" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt 6 ]; then
    echo "WATCHDOG $(date +%T): MemAvailable ${avail} GB < 6 GB, stopping F3 training" | tee -a "$L"
    pkill -TERM -P "$PID"; kill -TERM "$PID"; sleep 10; pkill -f run_vtr_flow
    exit 1
  fi
  sleep 30
done
wait "$PID"; rc=$?
echo "F3 training exited rc=$rc $(date +%T)" >> "$L"
[ "$rc" -eq 0 ] || exit "$rc"
~/.venv/bin/python f3_eval.py 6 > ../logs/f3_eval.log 2>&1
echo "F3 eval exited rc=$? $(date +%T)" >> "$L"
