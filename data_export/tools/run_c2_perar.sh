#!/usr/bin/env bash
# Corrected C2 sweep, memory-safe after the 2026-09-18 OOM reboot:
#   * every process (and so every VTR job) is capped at 8 GB virtual memory;
#   * a watchdog kills the sweep if host available memory drops below 6 GB.
# Resumable: rerunning skips rows already in baseline_ar_sweep_perar.csv.
set -u
cd "$(dirname "$0")"
LOG=../logs/c2_perar.log
ulimit -v 8000000
~/.venv/bin/python c2_perar.py 10 >> "$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt 6 ]; then
    echo "WATCHDOG $(date +%T): MemAvailable ${avail} GB < 6 GB, stopping sweep" | tee -a "$LOG"
    pkill -TERM -P "$PID"; kill -TERM "$PID"; sleep 5; pkill -f run_vtr_flow
    exit 1
  fi
  sleep 15
done
echo "C2 per-AR finished $(date +%T)" >> "$LOG"
