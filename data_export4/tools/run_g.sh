#!/usr/bin/env bash
# Data request 4, in priority order: sweep (new six) -> G1 -> G2 -> G3.
# Must NOT run concurrently with F3 training (that combination exhausted memory
# at 11:47 on 2026-09-18). So:
#   * 2 concurrent VTR flows for stages touching the new circuits: synthesis of
#     lenet / stereovision1 peaks around 4.5-6 GB per yosys process (4 workers
#     dipped host memory to 4 GB at 12:0x); 4 for G3 (original six only);
#   * the memory watchdog kills the stage's whole process TREE (VTR's yosys
#     children escaped a process-group kill on the first attempt);
#   * stop if host MemAvailable drops below 8 GB, sampled every 5 s.
# No `ulimit -v`: g.py loads PyTorch, which reserves far more virtual address
# space than it uses. Every stage is resumable.
set -u
cd "$(dirname "$0")"
mkdir -p ../logs
L=../logs/run_g.log
descendants() { local c; for c in $(ps -o pid= --ppid "$1"); do descendants "$c"; echo "$c"; done; }
stage() {
  local W="$2"
  echo "=== $1 start $(date +%T) workers=$W" | tee -a "$L"
  setsid ~/.venv/bin/python g.py "$1" "$W" >> "../logs/$1.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt 8 ]; then
      echo "WATCHDOG $(date +%T): MemAvailable ${avail} GB < 8 GB, killing stage $1 only" | tee -a "$L"
      tree="$(descendants "$pid") $pid"
      kill -KILL $tree 2>/dev/null
      exit 1
    fi
    sleep 5
  done
  wait "$pid"; local rc=$?
  echo "=== $1 end rc=$rc $(date +%T)" | tee -a "$L"
  [ "$rc" -eq 0 ] || exit "$rc"
}
stage sweep 2
stage g1 2
stage g2 2
stage g3 4
echo "ALL DONE $(date +%T)" | tee -a "$L"
