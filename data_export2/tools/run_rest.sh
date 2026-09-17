#!/usr/bin/env bash
# Priority order: E1 (mismatched_netlist) -> E1b heuristic -> E2. Each step is resumable.
set -u
cd "$(dirname "$0")"
PY=~/.venv/bin/python
L=../logs
ts() { date +%H:%M:%S; }

echo "=== E1 mismatched_netlist VTR  $(ts)" | tee -a $L/timeline.txt
$PY e1.py vtr 16 >> $L/e1_vtr.log 2>&1
echo "=== E1b heuristic  $(ts)" | tee -a $L/timeline.txt
$PY e1b.py 6 > $L/e1b.log 2>&1
echo "=== E2 ar rank (no runs)  $(ts)" | tee -a $L/timeline.txt
$PY e2_ar_rank.py > $L/e2_ar_rank.log 2>&1
echo "=== E2 rollouts  $(ts)" | tee -a $L/timeline.txt
$PY e2.py rollouts > $L/e2_rollouts.log 2>&1
echo "=== E2 VTR  $(ts)" | tee -a $L/timeline.txt
$PY e2.py vtr 16 > $L/e2_vtr.log 2>&1
echo "=== ALL DONE  $(ts)" | tee -a $L/timeline.txt
