#!/bin/bash
# Tier C orchestrator: priority order C2 -> C3 -> C4 -> C5 (C1 reuses existing runs/inpool_seed*.json).
cd /home/digital-2/workspace/rl_gnn/vanilla_rl/data_export/tools
PY=/home/digital-2/.venv/bin/python
W=${W:-16}
stamp(){ echo "$(date -Is) $*" >> ../logs/tier_c_timeline.txt; }
stamp "START host=$(hostname) cpu='$(lscpu | grep 'Model name' | sed 's/.*: *//')' workers=$W"
stamp "C2 start";  $PY c2_ar_sweep.py $W      > ../logs/c2.log 2>&1; stamp "C2 end rc=$?"
stamp "C3 start";  $PY c345.py c3 20 $W       > ../logs/c3.log 2>&1; stamp "C3 end rc=$?"
stamp "C4 start";  $PY c345.py c4 10          > ../logs/c4.log 2>&1; stamp "C4 end rc=$?"
stamp "C5 start";  $PY c345.py c5 10 $W       > ../logs/c5.log 2>&1; stamp "C5 end rc=$?"
stamp "ALL DONE"
