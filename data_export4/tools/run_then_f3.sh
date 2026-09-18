#!/usr/bin/env bash
# Run data request 4 alone, then restart F3 from scratch alone.
# (Running both at once exhausted memory on 2026-09-18 11:47.)
cd "$(dirname "$0")"
bash run_g.sh
echo "run_g.sh exited rc=$? $(date +%T); starting F3 fresh" >> ../logs/run_g.log
bash ../../data_export3/tools/run_f3.sh
