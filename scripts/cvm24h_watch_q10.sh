#!/bin/bash
# Unattended tail: wait until the Q10 runner exits, then score full-5000 cells.
set -u
cd "$(git rev-parse --show-toplevel)"
L=results/cvm2027_supplement_20261026/logs
while pgrep -f "cvm24h_run_q10" > /dev/null; do sleep 120; done
sleep 5
bash scripts/cvm24h_score_q10.sh > $L/q10_score.log 2>&1
echo "WATCH_DONE $(date +%T)" >> $L/q10_score.log
