#!/bin/bash
# Retry pushing the supplement branch while GitHub HTTPS is flaky (10-min backoff,
# up to ~6h).  Plain push, no force, branch only.
set -u
cd "$(git rev-parse --show-toplevel)"
LOG=results/cvm2027_supplement_20261026/logs/push_retry.log
for i in $(seq 1 36); do
  if git push origin codex/cvm-24h-supplement >> "$LOG" 2>&1; then
    echo "OK $(date +%T)" >> "$LOG"; exit 0
  fi
  sleep 600
done
echo "GAVEUP $(date +%T)" >> "$LOG"
