#!/usr/bin/env bash
# Resume the one-day experiment after an already active optional cell exits.
# Usage: bash scripts/cvm24h_finish_corrected.sh WAIT_PYTHON_PID
# The old Q10 parent is intentionally stopped separately to keep the GPU serial.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
WAIT_PID=${1:?Provide the active, authorized evaluation Python PID}
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
Q=results/cvm2027_supplement_20261026/plan
L=results/cvm2027_supplement_20261026/logs
DEADLINE=1792998426
# Leave a minute for the last in-flight request and saving a partial cell.
SOFT_DEADLINE=$((DEADLINE-60))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[corrected queue] waiting for PID $WAIT_PID at $(date -Iseconds)"
while test -d "/proc/$WAIT_PID"; do sleep 5; done
echo "[corrected queue] active cell exited at $(date -Iseconds)"
# No other compute task may be running. Never stop or pre-empt a foreign job.
while test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; do
  if test "$(date +%s)" -ge "$DEADLINE"; then
    echo "[corrected queue] deadline waiting for GPU"; exit 2
  fi
  sleep 5
done
"$P" scripts/cvm24h_timing_v2.py --engine harness --limit 2 --repeats 1 \
  --out "$Q/timing_smoke_harness.json" --noop-reference "$Q/timing_v2_smoke_reference.pt" \
  --deadline-epoch "$SOFT_DEADLINE" > "$L/timing_smoke_harness.log" 2>&1
"$P" scripts/cvm24h_timing_v2.py --engine hilo --limit 2 --repeats 1 \
  --out "$Q/timing_smoke_hilo.json" --noop-reference "$Q/timing_v2_smoke_reference.pt" \
  --deadline-epoch "$SOFT_DEADLINE" > "$L/timing_smoke_hilo.log" 2>&1
echo "[corrected queue] mixed-attention runtime smoke passed at $(date -Iseconds)"
test "$(date +%s)" -lt "$SOFT_DEADLINE"
timeout --signal=TERM "$((DEADLINE-$(date +%s)))" "$P" scripts/cvm24h_timing_v2.py \
  --engine harness --out "$Q/timing_harness_corrected.json" --deadline-epoch "$SOFT_DEADLINE" \
  > "$L/timing_harness_corrected.log" 2>&1
echo "[corrected queue] harness exited at $(date -Iseconds)"
test "$(date +%s)" -lt "$SOFT_DEADLINE"
timeout --signal=TERM "$((DEADLINE-$(date +%s)))" "$P" scripts/cvm24h_timing_v2.py \
  --engine hilo --out "$Q/timing_hilo_corrected.json" --deadline-epoch "$SOFT_DEADLINE" \
  > "$L/timing_hilo_corrected.log" 2>&1
echo "[corrected queue] hilo exited at $(date -Iseconds)"
"$P" scripts/cvm24h_score_timing.py
"$P" scripts/cvm24h_build_tables.py
echo "[corrected queue] complete at $(date -Iseconds)"
