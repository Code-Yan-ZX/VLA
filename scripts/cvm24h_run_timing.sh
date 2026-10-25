#!/bin/bash
# Serial P3 timing (relaunch after p3_timing_ids.json was generated).
set -u
cd "$(git rev-parse --show-toplevel)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
Q=results/cvm2027_supplement_20261026/plan
L=results/cvm2027_supplement_20261026/logs
timeout 7200 $P scripts/cvm24h_timing.py --engine harness --out $Q/timing_harness.json >> $L/timing_harness.log 2>&1 \
  && echo "[t] harness OK $(date +%T)" >> $L/timing_status.log \
  || echo "[t] harness FAIL $(date +%T)" >> $L/timing_status.log
timeout 5400 $P scripts/cvm24h_timing.py --engine hilo --out $Q/timing_hilo.json >> $L/timing_hilo.log 2>&1 \
  && echo "[t] hilo OK $(date +%T)" >> $L/timing_status.log \
  || echo "[t] hilo FAIL $(date +%T)" >> $L/timing_status.log
