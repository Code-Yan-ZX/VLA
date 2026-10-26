#!/bin/bash
# Q10 extension (directive 11/30): TextVQA curve cells at keep .50/.75 from the
# 1000-subset UP TO full 5000, for all three compressed methods (25% full cells
# already exist). Launched ONLY after all mandatory cells done (time/resource
# trigger, not score). Hard stop 2026-10-26 12:00 (report protection margin).
set -u
cd "$(git rev-parse --show-toplevel)"
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HF=src/v3_premerger/baselines_hf.py
D=results/cvm2027_supplement_20261026/predictions
L=results/cvm2027_supplement_20261026/logs
Q=results/cvm2027_supplement_20261026/plan
STOP=$(($(cat /tmp/cvm24h_T0.txt)+19*3600))  # 14:07 stop = GPU deadline 15:07 minus 1h cell margin
run(){ local tag=$1; shift
  [ -f "$D/$tag.json" ] && { echo "[skip] $tag"; return; }
  [ "$(date +%s)" -gt "$STOP" ] && { echo "[q10] stop $(date +%T)"; return; }
  echo "[q10 start] $tag $(date +%T)"
  timeout 4000 "$@" --out $D/$tag.json > $L/$tag.log 2>&1
  echo "[q10 end] $tag rc=$? $(date +%T)"; }
FS=eval/full_splits/textvqa_val.jsonl
M="--model Qwen/Qwen3-VL-8B-Instruct --benchmark textvqa --subset $FS --n 5000 --max-pixels 4000000 --seed 0 --max-tokens 32"
run q10_pre_textvqa5k_keep0.50  $P $HF --mode pre  $M --r-pre 0.50 --mrope native
run q10_pre_textvqa5k_keep0.75  $P $HF --mode pre  $M --r-pre 0.75 --mrope native
run q10_fastv_textvqa5k_keep0.50 $P $HF --mode fastv $M --r 0.50 --fastv-k 3
run q10_fastv_textvqa5k_keep0.75 $P $HF --mode fastv $M --r 0.25 --fastv-k 3
run q10_hilo_textvqa5k_keep0.50  $P scripts/cvm24h_hilo_eval.py --benchmark textvqa --subset $FS --n 5000 --keep-frac 0.50 --max-pixels 4000000 --seed 0
run q10_hilo_textvqa5k_keep0.75  $P scripts/cvm24h_hilo_eval.py --benchmark textvqa --subset $FS --n 5000 --keep-frac 0.75 --max-pixels 4000000 --seed 0
echo "[q10] ALL DONE $(date +%T)"
