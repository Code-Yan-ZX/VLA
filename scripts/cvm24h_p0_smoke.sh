#!/bin/bash
# CVM2027 one-day supplement — P0 smoke gate (32 questions, TextVQA).
# Debug + timing estimation ONLY (prompt directive 4/5): never used to pick
# favorable params/samples. Serial, one GPU task at a time.
set -u
cd "$(git rev-parse --show-toplevel)"
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HF=src/v3_premerger/baselines_hf.py
OUT=results/cvm2027_supplement_20261026/predictions/smoke
LOGD=results/cvm2027_supplement_20261026/logs
FS=results/cvm2027_supplement_20261026/plan/smoke32_textvqa.jsonl
mkdir -p "$OUT" "$LOGD"
Q3=Qwen/Qwen3-VL-8B-Instruct
PX="--max-pixels 4000000"
COMMON="--model $Q3 --benchmark textvqa --subset $FS --n 32 $PX --seed 0 --max-tokens 32"
# tag | mode | extra  (r=DROP for fastv; r-pre/r-post=KEEP)
run() { # $1 tag $2 mode $3 extra
  local tag=$1 mode=$2 extra=$3
  if [ -f "$OUT/$tag.json" ]; then echo "[skip] $tag exists"; return 0; fi
  echo "[start] $tag $(date +%T)"
  timeout 3600 python "$HF" --mode "$mode" $COMMON $extra \
      --out "$OUT/$tag.json" > "$LOGD/smoke_$tag.log" 2>&1
  local rc=$?
  echo "[end] $tag rc=$rc $(date +%T)"
  [ $rc -ne 0 ] && tail -3 "$LOGD/smoke_$tag.log"
}
run none     none  ""
run noop_pre pre  "--r-pre 1.0 --mrope native"
run noop_post post "--r-post 1.0"
run noop_fastv fastv "--r 0.0 --fastv-k 3"
run pre_keep25  pre "--r-pre 0.25 --mrope native"
run pre_keep50  pre "--r-pre 0.50 --mrope native"
run pre_keep75  pre "--r-pre 0.75 --mrope native"
run fastv_keep75 fastv "--r 0.25 --fastv-k 3"
run fastv_keep50 fastv "--r 0.50 --fastv-k 3"
run fastv_keep25 fastv "--r 0.75 --fastv-k 3"
run pre_keep25_mimic pre "--r-pre 0.25 --mrope vllm-mimic"
echo "[P0 smoke] ALL DONE $(date +%T)"
