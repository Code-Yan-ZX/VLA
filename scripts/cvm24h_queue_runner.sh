#!/bin/bash
# CVM2027 supplement master serial GPU queue (frozen at 2026-10-25 19:5x).
# Runs AFTER the currently launched none-full jobs. ONE GPU task at a time.
# Never reordered or trimmed by scores; only by hard time/resource limits.
set -u
cd "$(git rev-parse --show-toplevel)"
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HF=src/v3_premerger/baselines_hf.py
D=results/cvm2027_supplement_20261026/predictions
L=results/cvm2027_supplement_20261026/logs
Q=results/cvm2027_supplement_20261026/plan
PX="--max-pixels 4000000"
Q3=Qwen/Qwen3-VL-8B-Instruct
GPU_DEADLINE=$(($(cat /tmp/cvm24h_T0.txt)+20*3600-3600))  # stop launching 1h early margin

wait_none(){ while [ ! -f $L/none_full.done ]; do sleep 60; done; }
past(){ [ "$(date +%s)" -gt "$GPU_DEADLINE" ] && echo "[ABORT] past gpu deadline" && exit 0; }
hfrun(){ # tag benchmark subset n mode extra
  local tag=$1 bench=$2 sub=$3 n=$4 mode=$5 extra=$6
  [ -f "$D/$tag.json" ] && { echo "[skip] $tag"; return; }
  past
  echo "[start] $tag $(date +%T)"
  timeout 14400 $P $HF --mode $mode --model $Q3 --benchmark $bench --subset $sub \
      --n $n $extra $PX --seed 0 --max-tokens 32 --out $D/$tag.json > $L/$tag.log 2>&1
  echo "[end] $tag rc=$? $(date +%T)"; }
hilorun(){ # tag bench subset n keep extra
  local tag=$1 bench=$2 sub=$3 n=$4 keep=$5 extra=$6
  [ -f "$D/$tag.json" ] && { echo "[skip] $tag"; return; }
  past
  echo "[start] $tag $(date +%T)"
  timeout 14400 $P scripts/cvm24h_hilo_eval.py --benchmark $bench --subset $sub \
      --n $n --keep-frac $keep $extra $PX --seed 0 --out $D/$tag.json > $L/$tag.log 2>&1
  echo "[end] $tag rc=$? $(date +%T)"; }

# make subset jsonls from pre-fixed ids (TextVQA 1000 stratified)
mksub(){ # idlist sourcejsonl outjsonl
  $P - "$1" "$2" "$3" <<'EOF'
import json,sys
ids=set(x.strip() for x in open(sys.argv[1]) if x.strip())
out=open(sys.argv[3],"w"); k=0
for l in open(sys.argv[2]):
    if json.loads(l)["id"] in ids: out.write(l); k+=1
out.close(); print(sys.argv[3],k)
EOF
}
mksub $Q/p2_textvqa_1000_ids.txt eval/full_splits/textvqa_val.jsonl $Q/p2_textvqa_1000.jsonl
head -500 $Q/p2_docvqa_500_ids.txt > $Q/p2_docvqa_500_ids_head500.txt
mksub $Q/p2_docvqa_500_ids.txt eval/full_splits/docvqa_val.jsonl $Q/p2_docvqa_500.jsonl

wait_none
# ---- DocVQA 32q timing smoke (directive 12: OOM/cap decision BEFORE formal runs)
head -32 eval/full_splits/docvqa_val.jsonl > $Q/smoke32_docvqa.jsonl
hfrun s32_doc_none docvqa $Q/smoke32_docvqa.jsonl 32 none
hfrun s32_doc_pre25 docvqa $Q/smoke32_docvqa.jsonl 32 pre "--r-pre 0.25 --mrope native"
hfrun s32_doc_fastv25 docvqa $Q/smoke32_docvqa.jsonl 32 fastv "--r 0.75 --fastv-k 3"
hilorun s32_doc_hilo25 docvqa $Q/smoke32_docvqa.jsonl 32 0.25
# ---- P1 main table: HiLo at keep .25 full splits (RBM/FastV reused, None running now)
hilorun p1_hilo_textvqa_full textvqa eval/full_splits/textvqa_val.jsonl 5000 0.25
hilorun p1_hilo_ocrbench_full ocrbench eval/full_splits/ocrbench.jsonl 1000 0.25
# ---- P2 budget curves on pre-fixed subsets (25% cells = ID slices of P1 preds)
for KEEP in 0.50 0.75; do
  DR=$(python3 -c "print(round(1-$KEEP,2))")
  hfrun c25_pre_textvqa1k_keep$KEEP textvqa $Q/p2_textvqa_1000.jsonl 1000 pre "--r-pre $KEEP --mrope native"
  hfrun c25_fastv_textvqa1k_keep$KEEP textvqa $Q/p2_textvqa_1000.jsonl 1000 fastv "--r $DR --fastv-k 3"
  hilorun c25_hilo_textvqa1k_keep$KEEP textvqa $Q/p2_textvqa_1000.jsonl 1000 $KEEP
done
for KEEP in 0.50 0.75; do
  DR=$(python3 -c "print(round(1-$KEEP,2))")
  hfrun c25_pre_ocrbench_keep$KEEP ocrbench eval/full_splits/ocrbench.jsonl 1000 pre "--r-pre $KEEP --mrope native"
  hfrun c25_fastv_ocrbench_keep$KEEP ocrbench eval/full_splits/ocrbench.jsonl 1000 fastv "--r $DR --fastv-k 3"
  hilorun c25_hilo_ocrbench_keep$KEEP ocrbench eval/full_splits/ocrbench.jsonl 1000 $KEEP
done
# ---- P2 DocVQA 500 stress (subset, 25%, same HF protocol)
hfrun doc500_none docvqa $Q/p2_docvqa_500.jsonl 500 none
hfrun doc500_pre25 docvqa $Q/p2_docvqa_500.jsonl 500 pre "--r-pre 0.25 --mrope native"
hfrun doc500_fastv25 docvqa $Q/p2_docvqa_500.jsonl 500 fastv "--r 0.75 --fastv-k 3"
hilorun doc500_hilo25 docvqa $Q/p2_docvqa_500.jsonl 500 0.25
echo "[queue] MAIN QUEUE DONE $(date +%T)"
# ---- P3 timing after this (separate script), then optional extensions
