#!/bin/bash
# DCC pre-submission Experiment B: Qwen3-VL TextVQA FULL, both arms under the
# SAME HF harness (baselines_hf.py) -- protocol mirrors P1
# (scripts/run_p1_fastv_hf_ocrbench.sh), benchmark swapped to textvqa:
#   RBM   : --mode pre --r-pre 0.25 --mrope native   (keep 25% merger-input units)
#   FastV : --mode fastv --r 0.75 --fastv-k 3        (keep 25% after LLM layer 3)
# Same model (Qwen3-VL-8B), sample IDs (eval/full_splits/textvqa_val.jsonl),
# image setting (--max-pixels 4000000, non-binding for TextVQA small images but
# kept for protocol identity with P1), prompt, greedy temp=0 / max_tokens=32,
# seed=0, and the same scorer inside baselines_hf.py.
#
# Old FastV full-TextVQA (runs/r2_same_scope/r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000.json)
# is a vLLM-harness run (v3_premerger_runner.py, native px) -- NOT the HF
# protocol, so it is NOT reused; both arms run fresh here.
#
# Usage: bash scripts/run_dcc_expB_hf_textvqa.sh [smoke|full|all]
# Outputs -> results/dcc_presubmit/expB/  (never overwrites historical dirs)
set -u
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
       VLLM_USE_MODELSCOPE=False VLLM_ENABLE_V1_MULTIPROCESSING=0

OUT=results/dcc_presubmit/expB
mkdir -p $OUT
HF=src/v3_premerger/baselines_hf.py
FS=eval/full_splits/textvqa_val.jsonl
N=5000
PX="--max-pixels 4000000"
Q3=Qwen/Qwen3-VL-8B-Instruct
STAGE=${1:-all}

wait_gpu(){
  for i in $(seq 1 240); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$FREE" -gt 40000 ] && { echo "[expB] GPU free ${FREE} MiB"; return 0; }
    sleep 30
  done
  echo "[expB][ABORT] GPU busy after wait"; return 1
}

hfrun(){ # mode extra tag
  timeout 43200 python $HF --mode $1 --model $Q3 --benchmark textvqa \
    --subset $FS --n $N $2 $PX --seed 0 --max-tokens 32 \
    --out $OUT/$3.json > $OUT/$3.log 2>&1 \
    && echo "[done] $3 $(python -c "import json;d=json.load(open('$OUT/$3.json'));print('acc=%.4f ptid=%.0f skip=%d'%(d['acc'],d['mean_ptid_len'],d['n_skipped']))" 2>/dev/null)" \
    || echo "[fail] $3 (see log)"
}

wait_gpu
if [ "$STAGE" = "smoke" ] || [ "$STAGE" = "all" ]; then
  echo "=== [smoke] textvqa n=8 both arms ==="
  for ARMX in "pre r25 --r-pre 0.25 --mrope native" "fastv k3 --r 0.75 --fastv-k 3"; do
    set -- $ARMX; MODE=$1; TAGX=$2; shift 2; EXTRA="$*"
    timeout 1800 python $HF --mode $MODE --model $Q3 --benchmark textvqa \
      --subset $FS --n 8 $EXTRA $PX --seed 0 --max-tokens 32 \
      --out $OUT/smoke_textvqa_$TAGX.json > $OUT/smoke_textvqa_$TAGX.log 2>&1
    python -c "import json;d=json.load(open('$OUT/smoke_textvqa_$TAGX.json'));assert d['n_skipped']==0,d['n_skipped'];print('[smoke] $MODE $TAGX PASS acc=%.3f n_image_kept=%s'%(d['acc'],d['per_sample'][0].get('n_image_kept')))" \
      || { echo "[expB][ABORT] smoke $MODE $TAGX failed"; exit 1; }
  done
fi

if [ "$STAGE" = "full" ] || [ "$STAGE" = "all" ]; then
  wait_gpu
  echo "=== [full] RBM pre r0.25 n=5000 ==="
  hfrun pre "--r-pre 0.25 --mrope native" full_textvqa_rbm_pre_r25
  wait_gpu
  echo "=== [full] FastV r0.75 k3 n=5000 ==="
  hfrun fastv "--r 0.75 --fastv-k 3" full_textvqa_fastv_k3
  python scripts/analyze_dcc_expB.py
fi
echo "[expB] stage '$STAGE' done"
