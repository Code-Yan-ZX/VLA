#!/bin/bash
# P1: RBM vs FastV-k3 on OCRBench FULL 1000, BOTH models, SAME HF harness.
#
# Both arms run under baselines_hf.py (HF transformers eager-attention harness):
#   RBM   : --mode pre  --r-pre 0.25   (keep 25% of merger-input units, L2,
#           query-blind -- the paper's Rank-Before-Merge)
#   FastV : --mode fastv --fastv-k 3 --r 0.75   (keep 25% of image tokens after
#           LLM layer 3 by last-query attention; K=3 is the paper's best-K)
# Same processor / pixel cap (max_pixels=4000000, matching P0-2 so both arms of
# the whole campaign share one OCRBench imaging config) / sample IDs
# (eval/full_splits/ocrbench.jsonl) / prompt / greedy decoding (max_tokens=32).
#
# NO vLLM numbers are mixed into this table; every number comes from HF.
#
# Outputs -> results/acmmm_final_controls/p1/*.json + *.log
set -u
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
       VLLM_USE_MODELSCOPE=False VLLM_ENABLE_V1_MULTIPROCESSING=0

OUT=/media/disk2/YZX/research/vla/results/acmmm_final_controls/p1
mkdir -p $OUT
HF=src/v3_premerger/baselines_hf.py
FS=eval/full_splits/ocrbench.jsonl
N=1000
PX="--max-pixels 4000000"
Q3=Qwen/Qwen3-VL-8B-Instruct
Q2=Qwen/Qwen2.5-VL-7B-Instruct

echo "[P1] $(date -u '+%F %T') start; waiting for >= 40000 MiB free"
for i in $(seq 1 240); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[P1] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[P1][ABORT] GPU busy after wait"; exit 1; }

hfrun(){ # model famtag mode tag-suffix extra-args
  local MODEL=$1 FAM=$2 MODE=$3 TAGX=$4 EXTRA=$5
  local TAG=$(printf "p1_%s_%s_ocrbench_%s_full" "$FAM" "$MODE" "$TAGX")
  if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; return; fi
  echo "=== $TAG n=$N ==="
  timeout 14400 python $HF --mode $MODE --model $MODEL --benchmark ocrbench \
    --subset $FS --n $N $EXTRA $PX --seed 0 --out $OUT/$TAG.json \
    > $OUT/$TAG.log 2>&1 \
    && echo "[done] $TAG $(python -c "import json;d=json.load(open('$OUT/$TAG.json'));print('acc=%.3f ptid=%.0f skip=%d'%(d['acc'],d['mean_ptid_len'],d['n_skipped']))" 2>/dev/null)" \
    || echo "[fail] $TAG (see log)"
}

# ---- smoke gates (n=8) for the untested HF paths: qwen2vl pre, both fastv-k3 ----
smoke(){ # model famtag mode tagx extra
  local SMK=$OUT/p1_smoke_$2_$3_$4_n8.json
  if [ ! -s $SMK ]; then
    timeout 1200 python $HF --mode $3 --model $1 --benchmark ocrbench \
      --subset $FS --n 8 $5 $PX --seed 0 --out $SMK > ${SMK%.json}.log 2>&1
  fi
  python -c "import json;d=json.load(open('$SMK'));assert d['n_skipped']==0, d['n_skipped'];print('[smoke] $2 $3 $4 PASS acc=%.3f'%d['acc'])" 2>/dev/null \
    || { echo "[smoke] $2 $3 $4 FAIL -- see ${SMK%.json}.log"; return 1; }
  return 0
}
smoke $Q2 qwen2vl pre r25 "--r-pre 0.25" || exit 1
smoke $Q2 qwen2vl fastv k3  "--r 0.75 --fastv-k 3" || exit 1
smoke $Q3 qwen3vl pre r25 "--r-pre 0.25" || exit 1
smoke $Q3 qwen3vl fastv k3  "--r 0.75 --fastv-k 3" || exit 1

# ---- full 1000 cells ----
hfrun $Q3 qwen3vl pre    r25 "--r-pre 0.25"
hfrun $Q3 qwen3vl fastv  k3  "--r 0.75 --fastv-k 3"
hfrun $Q2 qwen2vl pre    r25 "--r-pre 0.25"
hfrun $Q2 qwen2vl fastv  k3  "--r 0.75 --fastv-k 3"

echo "=== P1 DONE $(date -u '+%F %T') ==="
