#!/usr/bin/env bash
# J4 baseline-harness self-check (equivalence + smoke). NEEDS THE A40 GPU -- run
# only when the GPU is free (J1 Qwen2.5-VL campaign currently holds it).
#
# Two parts:
#   STEP 0  CPU-only logic self test (no GPU): baselines_hf.py --dry-check.
#           Verifies the manual layer loop == native forward at r=0 (bitwise),
#           FastV/Pyramid keep counts + cache cropping, and the vision-capture
#           path on a tiny random model. Runs in seconds; safe to run anytime.
#   STEP 1  EQUIVALENCE (GPU): mode=none HF-transformers vs the vLLM runner,
#           SAME model/prompt/sampling on a GQA subset n=16. At temp=0 the two
#           engines should agree on >=15/16 answers (residual = eager vs flash
#           kernel epsilon). This is the go/no-go for trusting the HF harness.
#   STEP 2  SMOKE (GPU, placeholders): FastV (n=8) and PyramidDrop (n=8) cells
#           just to confirm they run end-to-end on real weights and emit the
#           runner-compatible JSON. NOT timed/claimed -- efficiency uses vLLM.
#
# No `set -e` (a cell failure must not abort the rest). Idempotent: skip JSONs
# that already exist. Sparse [skip]/[done]/[fail] lines -- feed to a Monitor.

cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/j4_baselines_hf
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py
HF=src/v3_premerger/baselines_hf.py
SUB=eval/subsets/gqa_200.jsonl
MODEL=Qwen/Qwen3-VL-8B-Instruct          # default; swap to Qwen2.5-VL-7B for cross-family

echo "==================== STEP 0: CPU dry-check (no GPU) ===================="
python $HF --dry-check 2>&1 | sed 's/^/[dry] /' || echo "[fail] dry-check"

echo "==================== STEP 1: none HF vs vLLM (GPU, n=16) ===================="
RUNNER_JSON=$OUT/runner_none_gqa_n16.json
HF_JSON=$OUT/hf_none_gqa_n16.json
# 1a) vLLM runner (mode none), GQA standard config (mns 16), seed 0.
if [ -f "$RUNNER_JSON" ]; then echo "[skip] runner_none_gqa_n16"; else
  python $RUN --mode none --r 0.0 --benchmark gqa --subset $SUB --n 16 \
    --selector l2 --seed 0 --max-model-len 32768 --max-num-seqs 16 \
    --out $RUNNER_JSON > $OUT/runner_none_gqa_n16.log 2>&1 \
    && echo "[done] runner_none_gqa_n16" || echo "[fail] runner_none_gqa_n16 (see log)"
fi
# 1b) HF harness (mode none), SAME model/prompt/sampling/pixels.
if [ -f "$HF_JSON" ]; then echo "[skip] hf_none_gqa_n16"; else
  python $HF --mode none --model $MODEL --benchmark gqa --subset $SUB --n 16 \
    --seed 0 --max-pixels 0 --out $HF_JSON > $OUT/hf_none_gqa_n16.log 2>&1 \
    && echo "[done] hf_none_gqa_n16" || echo "[fail] hf_none_gqa_n16 (see log)"
fi
# 1c) per-sample agreement by id (answer string + runner/online correct).
if [ -f "$RUNNER_JSON" ] && [ -f "$HF_JSON" ]; then
python - "$RUNNER_JSON" "$HF_JSON" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ra = {str(s["id"]): s for s in a["per_sample"] if not s.get("skipped")}
hb = {str(s["id"]): s for s in b["per_sample"] if not s.get("skipped")}
ids = [i for i in ra if i in hb]
ans_eq = sum(1 for i in ids if ra[i]["answer"].strip() == hb[i]["answer"].strip())
cor_eq = sum(1 for i in ids if int(ra[i]["correct"]) == int(hb[i]["correct"]))
n = len(ids)
print(f"[equiv] n_common={n}  answer_match={ans_eq}/{n}  correct_match={cor_eq}/{n}  "
      f"runner_acc={a.get('acc')}  hf_acc={b.get('acc')}  "
      f"({'PASS >=15/16' if ans_eq>=15 else 'CHECK: engine epsilon or bug'})")
PY
else
  echo "[skip] equiv compare (missing a JSON)"
fi

echo "==================== STEP 2: FastV / Pyramid smoke (GPU, n=8) ===================="
# FastV: prune after layer K=2, keep round(n_img*(1-r)); r=0.5 -> keep 50%.
FV=$OUT/hf_fastv_gqa_r0.50_n8.json
if [ -f "$FV" ]; then echo "[skip] hf_fastv_gqa_r0.50_n8"; else
  python $HF --mode fastv --model $MODEL --benchmark gqa --subset $SUB --n 8 \
    --r 0.5 --fastv-k 2 --seed 0 --max-pixels 0 --out $FV \
    > $OUT/hf_fastv_gqa_r0.50_n8.log 2>&1 \
    && echo "[done] hf_fastv_gqa_r0.50_n8 $(python -c "import json;d=json.load(open('$FV'));print('acc=%.3f ptid=%.0f skip=%d'%(d['acc'],d['mean_ptid_len'],d['n_skipped']))" 2>/dev/null)" \
    || echo "[fail] hf_fastv_gqa_r0.50_n8 (see log)"
fi
# PyramidDrop: 4 bands keep [1.0,0.75,0.5,0.25] -> keep_equiv=0.625 (r_equiv=0.375).
PD=$OUT/hf_pyramid_gqa_n8.json
if [ -f "$PD" ]; then echo "[skip] hf_pyramid_gqa_n8"; else
  python $HF --mode pyramid --model $MODEL --benchmark gqa --subset $SUB --n 8 \
    --pyramid-ratios 1.0,0.75,0.5,0.25 --seed 0 --max-pixels 0 --out $PD \
    > $OUT/hf_pyramid_gqa_n8.log 2>&1 \
    && echo "[done] hf_pyramid_gqa_n8 $(python -c "import json;d=json.load(open('$PD'));print('acc=%.3f ptid=%.0f keep_equiv=%.3f skip=%d'%(d['acc'],d['mean_ptid_len'],d['pyramid_keep_equiv'],d['n_skipped']))" 2>/dev/null)" \
    || echo "[fail] hf_pyramid_gqa_n8 (see log)"
fi

echo "==================== J4 BASELINE CHECK DONE ===================="
