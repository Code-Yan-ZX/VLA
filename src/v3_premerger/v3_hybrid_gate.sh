#!/usr/bin/env bash
# Merger-aware hybrid selection GATE battery (Task 5; drafts/v3_merger_aware_design.md
# §4c/§5). keep=25% (r=0.75), L2 selector, seed=0, short-answer subsets, n=100,
# one fresh vLLM process per cell. NO scaling to n=200 (the main window decides).
#
# Stage 1: tune --hybrid-text-frac on textvqa @n=100 (sweep {0.0, 0.5, 1.0}).
# Stage 2: pick frac = argmax textvqa VQA-acc (OFFICIAL metric, offline rescore
#          of the 3 sweep cells; tie -> lower frac); echo choice honestly.
# Stage 3: hybrid (chosen frac) on ocrbench/gqa @n=100; gqa pre/post/none @n=100
#          (fresh short-answer references); gqa pre/post on the 64 captured
#          router ids (cap64, for the offline disagreement router); attn-selector
#          pre/post on textvqa @n=100 (selector invariance, Task 3).
#
# Invocations mirror the reference cells: textvqa/gqa = rescore_rerun config
# (--max-model-len 32768 --max-num-seqs 16); ocrbench = v3_sota_matrix BIG
# config (--max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4).
# GPU-polite: waits at the top until the shared A40 has <20GB used.
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_merger_aware/hybrid_gate
ROUTER=runs/v3_merger_aware/router
mkdir -p $OUT $ROUTER
RUN=src/v3_premerger/v3_premerger_runner.py
R=0.75

echo "=== GPU WAIT: holding until the shared A40 is free (<20GB used) ==="
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$USED" -lt 20000 ]; then break; fi
  echo "    gpu busy (${USED} MiB used); sleeping 30s"
  sleep 30
done
echo "=== GPU FREE (${USED} MiB used): starting gate battery $(date) ==="

run_cell() {  # <out_json> <extra runner args...>
  local out="$1"; shift
  if [ -s "$out" ]; then
    echo "====== SKIP (exists): $(basename "$out") ======"
    return 0
  fi
  echo "====== RUN: $(basename "$out") ======"
  python $RUN "$@" --out "$out" > "${out%.json}.log" 2>&1 \
    && echo "====== DONE: $(basename "$out") ======" \
    || echo "!!!!!! FAIL: $(basename "$out") (see ${out%.json}.log) !!!!!!"
}

# ---------- Stage 1: hybrid-text-frac sweep on textvqa @n100 ----------
for TF in 0.0 0.5 1.0; do
  run_cell $OUT/hybrid_textvqa_r0.750_l2_tf${TF}_n100.json \
    --mode hybrid --hybrid-text-frac $TF --save-unit-scores \
    --r $R --selector l2 --benchmark textvqa \
    --subset eval/subsets/textvqa_200.jsonl --n 100 --seed 0 \
    --max-model-len 32768 --max-num-seqs 16
done

# ---------- Stage 2: pick frac (argmax official textvqa VQA-acc) ----------
python - <<'EOF' > $OUT/chosen_frac.txt
import json, sys
sys.path.insert(0, "src/v3_premerger")
from official_scorers import score_textvqa_vqaacc
best_tf, best_acc = None, -1.0
for tf in ("0.0", "0.5", "1.0"):
    p = f"runs/v3_merger_aware/hybrid_gate/hybrid_textvqa_r0.750_l2_tf{tf}_n100.json"
    try:
        d = json.load(open(p))
        acc = sum(score_textvqa_vqaacc(s["answer"], s["gt"])
                  for s in d["per_sample"] if not s.get("skipped")) / d["n"]
    except Exception as e:
        print(f"  [pick] tf={tf} unreadable ({type(e).__name__})", file=sys.stderr)
        continue
    print(f"  [pick] tf={tf} textvqa VQA-acc={acc:.4f}", file=sys.stderr)
    if acc > best_acc + 1e-12:          # strict > -> ties keep the LOWER frac
        best_tf, best_acc = tf, acc
print(best_tf if best_tf is not None else "0.5")
EOF
TF=$(cat $OUT/chosen_frac.txt | tail -1)
echo "=== CHOSEN hybrid-text-frac = $TF ==="

# ---------- Stage 3a: hybrid (chosen frac) on ocrbench + gqa @n100 ----------
run_cell $OUT/hybrid_ocrbench_r0.750_l2_tf${TF}_n100.json \
  --mode hybrid --hybrid-text-frac $TF --save-unit-scores \
  --r $R --selector l2 --benchmark ocrbench \
  --subset eval/subsets/ocrbench_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-batched-tokens 32768 \
  --max-pixels 1500000 --max-num-seqs 4

run_cell $OUT/hybrid_gqa_r0.750_l2_tf${TF}_n100.json \
  --mode hybrid --hybrid-text-frac $TF --save-unit-scores \
  --r $R --selector l2 --benchmark gqa \
  --subset eval/subsets/gqa_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16

# ---------- Stage 3b: gqa pre/post/none references @n100 (short-answer) ----------
run_cell $OUT/none_gqa_r0.000_l2_n100.json \
  --mode none --r 0.0 --selector l2 --benchmark gqa \
  --subset eval/subsets/gqa_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16
run_cell $OUT/post_gqa_r0.750_l2_n100.json \
  --mode post --r $R --selector l2 --benchmark gqa \
  --subset eval/subsets/gqa_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16
run_cell $OUT/pre_gqa_r0.750_l2_n100.json \
  --mode pre --r $R --selector l2 --benchmark gqa \
  --subset eval/subsets/gqa_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16

# ---------- Stage 3c: gqa pre/post on the 64 captured router ids ----------
python - <<'EOF'
import json
meta = json.load(open("runs/v3_merger_aware/survival_capture/gqa_meta.json"))
idx = meta["sampled_line_indices"]
lines = open("eval/subsets/gqa_200.jsonl").read().splitlines()
out = [lines[i] for i in idx]
with open("runs/v3_merger_aware/router/gqa_cap64.jsonl", "w") as f:
    f.write("\n".join(out) + "\n")
print(f"[router] wrote gqa_cap64.jsonl: {len(out)} lines (sampled ids match capture)")
EOF
run_cell $ROUTER/pre_gqa_cap64_r0.750_l2_n64.json \
  --mode pre --r $R --selector l2 --benchmark gqa \
  --subset $ROUTER/gqa_cap64.jsonl --n 64 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16
run_cell $ROUTER/post_gqa_cap64_r0.750_l2_n64.json \
  --mode post --r $R --selector l2 --benchmark gqa \
  --subset $ROUTER/gqa_cap64.jsonl --n 64 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16

# ---------- Stage 3d: attn-selector invariance (textvqa @n100, pre/post) ----------
run_cell $OUT/post_textvqa_r0.750_attn_n100.json \
  --mode post --r $R --selector attn --benchmark textvqa \
  --subset eval/subsets/textvqa_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16
run_cell $OUT/pre_textvqa_r0.750_attn_n100.json \
  --mode pre --r $R --selector attn --benchmark textvqa \
  --subset eval/subsets/textvqa_200.jsonl --n 100 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16

echo "=== HYBRID GATE BATTERY COMPLETE $(date) ==="
