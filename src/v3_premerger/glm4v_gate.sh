#!/bin/bash
# GLM-4V (GLM-4.1V-9B-Thinking) fourth-family STAGE-LAW GATE.
#   none/pre/post x {textvqa_200, docvqa_200, gqa_200} @ r0.75 (keep 25%),
#   selector l2, fresh process per cell, outputs -> runs/glm4v_gate/.
#   Ends with the official offline rescore (textvqa VQA-acc, docvqa ANLS,
#   gqa exact-match) into runs/glm4v_gate/glm4v_gate_official_summary.json.
#
# Model: ZhipuAI/GLM-4.1V-9B-Thinking ModelScope snapshot at
#   /data/models/modelscope/hub/models/ZhipuAI/GLM-4___1V-9B-Thinking
# (GLM-4.6V-Flash was the audit TOP-1 but its 5.0rc-era Glm46VProcessor
# classes do not exist in the pinned transformers 4.57.6 -> pre-registered
# fallback; identical Glm4v architecture, config targets 4.57.1.)
#
# Hooks (runner): PRE = patched visual.forward unit slice on the ViT stream
# before the downsample-conv + merger MLP; POST = _process_image_input split
# prune (family-agnostic). mrope fix ON (block sections [8,12,12]).
# Resolution: the model's DEFAULT processor budget
# (size={shortest_edge 12544, longest_edge 9633792} areas; patch 14, merge 2)
# -> NO --max-pixels (Glm4vImageProcessor has no such kwarg anyway).
# max_tokens 1024: GLM-4.1V-9B-Thinking is a thinking model (spontaneous
# <think> blocks); 32 would truncate reasoning before any answer, and the
# n=8 smoke showed degraded arms loop inside <think> (uncertainty) and need
# room to still converge on the boxed answer. Identical for all arms, so only
# the within-model PATTERN is read (absolute values are not cross-family
# comparable anyway).
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
# Weights are a LOCAL modelscope snapshot path -> no hub resolution at all.
export VLLM_USE_MODELSCOPE=False

OUT=runs/glm4v_gate
mkdir -p $OUT
FAM=glm4v
R=0.75
MAXTOK=1024

# ---- (0) wait for an idle GPU (>= 40000 MiB free) ----
echo "[glm4v_gate] $(date -u '+%F %T') waiting for >= 40000 MiB free GPU"
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 40000 ]; then echo "[glm4v_gate] GPU free ${FREE} MiB"; break; fi
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 40000 ]; then echo "[glm4v_gate][ABORT] GPU busy after wait"; exit 1; fi

GMU=0.9
STD="--max-num-seqs 16 --chunk-size 250 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization $GMU"
# DOC: mns 4 + gmu 0.85 -- the pruned (pre/post) arms have SHORT placeholders,
# so the scheduler packs max_num_seqs concurrent long-doc prefills at once;
# with mns 8 / gmu 0.9 that stack of ViT activations (20k+-patch documents)
# OOM'd at step 0. Lower concurrency + 2.2GB extra headroom fixes it (none
# arm completed fine at mns 8 but the same safe flags apply, idempotent skip).
DOC="--max-num-seqs 4 --chunk-size 250 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.85"
declare -A JSONL=( [textvqa]=eval/subsets/textvqa_200.jsonl
                   [docvqa]=eval/subsets/docvqa_200.jsonl
                   [gqa]=eval/subsets/gqa_200.jsonl )
N=200

run_cell(){ # bench mode r "flags" tag
  timeout 5400 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $1 \
    --subset ${JSONL[$1]} --n $N --r $3 --mode $2 --selector l2 --max-tokens $MAXTOK \
    $4 --out $OUT/$5.json > $OUT/$5.log 2>&1
  tail -2 $OUT/$5.log
}
skip_ratio(){ $PY -c "
import json
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
cell(){ # mode bench r
  local MODE=$1 B=$2 R=$3
  local FLAGS=$STD; [ "$B" = "docvqa" ] && FLAGS=$DOC
  local TAG=$(printf "glm4v_%s_%s_r%.3f_n200" "$MODE" "$B" "$R")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip_ratio=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip_ratio=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N ==="
  run_cell $B $MODE "$R" "$FLAGS" $TAG
  echo "=== $TAG done (skip_ratio=$(skip_ratio $TAG)) ==="
}

# ---- Wave 1: none (baseline) ----
for B in textvqa docvqa gqa; do cell none $B 0.0; done
# ---- Wave 2: pre @ r0.75 (rank-before-merge) ----
for B in textvqa docvqa gqa; do cell pre $B $R; done
# ---- Wave 3: post @ r0.75 ----
for B in textvqa docvqa gqa; do cell post $B $R; done

echo "=== glm4v official rescore (CPU) ==="
$PY - <<'EOF'
import json, glob, sys, os
sys.path.insert(0, 'src')
from v3_premerger.official_scorers import (score_textvqa_vqaacc,
                                             score_docvqa_anls, score_gqa)
SCORERS = {'textvqa': score_textvqa_vqaacc,
           'docvqa': score_docvqa_anls,
           'gqa': score_gqa}
rows = []
for f in sorted(glob.glob('runs/glm4v_gate/glm4v_*.json')):
    if f.endswith('_official_summary.json'):
        continue
    d = json.load(open(f))
    b = d.get('benchmark')
    ps = d.get('per_sample') or []
    if b not in SCORERS or not ps:
        continue
    sc = SCORERS[b]
    vals, n_skip = [], 0
    for p in ps:
        if p.get('skipped'):
            n_skip += 1
            continue
        vals.append(float(sc(p.get('answer', ''), str(p.get('gt', '')))))
    off = sum(vals) / len(vals) if vals else 0.0
    row = {'file': os.path.basename(f), 'benchmark': b, 'mode': d.get('mode'),
           'r': d.get('r'), 'n': len(ps), 'n_skipped': n_skip,
           'runner_acc': d.get('acc'), 'official_metric':
           {'textvqa': 'vqa_acc', 'docvqa': 'anls', 'gqa': 'exact_match'}[b],
           'official_score': round(off, 4),
           'mean_ptid_len': d.get('mean_ptid_len'),
           'wall_s': d.get('wall_s'), 'load_s': d.get('load_s')}
    # token-count diagnostics: pre/post placeholder (n,k) pairs
    diag = d.get('diag') or {}
    row['diag_nk'] = (diag.get('nk') or [])[:8]
    row['proc_placeholder_counts'] = d.get('proc_placeholder_counts') or []
    rows.append(row)
    print(f"[rescore] {row['file']}: official {row['official_metric']}="
          f"{row['official_score']:.4f} (runner acc={row['runner_acc']}, "
          f"skip={n_skip}/{len(ps)})")
summary = {'gate': 'glm4v_stage_law', 'keep_frac': 0.25, 'selector': 'l2',
           'rows': rows}
with open('runs/glm4v_gate/glm4v_gate_official_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
# ---- verdict table ----
tab = {}
for r in rows:
    tab[(r['benchmark'], r['mode'])] = r['official_score']
print('\n=== OFFICIAL SCORES (none/pre/post) ===')
for b in ('textvqa', 'docvqa', 'gqa'):
    n, pre, post = tab.get((b, 'none')), tab.get((b, 'pre')), tab.get((b, 'post'))
    d_pp = None if (pre is None or post is None) else round((pre - post) * 100, 2)
    print(f"{b:8s} none={n}  pre={pre}  post={post}  d(pre-post)pp={d_pp}")
EOF
echo "=== GLM4V GATE DONE ==="
