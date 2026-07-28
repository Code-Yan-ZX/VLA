#!/bin/bash
# InternVL3-8B third-family MAIN MATRIX (architecture-generalization evidence).
#   none/pre/post x {textvqa,docvqa,ocrbench,gqa} @ r0.75  +  textvqa/docvqa @ r0.875
#   full splits; docvqa gets --max-pixels 4000000 (PIL pre-cap防巨图 OOM); robust
#   mns4/chunk500 throughout; idempotent (skip_ratio<=0.25 -> resume); ends with
#   the official rescore (official_scorers; OCRBench category = TOP-LEVEL jsonl
#   field). Output -> runs/internvl3/.
#
# InternVL3 hooks (see runner): PRE = extract_feature pixel-shuffle-unit prune;
# POST = _process_vision_input split prune. No deepstack / no mrope (LLM=Qwen2,
# 1-D RoPE) -> no position fix needed. Weights live in the DEFAULT HF cache
# (~/.cache/huggingface -> /data/models/huggingface/hub, same as the Qwen models;
# runtime is HF_HUB_OFFLINE=1). Self-serializes on the A40: waits for the
# weights to finish downloading AND for the GPU to be idle (R2 campaign + cascade
# gate run ahead of this).
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/internvl3
mkdir -p $OUT
FAM=internvl3
# Weights source: the COMPLETE 15GB copy lives in the default HF cache
# (~/.cache/huggingface -> /data/models/huggingface/hub). The global env sets
# VLLM_USE_MODELSCOPE=True (Qwen models live in the modelscope cache), but the
# InternVL3-8B modelscope copy is still downloading -- so for THIS family we
# point vLLM at the finished HF cache. Localized override; Qwen scripts are
# untouched and keep using modelscope.
export VLLM_USE_MODELSCOPE=False

# ---- (0) wait for the 4 weight shards to be present (download may still run) ----
HUB=/home/dell/.cache/huggingface/hub/models--OpenGVLab--InternVL3-8B
echo "[internvl3] $(date -u '+%F %T') waiting for weights (4 shards) in $HUB"
for i in $(seq 1 360); do
  NS=$(ls $HUB/snapshots/*/model-0000*-of-00004.safetensors 2>/dev/null | wc -l)
  if [ "$NS" -ge 4 ]; then echo "[internvl3] weights present (4 shards)"; break; fi
  sleep 60
done
NS=$(ls $HUB/snapshots/*/model-0000*-of-00004.safetensors 2>/dev/null | wc -l)
if [ "$NS" -lt 4 ]; then echo "[internvl3][ABORT] weights incomplete after wait"; exit 1; fi

# ---- (1) wait for an idle GPU (>= 40000 MiB free; R2/cascade run ahead) ----
echo "[internvl3] $(date -u '+%F %T') waiting for >= 40000 MiB free GPU"
for i in $(seq 1 360); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 40000 ]; then echo "[internvl3] GPU free ${FREE} MiB"; break; fi
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 40000 ]; then echo "[internvl3][ABORT] GPU busy after 6h"; exit 1; fi

GMU=0.9
STD="--max-num-seqs 4 --chunk-size 500 --max-model-len 8192 --gpu-memory-utilization $GMU"
DOC="--max-num-seqs 4 --chunk-size 500 --max-model-len 32768 --max-num-batched-tokens 32768 --max-pixels 4000000 --gpu-memory-utilization $GMU"
FS=eval/full_splits
declare -A JSONL=( [textvqa]=$FS/textvqa_val.jsonl [docvqa]=$FS/docvqa_val.jsonl [ocrbench]=$FS/ocrbench.jsonl [gqa]=$FS/gqa_testdev.jsonl )
declare -A N=( [textvqa]=5000 [docvqa]=5349 [ocrbench]=1000 [gqa]=12578 )

run_cell(){ # bench mode r "flags" tag
  timeout 21000 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $1 \
    --subset ${JSONL[$1]} --n ${N[$1]} --r $3 --mode $2 $4 --out $OUT/$5.json > $OUT/$5.log 2>&1
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
  local TAG=$(printf "internvl3_%s_%s_r%.3f_full" "$MODE" "$B" "$R")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.25 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip_ratio=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip_ratio=$SR > 0.25 -> safe-flags retry ==="
  fi
  echo "=== $TAG n=${N[$B]} ==="
  run_cell $B $MODE "$R" "$FLAGS" $TAG
  SR=$(skip_ratio $TAG)
  if [ "$($PY -c "print(1 if float('$SR')<=0.25 else 0)")" != "1" ]; then
    echo "=== $TAG retry (skip_ratio=$SR): mns4/chunk300 + pixel cap ==="
    local SAFE="--max-num-seqs 4 --chunk-size 300 --max-model-len 8192 --gpu-memory-utilization 0.9 --max-pixels 4000000"
    [ "$B" = "docvqa" ] && SAFE="--max-num-seqs 4 --chunk-size 300 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9 --max-pixels 2000000"
    run_cell $B $MODE "$R" "$SAFE" $TAG
    echo "=== $TAG retry done (skip_ratio=$(skip_ratio $TAG)) ==="
  fi
}

# ---- Wave 1: none (baseline) all benches ----
for B in textvqa docvqa ocrbench gqa; do cell none $B 0.0; done
# ---- Wave 2: pre @ r0.75 all benches (headline mechanism) ----
for B in textvqa docvqa ocrbench gqa; do cell pre $B 0.75; done
# ---- Wave 3: post @ r0.75 all benches ----
for B in textvqa docvqa ocrbench gqa; do cell post $B 0.75; done
# ---- Wave 4: deep budget r0.875 (12.5% keep) text-dense benches, pre+post ----
for B in textvqa docvqa; do cell pre  $B 0.875; cell post $B 0.875; done

echo "=== internvl3 official rescore (CPU) ==="
$PY - <<'EOF'
import json, glob, sys
sys.path.insert(0, 'src/v3_premerger')
import official_scorers as S
# OCRBench question_type/category are TOP-LEVEL fields in the full-split jsonl
qt = {}
for src in ['eval/full_splits/ocrbench.jsonl']:
    try:
        for ln in open(src):
            o = json.loads(ln)
            qt.setdefault(str(o['id']), (o.get('question_type', ''), o.get('category')))
    except Exception as e:
        print('meta warn', src, e)
rows = []
for f in sorted(glob.glob('runs/internvl3/internvl3_*.json')):
    if f.endswith('_official_summary.json'):
        continue
    d = json.load(open(f)); ps = d.get('per_sample') or []; b = d.get('benchmark'); n = len(ps)
    if not n:
        print(f, 'NO per_sample'); continue
    preds = [str(p.get('answer', '')) for p in ps]
    gts = [str(p.get('gt', '')) for p in ps]
    if b == 'textvqa':
        off = sum(S.score_textvqa_vqaacc(a, g) for a, g in zip(preds, gts)) / n; m = 'VQA-acc'
    elif b == 'docvqa':
        off = sum(S.score_docvqa_anls(a, g) for a, g in zip(preds, gts)) / n; m = 'ANLS'
    elif b == 'gqa':
        off = S.score_gqa_batch(preds, gts)['acc']; m = 'acc-official'
    elif b == 'ocrbench':
        items = [(a, g) + tuple(qt.get(str(p.get('id')), ('', '')))
                 for a, g, p in zip(preds, gts, ps)]
        r = S.score_ocrbench_batch(items); off = r.get('acc')
        m = 'acc (Final/1000=%s)' % r.get('final_score')
    else:
        off = None; m = '?'
    rows.append({'file': f.split('/')[-1], 'bench': b, 'mode': d.get('mode'),
                 'r': d.get('r'), 'model': d.get('model'), 'n': n,
                 'official': round(off, 4) if isinstance(off, float) else off,
                 'metric': m, 'ptid': d.get('mean_ptid_len')})
    print(f.split('/')[-1], '| official=',
          round(off, 4) if isinstance(off, float) else off, m,
          '| n=', n, '| ptid=', d.get('mean_ptid_len'))
json.dump(rows, open('runs/internvl3/internvl3_official_summary.json', 'w'),
          indent=1, ensure_ascii=False)
print('wrote runs/internvl3/internvl3_official_summary.json')
EOF
echo "[internvl3 done] $(date -u '+%F %T')"
