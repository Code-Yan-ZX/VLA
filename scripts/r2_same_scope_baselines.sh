#!/bin/bash
# R2 — same-scope baseline gap-fill (mock-review R2-1/R2-2; measurement only):
#  (1) FastV (HF harness) K sensitivity: textvqa n=500 qwen3vl r=0.75 --fastv-k {1,2,3,4}
#  (2) qwen3vl none docvqa FULL 5349 main-table anchor (runner --mode none,
#      --max-model-len 49152 --max-num-seqs 1 --chunk-size 200; huge docs may
#      still skip -> skip count reported; >15% -> digest falls back to the
#      subset none 0.976 n200 footnote)
#  (3) FastV k=2 FULL splits, qwen3vl r=0.75 (same scope as our full-split
#      cells; docvqa/ocrbench keep the existing n=500 cells):
#      textvqa_val 5000 + gqa_testdev 12578
# All cells resume (existing non-empty json -> skip), full logs per cell,
# official rescore at the end (TextVQA VQA-acc, GQA official acc, DocVQA ANLS).
set -u
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/r2_same_scope
HF=src/v3_premerger/baselines_hf.py
RUN=src/v3_premerger/v3_premerger_runner.py
FS=eval/full_splits
Q3=Qwen/Qwen3-VL-8B-Instruct
mkdir -p $OUT
echo "[R2] $(date -u '+%F %T') campaign start"

wait_gpu(){ # shared-machine etiquette: poll every 60s until >= 30000 MiB free
  local tag=$1
  for i in $(seq 1 720); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$FREE" -gt 30000 ]; then echo "[$tag] GPU free ${FREE} MiB"; return 0; fi
    sleep 60
  done
  echo "[$tag][ABORT] GPU still busy after 12h wait"; return 1
}

hfcell(){ # tag bench subset n k timeout_s [extra px]
  local TAG=$1 B=$2 SUB=$3 N=$4 K=$5 TMO=$6
  if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; return; fi
  wait_gpu $TAG || exit 1
  echo "=== $TAG start $(date -u '+%F %T') ==="
  timeout $TMO python $HF --mode fastv --model $Q3 --model-family qwen3vl --benchmark $B \
    --subset $SUB --n $N --r 0.75 --fastv-k $K --seed 0 \
    --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
  echo "=== $TAG rc=$? $(date -u '+%F %T') ==="
  tail -1 $OUT/$TAG.log
}

# ---- (1) FastV K sensitivity (shortest cells first) ----
for K in 1 2 3 4; do
  hfcell r2_qwen3vl_fastv_k${K}_textvqa_r0.75_n500 textvqa $FS/textvqa_val.jsonl 500 $K 5400
done

# ---- (2) qwen3vl none docvqa full-5349 main-table anchor ----
TAG=r2_qwen3vl_none_docvqa_full5349
if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; else
  wait_gpu $TAG || exit 1
  echo "=== $TAG start $(date -u '+%F %T') ==="
  timeout 14400 python $RUN --model-family qwen3vl --benchmark docvqa --mode none --r 0.0 \
    --subset $FS/docvqa_val.jsonl --n 5349 --max-model-len 49152 --max-num-seqs 1 \
    --chunk-size 200 --gpu-memory-utilization 0.9 \
    --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
  echo "=== $TAG rc=$? $(date -u '+%F %T') ==="
  tail -1 $OUT/$TAG.log
  python -c "
import json
d=json.load(open('$OUT/$TAG.json'))
n=len(d.get('per_sample') or []); sk=d.get('n_skipped',0)
print(f'[$TAG] n_scored={n} skipped={sk} ({100.0*sk/max(n+sk,1):.1f}%) acc={d.get(\"acc\")}')
print('[$TAG] FALLBACK: skip>15% -> report subset none 0.976 n200 in footnote' if sk>0.15*(n+sk) else '[$TAG] skip<=15% -> full anchor usable')
"
fi

# ---- (3) FastV k=2 full splits ----
hfcell r2_qwen3vl_fastv_k2_textvqa_r0.75_full5000 textvqa $FS/textvqa_val.jsonl 5000 2 25200
hfcell r2_qwen3vl_fastv_k2_gqa_r0.75_full12578    gqa     $FS/gqa_testdev.jsonl 12578 2 43200

# ---- official rescore ----
echo "=== R2 official rescore $(date -u '+%F %T') ==="
python - <<'EOF'
import json, glob, sys
sys.path.insert(0, 'src/v3_premerger')
import official_scorers as S
rows = []
for f in sorted(glob.glob('runs/r2_same_scope/r2_*.json')):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f.split('/')[-1], 'UNREADABLE', e); continue
    ps = d.get('per_sample') or []
    b = d.get('benchmark'); n = len(ps)
    if not n:
        print(f.split('/')[-1], 'EMPTY'); continue
    preds = [str(p.get('answer', '')) for p in ps]
    gts = [str(p.get('gt', '')) for p in ps]
    if b == 'textvqa':
        off = sum(S.score_textvqa_vqaacc(a, g) for a, g in zip(preds, gts)) / n; m = 'VQA-acc'
    elif b == 'docvqa':
        off = sum(S.score_docvqa_anls(a, g) for a, g in zip(preds, gts)) / n; m = 'ANLS'
    elif b == 'gqa':
        off = S.score_gqa_batch(preds, gts)['acc']; m = 'acc'
    else:
        off = None; m = '?'
    rows.append({'file': f.split('/')[-1], 'bench': b, 'mode': d.get('mode'),
                 'fastv_k': (d.get('cfg') or {}).get('fastv_k') if isinstance(d.get('cfg'), dict) else d.get('fastv_k'),
                 'r': d.get('r'), 'official': round(off, 4) if isinstance(off, float) else off,
                 'metric': m, 'n': n, 'skipped': d.get('n_skipped'),
                 'mean_ptid_len': d.get('mean_ptid_len')})
    print(f"{f.split('/')[-1]} | {m}={off:.4f} n={n} skip={d.get('n_skipped')} ptid={d.get('mean_ptid_len')}")
json.dump(rows, open('runs/r2_same_scope/r2_official_summary.json', 'w'),
          indent=1, ensure_ascii=False)
print('wrote runs/r2_same_scope/r2_official_summary.json')
EOF
echo "[R2 done] $(date -u '+%F %T')"
