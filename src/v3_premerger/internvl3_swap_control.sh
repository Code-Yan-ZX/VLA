#!/usr/bin/env bash
# InternVL3 M3 ranking-swap + kept-set-identity mechanism control (P1-1).
#
# GENERALIZES the paper's M3 causal claim (Qwen-only so far) to a 3rd family.
# The M3 claim: the RESULT of pruning is determined by WHICH merge-units are
# kept (the RANKING), NOT by the forward path used to apply it. The control:
#   --mode post --mask-ranking swap  =  run the FULL POST forward path
#     (_process_vision_input -> extract_feature -> pixel_shuffle -> mlp1 on ALL
#     units, numerically untouched) but SELECT the kept units with the PRE
#     ranking (per-tile top-k on the pixel-shuffle unit L2 scores, captured via
#     an mlp1 forward pre_hook -- bitwise the same pre-merger representation
#     "pre" mode scores). Because pixel_shuffle+mlp1 have NO cross-unit
#     receptive field, a kept unit's merged token is IDENTICAL at either stage,
#     so swap must reproduce PRE-standard accuracy => the pre>post gap is 100% a
#     RANKING effect, forward path held constant.
#
# 3 arms (greedy temp=0, max_tokens=32, r=0.75 keep=25%, selector l2, n=200):
#   pre   stage ranking on the pixel-shuffle units (rank-before-merge)
#   post  stage ranking on the merged tokens (rank-after-merge)
#   swap  POST forward path + PRE ranking selection (the M3 cross)
# pre and swap also pass --save-unit-scores so the runner attaches per-image
# kept_indices (R1-1) in the SAME [0, n_tiles*U) space -> Jaccard(pre,swap).
#
# DECISION RULE (P1-1):
#   If swap ~= pre (official score recovered) AND Jaccard(pre,swap) ~= 1 AND
#   answer-identity(pre,swap) high  ->  mechanism GENERALIZES to InternVL3
#   (architecture-general claim OK; write into paper as 3rd-family replication).
#   If swap ~= post (pre ranking did NOT recover pre) OR Jaccard low  ->  the
#   mechanism does NOT generalize -> claim LIMITED to Qwen. FLAG, do NOT
#   self-package -- main window escalates.
#
# Usage: bash src/v3_premerger/internvl3_swap_control.sh
# Outputs -> runs/internvl3_swap_control/
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

OUT=runs/internvl3_swap_control
mkdir -p $OUT
FAM=internvl3
R=0.75
N=200
MAXTOK=32
MODES="pre post swap"   # swap = post + --mask-ranking swap

# ---- wait for an idle GPU (>= 40000 MiB free) ----
echo "[internvl3-swap] $(date -u '+%F %T') modes='$MODES' waiting >= 40000 MiB free"
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[internvl3-swap] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[internvl3-swap][ABORT] GPU busy after wait"; exit 1; }

# Canonical internvl3 config (matches runs/internvl3_*): max_num_seqs=4. docvqa
# big-document safety: max-model-len 32768 + max-num-batched-tokens 32768 +
# max-pixels 4000000 (PIL pre-resize cap, since InternVLImageProcessor rejects
# the max_pixels kwarg). textvqa/gqa: max-model-len 8192, no pixel cap.
STD="--max-num-seqs 4 --chunk-size 250 --max-model-len 8192 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --chunk-size 250 --max-model-len 32768 --max-num-batched-tokens 32768 --max-pixels 4000000 --gpu-memory-utilization 0.85"
declare -A JSONL=( [textvqa]=eval/subsets/textvqa_200.jsonl
                   [docvqa]=eval/subsets/docvqa_200.jsonl
                   [gqa]=eval/subsets/gqa_200.jsonl )

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
  local TAG=$(printf "internvl3_%s_%s_r%.3f_n%d" "$MODE" "$B" "$R" "$N")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N (greedy temp=0) ==="
  local SWAPF=""
  [ "$MODE" = "swap" ] && SWAPF="--mask-ranking swap"
  # pre and swap record kept-set identity (R1-1) for Jaccard(pre,swap)
  local KEEPF=""
  { [ "$MODE" = "pre" ] || [ "$MODE" = "swap" ]; } && KEEPF="--save-unit-scores"
  run_cell $B "$MODE" "$R" "$FLAGS $SWAPF $KEEPF" $TAG
}

for B in textvqa docvqa gqa; do
  for M in $MODES; do
    cell $M $B $R
  done
done
echo "=== INTERNVL3 SWAP CELLS DONE ==="

# ---- rescore: official scores + kept-set Jaccard(pre,swap) + answer identity ----
echo "=== internvl3_swap rescore (CPU) ==="
$PY - <<'EOF'
import json, glob, os, sys, statistics as st
sys.path.insert(0,'src')
from v3_premerger.official_scorers import score_textvqa_vqaacc, score_docvqa_anls, score_gqa
SCORERS={'textvqa':score_textvqa_vqaacc,'docvqa':score_docvqa_anls,'gqa':score_gqa}
OUT='runs/internvl3_swap_control'

def load(mode,bench):
    tag=f"internvl3_{mode}_{bench}_r0.750_n200"
    p=os.path.join(OUT,tag+'.json')
    if not os.path.exists(p): return None
    return json.load(open(p))

def official(d,bench):
    if not d: return None
    sc=SCORERS[bench]; ps=d.get('per_sample') or []; vals=[]
    for p in ps:
        if p.get('skipped'): continue
        vals.append(float(sc(p.get('answer',''),str(p.get('gt','')))))
    return round(sum(vals)/len(vals),4) if vals else None

def norm(a): return ' '.join(str(a).strip().lower().split())
rows=[]
for b in ('textvqa','docvqa','gqa'):
    dpre=load('pre',b); dpost=load('post',b); dswap=load('swap',b)
    pre=official(dpre,b); post=official(dpost,b); swap=official(dswap,b)
    # kept-set Jaccard(pre,swap) + answer identity(pre,swap)
    jac=[]; ans_id=0; ans_tot=0; kept_n=0
    if dpre and dswap:
        pm={p['id']:p for p in (dpre.get('per_sample') or [])}
        sm={p['id']:p for p in (dswap.get('per_sample') or [])}
        for sid in sorted(set(pm)&set(sm)):
            pp,sp=pm[sid],sm[sid]
            if pp.get('skipped') or sp.get('skipped'): continue
            ans_tot+=1
            if norm(pp.get('answer',''))==norm(sp.get('answer','')): ans_id+=1
            ki=pp.get('kept_indices'); kj=sp.get('kept_indices')
            if ki is not None and kj is not None:
                A=set(ki); B=set(kj); u=len(A|B)
                jac.append(len(A&B)/u if u else 1.0); kept_n+=1
    mean_jac=round(st.mean(jac),4) if jac else None
    ans_id_frac=round(ans_id/ans_tot,4) if ans_tot else None
    ptid=dswap.get('mean_ptid_len') if dswap else None
    swap_fb=None
    if dswap and dswap.get('diag'):
        dg=dswap['diag']
        swap_fb={'consumed':dg.get('consumed'),'fallback':dg.get('fallback_stage'),
                 'leftover':dg.get('swap_queue_leftover',0)}
    rows.append({'benchmark':b,'pre':pre,'post':post,'swap':swap,
                 'jaccard_pre_swap':mean_jac,'n_jaccard':kept_n,
                 'answer_identity':ans_id_frac,'n_ans':ans_tot,
                 'mean_ptid':ptid,'swap_diag':swap_fb})
    print(f"[rescore] {b:8s} pre={pre} post={post} swap={swap} "
          f"Jaccard(pre,swap)={mean_jac}(n={kept_n}) "
          f"ans_id(pre==swap)={ans_id_frac}(n={ans_tot}) ptid={ptid} "
          f"swap_diag={swap_fb}")

# DECISION RULE
print('\n=== P1-1 DECISION (InternVL3 M3 ranking-swap generalization) ===')
all_ok=True
for r in rows:
    b=r['benchmark']; pre=r['pre']; post=r['post']; swap=r['swap']
    jac=r['jaccard_pre_swap']; ai=r['answer_identity']
    if pre is None or swap is None:
        print(f"  {b:8s} INCONCLUSIVE (missing pre/swap run)"); all_ok=False; continue
    d_swap_pre=round((swap-pre)*100,2)            # pp; ~0 => swap recovers pre
    recovers = (abs(swap-pre) <= 0.02)            # within 2pp
    clean   = (jac is not None and jac >= 0.999)  # Jaccard~=1 by construction
    ident   = (ai is not None and ai >= 0.95)
    verdict = "GENERALIZES" if (recovers and clean and ident) else "DOES NOT GENERALIZE"
    if verdict!="GENERALIZES": all_ok=False
    print(f"  {b:8s} swap-pre={d_swap_pre:+.2f}pp Jaccard={jac} ans_id={ai} -> {verdict}")
summary={'gate':'internvl3_m3_ranking_swap','family':'internvl3','keep_frac':0.25,
         'selector':'l2','decoding':'greedy','n':200,'rows':rows,
         'mechanism_generalizes':bool(all_ok)}
with open(os.path.join(OUT,'internvl3_swap_summary.json'),'w') as fh: json.dump(summary,fh,indent=2)
print(f"\n[mechanism_generalizes={all_ok}] summary -> {OUT}/internvl3_swap_summary.json")
if not all_ok:
    print("FLAG: mechanism did NOT cleanly generalize on >=1 benchmark -- do NOT "
          "self-package; main window escalates.")
EOF
echo "=== INTERNVL3 SWAP CONTROL DONE ==="
