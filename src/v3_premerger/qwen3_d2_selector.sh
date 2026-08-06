#!/bin/bash
# Qwen3-VL D2 "merger-loss-aware" pre-merger SELECTOR gate.
#
# Hypothesis (mechanism-derived, drafts/v3_merger_aware_design.md / mechanism
# results): the lossy 2x2 averaging merger DESTROYS high-frequency/text units
# (averaging flattens spatial structure). A selector that scores pre-merger
# units by how much the merger would destroy them -- Sobel-style spatial
# gradient ("edge") or within-unit feature variance ("var") -- should therefore
# preferentially PRESERVE text/edge units, beating the magnitude-only L2
# baseline on text-dense benchmarks (TextVQA/DocVQA) at iso-token budget.
#
# All cells: --mode pre (rank BEFORE the 2x2 main merger), r=0.75 (keep 25% of
# merge-units), greedy temp=0, max_tokens=32, n=200. The ONLY varied factor is
# --selector in {l2, edge, var}. l2 = the established v3 baseline (mean L2-norm
# of the unit's 4 patch vectors). edge = Sobel-style spatial gradient across
# the 2x2 unit's feature quadrants (high-frequency proxy). var = within-unit
# feature variance across the 4 patches (omnidirectional merger-loss proxy).
# Both edge/var are computed on the SAME pre-merger feature tensor L2 uses.
#
# ISO-TOKEN CONTRACT: k_i = round(full_i*(1-r)) per image is INDEPENDENT of the
# selector (top-k keeps k_i units regardless of how they are scored). The
# rescore step VERIFIES prompt_token_ids match across selectors per sample
# (edge/var vs l2) -- a selector change must NOT alter the token budget.
#
# DECISION RULE (D2):
#   If edge OR var beats l2 by >= 2pp on TextVQA OR DocVQA
#       -> MECHANISM-DERIVED method: the merger-loss-aware selector is a real
#          improvement; write into paper as a mechanism-derived selector.
#   If |delta| < 2pp on TextVQA AND DocVQA (ties l2)
#       -> PRINCIPLED INSTANTIATION: edge/var is the selector the mechanism
#          PREDICTS (preserve high-loss units); it ties L2, confirming the
#          mechanism's prediction that L2 already approximates "preserve
#          high-information units". Still a valid mechanism-grounded result.
#   If edge/var is WORSE than l2 by >= 2pp
#       -> the merger-loss-aware proxy does NOT capture what L2 captures; flag.
#
# Usage: bash src/v3_premerger/qwen3_d2_selector.sh
# Outputs -> runs/qwen3_d2_selector/
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

OUT=runs/qwen3_d2_selector
mkdir -p $OUT
FAM=qwen3vl
R=0.75
N=200
MAXTOK=32
SELECTORS="l2 edge var"
BENCHES="textvqa docvqa gqa"

# ---- wait for an idle GPU (>= 40000 MiB free) ----
echo "[d2] $(date -u '+%F %T') selectors='$SELECTORS' benches='$BENCHES' waiting >= 40000 MiB free"
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[d2] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[d2][ABORT] GPU busy after wait"; exit 1; }

STD="--max-num-seqs 16 --chunk-size 250 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --chunk-size 250 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.85"
declare -A JSONL=( [textvqa]=eval/subsets/textvqa_200.jsonl
                   [docvqa]=eval/subsets/docvqa_200.jsonl
                   [gqa]=eval/subsets/gqa_200.jsonl )

run_cell(){ # bench selector r "flags" tag
  timeout 5400 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $1 \
    --subset ${JSONL[$1]} --n $N --r $3 --mode pre --selector $2 --max-tokens $MAXTOK \
    $4 --out $OUT/$5.json > $OUT/$5.log 2>&1
  tail -2 $OUT/$5.log
}
skip_ratio(){ $PY -c "
import json
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
cell(){ # selector bench
  local SEL=$1 B=$2
  local FLAGS=$STD; [ "$B" = "docvqa" ] && FLAGS=$DOC
  local TAG=$(printf "qwen3_pre_%s_%s_r%.3f_n%d" "$B" "$SEL" "$R" "$N")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N (greedy temp=0, mode=pre, selector=$SEL) ==="
  run_cell $B $SEL "$R" "$FLAGS" $TAG
  echo "=== $TAG done (skip=$(skip_ratio $TAG)) ==="
}

for SEL in $SELECTORS; do
  for B in $BENCHES; do cell $SEL $B; done
done

echo "=== D2 rescore + iso-token verify + paired delta (CPU) ==="
$PY - <<'PYEOF'
import json, glob, os, sys, statistics as st
sys.path.insert(0,'src')
from v3_premerger.official_scorers import score_textvqa_vqaacc, score_docvqa_anls, score_gqa
OUT='runs/qwen3_d2_selector'
SCORERS={'textvqa':score_textvqa_vqaacc,'docvqa':score_docvqa_anls,'gqa':score_gqa}
SELECTORS=['l2','edge','var']
BENCHES=['textvqa','docvqa','gqa']
R=0.75; N=200

def tag(b,sel): return f"qwen3_pre_{b}_{sel}_r{R:.3f}_n{N}"

# ---- per (benchmark x selector) official score ----
rows=[]
for sel in SELECTORS:
  for b in BENCHES:
    f=os.path.join(OUT,tag(b,sel)+'.json')
    if not os.path.exists(f):
        print(f"[rescore] MISSING {tag(b,sel)}"); continue
    d=json.load(open(f)); ps=d.get('per_sample') or []
    sc=SCORERS[b]; vals=[]; n_skip=0
    for p in ps:
        if p.get('skipped'): n_skip+=1; continue
        vals.append(float(sc(p.get('answer',''),str(p.get('gt','')))))
    n_ans=len(vals)
    off=sum(vals)/n_ans if n_ans else 0.0
    rows.append({'tag':tag(b,sel),'benchmark':b,'selector':sel,
                 'n':len(ps),'n_skipped':n_skip,'n_answered':n_ans,
                 'official_score':round(off,4)})
    print(f"[rescore] {b:8s} {sel:5s} off={off:.4f} n_ans={n_ans} skip={n_skip}")

def load_ps(bench, sel):
    f=os.path.join(OUT,tag(bench,sel)+'.json')
    if not os.path.exists(f): return {}
    d=json.load(open(f))
    return {p['id']:p for p in (d.get('per_sample') or [])}

# ---- iso-token verification: edge/var vs l2 prompt_token_ids match ----
print('\n=== ISO-TOKEN CHECK (selector vs l2 prompt_token_ids) ===')
iso_rows=[]
for b in BENCHES:
    base=load_ps(b,'l2')
    for sel in ('edge','var'):
        other=load_ps(b,sel)
        ids=sorted(set(base)&set(other))
        if not ids: print(f"{b:8s} {sel}: no paired samples"); continue
        match=0; mismatch=0
        for i in ids:
            if base[i].get('skipped') or other[i].get('skipped'): continue
            pt_b=base[i].get('prompt_token_ids',0); pt_o=other[i].get('prompt_token_ids',0)
            if pt_b==pt_o: match+=1
            else: mismatch+=1
        status='PASS' if mismatch==0 else 'FAIL'
        iso_rows.append({'benchmark':b,'selector':sel,'n_paired':match+mismatch,
                         'match':match,'mismatch':mismatch})
        print(f"{b:8s} {sel}-l2 {status}  match={match} mismatch={mismatch}"
              f"{' (iso-token confirmed)' if mismatch==0 else ' (ISO-TOKEN VIOLATION!)'}")

# ---- paired delta (edge-l2, var-l2) ----
print('\n=== PAIRED DELTA (selector - l2) ===')
delta_rows=[]
for b in BENCHES:
    base=load_ps(b,'l2')
    for sel in ('edge','var'):
        other=load_ps(b,sel)
        ids=sorted(set(base)&set(other))
        sc=SCORERS[b]
        diffs=[float(sc(other[i].get('answer',''),str(other[i].get('gt',''))))
               -float(sc(base[i].get('answer',''),str(base[i].get('gt',''))))
               for i in ids if not base[i].get('skipped') and not other[i].get('skipped')]
        mean=st.mean(diffs) if diffs else 0.0
        sd=st.stdev(diffs) if len(diffs)>1 else 0.0
        se=sd/(len(diffs)**0.5) if diffs else 0.0
        d_pp=round(mean*100,2)
        delta_rows.append({'benchmark':b,'selector':sel,'n_paired':len(diffs),
                           'mean_delta_pp':d_pp,'stderr_pp':round(se*100,3),
                           'mean_delta':round(mean,4)})
        print(f"{b:8s} {sel}-l2: d={d_pp}pp  paired(n={len(diffs)}) "
              f"meanD={mean:.4f}+-{se:.4f}")

# ---- decision rule ----
print('\n=== DECISION RULE (threshold = 2pp on TextVQA/DocVQA) ===')
THRESH=2.0
verdicts=[]
for b in ('textvqa','docvqa'):
    for dr in delta_rows:
        if dr['benchmark']!=b: continue
        d=dr['mean_delta_pp']; sel=dr['selector']
        if d>=THRESH:
            v=f"{b:8s} {sel}-l2={d}pp >= {THRESH}pp -> MECHANISM-DERIVED: {sel} " \
              f"beats l2 on {b}. Write into paper."
        elif d<=-THRESH:
            v=f"{b:8s} {sel}-l2={d}pp <= -{THRESH}pp -> WORSE than l2. {sel} is " \
              f"NOT the right proxy; flag (mechanism prediction not confirmed)."
        else:
            v=f"{b:8s} {sel}-l2={d}pp (|d|<{THRESH}pp) -> PRINCIPLED INSTANTIATION: " \
              f"{sel} ties l2 (the selector the mechanism predicts)."
        print(f"  {v}"); verdicts.append(v)
# aggregate verdict across TextVQA/DocVQA
best=max((dr['mean_delta_pp'] for dr in delta_rows
          if dr['benchmark'] in ('textvqa','docvqa')), default=-999)
if best>=THRESH:
    print(f"\n  AGGREGATE: best delta {best}pp >= {THRESH}pp -> MECHANISM-DERIVED method.")
else:
    print(f"\n  AGGREGATE: best delta {best}pp < {THRESH}pp -> PRINCIPLED INSTANTIATION "
          f"(edge/var tie l2; the mechanism-predicted selector).")

summary={'gate':'qwen3_d2_merger_loss_aware_selector','protocol':'greedy temp=0',
         'r':R,'keep_frac':1.0-R,'n':N,'selectors':SELECTORS,'benches':BENCHES,
         'model':'Qwen/Qwen3-VL-8B-Instruct','rows':rows,'paired_delta':delta_rows,
         'iso_token':iso_rows,'threshold_pp':THRESH}
with open(os.path.join(OUT,'d2_selector_summary.json'),'w') as fh: json.dump(summary,fh,indent=2)
print("\n=== summary -> runs/qwen3_d2_selector/d2_selector_summary.json ===")
PYEOF
echo "=== D2 SELECTOR GATE DONE ==="
