#!/bin/bash
# Qwen3-VL-8B EFFICIENCY REPEATABILITY (P1-3).  No accuracy claim here -- this
# gate quantifies the THROUGHPUT / latency COST of pre- vs post-merger pruning
# at iso-batch, to decide whether the paper's "stage-neutral <=3% efficiency"
# claim survives a 5-rep mean+-std measurement (the claim is only retained if
# the pre/post req/s gap is within +-3% AND the repeat-variance noise floor
# supports that conclusion).
#
# Protocol (per the P1-3 plan):
#   * Qwen3-VL-8B-Instruct, TextVQA n=200, GREEDY (temp=0, max_tokens=32),
#     selector l2, subset eval/subsets/textvqa_200.jsonl.
#   * Modes: none (unpruned anchor) / pre / post.
#   * Keep ratios 75/50/25%  <=>  prune r = 0.25 / 0.50 / 0.75.
#   * ISO-BATCH across every cell: --max-num-seqs 16 --chunk-size 250
#     --max-model-len 8192 --max-num-batched-tokens 32768
#     --gpu-memory-utilization 0.9  (== qwen3_prefinal_control.sh STD config,
#     the Qwen3 pre-final control, for cross-gate consistency).
#   * The runner runs ONE config per invocation.  For each config we do
#     1 warm-up invocation (discard timing -- primes the OS page cache so the
#     model-weight load_s is stable) then 5 INDEPENDENT timed invocations,
#     each writing its own JSON.  The runner ALSO does an internal single-
#     request eager-kernel warmup before its own timed window (runner L3426),
#     so each rep's wall_s already excludes in-process kernel priming.
#
# none is r-INVARIANT: the runner forces r=0.0 for --mode none (L3178-3179) and
# patch_processor is a no-op at r=0 -> none's efficiency does not depend on r.
# We therefore run none ONCE (5 reps) instead of 3x per r; the 3x would be
# pure redundant compute.  pre/post ARE r-dependent and run for each r.
#
# Efficiency fields the runner emits per run (confirmed via grep of
# v3_premerger_runner.py L3538-3541 + per_sample L3480-3485):
#   req_per_s        requests/sec  (n_scored / wall_s)
#   wall_s           generation wall time (excludes model load; includes the
#                    internal 1-request warmup? NO -- wall is t0..t_end around
#                    the n=200 batch only, AFTER the internal warmup fwd)
#   load_s           model LOAD time (vLLM engine construct + weight load), NOT
#                    prefill latency.  Reported as the load-cost proxy; there is
#                    NO isolated prefill-latency field in the runner (GAP -- see
#                    digest).  Per-request latency is approximated as
#                    wall_s / n_answered in the rescore step.
#   mean_ptid_len    mean INPUT prompt-token length across the 200 samples
#                    (sum of per-sample prompt_token_ids / n).
#   n_answered       count of non-empty answers (== n_scored for req/s).
#   per_sample[i].gen_len  OUTPUT token count per sample (len(out0.token_ids)).
#
# Decision rule (rescore): for each r in {0.25,0.50,0.75}, compare pre vs post
# req_per_s.  rel_diff = 100*(pre-post)/mean(pre,post).  The repeat-std (larger
# of pre/post req/s std across 5 reps, as % of mean) is the NOISE FLOOR.
#   |rel_diff| <= 3%  -> STAGE-NEUTRAL; if noise_floor >= |rel_diff| the diff is
#     indistinguishable from noise -> flagged inconclusive (claim not retained).
#   |rel_diff| >  3%  -> NOT stage-neutral; if noise_floor >= 3% the measurement
#     is too noisy to conclude either way.
# Only retain the paper's "stage-neutral <=3%" claim when, for every r, the
# verdict is STAGE-NEUTRAL with noise_floor < |rel_diff| (or noise_floor < 3%
# and |rel_diff| <= 3%).
#
# Usage: bash src/v3_premerger/qwen3_efficiency_repeatability.sh
# Outputs -> runs/qwen3_efficiency/{mode}_r{r}_{warmup,rep0..4}.json + .log
#           + runs/qwen3_efficiency/efficiency_summary.json
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_MODEL_SCOPE=False

OUT=runs/qwen3_efficiency
mkdir -p $OUT
FAM=qwen3vl
BENCH=textvqa
SUBSET=eval/subsets/textvqa_200.jsonl
N=200
MAXTOK=32
N_REP=5
RS="0.25 0.50 0.75"     # prune ratios  <=> keep 75/50/25%
# ISO-BATCH (== qwen3_prefinal_control.sh STD; Qwen3 pre-final control config)
STD="--max-num-seqs 16 --chunk-size 250 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"

# ---- wait for an idle GPU (>= 40000 MiB free) ----
echo "[efficiency] $(date -u '+%F %T') modes='none pre post' r='$RS' waiting >= 40000 MiB free"
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[efficiency] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[efficiency][ABORT] GPU busy after wait"; exit 1; }

# run_cell mode r tag  -- one runner invocation (one fresh vLLM process).
run_cell(){ # mode r tag
  timeout 5400 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM \
    --benchmark $BENCH --subset $SUBSET --n $N --r $2 --mode $1 --selector l2 \
    --max-tokens $MAXTOK $STD --out $OUT/$3.json > $OUT/$3.log 2>&1
  tail -2 $OUT/$3.log
}
skip_ratio(){ $PY -c "
import json
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
# ensure tag mode r  -- run if missing or high-skip; else skip (resumable).
ensure(){ # tag mode r
  local TAG=$1 MODE=$2 R=$3
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return 0
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N mode=$MODE r=$R (greedy temp=0) ==="
  run_cell $MODE "$R" $TAG
  echo "=== $TAG done (skip=$(skip_ratio $TAG)) ==="
}
# config mode r  -- 1 warm-up (discard) + N_REP independent timed reps.
config(){ # mode r
  local MODE=$1 R=$2
  local RTAG; RTAG=$(printf "r%.2f" "$R")
  ensure "${MODE}_${RTAG}_warmup" "$MODE" "$R"          # discard timing
  for i in $(seq 0 $((N_REP-1))); do
    ensure "$(printf '%s_%s_rep%d' "$MODE" "$RTAG" "$i")" "$MODE" "$R"
  done
}

# none is r-invariant (runner forces r=0.0) -> run once.
config none 0.0
for R in $RS; do
  config pre  "$R"
  config post "$R"
done

echo "=== efficiency rescore + +-3% stage-neutral decision (CPU) ==="
$PY - <<'PYEOF'
import json, glob, os, math, statistics as st
OUT='runs/qwen3_efficiency'
N_REP=5
# (mode, r) cells. none is r-invariant -> single cell at r=0.00.
CONFIGS=[('none',0.00),('pre',0.25),('pre',0.50),('pre',0.75),
         ('post',0.25),('post',0.50),('post',0.75)]

def load_reps(mode, r):
    reps=[]
    for f in sorted(glob.glob(os.path.join(OUT, f'{mode}_r{r:.2f}_rep*.json'))):
        try:
            d=json.load(open(f))
        except Exception:
            continue
        ps=d.get('per_sample') or [1]
        if (d.get('n_skipped') or 0) > 0.10*max(len(ps),1):
            continue                       # drop high-skip reps
        reps.append(d)
    return reps

def stats(vals):
    vals=[float(v) for v in vals if v is not None]
    if not vals: return {'n':0,'mean':0.0,'std':0.0}
    m=sum(vals)/len(vals)
    sd=st.stdev(vals) if len(vals)>1 else 0.0
    return {'n':len(vals),'mean':round(m,4),'std':round(sd,4)}

summary={'gate':'qwen3_efficiency_repeatability',
         'protocol':'greedy temp=0, max_tokens=32',
         'model':'Qwen/Qwen3-VL-8B-Instruct','benchmark':'textvqa','n':200,
         'batch':{'max_num_seqs':16,'chunk_size':250,'max_model_len':8192,
                  'max_num_batched_tokens':32768,'gpu_memory_utilization':0.9},
         'n_reps_target':N_REP,'configs':{}}

print('=== QWEN3 EFFICIENCY REPEATABILITY (mean+-std over reps) ===')
hdr=f"{'config':<16} {'n':>3} {'req/s':>16} {'wall_s':>16} {'load_s':>16} {'per-req_s':>16} {'in_tok':>9} {'out_tok':>16}"
print(hdr); print('-'*len(hdr))
for mode,r in CONFIGS:
    reps=load_reps(mode,r)
    key=f'{mode}_r{r:.2f}'
    if not reps:
        print(f"{key:<16}  MISSING"); continue
    req=stats([d.get('req_per_s',0) for d in reps])
    wall=stats([d.get('wall_s',0) for d in reps])
    load=stats([d.get('load_s',0) for d in reps])
    ptid=stats([d.get('mean_ptid_len',0) for d in reps])
    # per-request latency proxy = wall_s / n_answered (prefill+decode per req).
    prel=stats([d.get('wall_s',0)/max(d.get('n_answered',1),1) for d in reps])
    # output tokens = sum of per-sample gen_len (only answered samples).
    out_tok=stats([sum(p.get('gen_len',0) for p in (d.get('per_sample') or [])
                       if not p.get('skipped')) for d in reps])
    summary['configs'][key]={'mode':mode,'r':r,'n_reps':req['n'],
        'req_per_s':req,'wall_s':wall,'load_s':load,
        'mean_ptid_len':ptid,'per_request_s':prel,'output_tokens':out_tok,
        'note':'load_s=model-load time, NOT prefill (no prefill-latency field)'}
    f=lambda s: f"{s['mean']:.3f}+-{s['std']:.3f}"
    print(f"{key:<16} {req['n']:>3} {f(req):>16} {f(wall):>16} {f(load):>16} "
          f"{f(prel):>16} {ptid['mean']:>9.1f} {f(out_tok):>16}")

# ---- decision rule: pre vs post req/s per r (+-3% stage-neutral, repeat-std = noise floor) ----
print('\n=== DECISION RULE: pre vs post req/s per r  (stage-neutral <=3%; noise floor = repeat-std) ===')
decisions=[]
for r in (0.25,0.50,0.75):
    pre=summary['configs'].get(f'pre_r{r:.2f}')
    post=summary['configs'].get(f'post_r{r:.2f}')
    if not pre or not post or pre['n_reps']<1 or post['n_reps']<1:
        print(f"  r={r:.2f}: MISSING pre or post reps -> cannot decide")
        decisions.append({'r':r,'keep_frac':round(1-r,2),'verdict':'INCONCLUSIVE (missing reps)'})
        continue
    pm,ps=pre['req_per_s']['mean'],pre['req_per_s']['std']
    qm,qs=post['req_per_s']['mean'],post['req_per_s']['std']
    mid=(pm+qm)/2.0 if (pm+qm)>0 else 1.0
    rel=100.0*(pm-qm)/mid                      # + => pre faster than post
    noise=100.0*max(ps,qs)/mid                 # larger repeat-std as % of mean
    np_,nq=pre['n_reps'],post['n_reps']
    ci=1.96*math.sqrt(ps**2/np_+qs**2/nq)/mid*100 if min(np_,nq)>1 else float('inf')
    within3=abs(rel)<=3.0
    noise_swallows=noise>=abs(rel)
    if within3:
        verdict='STAGE-NEUTRAL (|diff|<=3%)'
        if noise_swallows:
            verdict+=' BUT within noise floor -> inconclusive (claim NOT retained)'
        else:
            verdict+=' AND above noise floor -> claim RETAINED for this r'
    else:
        verdict='NOT stage-neutral (|diff|>3%)'
        if noise>=3.0:
            verdict+='; noise floor >=3% (measurement too noisy to conclude)'
    print(f"  r={r:.2f} (keep {100*(1-r):.0f}%): pre={pm:.3f}+-{ps:.3f}  "
          f"post={qm:.3f}+-{qs:.3f}  rel_diff={rel:+.2f}%  noise_floor={noise:.2f}%  "
          f"ci95_diff={ci:.2f}%  -> {verdict}")
    decisions.append({'r':r,'keep_frac':round(1-r,2),
        'pre_req_per_s_mean':pm,'pre_req_per_s_std':ps,
        'post_req_per_s_mean':qm,'post_req_per_s_std':qs,
        'rel_diff_pct':round(rel,3),'noise_floor_pct':round(noise,3),
        'ci95_diff_pct':round(ci,3),'verdict':verdict})

# overall: retain the <=3% stage-neutral claim only if EVERY r is STAGE-NEUTRAL
# AND above its noise floor.
retain=all(d['verdict'].startswith('STAGE-NEUTRAL') and 'claim RETAINED' in d['verdict']
           for d in decisions) if decisions else False
overall=('RETAIN the stage-neutral <=3% efficiency claim (all r stage-neutral & above noise floor)'
         if retain else
         'DO NOT RETAIN the stage-neutral <=3% claim as-is (>=1 r fails or is inconclusive) -> flag for review')
print(f"\n=== OVERALL: {overall} ===")
summary['decision_rule']=decisions
summary['overall']=overall
summary['claim_retained']=retain
with open(os.path.join(OUT,'efficiency_summary.json'),'w') as fh:
    json.dump(summary,fh,indent=2)
print('=== summary written to runs/qwen3_efficiency/efficiency_summary.json ===')
PYEOF
echo "=== QWEN3 EFFICIENCY REPEATABILITY DONE ==="
