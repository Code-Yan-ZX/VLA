#!/bin/bash
# Score the Q10 full-5000 TextVQA curve cells vs the full None reference,
# plus keep .25 full cells already scored in P1 (not repeated here).
set -u
cd "$(git rev-parse --show-toplevel)"
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
D=results/cvm2027_supplement_20261026/predictions
S=results/cvm2027_supplement_20261026/plan/scores
NONE=$D/p1_none_textvqa_full.json
i=80
declare -A MAP=([rb]=q10_pre_textvqa5k_keep [fv]=q10_fastv_textvqa5k_keep [hl]=q10_hilo_textvqa5k_keep)
for M in rb fv hl; do
  for K in 0.50 0.75; do
    F=$D/${MAP[$M]}$K.json
    if [ -f "$F" ] && [ ! -f "$S/full_q10_${M}_$K.json" ]; then
      $P scripts/cvm24h_score.py textvqa "$F" $NONE --seed-int $i \
         --out $S/full_q10_${M}_$K.json > $S/full_q10_${M}_$K.log 2>&1 || echo FAIL $M $K
    fi
    i=$((i+1))
  done
done
echo "Q10_SCORED $(date +%T)" > results/cvm2027_supplement_20261026/logs/q10_scored.flag
