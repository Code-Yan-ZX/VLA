#!/bin/bash
# Score the 9 OCRBench budget-curve cells early (CPU; idempotent - handoff skips
# already-scored files). None@100% is the curve reference on identical IDs.
set -u
cd "$(git rev-parse --show-toplevel)"
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
D=results/cvm2027_supplement_20261026/predictions
S=results/cvm2027_supplement_20261026/plan/scores
P1=/media/disk2/YZX/research/vla/results/acmmm_final_controls/p1
mkdir -p $S
cp -f $P1/p1_qwen3vl_pre_ocrbench_r25_full.json $D/c25_rb_ocrbench_keep0.25.json
cp -f $P1/p1_qwen3vl_fastv_ocrbench_k3_full.json $D/c25_fv_ocrbench_keep0.25.json
cp -f $D/p1_hilo_ocrbench_full.json $D/c25_hl_ocrbench_keep0.25.json
i=40
for K in 0.25 0.50 0.75; do
  for M in rb fv hl; do
    if [ "$K" = "0.25" ]; then F=$D/c25_${M}_ocrbench_keep0.25.json
    else
      case $M in
        rb) F=$D/c25_pre_ocrbench_keep$K.json;;
        fv) F=$D/c25_fastv_ocrbench_keep$K.json;;
        hl) F=$D/c25_hilo_ocrbench_keep$K.json;;
      esac
    fi
    if [ -f "$F" ] && [ ! -f "$S/curve_oc_${M}_$K.json" ]; then
      $P scripts/cvm24h_score.py ocrbench "$F" $D/p1_none_ocrbench_full.json \
         --seed-int $i --out $S/curve_oc_${M}_$K.json > $S/curve_oc_${M}_$K.log 2>&1 || echo "FAIL oc $M $K"
    fi
    i=$((i+1))
  done
done
echo "OC_CURVE_SCORED $(date +%T)"
