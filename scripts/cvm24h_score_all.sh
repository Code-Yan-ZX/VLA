#!/bin/bash
# Score all supplement cells/pairs using the pinned official metric driver.
# Run AFTER GPU queue. CPU only. Seeds: fixed 20261200+idx (driver).
set -u
cd "$(git rev-parse --show-toplevel)"
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
D=results/cvm2027_supplement_20261026/predictions
Q=results/cvm2027_supplement_20261026/plan
S=$Q/scores; mkdir -p $S
EXPB=/media/disk2/YZX/research/vla/results/dcc_presubmit/expB
P1DIR=/media/disk2/YZX/research/vla/results/acmmm_final_controls/p1

# --- build subset slices (same IDs across budgets; directive 11/30) ---
$P scripts/cvm24h_extract_slice.py $EXPB/full_textvqa_rbm_pre_r25.json $Q/p2_textvqa_1000_ids.txt $D/c25_rb_textvqa1k_keep0.25.json
$P scripts/cvm24h_extract_slice.py $EXPB/full_textvqa_fastv_k3.json       $Q/p2_textvqa_1000_ids.txt $D/c25_fv_textvqa1k_keep0.25.json
$P scripts/cvm24h_extract_slice.py $D/p1_hilo_textvqa_full.json           $Q/p2_textvqa_1000_ids.txt $D/c25_hl_textvqa1k_keep0.25.json
$P scripts/cvm24h_extract_slice.py $D/p1_none_textvqa_full.json           $Q/p2_textvqa_1000_ids.txt $D/c25_none_textvqa1k.json
cp $P1DIR/p1_qwen3vl_pre_ocrbench_r25_full.json  $D/c25_rb_ocrbench_keep0.25.json
cp $P1DIR/p1_qwen3vl_fastv_ocrbench_k3_full.json $D/c25_fv_ocrbench_keep0.25.json
cp $D/p1_hilo_ocrbench_full.json                 $D/c25_hl_ocrbench_keep0.25.json
cp $D/p1_none_ocrbench_full.json                 $D/c25_none_ocrbench.json

sc(){ # name bench a b ids seed
  $P scripts/cvm24h_score.py "$2" "$3" "$4" ${5:+--ids $5} --seed-int "$6" --out $S/"$1".json > $S/"$1".log 2>&1 || echo "[score-fail] $1"
}
# --- P1 main table keep .25 (full denominators) ---
sc p1_tv_rb_fv   textvqa $EXPB/full_textvqa_rbm_pre_r25.json $EXPB/full_textvqa_fastv_k3.json "" 1
sc p1_tv_rb_none textvqa $EXPB/full_textvqa_rbm_pre_r25.json $D/p1_none_textvqa_full.json "" 2
sc p1_tv_fv_none textvqa $EXPB/full_textvqa_fastv_k3.json $D/p1_none_textvqa_full.json "" 3
sc p1_tv_hl_none textvqa $D/p1_hilo_textvqa_full.json $D/p1_none_textvqa_full.json "" 4
sc p1_tv_hl_rb   textvqa $D/p1_hilo_textvqa_full.json $EXPB/full_textvqa_rbm_pre_r25.json "" 5
sc p1_tv_hl_fv   textvqa $D/p1_hilo_textvqa_full.json $EXPB/full_textvqa_fastv_k3.json "" 6
sc p1_oc_rb_fv   ocrbench $D/c25_rb_ocrbench_keep0.25.json $D/c25_fv_ocrbench_keep0.25.json "" 7
sc p1_oc_rb_none ocrbench $D/c25_rb_ocrbench_keep0.25.json $D/p1_none_ocrbench_full.json "" 8
sc p1_oc_fv_none ocrbench $D/c25_fv_ocrbench_keep0.25.json $D/p1_none_ocrbench_full.json "" 9
sc p1_oc_hl_none ocrbench $D/c25_hl_ocrbench_keep0.25.json $D/p1_none_ocrbench_full.json "" 10
sc p1_oc_hl_rb   ocrbench $D/c25_hl_ocrbench_keep0.25.json $D/c25_rb_ocrbench_keep0.25.json "" 11
sc p1_oc_hl_fv   ocrbench $D/c25_hl_ocrbench_keep0.25.json $D/c25_fv_ocrbench_keep0.25.json "" 12
# --- P2 curves vs None (same ID sets) ---
i=20
for K in 0.25 0.50 0.75; do
  for M in rb fv hl; do
    case $M in rb) FULL=pre;; fv) FULL=fastv;; hl) FULL=hilo;; esac
    if [ "$K" = "0.25" ]; then
      TVF=$D/c25_${M}_textvqa1k_keep0.25.json; OCF=$D/c25_${M}_ocrbench_keep0.25.json
    else
      TVF=$D/c25_${FULL}_textvqa1k_keep$K.json; OCF=$D/c25_${FULL}_ocrbench_keep$K.json
    fi
    sc "curve_tv_${M}_$K" textvqa $TVF $D/c25_none_textvqa1k.json "" $i; i=$((i+1))
    sc "curve_oc_${M}_$K" ocrbench $OCF $D/c25_none_ocrbench.json "" $i; i=$((i+1))
  done
done
# --- DocVQA 500 ---
for M in pre fastv hilo; do
  case $M in pre) A=$D/doc500_pre25.json;; fastv) A=$D/doc500_fastv25.json;; hilo) A=$D/doc500_hilo25.json;; esac
  sc doc_${M} docvqa $A $D/doc500_none.json "" $((i++))
done
echo "[score-all] done; results in $S"
