#!/bin/bash
# Autonomous GPU chain (runs without API): D2 rerun (missing cells) -> P1-1 -> P1-3 -> GLM.
# Each sub-script has its own GPU-wait loop (>=40000 MiB free) + idempotent skip,
# so they serialize cleanly and resume on OOM/interruption.
set -u
cd /media/disk2/YZX/research/vla
echo "=== CHAIN START $(date -u '+%F %T') ==="
echo "### [1/4] D2 selector (rerun missing textvqa edge/var + gqa cells) ###"
bash src/v3_premerger/qwen3_d2_selector.sh
echo "### [2/4] P1-1 InternVL3 ranking-swap ###"
bash src/v3_premerger/internvl3_swap_control.sh
echo "### [3/4] P1-3 efficiency repeatability ###"
bash src/v3_premerger/qwen3_efficiency_repeatability.sh
echo "### [4/4] GLM pre/post (seed 0) ###"
bash src/v3_premerger/glm4v_sampling_gate.sh 0 pre post
echo "=== CHAIN DONE $(date -u '+%F %T') ==="