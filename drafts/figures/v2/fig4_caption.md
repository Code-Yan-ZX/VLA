## Figure 4 — Architecture-conditional amplification (honest boundary)

**r75/r0 served-req/s speedup vs concurrency (log x), two architectures.**
LLaVA-1.5-7B (fixed 576 visual tokens, blue) rises steeply: **1.19× (c1) → 1.53×
(c4) → 1.96× (c16) → 2.22× (c64)**. Qwen3-VL-8B-Instruct (dynamic resolution,
native 2×2 MLP merger, violet) is **attenuated**: 1.08× (c1) → 1.29× (c12) →
1.34× (c64, r50/r0 — r75 not measured on Qwen3-VL c64). The concurrency-
amplification mechanism (F1) is robust to the architecture (Qwen3-VL r75 bonus
c1→c12 = +0.21 > 0) but is **~1/3 the strength** of LLaVA-1.5 (+0.21 vs +0.65).

**Mechanism — the native 2×2 merger and a post-hoc pruner are SUBSTITUTES, not
complements.** The merger compresses 4 patches → 1 token *before* our pruner; on
GQA only ~260 post-merger tokens survive (vs LLaVA's fixed 576), so the pruner
has less to remove → smaller KV/prefill relief → smaller F1 speedup. The
attenuation is honest, mechanism-explained science that surfaces **pre-merger
pruning** as the next lever on natively-compressed architectures.

*Source:* LLaVA — `runs/v2_p2/batch_c{1,4,16,64}_r{0,75}.json`; Qwen3-VL c1/c12 —
`notes/v2_p1_qwen3vl.md` §2; Qwen3-VL c64 — `runs/v2_p2/qwen3vl_c64_r{0,50}.json`.
*Generator:* `gen_fig4.py`.
