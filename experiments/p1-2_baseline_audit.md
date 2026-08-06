# P1-2: Strong-baseline fairness audit

**Date:** 2026-08-06 · **Verdict:** Hi-Lo Prune / QuietPrune / IF-Prune are NON-reproducible (no usable code) -> non-reproducibility digest formed, no GPU wasted (per user rule). PixelPrune is the only usable training-free Qwen3-VL baseline (gate deferred: different mechanism family + integration cost + GPU priorities). Table 3 mixed-scope concern already mitigated by P0-2b qualifications.

## Audit of the three named baselines

1. **Hi-Lo Prune** (CVPR'26, Sun et al.; poster cvpr.thecvf.com/virtual/2026/poster/40023). Training-free; paper evaluates on Qwen3-VL-8B / Qwen2.5-VL-7B / Qwen2-VL-2B / LLaVA-1.5. **Official repo [github.com/sealost/Hi-Lo_Prune](https://github.com/sealost/Hi-Lo_Prune) is an EMPTY placeholder** (`# Hi-Lo-Prune`, 0 KB, 1 commit, pushed 2026-03-19). **Code NOT released.** -> Not usable; do NOT hand-reproduce (policy). This is the ideal paper match (training-free + Qwen3-VL-8B + hierarchical fixed-budget selection) but cannot be run; revisit if code is released.

2. **QuietPrune** (CVPR'26, Gao et al.; openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_...). CVPR'26 Papernotes lists "Code: None." **Requires training** a lightweight adapter via self-distillation (~10K samples, 40 min A100, VLM frozen) - NOT training-free. Base: Qwen3-VL / InternVL3. -> Not usable (no code + not training-free).

3. **IF-Prune** (CVPR'26, Snap Research; [github.com/snap-research/EVLM-IF-Prune](https://github.com/snap-research/EVLM-IF-Prune), project gz6905.github.io/EVLM-IF-Prune-website). Official code present and runnable (train/test/eval, lmms-eval integration). **But requires LoRA fine-tuning (not training-free)** and supports only **Qwen2-VL + InternVL2/2.5** - no Qwen3-VL/Qwen2.5-VL. -> Has code but wrong fit for a training-free Qwen3-VL gate.

**Non-reproducibility digest:** of the three named baselines, zero are runnable as a training-free Qwen3-VL-8B comparison. Hi-Lo Prune (the closest match) has no code; QuietPrune needs training + has no code; IF-Prune needs training + wrong base. Per the project policy (no hand-reproduction from the paper when no code exists), no GPU is spent on them.

## PixelPrune - the one usable training-free Qwen3-VL baseline (gate deferred)

**PixelPrune** ([github.com/OPPO-Mente-Lab/PixelPrune](https://github.com/OPPO-Mente-Lab/PixelPrune), arXiv:2604.00886). Training-free, no learnable params, explicit `Qwen/Qwen3-VL-8B-Instruct` support (HF + vLLM), VLMEvalKit eval, native DocVQA_VAL + OCRBench. The ONLY training-free method with usable public code on Qwen3-VL-8B.

**Caveats (why a gate is deferred):**
- Control is a similarity THRESHOLD (`PIXELPRUNE_THRESHOLD`), NOT a fixed keep-ratio -> matching keep=25% requires a τ sweep to hit ~25% avg retention (non-trivial iso-budget matching).
- Mechanism is pre-ViT pixel deduplication - a DIFFERENT family from the paper's merger-stage (pre/post) pruning. Not apples-to-apples with RBM/FastV; it would be a "another training-free Qwen3 method" data point, not a direct stage comparison.
- Integration cost: clone + env setup + τ sweep + n=200×4 benchmarks.
- GPU is serial and prioritized for P1-1 (mechanism) + P1-3 (efficiency), which are core claims.

**Decision:** document PixelPrune as available; DEFER the gate (moderate value, high cost, different mechanism). Flag for the user: a PixelPrune n=200 gate is possible if they want a 2026 training-free Qwen3 baseline in Table 3, but it is not on the critical path.

## Other candidates (2025-2026, with code)

- **VisionZip** ([github.com/dvlab-research/VisionZip](https://github.com/dvlab-research/VisionZip), arXiv:2502.18804, 444★): training-free mode, supports LLaVA + **Qwen2.5-VL** (`Qwen2_5_VL/` dir), fixed keep-ratio. No Qwen3-VL. Best secondary if a Qwen2.5-VL gate is acceptable. (Already referenced in the paper as a post-merger SOTA; the v3_sota_matrix has vz_* cells.)
- **SparseVLM** (arXiv:2410.04417, ICML'25): training-free, LLaVA-only. No Qwen.
- **FasterVLM** (arXiv:2412.01818): training-free ([CLS]-attn), LLaVA-1.5/1.6 only. No Qwen.
- **ToMe-for-VLM**: only unofficial LLaVA ports; original ToMe is ViT-only. No Qwen3-VL.
- **AdaptPrune** (arXiv:2503.08019): Qwen3-VL compatibility unverified.
- ERA / AdaVIPS: no method found by these names (unverified).

## Table 3 mixed-scope evaluation (FastV full-split extension)

Existing FastV cells: `runs/full_matrix/j7hf_{qwen3vl,qwen2vl}_fastv_{textvqa,docvqa,gqa,ocrbench}_r0.750_n500.json` (n=500, NOT full split) + `runs/r2_same_scope/r2_qwen3vl_fastv_k2_textvqa_r0.75_full5000.json` (1 full-split TextVQA). So Table 3 has mixed scope (some full-5000, some n=200/n=500).

- The mixed-scope concern is ALREADY mitigated: P0-2b downgraded the two weakest RBM edges (Qwen2.5-VL GQA +4.5pp, DocVQA +2.1pp, CIs cross 0) to "exploratory/inconclusive". The FastV text-dense wins are significant (P0-2 paired stats). So no "winner claim" rests on an unstable mixed-scope cell.
- Extending FastV to full split (TextVQA/OCR-Bench/GQA, both families) would make scope consistent but costs significant GPU (full splits = 5000 samples × 3 benchmarks × 2 families). **Deferred** - not on the critical path; the P0-2b qualifications suffice for submission. Flagged as a possible robustness improvement if GPU permits after P1-1/P1-3.

## Recommendation / decision

- Core P1-2 (audit the 3 named baselines): DONE -> non-reproducibility digest, no GPU wasted.
- PixelPrune gate: available but DEFERRED (different mechanism + cost + GPU priorities).
- FastV full-split extension: evaluated, DEFERRED (P0-2b qualifications mitigate the mixed-scope concern).
- Table 3 keeps FastV (in-LLM) + RBM (the paper's method) + VisionZip (post-merger SOTA, already in v3_sota_matrix) as the baseline comparison, with the audit documenting why Hi-Lo/QuietPrune/IF-Prune are not included.

## Artifacts
- This digest: `experiments/p1-2_baseline_audit.md`.
- Baseline cells: `runs/full_matrix/j7hf_*_fastv_*`, `runs/v3_sota_matrix/vz_*`, `runs/r2_same_scope/r2_*_fastv_*`.
- P0-2 stats: `experiments/paired_metric_statistics.md` (Table 3 cell-by-cell CIs).
