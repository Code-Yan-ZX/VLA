# V3 merger-aware token selection — consolidated results (2026-07-23)

> Round goal (user): make V3 a METHOD paper ("merger-aware token selection"), correct the
> mechanism, fix metrics, explore adaptive/hybrid. This consolidates the three detailed
> reports: `rescore_rerun_report.md` (metrics), `mechanism_verification_report.md` (mechanism),
> `method_gate_report.md` (method gate). All Qwen3-VL-8B-Instruct, vLLM 0.19 V1, short-answer
> prompting, official metrics, seed=0.

## Headline
1. **Corrected mechanism PROVEN (causal):** the native 2×2 merger *rewrites token
   saliency/ranking, anti-text*; a post-merger selector drops text. A ranking-swap control
   shows the entire pre>post gap is the RANKING (forward path held constant; swap≡pre exactly).
2. **Metrics fixed; main conclusions HOLD:** with official VQA-acc/ANLS + short-answer prompts,
   pre-merger still beats post-merger on text-dense (TextVQA +38.3pp, DocVQA +26.5pp, ~6σ).
3. **Claim correction:** the "post wins on object QA (GQA)" crossover was a verbose-generation +
   containment-scorer artifact. Under proper eval **GQA pre==post==0.51**. The honest stage law is
   **pre-merger weakly DOMINATES post: ties on object, wins big on text-dense** — no crossover.
4. **Method-search honest negative:** the merger-aware HYBRID (agreement units + disagreement
   budget routed to text) and the disagreement-based adaptive ROUTER both **fail to beat plain
   pre-merger selection** — because pre ≥ post on 84–97% of images (no regime/region where post is
   better), so routing/mixing toward post only hurts. Residual oracle headroom (+8.2pp) is
   per-image QUERY-dependent, unreachable from image-level signals.

## What survives as the method
**"Rank-before-merge" selection** = compute saliency on PRE-merger (ViT-block-8) features, select
units, THEN merge. Concrete, novel (pre-merger × native-merger cell verified empty), mechanism-
grounded (swap control proves ranking is the lever), and weakly dominant under proper eval
(≥ post everywhere, ≫ on text-dense; selector-invariant: L2 + attn both +35pp). The adaptive/hybrid
extensions are reported as bounding negatives (even ranking-informed post-stage selection cannot
improve on committing to the pre ranking) — which *strengthens* the mechanism claim.

## Evidence → claim map
| Claim | Evidence | Status |
|---|---|---|
| Merger corrupts unit ranking, anti-text | M1 ρ=0.14(doc)/0.33(text)/0.36(gqa); M2 rank_shift↔edge +0.44(doc), pre-kept/post-dropped edge 0.64 vs 0.12 | SOLID (n=64/bench) |
| Gap is ranking-only (not forward path) | M3 swap≡pre: doc 0.465=0.465 (200/200), text 0.603≈0.598 (198/200) | SOLID (causal) |
| pre>post text-dense, official metrics | TextVQA VQA-acc pre .598/post .215; DocVQA ANLS pre .465/post .200 (n=200, ~6σ) | SOLID |
| pre dominates (ties object) | GQA pre==post==0.51 (n=100, short-answer exact-match); pre≥post 84–97% of images | n=100 (confirm GQA tie @n=200 before paper) |
| selector-invariant | attn pre .553 > post .200 (+35pp) under official VQA-acc (n=100) | SOLID |
| hybrid/router don't beat pre | hybrid OCRBench −8pp vs pre; router 0.484 ≤ always-pre 0.494; oracle +8.2pp query-dep | SOLID (honest negative) |

## Tension with goal + open decision
User wanted a *method* beyond "pre>post diagnostic." The clever method (adaptive/hybrid) did not
pan out; the surviving method is the simple-but-principled "rank-before-merge." Options for the
paper direction are escalated to the user (see DECISIONS 2026-07-23 method-gate entry):
(A) mechanism-led method paper with rank-before-merge as the method + honest bounding negatives;
(B) invest GPU in query-aware pre-merger selection to chase the +8.2pp oracle headroom (riskier —
query-aware failed 3× on boundary in 2026-07, but untested on pre-merger); (C) A now, B as extension.

## Assets
- Metrics: src/v3_premerger/official_scorers.py; scripts/{rescore_official,rescore_rerun,fix_shortanswer_subsets,fix_shortanswer_gqa_subset}.py; cells runs/v3_merger_aware/rescore_rerun/.
- Mechanism: scripts/mechanism_token_survival.py (M1/M2); runner --mask-ranking swap; runs/v3_merger_aware/{survival_capture,swap}/; figs drafts/figures/token_survival_*.
- Method: runner --mode hybrid/--hybrid-text-frac/--save-unit-scores; src/v3_premerger/v3_hybrid_gate*.sh; scripts/{rescore_hybrid_gate,router_disagreement_analysis}.py; runs/v3_merger_aware/{hybrid_gate,router}/.
- Commits: a3e7780, 638c09a, 6379bcf, 989d386, 84f06d5 (Code-Yan-ZX, no AI attribution).
