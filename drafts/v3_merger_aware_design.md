# V3 → "Merger-Aware Token Selection" — design spec (2026-07-23)

> Purpose: turn V3 from a diagnostic ("pre > post") into a METHOD paper. Records the corrected
> mechanism, the decisive controls, and the method designs. Handed to GPU-execution agents.

## 1. Corrected mechanism (user correction, verified against code)

Pre-merger pruning does **NOT** bypass the merger. The prune mask is at **2×2 merge-unit (32px)
granularity**: whole units are kept/dropped, and a kept unit's merger sees all 4 patches in BOTH
pre and post mode. => The merged token for any kept unit is **identical** regardless of stage.

Therefore pre vs. post differ in exactly ONE thing: **which ranking selects the units.**
- pre  = rank units by **raw ViT-block-8 saliency** (deepstack[0] input), then merge survivors.
- post = merge everything, rank units by **merged-token saliency**, then select.

Hypothesis (the real mechanism): the learned 2×2 merger **rewrites token saliency/ranking**; for
text/high-frequency units it demotes them in the post-merger ranking, so a post-merger selector
drops text. Pre commits its selection on the pre-distortion ranking. The advantage is 100% a
RANKING effect, not information destruction in the forward path.

## 2. Decisive mechanism experiments (M1–M3)

Tool: extend `scripts/mechanism_token_survival.py` (already captures per-unit PRE scores from
`ds_in[0]` and POST scores from cat(main,ds0..2), and Sobel edge energy/unit + top-k Jaccard).

- **M1 ranking overlap** (CPU, add to `analyze()`): Spearman + Kendall between pre_scores and
  post_scores per image; top-k Jaccard@keep. Report distribution over n>=50 images, split
  text-dense (docvqa/textvqa) vs object (gqa). Claim: merger substantially reshuffles rank.
- **M2 disagreement x text-density** (CPU): for each unit compute rank shift |pre_rank-post_rank|;
  correlate with Sobel edge energy. Claim: the units the merger most demotes (pre-high/post-low)
  are disproportionately high-edge/text units; post drops them, pre keeps them. Quantify fraction.
  (This is the mechanism behind the DocVQA "selection-misguided" layer; for TextVQA also check the
  feature-degradation layer found 2026-07-23 — report honestly.)
- **M3 ranking-swap control** (GPU, new runner mode): because of the unit equivalence, "post forward
  path + PRE ranking" must equal pre-standard accuracy, and "pre forward path + POST ranking" must
  equal post-standard. Run the one non-trivial cell **mode=post, mask-ranking=pre** on
  textvqa/docvqa/gqa @keep=25% n=100. If it matches pre-standard => the gap is proven attributable
  to RANKING alone (forward path held constant). This is the shared-mask control the user asked for.

Implementation for M3: add `--mask-ranking {stage,swap}` to runner. With mode=post + swap: hook the
merger to capture ViT-block-8 feats, compute unit scores (same `_score_units`), and use THAT mask to
filter merged tokens in `setup_post_merger._patched`. (mode=pre+swap ≡ post-standard by equivalence;
optional sanity.)

## 3. Metrics (official, applied offline first)

- TextVQA -> **VQA accuracy** (min(count_match_over_10/3,1), canonical VQA-eval normalization).
- DocVQA  -> **ANLS** (max over variants of s if s>=0.5 else 0; s=1-norm_levenshtein).
- Standalone `src/v3_premerger/official_scorers.py` + `scripts/rescore_official.py` (Agent A, CPU).
- Rescore cells with saved `per_sample` first; rerun only missing cells (deep keep=12.5% pre,
  baselines, n=500 tighten). GATE: if pre>post (text-dense) or post>pre (GQA) flips -> escalate.

**RESOLVED (2026-07-23): HOLD, no flip.** Offline rescore first revealed the saved predictions were
VERBOSE sentences (runner prompt = raw question, no short-answer instruction) -> official VQA-acc/ANLS
= ~0 for everyone (containment metric had overstated ~100x). Root cause = prompt protocol, not method.
Fix (same convention as ChartQA/OCRBench): bake "Answer the question using a single word or phrase."
into the TextVQA/DocVQA subsets (scripts/fix_shortanswer_subsets.py) + re-run. OFFICIAL-METRIC results
(keep=25%, n=200, L2, seed=0; runs/v3_merger_aware/rescore_rerun/, drafts/rescore_rerun_report.md):
- TextVQA VQA-acc: baseline 85.8+-2.5 / post 21.5+-2.9 / pre 59.8+-3.5 -> **pre-post +38.3pp +-4.5 (~6sigma) HOLD** (ret pre 70% / post 25%).
- DocVQA ANLS:     baseline 97.6+-1.1 / post 20.0+-2.8 / pre 46.5+-3.5 -> **pre-post +26.5pp +-4.5 (~6sigma) HOLD** (ret pre 48% / post 21%).
- GQA still verbose (raw-question prompt) -> GQA exact-match also compromised; FIX GQA subset before the
  method gate (task 5 uses GQA). Old containment GQA post>pre (-6pp) not yet revalidated under short-answer.

## 4. Method designs (the "merger-aware token selection" contribution)

### 4a. Selector invariance (Task 3)
attn (centroid-distance) already run 2026-07-23. Re-confirm under official metrics (offline rescore
of v3_attn_robust). Optionally add one more cheap text-agnostic family (e.g. per-unit feature
variance) to show the ranking effect is not L2-specific. Keep light.

### 4b. Adaptive stage selector (Task 4) — signal = ranking disagreement / text-density
Per image, compute BOTH pre and post unit rankings (one forward captures both), measure
image-level disagreement (mean |rank shift| or 1-Jaccard@k) and text-density (mean Sobel edge).
Route: high disagreement x high text-density -> use PRE ranking; else POST ranking.
- Runner: add `--save-unit-scores` to stash per-image pre_scores+post_scores (or summary stats)
  into per_sample. Then build router OFFLINE: oracle = per-sample best(pre,post); compare
  disagreement-router vs old ptid-threshold router vs always-pre/always-post.
- This is a BETTER signal than ptid because it directly measures "how much the merger corrupted
  the ranking for THIS image" — the corrected-mechanism signal.

### 4c. Hybrid merger-aware selection (Task 5) — the headline method
Budget = k units. Per image:
- **Agreement set A** = units in top-k under BOTH rankings (stage-robust, keep all).
- **Disagreement budget** = k - |A|. Allocate toward high-frequency text: among contested units
  (pre-high/post-low or vice-versa), pick by (pre-ranking AND high edge density) first -> protects
  text/OCR; fill remaining from post-ranking in low-text (object) regions -> recovers GQA.
- Net: per-REGION adaptive — text regions trust pre-ranking, non-text trusts post-ranking, agreement
  anchors. Target: keep OCR advantage AND reduce GQA loss vs always-pre.
- Runner: add `--mode hybrid --hybrid-alpha <agreement fraction>` (or a threshold). Implement unit
  selection = agreement ∪ routed-disagreement.

## 5. Gate (Task 5, before scaling)

Small gate: **TextVQA / OCRBench / GQA, keep=25%, n=100**. Compare hybrid vs pre-standard vs
post-standard (all under official metrics). Scale to n=200 / full split ONLY IF:
- text-dense (TextVQA VQA-acc, OCRBench) hybrid >= pre-standard (no OCR regression), AND
- GQA hybrid notably better than pre-standard (closes gap toward post) without hurting text-dense.
Do NOT expand to more benchmarks; do NOT make paper figures yet.

## 6. Deliverables
- `drafts/rescore_official_report.md` (Agent A) — official-metric gaps + HOLD/FLIP.
- `drafts/figures/token_survival_*` + `token_survival_stats.json` — M1/M2 extended.
- `runs/v3_merger_aware/` — M3 swap cells, adaptive-router scores, hybrid gate cells.
- `drafts/v3_merger_aware_results.md` — consolidated mechanism+method results + method claim.
- Update STATE.md / DECISIONS.md; commit+push as Code-Yan-ZX (no AI attribution).
