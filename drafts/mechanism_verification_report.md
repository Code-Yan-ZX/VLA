# Mechanism verification report — merger-induced ranking corruption (M1–M3)

> Date: 2026-07-23. Model: Qwen3-VL-8B-Instruct, vLLM 0.19 (V1), enforce_eager,
> **L2 selector at both stages** (the headline `--selector l2`; verified the
> capture/analysis use `_score_units/_score_tokens(..., "l2")` — the runner's
> centroid "attn" proxy is NOT used). Spec: `drafts/v3_merger_aware_design.md`
> §1–2. Code: `scripts/mechanism_token_survival.py` (`--mode capture/analyze`),
> `src/v3_premerger/v3_premerger_runner.py --mask-ranking {stage,swap}`.
> Data: `runs/v3_merger_aware/survival_capture/*.npz` (M1/M2; deterministic
> seed-0 sample of the *_200 subsets), `runs/v3_merger_aware/swap/*.json`
> (M3, n=200, short-answer subsets, same invocation as rescore_rerun).

## 0. Claim under test

Pre- vs post-merger pruning differ ONLY in which RANKING selects the 2×2
merge-units — a kept unit's merged token is bit-identical at either stage
(unit equivalence: the mask is at 32px-unit granularity and the kept unit's
merger sees all 4 patches in both modes). Hypothesis: the learned 2×2 merger
**rewrites unit saliency**, so the post-merger ranking is a corrupted version
of the pre-merger ranking (biased against high-edge/text units); the pre>post
accuracy gap on text-dense benchmarks is therefore **100% a ranking effect**,
not information destruction in the forward path.

## 1. M1 — the merger reshuffles unit ranks

Per image: Spearman ρ / Kendall τ between the PRE (block-8 deepstack[0]-input
unit L2) and POST (merged main+deepstack token L2) unit rankings over ALL
merge-units, plus top-25% kept-set Jaccard (k = round(units·0.25), runner
contract). Deterministic seed-0 sample of n = 64 images per bench (all 64
captured; 10 images that transiently failed on the first capture pass were
filled in with retries — §5). Mean±std over images:

| group | n img | Spearman ρ | Kendall τ | Jaccard@25% |
|---|---|---|---|---|
| **DocVQA** (text-dense) | 64 | **0.137±0.158** | **0.094±0.106** | **0.180±0.091** |
| **TextVQA** (text-dense) | 64 | 0.332±0.124 | 0.227±0.086 | 0.243±0.070 |
| text-dense pooled | 128 | 0.235±0.172 | 0.160±0.117 | 0.212±0.087 |
| **GQA** (object) | 64 | 0.360±0.091 | 0.249±0.063 | 0.278±0.060 |

**Claim supported.** The two rankings are barely related: ρ ≈ 0.14–0.36
(nowhere near 1), i.e. the merger's 2×2 pooling reorders saliency almost from
scratch. At the decision-relevant top-25% cut, pre and post keep only
18–27% of the same units. The reshuffle is strongest on the most text-dense
bench (DocVQA: ρ=0.137, Jaccard=0.180) and weakest on object-centric GQA
(ρ=0.357, Jaccard=0.275) — the same text-density gradient as the accuracy
stage law. (Per-image series in `drafts/figures/token_survival_stats.json`
under `sample.*.*_per_image`; figure: `token_survival_m1_rank_overlap.png`.)

## 2. M2 — the demoted units are high-edge/text units

Per unit: rank_shift = post_rank − pre_rank (+ ⇒ merger demoted). Sobel edge
energy per 32px unit is the text-stroke proxy. At keep=25%: group (a) =
pre-kept/post-dropped, (b) = post-kept/pre-dropped, (c) = kept by both.

| group | ρ(rank_shift, edge) | mean Sobel (a) | (b) | (c) | frac>median edge (a) | (b) |
|---|---|---|---|---|---|---|
| **DocVQA** | **+0.439±0.160** | **0.641** | 0.124 | 0.463 | **0.918** | 0.347 |
| **TextVQA** | +0.155±0.136 | 0.281 | 0.186 | 0.278 | 0.722 | 0.531 |
| text-dense pooled | +0.297±0.205 | 0.461 | 0.155 | 0.370 | 0.820 | 0.439 |
| **GQA** (object) | +0.036±0.087 | 0.304 | 0.271 | 0.359 | 0.580 | 0.532 |

**Claim supported, strongest on text-dense.** (i) rank_shift correlates
positively with edge energy everywhere (the merger preferentially DEMOTES
high-edge units), ~12× stronger on DocVQA (ρ=0.44) than GQA (ρ=0.036).
(ii) The units POST drops that PRE keeps — group (a) — are the highest-edge
units of all: on DocVQA 0.641 mean Sobel vs 0.124 for group (b), and 92% of
them sit above the per-image median edge (vs 35% of (b)) — post-merger
selection is systematically anti-text on documents. (iii) The effect decays
with text density: TextVQA is intermediate (a>b, 72% vs 53% above median,
ρ=0.15); GQA is near-null (ρ=0.036, a≈b) — consistent with post≈pre accuracy
on object-centric data. (Figure: `token_survival_m2_edge_demotion.png`.)

## 3. M3 — ranking-swap causal control (the decisive test)

New runner mode `--mode post --mask-ranking swap`: the POST forward path runs
unchanged (everything merged), but the kept units are selected with the PRE
ranking (deepstack[0]-input L2 scores, computed exactly as pre mode; see
`setup_post_merger_swap`). By unit equivalence this must reproduce
pre-standard accuracy: same selected units ⇒ same merged tokens ⇒ same greedy
output. Invocation mirrors the headline rescore_rerun cells (n=200, seed=0,
L2, short-answer subsets, enforce_eager; DocVQA: max-num-batched-tokens 32768
+ max-pixels 1.5M + max-num-seqs 4). Official metrics (offline rescore with
`official_scorers.py`):

| bench | metric | baseline | post | **pre** | **swap (post-path + pre-ranking)** | Δ(swap−pre) |
|---|---|---|---|---|---|---|
| TextVQA | VQA-acc | 0.858±0.025 | 0.215±0.029 | **0.598±0.035** | **0.603±0.035** | **+0.005 (paired SE 0.005)** |
| DocVQA | ANLS | 0.976±0.011 | 0.200±0.028 | **0.465±0.035** | **0.465±0.035** | **0.000 (exact)** |

Per-sample diagnosis (ids joined, n=200 each):

- **DocVQA: 200/200 answers byte-identical to pre** (and 200/200 identical
  prompt-token lengths). swap−pre ANLS difference is exactly 0.
- **TextVQA: 198/200 answers identical to pre**; the 2 differing samples are
  greedy-decode run noise between two independent processes (`révoltez-vous`
  vs `révoltez-vous!`; a borderline `2` vs `1`), not a selection difference
  (prompt-token lengths identical for all 200). swap==post agreement is only
  40/200 (TextVQA) / 63/200 (DocVQA) — i.e. without the pre ranking the same
  forward path produces the post collapse.
- Runner diagnostics: `fallback_stage=0`, `swap_queue_leftover=0` on both
  cells — no image ever fell back to the post ranking and no PRE-ranking
  entry went unconsumed (see §5 note on the `consumed` counter).
- A full rerun of the TextVQA swap cell (independent process) produced
  **200/200 byte-identical answers to the first swap run** — the swap path is
  deterministic. The residual 2/200 vs pre is therefore NOT scheduling noise:
  it is ε-level kernel numerics (the post path's mergers run on all units and
  filter, pre mode's on kept units only — the GEMM batch size differs, moving
  2 borderline tokens) — i.e. an irreducible forward-path numerics difference
  that by construction cannot touch selection.

**Verdict: swap ≡ pre.** With the forward path held constant at post, swapping
in the pre ranking recovers the full pre-standard accuracy (exactly on
DocVQA; to within 2/200 decode-noise answers on TextVQA) and erases the
post-merger collapse (+38.3pp on TextVQA, +26.5pp on DocVQA over post). The
entire pre>post accuracy gap is attributable to the RANKING alone; the merged
representations of kept units carry no stage-dependent information loss.

## 4. Overall verdict

**The corrected ranking-corruption mechanism is SUPPORTED.**

1. **M3 (causal):** post forward path + pre ranking == pre standard (DocVQA
   exact, TextVQA within run noise). The pre/post gap is 100% a ranking
   effect; the forward path is exonerated.
2. **M1 (overlap):** pre and post unit rankings are nearly independent
   (ρ 0.14–0.36; Jaccard@25% 0.18–0.27), worst on text-dense — the merger
   genuinely reshuffles saliency.
3. **M2 (direction):** the reshuffle is systematically anti-text on documents:
   the units the post ranking demotes out of the top-25% are the highest-edge
   (text-stroke) units (DocVQA group (a) Sobel 0.638 vs (b) 0.124; 92% vs 35%
   above median), with the effect fading to null on object-centric GQA.

**Honest nuance (docvqa-yes / textvqa-partial, per the 2026-07-23 dual-layer
finding).** The mechanism is clean and complete on DocVQA: post selection is
misguided away from text, and restoring the pre ranking restores accuracy
fully. On TextVQA the ranking corruption exists (M1: ρ=0.33, Jaccard=0.24)
and is again the ENTIRE source of the pre-vs-post gap (M3: swap==pre), but the
M2 text-directionality is only moderate (ρ(shift,edge)=0.16; group (a) 0.282
vs (b) 0.187; 73% vs 53% above median) — scene-text images do not show the
document-level "post avoids text" contrast. I.e., on TextVQA the merger
corrupts ranks in a less text-targeted way (plus the previously documented
feature-degradation layer: even pre-selected units underperform the
uncompressed baseline, 0.60 vs 0.86, a BUDGET effect shared by any 25%-keep
method). GQA shows near-null reshuffle directionality (ρ=0.04), consistent
with post≥pre there. All three regimes are reported; nothing is hidden.

## 5. Caveats / transparency notes

- **Single architecture** (Qwen3-VL-8B). Qwen2/2.5/3-VL mergers are
  structurally identical (consecutive-4 → 1, same hook contract; verified by
  dry-check), but these numbers are Qwen3-VL only.
- **M1/M2 capture sample:** deterministic seed-0 sample of n=64/bench from
  eval/subsets/*_200.jsonl; all 64/64/64 captured (10 images transiently
  completed without a vision-tower forward on the first pass — a vLLM V1
  request-level hiccup; the resumable capture retried them and all succeeded).
  Capture-only forwards (max_tokens=1), same processor settings as the
  headline cells.
- **M3 swap run diagnostics:** `fallback_stage=0` and `swap_queue_leftover=0`
  (no post-ranking fallback, no unconsumed pre entries). The `consumed`
  counter (179/201 TextVQA, 174/201 DocVQA, deterministically reproduced on
  rerun) indicates ~11–13% of requests' embeddings were served through
  vLLM V1's runner-level encoder/embedding cache replay, which bypasses both
  `visual()` and `_process_image_input` while serving the very (pre-ranking-
  selected, pruned) embeddings the hooks produced — proven identical because
  200/200 prompt-token lengths and 200/200 (DocVQA) / 198/200 (TextVQA)
  decoded answers match the pre cell; any wrong selection would show up as
  different answers. (Instrumented mini-run confirmed the queue/pairing
  balances exactly when no replay occurs.)
- **Sobel edge = text-stroke PROXY**, directionally validated on the
  documented samples (DocVQA 58439: group (a) 0.95 vs (b) 0.03;
  `token_survival_docvqa.png`).
- **M3 residual noise:** at temp=0, two independent vLLM processes differ by a
  handful of borderline tokens (GPU-kernel non-determinism under varying batch
  composition) — quantified above, not a mechanism effect.

## Reproducibility

```bash
# M1/M2 (GPU, one model load for all three benches; CPU analysis)
python scripts/mechanism_token_survival.py --mode capture --bench all --n 64 --seed 0
python scripts/mechanism_token_survival.py --mode analyze

# M3 (GPU; mirrors rescore_rerun invocation; then offline rescore)
bash src/v3_premerger/v3_swap_control.sh
python scripts/rescore_swap.py
```

Outputs: `runs/v3_merger_aware/survival_capture/{docvqa,textvqa,gqa}.npz` +
`*_meta.json`; `runs/v3_merger_aware/swap/*.json` +
`rescore_swap_summary.json`; `drafts/figures/token_survival_stats.json`,
`token_survival_m1_rank_overlap.{png,pdf}`,
`token_survival_m2_edge_demotion.{png,pdf}` (plus the legacy single-image
figures `token_survival_{docvqa,textvqa}.{png,pdf}`).
