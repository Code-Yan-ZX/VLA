# Merger-aware token selection — method gate report (2026-07-23)

> Scope: build the hybrid merger-aware selection method + disagreement router, and run
> the user-defined gate BEFORE any scaling. keep=25% (r=0.75), Qwen3-VL-8B-Instruct,
> vLLM 0.19 V1 (enforce_eager), L2 selector, seed=0, short-answer subsets, **n=100**,
> official metrics. Spec: `drafts/v3_merger_aware_design.md` §4–5. Code: runner
> `--mode hybrid`/`--hybrid-text-frac`/`--save-unit-scores`
> (`src/v3_premerger/v3_premerger_runner.py`), `src/v3_premerger/v3_hybrid_gate.sh`,
> `src/v3_premerger/v3_hybrid_gate_sweep_sens.sh`,
> `scripts/{rescore_hybrid_gate,router_disagreement_analysis,fix_shortanswer_gqa_subset}.py`.
> Cells: `runs/v3_merger_aware/hybrid_gate/` (16 cells, zero failures) +
> `runs/v3_merger_aware/router/` (cap64 cells + `router_comparison.{json,md}`).

## 0. Protocol fixes

- **GQA subset fixed**: `eval/subsets/gqa_200.jsonl` now carries
  "\nAnswer the question using a single word or phrase." (pristine original backed up
  to `eval/subsets/_backup/gqa_200.jsonl`; ids/image/gt/order preserved, 200 lines;
  `scripts/fix_shortanswer_gqa_subset.py`, idempotent + verified). GQA metric = runner
  `score_gqa`: word-normalized **exact match** with singular handling + yes/no
  lead-token logic — the GQA convention; no additional normalization needed. Under the
  fixed prompt the median GQA answer is 1 word (was verbose sentences; the old
  containment metric had overstated GQA scores).
- Reference cells verified: textvqa/docvqa pre/post/none `rescore_rerun` n=200 sliced
  to first 100 ids == subset first-100 (verified True); OCRBench A/B/C (none/post/pre)
  `v3_sota_matrix` n=200/seed0 first 100 ids == subset first-100 (verified True; the
  none cell carries 6 skips in the first 100 -> n=94, a fallback artifact of that
  reference cell, not of this gate).

## 1. Hybrid method (runner `--mode hybrid`)

Post forward path (merge everything; by M3 unit equivalence, post-stage selection with
ANY unit mask == pre-stage selection) + per-image hybrid unit mask:
- **Agreement set A** = top-k(PRE) ∩ top-k(POST), kept. PRE = deepstack[0]-input unit
  L2 (exactly as pre mode), POST = merged-token L2.
- **Contested budget** (k−|A|) **routed to text**: `--hybrid-text-frac` t ∈ [0,1] of it
  goes to the PRE ranking among high-Sobel-edge units (above per-image contested
  median; text proxy), the rest to the POST ranking among low-edge units; overflow
  fills the other pool. **Exactly k units kept** (iso-token with pre/post). t=1 => all
  contested budget to pre/text; t=0 => all to post.
- Per-unit edge is reconstructed from the encoder's OWN input pixels (identical
  32px-unit Sobel pooling as the M1/M2 capture; no file I/O, immune to vLLM
  encoder-cache-replay mispairing because it is computed in the same visual-forward
  call that produces the PRE scores).

Diagnostics on every n=100 hybrid cell: `fallback_stage=0`, `hybrid_queue_leftover=0`,
`edge_fallback=0`. The agreement set never reaches k (|A|/k ≈ 0.40 mean on textvqa) so
the routed branch always fires. `--save-unit-scores` attaches per-image
{n_units, k, agree_n, Jaccard@k, Spearman(pre,post), mean edge, branch} to per_sample
(FIFO guarded by offline-recomputed unit counts; encoder-cache-replayed requests
correctly get no entry: e.g. textvqa 92/100 attached, 8 replays, 0 misattachments).

## 2. hybrid-text-frac sweep @n=100 (official metrics; frac tuned on textvqa VQA-acc)

| text-frac | textvqa VQA-acc | ocrbench acc | gqa exact-match |
|---|---|---|---|
| 0.0 (all contested -> post) | 0.313±0.045 | 0.290±0.046 | 0.480±0.050 |
| **0.5 (chosen)** | **0.560±0.048** | 0.510±0.050 | 0.500±0.050 |
| 1.0 (all contested -> pre/text) | 0.533±0.049 | 0.590±0.049 | 0.470±0.050 |
| pre reference | 0.537±0.048 | 0.590±0.049 | 0.510±0.050 |
| post reference | 0.187±0.037 | 0.190±0.039 | 0.510±0.050 |

(± sample stderr; n=100 each. Chosen frac = argmax textvqa VQA-acc, strict > so ties
keep the lower frac.) **Sensitivity, reported honestly (in-sample pick, n=100):**
text-routing the contested budget is what saves text-dense accuracy (t=0.0 collapses
toward post: 0.313/0.290); for t≥0.5 textvqa saturates (0.560 vs 0.533, +2.7pp, inside
the paired SE). The frac trades benches: **ocrbench rises monotonically with t to
exactly pre (0.590 at t=1.0), while gqa slightly falls (0.480 -> 0.500 -> 0.470, all
≈ pre == post == 0.510 within noise)**. No single frac wins all three.

## 3. Gate table: hybrid vs pre vs post @n=100 (official metrics)

| bench | metric | baseline (none) | post | pre | **hybrid (t=0.5)** | Δ(hybrid−pre), paired |
|---|---|---|---|---|---|---|
| textvqa | VQA-acc | 0.807±0.037 | 0.187±0.037 | 0.537±0.048 | **0.560±0.048** | **+0.023±0.038** |
| ocrbench | containment-acc | 0.787±0.042 (n=94) | 0.190±0.039 | 0.590±0.049 | **0.510±0.050** | **−0.080±0.039** |
| gqa | exact-match | 0.640±0.048 | 0.510±0.050 | 0.510±0.050 | **0.500±0.050** | **−0.010±0.033** |

**Gate criterion (user-defined):** PASS iff text-dense (textvqa VQA-acc AND ocrbench)
hybrid ≥ pre-standard (no OCR regression) AND gqa hybrid notably better than
pre-standard (closes ≥50% of the post−pre gap) without hurting text-dense.

### VERDICT: **FAIL**

- textvqa no-regression: **PASS** (0.560 ≥ 0.537; +2.3pp, paired SE 3.8pp — within
  noise of pre, not demonstrably above it).
- ocrbench no-regression: **FAIL** (0.510 < 0.590; −8.0pp, paired SE 3.9pp ≈ 2σ below
  pre — a real OCR regression at the tuned frac).
- gqa better-than-pre: **FAIL** (0.500 < 0.510). Moreover the premise collapsed: under
  the short-answer protocol **gqa pre == post == 0.510** (the old containment-metric
  "post beats pre by 6pp" does not survive official exact-match at n=100), so there is
  **no post−pre gap to close** (gap-closure undefined).

**Reading.** Hybrid at t=0.5 buys +2.3pp on textvqa (noise-level) at the cost of −8pp
on OCRBench, and the contested post-favored slots bring nothing on GQA (pre==post).
t=1.0 recovers ocrbench to exactly pre but gives back the textvqa bump (0.533 ≤ 0.537)
and still doesn't improve gqa (0.470) — the agreement∪routing mask is NOT pre-top-k
even at t=1.0 (the agreement set constrains it), and it is never better than plain pre
on any bench. The construction is iso-token and stable (zero fallbacks), but the
allocation it optimizes (text vs non-text UNITS) is not the allocation the accuracy
differences need — consistent with the M2 nuance (textvqa's gap is only moderately
text-directional) and with the router result below (the residual pre/post choice is
per-image query-dependent, unreachable from image-level signals).

## 4. Adaptive stage router (offline; survival-capture n=64/bench + official correctness)

Signals per image: disagreement = 1−Spearman(pre,post) over all units; 1−Jaccard@k;
mean Sobel edge. Correctness: textvqa VQA-acc + docvqa ANLS from `rescore_rerun`
(n=200, all 64 captured ids joined); gqa exact-match from fresh short-answer cap64
cells (`router/{pre,post}_gqa_cap64_r0.750_l2_n64.json`, 64/64 joined; gqa cap64:
pre == post == 0.516, exactly tied). Thresholds swept on the POOLED sample (in-sample;
reported as sensitivity, not out-of-sample).

| pool | n | always-pre | always-post | oracle | ptid-router | dis-router | text-gated (AND/OR) |
|---|---|---|---|---|---|---|---|
| textvqa | 64 | 0.562±0.060 | 0.229±0.050 | 0.661±0.057 | — | — | — |
| docvqa | 64 | 0.404±0.059 | 0.198±0.047 | 0.473±0.060 | — | — | — |
| gqa | 64 | 0.516±0.063 | 0.516±0.063 | 0.594±0.062 | — | — | — |
| **mixed pooled** | **192** | **0.494±0.035** | **0.314±0.033** | **0.576±0.035** | **0.494** (t=124, pre%=68) | **0.484** (τ=0.53, pre%=95) | AND 0.365 / OR 0.473 |

(jaccard-router, a disagreement variant: 0.499 at pre%=95; dis-sweep range
0.329–0.484 with the best point at pre%=95% ≈ always-pre; ptid-sweep 0.327–0.494.)

**Mixed-traffic test (does disagreement-router beat always-pre AND always-post by
keeping pre on text and switching to post on GQA?):** it beats always-post by +17pp
(0.484 vs 0.314) but does **NOT** beat always-pre (−1.0pp) and ties the ptid-router;
all three sit 8.2pp under the oracle 0.576. It never learns the regime switch: its
best threshold routes 95% of images to pre, because disagreement does not separate the
regimes (median dis: docvqa 0.90, textvqa 0.64, gqa 0.63 — GQA overlaps the
text-dense benches; docvqa is simply the most shuffled everywhere). Splitting each
bench at its disagreement median: pre−post is +0.30/+0.22 (high/low-dis) on
textvqa, +0.22/+0.19 on docvqa — pre wins in BOTH halves — and −0.03/+0.03 on gqa
(the sign a switch-router would need, at ~3pp magnitude, with pre==post overall).
Pre is a near-dominant strategy at the IMAGE level too (pre ≥ post on 84–97% of images
in every bench), so any router sending a meaningful fraction to post loses; the
oracle's +8.2pp headroom is per-image query-dependent (which text/region the QUESTION
needs — cf. the 2026-07-21 4b decomposition: 73% sample-level), unreachable from
image-level signals. This corroborates rather than contradicts the mechanism: the
merger corrupts rankings most on text-dense images (high dis there), but on those
images "corrupted" means "pre is right", and pre is ALSO right on most low-dis and gqa
images.

## 5. Selector invariance (Task 3): attn (centroid-distance) selector, textvqa @n=100

| selector | pre VQA-acc | post VQA-acc | pre>post? | gap |
|---|---|---|---|---|
| **attn** (centroid-distance) | **0.553±0.048** | **0.200±0.038** | **YES** | +35.3pp |
| l2 (reference) | 0.537±0.048 | 0.187±0.037 | YES | +35.0pp |

**pre>post holds under a non-L2 selector AND the official VQA-acc metric** — the stage
effect is selector-invariant (matches the 2026-07-23 attn-robustness finding at n=200
under containment, now reconfirmed at n=100 under the official metric).

## 6. Recommendation

**Do NOT scale hybrid to n=200; the gate returned FAIL and the evidence is coherent.**

1. **The method that survives is pre-standard itself** — at n=100 under official
   metrics it is within noise of the best hybrid on textvqa, strictly better on
   ocrbench (+8pp at the tuned frac; hybrid reaches pre there only at t=1.0, which
   loses textvqa), tied with post on gqa (0.510), and selector-invariant. Always-pre
   is also the best image-level routing policy (0.494 pooled, never collapses).
2. **Drop the hybrid-mask headline.** Its per-region allocation (agreement ∪
   text-routed contested) is mechanically motivated but empirically trades ocrbench
   for a noise-level textvqa gain and nothing on gqa; no global text-frac passes.
3. **Drop the disagreement router as an accuracy improver** — report it as a
   negative / mechanism-corroboration result: image-level signals (disagreement,
   Jaccard, edge, ptid) do not beat always-pre; the oracle headroom is query-dependent
   (would need question-aware routing, out of scope here).
4. **Paper spine unchanged** (per 2026-07-21 user decision A): lossy-merger ranking
   mechanism + workload-conditional stage law + post-merger SOTA fragility / pre-merger
   robust fix, with this gate reported HONESTLY as the method-search negative result
   ("we tried to beat pre with a merger-aware mask and a disagreement router; pre is
   the fixed point") — which strengthens the mechanism claim (even ranking-informed
   selection at the post stage cannot improve on committing to the pre ranking) rather
   than weakening the paper.
5. Main window decides any n=200 confirmation of the gqa pre==post tie (n=100 only;
   SE ±0.05) before it enters the paper.

### Caveats

- n=100 gate (sample/binomial SE ±0.03–0.05); router thresholds tuned in-sample on the
  pooled n=192 capture sample (sensitivity, not out-of-sample). The pooled router
  metric mixes VQA-acc/ANLS/exact-match across benches (per-bench rows given).
- Single architecture (Qwen3-VL-8B). Chosen frac tuned on the same textvqa n=100 it is
  reported on (in-sample).
- GPU: 12 gate cells (~45 min) + 4 sensitivity cells (~13 min) ≈ **58 GPU-min** on the
  shared A40 (serial, gpu_memory_utilization 0.90), zero failures.
