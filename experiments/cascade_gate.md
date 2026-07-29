# Cascade two-stage gate — digest (pre-registered, DECISIONS 2026-07-28)

## Method
Single-budget two-stage visual-token compression:
**Stage 1 (pre-merger, query-blind)**: RBM-style L2 top-κ over merger-input
UNITS (4 consecutive ViT patches = 1 merge unit; score = mean per-patch L2;
k_i = max(1, round(f_i·X)) per image, X = `--r-pre` KEEP fraction) → only
survivors pass the native PatchMerger. Motivation: keep raw-patch information
for text-dense/OCR units, avoiding merger distortion on the 75% discarded
content. **Stage 2 (FastV, query-conditioned)**: at LLM layer K, prune the
SURVIVING image tokens by averaged layer-K attention (last-query row), keeping
fraction (1−Y) of the Stage-1 remainder (Y = `--r`). **Total keep = X·(1−Y)**
(per-image rounding). Main config X=0.5/Y=0.5 → 25%; aux X=0.5/Y=0.75 → 12.5%.

## Implementation (src/v3_premerger/baselines_hf.py; runner UNTOUCHED)
- `--mode pre` / `--mode cascade` added to the HF-transformers harness
  (same prompt/sampling/scorer contract as the vLLM runner; eager attn).
- Pre mask mirrors the runner's PreMergerPruner EXACTLY: unit grouping, L2
  score, per-image k, top-k; ONE mask from the FIRST merger called (Qwen3-VL:
  deepstack[0] input = layer-8 features — runner diag
  `mask_computed_at='deepstack_0'`; Qwen2.5-VL: single main merger) applied
  consistently to main + every deepstack merger's row set (all mergers share
  the same block-major patch order — verified in
  transformers `Qwen3VLVisionModel.forward`: deepstack mergers fire inside the
  block loop on intermediate hidden_states, main merger last).
- Mechanics: full native vision stage runs; survivors are index-selected out
  of the captured inputs_embeds / position_ids / deepstack rows. Valid because
  the PatchMerger is per-unit (LayerNorm+MLP over each 4-patch group
  independently) → a kept unit's merged token is BIT-IDENTICAL pre/post merge
  (the runner's M3 swap≡pre identity). Survivors keep their native mrope 3-D
  coordinates (index_select on get_rope_index positions).
- Both families: Qwen3-VL deepstack (3 mergers + main, `MergerTap` hooks all,
  first-call wins) and Qwen2.5-VL (main merger only; no deepstack attr → skipped).
- Output JSON: `r` = TOTAL drop (runner convention); `r_pre` / `r_fastv` /
  `total_keep` + per-sample `diag.pre{...}` carry the stage-wise definition.

## Degenerate self-tests
- CPU `--dry-check` ALL PASS: premerger-mask pure-function test (runner
  semantics: per-image top-k, keep=1.0 identity, k floors at 1);
  apply_premerger (survivor positions + deepstack row alignment);
  vllm-mimic mrope unit test (grid-block + truncation, 3- and 4-row);
  **Y=0.0 → cascade ≡ pre-alone** (hidden allclose ≤1e-5); **X=1.0 → cascade ≡
  FastV-alone** (identity apply + hidden allclose); two-stage count check
  (6 units → pre 3 → FastV 2, L 12→9→8, cache cropped).
- py_compile OK; `bash -n` all scripts OK.
- Real-weights degeneracies (textvqa n8): **X=1 cascade(r_pre=1,r_fastv=.5) vs
  FastV(.5): 8/8 EXACT answers; Y=0 cascade(r_pre=.25,r_fastv=0) vs pre(.25):
  8/8 EXACT answers** (index-identical paths, as designed).

## Engine consistency (HF-pre vs vLLM-pre, textvqa n16, matched by id)
Reference: runs/full_matrix/j7_qwen3vl_pre_textvqa_r0.750_full.json (vLLM
0.19, l2 pre, r0.75). Pre-registered pass: ≥14/16 answer strings equal.
- First attempt: **12/16 → FAIL**. Diagnosis:
  - Kept-unit-set Jaccard vs the runner (`--save-unit-scores` cell
    runs/cascade/jac_runner_pre_textvqa_n16.json, compare_kept_sets.py):
    sizes/k IDENTICAL on all 15 common samples, 4/15 bit-exact,
    **mean Jaccard 0.986** → selection logic is the runner's formula exactly;
    the ~1.4% unit flips are engine ε (vLLM flash vs HF eager ViT features)
    breaking near-ties at the top-k boundary — NOT a mask bug.
  - ROOT CAUSE of the answer flips = **mrope position representation**:
    vLLM `_get_mrope_input_positions` (qwen3_vl.py) lays down the FULL (1,H,W)
    grid block at the image location while the runner scales placeholders to
    k, and the runner truncates the over-long array to the token count — so
    survivors carry ROW-MAJOR FIRST-k grid coordinates and post-image text
    inherits the grid continuation (the unfixed "Qwen3 interleaved
    self-consistent" path the J7 cells ran with). HF originally kept each
    survivor's native sparse coordinate.
- FIX: `mimic_vllm_pre_positions` (default `--mrope vllm-mimic` for
  pre/cascade; `native` retained as diagnostic) reproduces that layout
  exactly (unit-tested).
- After fix: **14/16 → PASS** (≥14 threshold). Residual 2 = irreducible
  cross-engine ε (kernel ε + boundary unit flips at Jaccard 0.986).
- Guardrails added after OOM/poison incidents: `wait_gpu` mem<3000MiB with
  60s STABILITY re-check (co-tenant load-ramp collision guard; sunlogin
  phantom-util ignored), `cell_done` rejects OOM-poisoned cells (skipped >
  n/4) so they re-run instead of being resume-skipped.

## Gate (Qwen3-VL-8B, n=200 × 4 benchmarks, official rescore)
Rule (LOCKED): GO = Pareto — EVERY benchmark cascade ≥ max(pre-alone,
fastv-alone) − 0.5pp AND ≥1 benchmark strictly beats BOTH arms (paired
per-sample correct, McNemar z ≥ 1.5). Metrics: textvqa VQA-acc, docvqa ANLS,
gqa exact-match, ocrbench official 0/1 (+ 5-category /1000 extrap). Pixels:
docvqa 600k cap (HF eager-attn constraint, disclosed), others processor default.

### Main — total keep 25% (cas r_pre=.5/Y=.5, K=2 vs pre@25% vs FastV@25%)
| benchmark | cascade | pre-alone | fastv-alone | z vs pre | z vs fastv | Pareto (≥max−0.5pp) | strict (both z≥1.5) |
|---|---|---|---|---|---|---|---|
| textvqa | 0.5950 | 0.5967 | **0.6800** | −0.152 | −2.335 | ❌ (0.595 ≪ 0.675) | no |
| docvqa | 0.4779 | 0.4239 | **0.5183** | +2.066 | −1.500 | ❌ (0.478 < 0.513) | no |
| ocrbench | 0.3978 | **0.5801** | 0.1713 | −5.154 | +6.112 | ❌ (0.398 ≪ 0.575) | no |
| gqa | 0.4050 | 0.4150 | **0.4900** | −0.333 | −2.429 | ❌ (0.405 < 0.485) | no |

### Aux — total keep 12.5% (cas r_pre=.5/Y=.75 vs pre@12.5% vs FastV@12.5%)
| benchmark | cascade | pre-alone | fastv-alone | z vs pre | z vs fastv | Pareto | strict |
|---|---|---|---|---|---|---|---|
| textvqa | 0.4050 | **0.5000** | 0.4450 | −2.540 | −0.896 | ❌ | no |
| docvqa | 0.2070 | **0.2980** | 0.2779 | −2.012 | −2.064 | ❌ | no |
| ocrbench | 0.1713 | **0.3591** | 0.0884 | −5.013 | +3.441 | ❌ | no |
| gqa | 0.3500 | 0.3950 | **0.4550** | −1.286 | −3.130 | ❌ | no |

OCRBench /1000 extrap: cascade r25 = 370.5 (pre 580, fastv 171); r12 = 160.4 (pre 359, fastv 88).
Pairing integrity: n_common matched (200/200/181/200), mean_ptid IDENTICAL across the three arms in every cell → same images, same budget, fair paired comparison. All 24/24 gate cells present (missing_cells: []).

## Verdict — NO-GO
Cascade fails condition 1 on ALL 8 (budget×bench) cells — never within 0.5pp
of max(pre, fastv), typically 8–18pp below — and condition 2 holds nowhere
(the only positive cas-vs-pre z, docvqa r25 +2.07, is paired with a loss to
fastv, z −1.5). Pattern is structural, not noise: cascade sits BETWEEN the two
single methods and is dominated by the better one on every benchmark —
query-blind Stage-1 discards degrade the token set that query-conditioned
Stage-2 (FastV) attends over, while on text-dense/OCR the FastV stage
destroys exactly the raw-patch information Stage-1 was built to protect
(cas OCRBench 0.398 ≈ between pre 0.580 and fastv 0.171; same shape at r12).
Two training-free selectors composed in series do NOT complement. Consistent
with the earlier hybrid/router FAIL (pre is a fixed point; nothing downstream
recovers what pre- or fastv-stage threw away).
Locked consequences: NO-GO → cascade reported as negative result (paper §6,
"tried the cascade, no Pareto improvement"), method frozen = plain RBM, no
further hyper-parameter search. GO → full-split campaign
(runs/cascade/full_splits.sh): both model families on the official full splits
(textvqa 5000 / gqa 12578 / ocrbench 1000 / docvqa 600k n500) + sensitivity
rungs K=4 and r_pre=0.35 (r_fastv=0.2857) at n=500.

## Full-split numbers (only if GO)
N/A — NO-GO. `runs/cascade/full_splits.sh` NOT executed (pre-registered
consequence). Cascade reported as negative result in paper §6; method frozen =
plain RBM; no further cascade hyper-parameter search (DECISIONS 2026-07-29).

## Assets
- scripts: runs/cascade/{lib,check_engine_consistency,gate_n200,full_splits,run_all}.sh, runs/cascade/gate_analyze.py
- cells: runs/cascade/gate_*.json, runs/cascade/cons_*.json, runs/cascade/degen_*.json
- result: runs/cascade/gate_result.json
