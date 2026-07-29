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
| benchmark | cascade | pre-alone | fastv-alone | z vs pre | z vs fastv | Pareto | strict |
|---|---|---|---|---|---|---|---|
<!-- FILL from gate_result.json -->

### Aux — total keep 12.5% (cas r_pre=.5/Y=.75 vs pre@12.5% vs FastV@12.5%)
<!-- FILL -->

## Verdict
<!-- FILL: GO/NO-GO -->
Locked consequences: NO-GO → cascade reported as negative result (paper §6,
"tried the cascade, no Pareto improvement"), method frozen = plain RBM, no
further hyper-parameter search. GO → full-split campaign
(runs/cascade/full_splits.sh): both model families on the official full splits
(textvqa 5000 / gqa 12578 / ocrbench 1000 / docvqa 600k n500) + sensitivity
rungs K=4 and r_pre=0.35 (r_fastv=0.2857) at n=500.

## Full-split numbers (only if GO)
<!-- FILL or "N/A — NO-GO" -->

## Assets
- scripts: runs/cascade/{lib,check_engine_consistency,gate_n200,full_splits,run_all}.sh, runs/cascade/gate_analyze.py
- cells: runs/cascade/gate_*.json, runs/cascade/cons_*.json, runs/cascade/degen_*.json
- result: runs/cascade/gate_result.json
