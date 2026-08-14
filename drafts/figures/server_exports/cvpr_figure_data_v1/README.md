# cvpr_figure_data_v1 — server-side measured data for the CVPR-style figures

Data capture and provenance only. This bundle is built on the A40 server,
August 2026, by `scripts/export_cvpr_cases.py` (per-case capture),
`scripts/export_cvpr_mechanism_ocr.py` (OCRBench mechanism sweep), and
`scripts/export_cvpr_figure_data_compile.py` (assembly, validation, packaging).

The final paper figures are NOT drawn here. They are composed locally in
`F:\doct\VLA\drafts\figures\server_exports\cvpr_figure_data_v1\` by
deterministic Python renderers that consume this bundle.

## Scientific message supported

> Does the selector rank visual units before or after the native lossy merger?

RBM (ours) ranks raw 2x2 units by L2 *before* the native merger;
Post-L2 ranks the merged rows *after*; FastV-k3 is the query-conditioned
layer-2 baseline. Every mask, score, answer, and correctness value in this
bundle comes from a real model run or an audited artifact — nothing is
invented, and no image-derived mask or evidence annotation influenced any
selection.

## Layout

```text
cvpr_figure_data_v1/
  README.md                this file
  manifest.json            file list, schema version, sample-selection rules,
                           reuse-vs-rerun accounting
  provenance.json          git commit, model revision, package versions, seed,
                           commands, hook locations, processor geometry
  checksums.sha256         SHA-256 of every file in the bundle
  cases/<benchmark>_<id>/  per-case: case.json + pre_scores.npz +
                           post_scores.npz + fastv_scores.npz + aux.npz +
                           input.jpg
  mechanism/               mechanism_per_image.csv, mechanism_summary.json,
                           swap_control.json
  aggregate/               aggregate_main_results.csv,
                           aggregate_regime_map.csv, retention_curves.csv
  logs/commands.txt        exact commands + truncated run logs
```

## Selected cases (6; rules in manifest.json)

| case | benchmark | D | why it was selected (measured) |
|---|---|---|---|
| ocr0422 | OCRBench | airport board | RBM ✓ `A105-108`; Post-L2 ✗; FastV ✗ (strict three-arm flip) |
| ocr0804 | OCRBench | FUNSD doc | RBM ✓ `OCTOBER 1999`; Post-L2 ✗ `4/14/99`; FastV ✗ `3/3/2013` |
| textvqa_34982 | TextVQA | shirt logo | RBM ✓ `rock the stripes`; Post-L2 ✗; FastV ✓ — scene text, answer-bearing crop |
| textvqa_35164 | TextVQA | sash text | RBM ✓ `veedol`; Post-L2 ✗; FastV ✓ — scene text |
| gqa_201056134 | GQA | object | FastV ✓ `ball`; RBM ✗ — honest regime boundary (RBM wrong) |
| textvqa_35000 | TextVQA | recycle sign | all three arms ✓ — agreement case (not only failures) |

## Mechanism bundle (per-image)

`mechanism/mechanism_per_image.csv` — one row per image for every audited
survival capture: TextVQA / DocVQA / GQA (n=64 each, reused audited capture at
1.5 M-pixel config) plus OCRBench (n=200, capture-only sweep at the same
config, added for this bundle). Columns follow the task schema:

- `spearman_pre_post`, `kendall_pre_post`, `jaccard_topk` — how much the
  merger reshuffles unit saliency (claim 1);
- `mean_edge_pre_keep_post_drop`, `mean_edge_post_keep_pre_drop`,
  `frac_high_edge_*`, `rank_shift_edge_rho` — high-edge (text-stroke) units
  dropped by post but kept by RBM (claim 2);
- `n_units`, `keep_ratio`, `unit_grid_h/w`.

`mechanism/mechanism_summary.json` — per-benchmark mean/std/median/quantiles +
capture metadata + exact source paths.
`mechanism/swap_control.json` — Qwen3 (byte-exact causal: kept-set Jaccard
1.0, n=32 probe; n=200 answer identity), Qwen2.5 (kept-set corroboration),
InternVL3 (accuracy-level replication, n=200). Kept as separate evidence
tiers — never pooled into one unsupported statistic.

## Aggregates (Export C)

CSV rows linked to their source files:
- `aggregate/aggregate_main_results.csv` — Table 1 parity (pre vs post per
  model × benchmark × κ ∈ {0.25, 0.125}); paired-bootstrap 95% CI +
  sign-flip p from `experiments/paired_metric_statistics.md`
  (n_resamples=20000, seed=0).
- `aggregate/aggregate_regime_map.csv` — Table 2 parity (FastV-k3 vs RBM @25%).
- `aggregate/retention_curves.csv` — Qwen3/Qwen2.5 TextVQA+DocVQA at
  keep ∈ {0.75, 0.5, 0.25} (n=200), values per `experiments/j8_ablations.md`.

Units are printed per row (`pp`, `OCRBench points`, metric-native ANLS /
VQA-acc). Rounding is 3 decimals; manuscript tables print ≤4.

## Provenance conventions

- Displayed answers are the audited run strings (real vLLM/HF runs); the
  borrower verifier re-generates and compares. Long greedy outputs can diverge
  between vLLM kernels and eager bf16 (numerical; documented per case in
  `case.json` under `audit_consistency.regen` with a `byte_identity` flag);
  correctness parity is the contract everywhere, and the audited answer is
  authoritative.
- FastV per-token scores are the mean-headed last-query-row attention at the
  layer-3 prune point — a measured, query-conditioned score at one
  layer, not a general attention heatmap claim (gate 8).
- Sobel edge energy and within-unit variance are analysis features for the
  claim-2 text-density analysis. They are never used to select anything.

## Files safe to copy into the local repo / what stays server-only

Safe to copy (all of this bundle): the compressed transfer is
`output/figure_data/cvpr_figure_data_v1.tar.zst`. All files here are data +
provenance (no weights, no model caches, no raw inference logs, no datasets).

Server-only (NOT in this bundle): the vLLM/HF model cache
(`~/.cache/huggingface`), the raw run JSONs under `runs/` (they are referenced
by path + SHA-256 in `provenance.json`), and the large `survival_*.npz`
captures in `runs/v3_merger_aware/survival_capture/` (their checksums and a
compact summary are included in `mechanism_summary.json`).