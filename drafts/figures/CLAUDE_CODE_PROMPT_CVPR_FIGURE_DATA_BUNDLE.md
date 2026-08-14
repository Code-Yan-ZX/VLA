# Server Claude Code Task: export measured data for the redesigned CVPR-style figures

You are working in `/media/disk2/YZX/research/vla` on the A40 server. This task
is **data capture and provenance only**. Do not draw the final paper figures
and do not replace the current Overleaf figures. The final composition will be
assembled locally from your exported data with deterministic Python renderers.

Read `STATE.md` first and inspect only the existing runner, mask-capture,
validation, and figure-data scripts needed below. Reuse existing measured runs
where possible. Do not invent saliency maps, attention values, answers, masks,
or benchmark examples.

## Scientific message

The paper studies one controlled variable:

> Does the selector rank visual units before or after the native lossy merger?

The figures must make the following distinction visible:

```text
RBM (ours):       raw 2x2 units -> L2 rank/top-k -> native merger -> LLM
Post-L2 contrast: raw 2x2 units -> native merger -> L2 rank/top-k -> LLM
FastV-k3:         native model path -> query-conditioned layer-2 pruning
```

The server export must support four visual claims:

1. The merger reshuffles unit saliency (`pre` vs `post` rank correlation and
   top-k Jaccard).
2. Units that post selection drops but RBM keeps are high-edge/text-stroke
   units, especially on DocVQA.
3. The ranking-swap control recovers the pre result on the same post forward
   path, with identical kept-set identity where the existing audit claims it.
4. The qualitative examples are representative measured cases, while the
   full-split aggregate result remains the statistical evidence.

## Visual reference principles

Use the supplied CVPR-style reference images only for composition principles:

- a real input image plus question/answer block;
- aligned rows or columns of measured curves, 2-D token maps, and zoomed
  evidence crops;
- grouped sections with short direct labels such as `Rank before merger`,
  `Rank after merger`, `Kept units`, and `Dropped units`;
- consistent color semantics, generous whitespace, and no decorative dashboard
  cards;
- multiple real examples when showing qualitative behavior, never one example
  as the only evidence for a cross-model claim.

Do not copy their logos, typography, icons, colors, or exact layout. Our final
figures will use a restrained colorblind-safe palette: Qwen3 blue, Qwen2.5
terracotta, InternVL3 green, RBM amber, post/FastV blue-gray, neutral black,
and red only for measured incorrect-answer/evidence annotations.

## Locked experimental protocol

Use the existing audited protocol unless a required field is genuinely absent:

- primary visual model: `Qwen/Qwen3-VL-8B-Instruct`;
- greedy decoding, exact stored question, official answer scorer;
- native-resolution configuration used by the headline Qwen3 results;
- retention `kappa=0.25` (also export `kappa=0.125` where already available);
- equal per-image final visual-token budget for pre/post comparisons;
- methods: `RBM` (pre-merger L2), `Post-L2` (post-merger L2), `FastV-k3`
  (query-conditioned layer-2 baseline);
- no image-derived mask, evidence rectangle, or hand-written saliency may
  influence selection;
- every displayed answer, correctness value, mask, score, and token index must
  come from a real model run or an existing audited artifact.

If a required capture would exceed **6 GPU-hours**, stop before launching it and
report the estimate and missing artifact. Otherwise run the smallest
deterministic capture that satisfies the schemas below.

## Required export A: qualitative case bank

Build a candidate bank from the existing Qwen3-VL full/dev samples. Search the
already audited n=200 or full-split outputs first; only rerun a sample when its
per-method mask or per-unit scores are absent.

Select at least six cases, with selection rules written in the manifest:

- two OCR/document cases where RBM is correct and Post-L2 or FastV is wrong;
- two scene-text cases with a clear answer-bearing crop;
- one object-centric GQA case where FastV is correct and RBM is wrong (honest
  regime boundary);
- one near-null or agreement case, if available, to avoid only showing failures.

Do not choose cases by visual attractiveness alone. Selection must be based on
the measured prediction/correctness pattern and then checked for readable
evidence. Record every candidate considered and the deterministic ranking rule.

For each selected case, export:

```text
case.json
  benchmark, split, sample_id, question, ground_truth, aliases
  answers: {rbm, post_l2, fastv_k3}
  correctness: {rbm, post_l2, fastv_k3}
  model_revision, decoding, pixel_cap, keep_ratio
  raw_image_size, processor_image_size, unit_grid_h, unit_grid_w
  final_visual_tokens: {rbm, post_l2, fastv_k3}
  evidence_boxes_source_pixels   # annotations only; never used for selection
  source_run_paths, source_sha256

pre_scores.npz
  unit_l2_pre, unit_rank_pre, keep_pre, kept_indices_pre
post_scores.npz
  unit_l2_post, unit_rank_post, keep_post, kept_indices_post
fastv_scores.npz                 # if FastV per-unit/layer-2 data is available
  layer2_attention_or_score, keep_fastv, kept_indices_fastv
aux.npz
  within_unit_variance, sobel_edge_energy, unit_xy, merger_group_ids
input.jpg                         # exact source image, no crop or resize
```

Masks must be 2-D boolean arrays on the declared native unit grid, and integer
indices must be preserved separately. Store scores before any normalization.
Do not label L2/edge maps as attention. If a quantity is unavailable, write
`null` in the manifest and explain why; do not substitute a proxy.

## Required export B: mechanism arrays

Create one aggregate bundle computed from all available audited samples, not
only the selected qualitative cases:

```text
mechanism_per_image.csv
  benchmark, model, sample_id, n_units, keep_ratio
  spearman_pre_post, kendall_pre_post, jaccard_topk
  mean_edge_pre_keep_post_drop, mean_edge_post_keep_pre_drop
  frac_high_edge_pre_keep_post_drop, frac_high_edge_post_keep_pre_drop
  rank_shift_edge_rho

mechanism_summary.json
  per benchmark: mean/std/median/quantiles and sample count for every metric
  exact source run paths and scorer/hook definitions

swap_control.json
  model, benchmark, n, keep_ratio
  pre_accuracy, post_accuracy, swap_accuracy
  swap_vs_pre_answer_agreement
  jaccard_swap_pre_kept_set
  byte_or_tolerance_contract and verification result
```

The bundle must preserve the distinction between Qwen3-VL byte-exact causal
evidence, Qwen2.5-VL kept-set corroboration, and InternVL3 accuracy-level
replication. Do not pool them into one unsupported causal statistic.

## Required export C: aggregate figure data

Export machine-readable copies of the values used by the main figures, linked
to source files rather than hardcoded in a renderer:

```text
aggregate_main_results.csv
  model, benchmark, n, retention, rbm, post_l2, delta, ci_low, ci_high, p_value

aggregate_regime_map.csv
  model, benchmark, n, fastv_k3, rbm, winner, margin_pp, ci_low, ci_high

retention_curves.csv
  model, benchmark, n, retention, rbm, post_l2, delta, ci_low, ci_high
```

Include the exact units (`pp`, `OCRBench points`, or metric-native score) and
the paired bootstrap/sign-flip settings in the metadata. Values must match the
current Table 1/Table 2/Supplementary Table S3 within rounding tolerance.

## Output location and reproducibility

Write the canonical server bundle here:

```text
/media/disk2/YZX/research/vla/drafts/figures/server_exports/cvpr_figure_data_v1/
```

Required layout:

```text
cvpr_figure_data_v1/
  README.md
  manifest.json
  provenance.json
  checksums.sha256
  cases/<benchmark>_<sample_id>/...
  mechanism/mechanism_per_image.csv
  mechanism/mechanism_summary.json
  mechanism/swap_control.json
  aggregate/aggregate_main_results.csv
  aggregate/aggregate_regime_map.csv
  aggregate/retention_curves.csv
  logs/commands.txt
```

`manifest.json` must list every file, schema version, sample-selection rule,
and whether it was reused or rerun. `provenance.json` must include git commit,
model revision, package versions, random seed, exact commands, hook locations,
image processor geometry, and SHA-256 for every source artifact. Do not place
weights, model caches, raw inference logs, or unrelated datasets in this
bundle. Large arrays may remain server-side, but their checksums and a compact
summary must be included.

Also create a compressed transfer artifact at:

```text
/media/disk2/YZX/research/vla/output/figure_data/cvpr_figure_data_v1.tar.zst
```

The local renderer will consume the extracted copy at:

```text
F:\doct\VLA\drafts\figures\server_exports\cvpr_figure_data_v1\
```

Do not commit the archive, weights, or raw logs. The bundle README must state
which files are safe to copy into the local repository and which remain
server-only.

## Validation gates

Fail closed if any gate fails:

1. Every selected case has real image, question, GT, answers, correctness,
   source run, and SHA-256.
2. Pre/post masks have the declared shape, integer indices agree with masks,
   and the retained count equals `round(kappa * n_units)`.
3. Pre/post final visual-token counts are equal wherever the protocol claims
   iso-token control.
4. Correctness is recomputed with the authoritative scorer, not copied from a
   stale text file without verification.
5. Unit coordinates and merger-group IDs map back to the processor image at
   all four corners; no mask is derived from an evidence rectangle.
6. Aggregate values match the current manuscript tables within stated rounding.
7. `manifest.json`, `provenance.json`, checksums, and command log are complete.
8. No unsupported attention/heatmap claim is present in metadata.

## Final response to the local agent

Return only a concise digest (maximum 20 lines) containing:

- server git commit and exact commands run;
- selected cases and why each passed the measured selection rule;
- counts of reused versus rerun samples and GPU time;
- canonical bundle path and compressed transfer path;
- manifest/provenance/checksum paths;
- validation result and any missing fields;
- one recommendation for the local figure assembly.

Do not edit `main.tex`, do not replace `fig1.pdf`, `fig2.pdf`, or `fig3.pdf`,
and do not commit or push the final paper figures in this task.
