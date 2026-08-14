# CVPR-style figure redesign for Rank Before You Merge

This is the local assembly specification. The server task exports measured
arrays and provenance; local deterministic renderers assemble the final PDF,
SVG, and PNG. The reference images motivate the visual grammar only: real
input/question blocks, aligned measured curves and maps, grouped sections,
direct labels, and generous whitespace.

## Figure hierarchy

### Fig. 1 — Why the ranking stage matters

Purpose: establish the insight in three seconds, without making one qualitative
sample carry the cross-model claim.

```text
Question + input image | measured unit state | ordering intervention | outcome
                       |                   |                        |
                       | pre score map     | RBM: Rank -> Keep       |
                       | post score map    | Post: Merge -> Rank    |
                       | kept/drop mask     | FastV compact baseline |
```

Use two representative rows (one OCR/document, one scene-text or GQA boundary)
selected by measured correctness, not aesthetics. Each row contains the real
image, exact question/answer, a small evidence box, pre/post score curves or
maps, and retained-unit masks. Add a compact aggregate strip at the right or
bottom: full-split TextVQA/DocVQA/OCRBench deltas and a neutral GQA marker. The
aggregate strip is the claim anchor; the examples are explanatory insets.

Do not use attention language for L2 or edge maps. Label maps explicitly as
`pre-merger L2`, `post-merger L2`, `retained units`, and `dropped units`.

### Fig. 2 — Full-split stage effect

Purpose: main statistical result.

Keep the current paired-bootstrap forest plot with three model colors and four
benchmark rows. Add a thin subtitle or footer stating `25% retention | equal
post-merger token budget | pre - post | official full split`. Mark the zero line
as `no stage effect`. Keep GQA close to zero and explicitly label it as the
scope boundary. Do not replace this with a qualitative gallery or a bar chart.

### Fig. 3 — Mechanism: reshuffle, demotion, swap recovery

Purpose: make the causal explanation visually testable.

Use three aligned panels in the visual density of the references:

1. `Rank reshuffle`: per-image distributions of Spearman rho and top-k Jaccard
   for DocVQA, TextVQA, and GQA. Show individual points lightly, with median and
   interval overlays; retain the text-density ordering.
2. `Text-stroke demotion`: paired distributions or compact bars for edge energy
   in `pre-kept/post-dropped` versus `post-kept/pre-dropped`, plus the measured
   rank-shift/edge correlation. Avoid an invented heatmap.
3. `Ranking swap`: a compact flow showing post forward path with pre ranking,
   then numeric recovery (`swap ~= pre`, `Jaccard=1.000` where byte-exact). Use
   the Qwen3 causal contract and label Qwen2.5 as kept-set corroboration only.

The current retention curves become Supplementary Fig. S3, with paired CIs and
the shallow DocVQA negative points retained.

### Fig. 4 — Workload regime and honest qualitative boundary

Purpose: explain where RBM is and is not the default.

Panel (a): a benchmark x model regime map showing FastV wins, RBM wins, and
uncertain cells with direct labels and margins. Panel (b): two or three real
examples, including one GQA case where FastV is correct and RBM is wrong. Every
example must show exact question, answer, correctness, and same-budget status.

If the body page budget cannot support Fig. 4, keep Table 2 in the main text and
move this figure to the supplement. Never use a single airport example as the
only qualitative evidence.

## Shared visual system

- White or very light neutral background; no gradients, shadows, or decorative
  dashboard cards.
- Model identity: Qwen3 blue, Qwen2.5 terracotta, InternVL3 green.
- Method identity: RBM amber; Post/FastV blue-gray; incorrect annotation red.
- Color is always duplicated by direct labels, marker shape, or line style for
  grayscale and color-vision accessibility.
- Use vector text and measured raster inputs only. Minimum final text size is
  7 pt at ACM two-column width.
- Use short labels inside artwork; put interpretation and caveats in captions.
- Keep axes/heatmaps on shared grids where comparisons are intended. State
  units (`pp`, `OCRBench points`, `L2`, `edge energy`) on every independent axis.

## Data-to-panel contract

The local renderer must read only the server bundle:

```text
drafts/figures/server_exports/cvpr_figure_data_v1/
```

No score, answer, mask, coordinate, or confidence interval may be hardcoded in
the renderer. Every visible value must resolve to a manifest entry and source
hash. The renderer should emit a machine-readable `figure_provenance.json`
mapping each panel and visible number to its source file and run.

## Assembly order

1. Validate and unpack the server bundle; fail closed on schema/checksum errors.
2. Render Fig. 1 candidate layouts using the selected measured cases.
3. Render Fig. 3 mechanism panels from per-image arrays, not summary prose.
4. Re-render Fig. 2 from the aggregate CSV and compare values to Table 1.
5. Assemble optional Fig. 4 and the supplementary retention/qualitative figures.
6. Compile `drafts/overleaf_submission/main.tex`, render all body pages, and
   inspect at both native and ACM embedded widths.

The final paper should make the evidence order explicit: insight (Fig.1),
full-split result (Fig.2), mechanism (Fig.3), workload boundary (Fig.4 or
Supplement).
