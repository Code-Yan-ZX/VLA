# Figure audit and replacement data (2026-09-16)

No skills, external literature search, new model inference, or GPU experiments were used. This audit inspects existing source records and renders an existing-results chart. It does not independently rerun benchmark scoring.

## New full-split comparison figure

- Output: `figs/fig3_fullsplit_controls.pdf` (vector plot with embedded TrueType fonts), plus PNG preview.
- Generator: `scripts/plot_dcc_stage_controls_20260916.py` from the repository root.
- Machine-readable source: `results/acmmm_final_controls/analysis.json`, keys `P0_1_table1_audit` and `P0_1_pure_stage_control`.
- Protocol and interpretation boundary: `reports/acmmm_final_controls.md:24–45`.
- Each panel is Qwen3-VL-8B, 25% retained units, its complete reported evaluation split. No n=200 cells or different metric definitions enter the figure.
- The generator checks paired sample and token equality for pre-final/Post-L2, the expected split sizes, and equality of the Post-L2 scores across the table audit and control records.

| Benchmark | n | Metric on displayed [0,1] scale | Early-tap RBM | Matched-input pre-final | Post-L2 |
|---|---:|---|---:|---:|---:|
| TextVQA | 5000 | VQA accuracy | 0.6053 | 0.4985 | 0.2217 |
| DocVQA | 5349 | ANLS | 0.4806 | 0.2836 | 0.2377 |
| OCRBench | 1000 | total benchmark points / 1000 | 0.547 | 0.419 | 0.184 |
| GQA | 12578 | normalized exact match | 0.4488 | 0.4207 | 0.4771 |

Labels use conventional half-up rounding to three decimals. Plot heights use the full values above. OCRBench is divided by its nominal 1000-point scale, not by the count of answered samples. All panels share the numeric [0,1] range for readability; their different task metrics are not directly comparable across benchmarks. This is a single operating-point comparison, not a rate-distortion curve. No per-method confidence intervals are fabricated from the available paired-difference intervals.

The early-tap path scores at ViT layer-8 / deepstack[0]-input and slices all four mergers. The final-input pre-final path scores at the main merger's own input and leaves deepstack mergers full during their execution, as does its Post-L2 control. Post-L2 nevertheless scores the concatenated main-plus-deepstack output, so this is not a pure intervention on one merger. The early-tap/pre-final difference must also **not** be labelled a pure feature-depth contribution. The evidence supports a stricter final-input contrast plus a separately reported early-tap algorithm.

Suggested caption:

> Full-split Qwen3-VL-8B scores at 25% retained units. Each panel compares early-tap RBM, final-input pre-final, and Post-L2 using one benchmark's metric; OCRBench points are divided by 1000. Pre-final removes the early feature tap but Post-L2 still uses a concatenated output score. Early-tap RBM also differs in deepstack execution, so its gap to pre-final is not a depth-only estimate. Scores are not directly comparable across benchmarks.

Visual QA: PNG inspected after rendering; four panels and all labels are legible without clipping or overlap. Grid, three distinct colors, and hatch patterns distinguish methods; PDF contains vector bars, text, and paths.

The full-split bar chart is retained in the final manuscript alongside the two user-required qualitative figures. It was selected over the older `fig3_word.png` mechanism panel because the latter makes a stronger text-demotion claim than the audited evidence supports.

## Original figure disposition

### Figure 1: user-specified schematic — retain unchanged

Original asset: `figs/fig1_user_20260915_v2.png`. It is intentionally preserved. Its generic “vision encoder → native merger” drawing does not explicitly show the headline method's early layer-8/deepstack[0] tap. Its embedded `K=round(ρN)` uses rho for retention, whereas the old text uses kappa for retention and rho for RankBridge's protected-budget fraction. Resolve this in the new caption/text without editing the user-selected image. The diagram's “text evidence demoted” is a conceptual illustration, not a proved text-localization diagnostic.

Suggested clarification: “Schematic of unit-preserving selection. The evaluated Qwen3-VL RBM scores an early ViT tap; matched-input controls are reported separately. The image denotes retention by rho, corresponding to kappa in the text.” If the new text adopts rho consistently for retention and a different hybrid symbol, say so instead.

### Original displayed Figure 2: `fig2_word.png` — retained as selected illustrative cases

PDF text extraction and `fig2_word.png` show OCRBench airport-board (`A105-108`) and document (`OCTOBER 1999`) cases, with RBM marked correct and Post-L2/FastV-k3 incorrect. No visible correctness contradiction with the existing case records was found.

Direct provenance:

- `drafts/figures/server_exports/cvpr_figure_data_v1/cases/ocrbench_ocr0422/case.json`: audited n=200 subset, keep 0.25, 216 candidate units, 54 retained per method, RBM/Post/FastV correctness 1/0/0.
- `drafts/figures/server_exports/cvpr_figure_data_v1/cases/ocrbench_ocr0804/case.json`: same audited subset, keep 0.25, 744 units, 186 retained per method, correctness 1/0/0.
- `drafts/figures/server_exports/cvpr_figure_data_v1/manifest.json`: selection rule explicitly chooses two OCR/document cases where RBM is correct and the other two fail. They are selected examples, not an unbiased sample of performance.

The recorded generation contract states displayed answers/correctness come from audited runs, while masks are recaptured; RBM regenerated answer strings are not byte-identical for these cases. Do not call the overlays an exact end-to-end causal trace of the displayed original answers. Keep the caption at “selected illustrative cases / recorded retained-unit masks”; aggregate claims must rely on the full-split table.

### Original displayed Figure 3: `fig5_rate_distortion.pdf` — replace

Generator `scripts/plot_dcc_rate_distortion.py:21–41` hardcodes n=200 diagnostic numbers. The metric provenance is insufficient in that script, and the values do not match the newer official-score subsets. For example its 25% TextVQA values are .695/.255, DocVQA .725/.390, and GQA .320/.380. It is not legitimate to combine these with the present full-split official-score claims. The replacement uses only the explicitly audited full-split values above.

### Original displayed Figure 4: `fig3_vector.pdf` mechanism — remove from new main text

PDF text extraction and the corresponding PNG confirm the title “Merger demotes text-stroke units”, which exceeds a Sobel edge association. The swap panel also prints TextVQA RBM .598 versus Swap .603, despite the old prose claiming identical recovered accuracy and byte-identical retained outputs. This difference might reflect scope, scoring, or differing answer trajectories; the asset itself does not explain it. Its n=200 answer and n=30/31 kept-set controls should not be represented as one fully paired byte-identical end-to-end experiment. Use carefully scoped textual evidence if retained, not this unmodified strong-mechanism graphic.

### Original displayed Figure 5: `fig4_vector.pdf` survival maps — retained as separately scoped historical examples

The companion `fig4_word.png` embeds early diagnostic text: “PRE = block-8 (deepstack_0) unit L2; POST = merged token (main+deepstack cat) L2”, TextVQA-200 pre .695/post .255, DocVQA-200 pre .725/post .39. These are not final full-split or final-input-control scores. It also labels positive rank-edge correlation as merger demotion of high-edge/text, conflating text and edge energy. The local TextVQA example has a slightly negative edge-shift correlation, which prevents treating every example as evidence for edge demotion. The final caption therefore identifies the historical $n=200$ diagnostic and limits the figure to selected spatial-allocation examples rather than aggregate or merger-only evidence.
