# paper_v4 figures — generation notes

Generator: `gen_paper_v4_figures.py` (pure CPU, matplotlib 3.8 + the
colorblind-safe palette in `_style.py`). Re-run with
`python3 gen_paper_v4_figures.py` from this directory. Every figure is emitted
as **PDF (vector, 0 embedded rasters) + PNG (300 dpi)** and is sized to scale to
a single (3.5 in) or double (7 in) column via `\includegraphics[width=...]`.
Correctness is encoded by **✓ / ✗ marks in addition to color** (red = wrong,
green = correct), so no panel relies on hue alone.

| file | panels | native size (in) | column |
|---|---|---|---|
| `fig1_pipeline` | schematic, 2 rows | 6.31 × 2.20 | double (scales up) |
| `fig2_mechanism` | a M1 + b M3 | 7.20 × 3.06 | double |
| `fig3_retention_gap` | 2×2 bench×model | 7.10 × 4.73 | double |
| `fig4_qualitative` | 3 example cards | 7.00 × 4.53 | double |

---

## Fig 1 — `fig1_pipeline` (method schematic)
- **Conclusion:** pre- vs post-merger pruning differ *only* in where the saliency
  score is tapped; the native 2×2 merger (+ deepstack on Qwen3-VL) and the LLM
  are identical in both pipelines.
- **Design:** row (a) = this work (RBM): L2 rank on **raw** 32 px-unit features,
  keep top-κN, *then* merge — red solid box. Row (b) = published family
  (VisionZip-style): merge all units first, rank the **merged** tokens, keep
  top-κN — grey dashed box. Token-count labels (N → κN kept / N merged → κN)
  sit above each arrow.
- **Data source:** method definition (paper_v4 §3.1); no numeric data.
- **Notes:** pure `patches`/`arrows` diagram (vector). No chartjunk.

## Fig 2 — `fig2_mechanism` (M1 + M3)
- **Conclusion:** (a) the learned merger reshuffles unit saliency almost from
  scratch (pre↔post Spearman ρ = 0.14–0.36, Jaccard@25% = 0.18–0.28; strongest
  on text-dense DocVQA, weakest on object GQA); (b) holding the forward path at
  *post* and swapping in the *pre* ranking recovers pre accuracy (Δ ≤ 0.005,
  DocVQA byte-identical) → the entire pre–post gap is a **ranking** effect.
- **Panel a data:** `drafts/mechanism_verification_report.md` §1 — ρ and
  Jaccard@25% with per-image s.d. (n = 64 images/bench, Qwen3-VL-8B). Dashed
  reference line at ρ = 1 (identical rankings).
- **Panel b data:** same report §3 (n = 200, official metrics): Uncompressed /
  post / pre / swap. Pre and swap bars share a blue fill (swap hatched) joined
  by a "≡ (Δ ≤ 0.005)" bracket.
- **Gap / caveat:** single architecture (Qwen3-VL-8B) for M1–M3. The M1/M2
  raster PNGs (`token_survival_m1_rank_overlap.png`,
  `token_survival_m2_edge_demotion.png`) are retained as supplementary-grade
  per-image evidence; Fig 2 redraws the headline statistics as vector bars. M2
  (edge-directionality) is reported in the text/table, not as a panel.

## Fig 3 — `fig3_retention_gap` (CORE: retention-vs-gap curves)
- **Conclusion:** the pre-merger lead is ~0 (≈ tie, within noise) under shallow
  compression and **widens monotonically** with depth, reaching +38.4 / +26.1 pp
  (TextVQA) and +24.3 / +11.0 pp (DocVQA) at 25% retention — the accuracy-side
  signature of the lossy-merger mechanism.
- **Axes:** x = visual-token retention {100, 75, 50, 25 %} (compression depth
  increases to the right); y = official score (VQA-acc / ANLS). Blue = pre
  (ours), red = post (VZ-style).
- **Marker semantics:** ● = n = 200 subset; ○ = full split. 100 % = `none`
  full-split (`runs/full_matrix/j7_main_table.json`); 75 % / 50 % = L2 n = 200
  subset (`runs/full_matrix/ablations/j8_summary.json`); 25 % = L2 **full
  split** (`j7_main_table.json`, r = 0.75) overlaid on the subset curve to show
  subset↔full consistency; gap double-arrows annotated at 25 %.
- **Data gap / honesty:** Qwen3-VL **DocVQA has no 100 % point** (native
  resolution overflows context on huge images → partial skip); that panel starts
  at 75 % and is footnoted. The centroid-attention selector-invariance probe at
  25 % (`j8_ablations.md` §B) is intentionally *not* on this curve (different
  scorer family); report it in the table, not the figure.

## Fig 4 — `fig4_qualitative` (3 of 10 catalogued cases)
- **Conclusion:** post-merger methods erase / corrupt small-dense text after the
  2×2 average (text "vanishes" or a 1000× unit error, e.g. billion→million),
  while pre-merger selection protects it; the object-centric GQA case is the
  honest trade-off direction (post correct, pre wrong).
- **Selected:** TextVQA 35014 (date erased), DocVQA 58439 ($1.3 BILLION→million,
  post **and** VZ-style wrong identically), GQA 201370409 (post-only correct —
  balance). Source: `drafts/qualitative_examples.md`.
- **Design note:** original images are **omitted** (copyright / file size) and
  replaced by hatched placeholder boxes captioned "image omitted (see
  supplementary)"; each card is a structured Q → GT → pre/post(/VZ) schematic
  with a failure-signature tag. ptid is identical per image across conditions,
  so the contrast isolates selection order.
- **Data gap:** per-sample VZ-style predictions exist only for DocVQA, so the
  VZ row appears only in the DocVQA card; GQA has no VZ run.

---

## Superseded / not for submission
`retention_curves.{png,pdf}` and `gen_fig1..4.py` (→ `fig1_gap`,
`fig2_concurrency_prune`, `fig3_controller`, `fig4_pareto`) predate the
official-metric rescore and the v4 figure plan — kept for history only. Use the
`fig{1..4}_*` set above for paper_v4. `stage_law.png` and the
`token_survival_{textvqa,docvqa}.png` single-image panels remain
supplementary-grade.
