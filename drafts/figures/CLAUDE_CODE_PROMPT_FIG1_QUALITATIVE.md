# Claude Code server task: rebuild Fig. 1 as the paper frontispiece

You are working in `/media/disk2/YZX/research/vla` on the A40 server. Work
end-to-end; do not stop at a plan. Read `STATE.md` first, then only the files
named here. Do not read the whole repository.

## Design correction

The previous three-dataset x four-method layout is rejected. It produces a
contact sheet, not a strong opening figure. The visual benchmark is the current
`drafts/overleaf_submission/figs/fig1.pdf`: keep its clean left-to-right flow,
large imagery, restrained amber/blue palette, and two controlled method lanes.
Do not copy its clutter: remove repeated stage prose, the large legend, tiny
footer qualifications, and the arbitrary SpongeBob input.

The new Fig. 1 must communicate one sentence in three seconds:

> Rank before the native merger preserves answer-bearing text that later
> selection can no longer recover.

This is a mechanism-plus-outcome frontispiece, not a benchmark dashboard and
not a multi-example gallery.

## Locked flagship example

Use one real OCRBench sample, not three datasets.

Primary: `ocr0422` from
`drafts/figures/camera_ready/contact_sheet_manifest.json`.

- Image: airport information board with strong yellow text on dark blue.
- Question: `which counter is boarding?`
- Ground truth: `A105-108`.
- RBM answer: `A105-108` (correct).
- FastV-k3 answer: `no counter is currently boarding` (incorrect).
- Post-L2 is also incorrect/incomplete in the recorded run.
- All three arms have the same final prompt-token count (`ptid=69`) and the
  same 25% final visual-token budget.

This example is preferred because it is visually striking and the answer text
is spatially localizable. Use `ocr0804` only if the `A105-108` evidence cannot
be isolated into a clean readable crop. The fallback has GT/RBM
`OCTOBER 1999`, FastV `3/3/2013`, Post-L2 `4/14/99`, `ptid=219`.

Do not mine more candidates unless both locked examples fail the evidence-crop
gate. TextVQA `35014` and DocVQA `58439` belong in a later qualitative figure
or the supplement, not in opening Fig. 1.

## Scientific contract

Use Qwen3-VL-8B-Instruct, greedy decoding, the stored question, and 25% final
visual-token retention. Display only methods with real same-sample data:

1. `RBM (ours)`: merger-input L2 ranking;
2. `Post-L2`: same L2 selector after the native merger, the controlled stage
   contrast;
3. `FastV-k3`: the strong query-conditioned baseline, shown only in the compact
   answer comparison, not as a third full pipeline lane.

For each displayed method, recover the exact answer, authoritative correctness,
final retained-token indices/mask, processor geometry, and final visual-token
count. Run only this sample if a mask is absent. Never substitute aggregate
accuracy or derive a mask from image pixels.

Assert equal final token count across methods and abort on mismatch. Map every
mask back to the shared processor-image grid and document the mapping. Label
mask overlays `retained visual tokens`, never `attention` unless the values are
literally attention weights. Evidence rectangles are annotations only and must
not influence selection.

The aggregate OCRBench result may be used only as a small context anchor:
`RBM +16.0 pp over FastV-k3` on the audited Qwen3 comparison. Preserve its
scope/provenance from
`drafts/figures/camera_ready/fig1_overview_provenance.md`; do not imply overall
SOTA or universal superiority. FastV remains stronger overall on several
TextVQA/DocVQA settings.

## Required data bundle

Create `drafts/figures/frontispiece_fig1/data/` containing:

- the one selected source image;
- `manifest.json` with benchmark/id, exact question, GT aliases, exact answers,
  correctness, `ptid`, keep ratio, source run paths, processor size, unit grid,
  and evidence box coordinates in source pixels;
- one NPZ per method with boolean 2-D `keep` and raw integer kept indices;
- `provenance.json` with SHA-256 hashes, repo commit, model revision, package
  versions, seed, pixel cap, exact commands, and hook definitions.

Reuse these sources first:

- `drafts/figures/camera_ready/contact_sheet_manifest.json`
- `runs/cascade/gate_pre25_ocrbench.json`
- `runs/rankbridge/locked_fst3_ocrbench_n200.json`
- `runs/cascade/gate_post25_ocrbench.json`
- `eval/subsets/ocrbench_200.jsonl`
- `src/v3_premerger/baselines_hf.py`
- the grid/mask mapping code in
  `drafts/figures/real_data_pipeline/scripts/render_compare.py`

Do not commit weights, raw run logs, caches, or unrelated data.

## Locked composition

Write a deterministic renderer at
`drafts/figures/frontispiece_fig1/render_fig1.py`. Output a landscape ACM
`figure*`, approximately 7.0 x 3.6 inches. Use Matplotlib/Pillow, vector text,
embedded fonts, and the palette/typography of the existing Overleaf Fig. 1.

Use one continuous visual canvas with three zones, not cards and not a 2x2
dashboard:

```text
 INPUT + QUESTION       WHERE TO RANK                         WHAT SURVIVES / ANSWER
 +----------------+     +---------------------------+         +-----------------------+
 | original image | --> | (a) RBM: rank -> keep -> | ------> | zoom + token blocks   |
 | red evidence   |     |     native merger         |         | A105-108  CORRECT     |
 | box            |     +---------------------------+         +-----------------------+
 +----------------+     +---------------------------+         +-----------------------+
 Q: which counter      | (b) Post: native merger ->| ------> | zoom + token blocks   |
 is boarding?           |     rank -> keep           |         | wrong / not recovered |
                        +---------------------------+         +-----------------------+
                                      FastV answer shown as one compact baseline line
                                      Small footer: 25% kept, same 69-token budget
                                      Small result anchor: OCRBench +16.0 pp vs FastV-k3
```

### Zone 1: input

- Make the real image the largest single object in the figure.
- Draw one thin red rectangle tightly around the `A105-108 ... boarding`
  evidence. No more than one red box.
- Put the exact question directly below the image in a quiet light-gray prompt
  strip. The question is part of the model input, not explanatory prose.
- Do not repeat the full image elsewhere.

### Zone 2: mechanism

- Retain the current Overleaf figure's two horizontal lanes and amber-vs-blue
  identity, but reduce each to three large visual operations.
- Top amber lane: `Rank raw 2x2 units` -> `Keep 25%` -> `Native merger`.
- Bottom blue/gray lane: `Native merger` -> `Rank merged tokens` -> `Keep 25%`.
- Use icon-like token grids and arrows; each operation gets at most a four-word
  label. Do not write paragraphs or numbered explanatory subtitles.
- The only emphasized phrase should be `rank before information is mixed`.
- The two lanes must be geometrically aligned so the changed operation order is
  obvious without reading the caption.

### Zone 3: evidence and answer

- Show two large, identically cropped evidence zooms, one per lane, with the
  method's real retained-token mask overlaid as square pixel blocks.
- Keep selected regions legible; fade dropped regions. Use amber borders for
  RBM and neutral blue/gray borders for Post-L2.
- Under the RBM zoom: `A105-108` and `Correct` with a check icon/text.
- Under the Post-L2 zoom: its exact short answer or `not recovered`, followed
  by `Incorrect`. Do not rely on green/red alone.
- Add FastV only as a compact third answer line below the two zooms:
  `FastV-k3: no counter is currently boarding - Incorrect`.
- Print GT once: `Ground truth: A105-108`.
- A tiny final line may state `25% retained | identical final token count: 69`.
- Add a small, restrained badge or one-line bar:
  `OCRBench: RBM +16.0 pp vs FastV-k3`. It must occupy less than 8% of the
  figure area and must not turn the figure into a chart dashboard.

## Visual rules

- No title sentence inside the figure; the paper caption provides it. Start
  directly with small zone labels `Input`, `Where to rank`, `Evidence`.
- No outer cards, rounded dashboard panels, radar charts, dense legends,
  gradients, decorative shapes, or more than one accent color per method.
- Reserve red for the evidence rectangle and incorrect markers. Do not tint
  whole panels pink/green.
- Use direct labels instead of a legend.
- Minimum final font size: 7 pt. Main lane labels: 9-10 pt. Method names:
  10-11 pt semibold. Question/answers: 8-9 pt.
- Preserve image and crop aspect ratios. Do not stretch screenshots.
- Keep at least 0.12 inch whitespace between all text and image regions.
- At ACM print size, the question, GT, both answers, and `A105-108` inside the
  zoom must be readable without digital zoom.

## What not to build

- Do not put TextVQA, DocVQA, and OCRBench side by side.
- Do not show four full method rows.
- Do not reproduce the contact sheet.
- Do not make Fig. 1 a performance/efficiency dashboard.
- Do not use the old arbitrary SpongeBob image.
- Do not show repeated full-image masks; only the shared input and two evidence
  crops need imagery.
- Do not include implementation disclaimers inside the artwork. Put provenance
  and caveats in the caption or manifest.

## Validation gates

Implement `validate_bundle.py` and fail closed:

1. all source files, answers, masks, hashes, and provenance fields exist;
2. mask shapes match the declared unit grid and kept counts match 25%;
3. all displayed arms have identical final `ptid`;
4. correctness is recomputed with the authoritative scorer;
5. evidence crop contains the GT text and stays inside source bounds;
6. mask-to-image alignment is tested at all four corners;
7. PDF/300-dpi PNG are nonblank, fonts are embedded, and no artist is clipped;
8. every font is >=7 pt at final size;
9. raster inspection at 100% and at the embedded ACM width shows no overlap,
   stretched image, unreadable crop, or ambiguous lane flow.

Render three iterations before selecting the final:

- `candidate_a_balanced`: equal space for input/mechanism/evidence;
- `candidate_b_visual`: larger input and zoom, smaller mechanism;
- `candidate_c_minimal`: only essential labels and maximum whitespace.

Visually inspect all three. Choose the most legible and immediate version; do
not choose by file size or an automated aesthetic score. Save the alternatives
under `drafts/figures/frontispiece_fig1/candidates/`.

## Paper integration

After all gates pass:

- back up the current pipeline figure as
  `drafts/figures/frontispiece_fig1/previous_pipeline_fig1.pdf`;
- write the chosen `drafts/figs/fig1.pdf` and 300-dpi `fig1.png`;
- copy the same PDF to `drafts/overleaf_submission/figs/fig1.pdf`;
- verify both PDFs have identical SHA-256 hashes;
- update the Fig. 1 caption in `drafts/overleaf_submission/main.tex` to explain
  the real OCRBench example, same 25%/69-token budget, the controlled stage
  contrast, FastV answer line, and the small aggregate OCRBench context;
- keep detailed method/provenance qualifications in the caption, not artwork;
- update `drafts/figs/README.md`, `STATE.md` (<=30 lines), and `DECISIONS.md`.

Do not alter numerical tables or unrelated claims.

## Completion and Git

Run the validator and the relevant LaTeX build/smoke test. Commit and push to
`origin main` with the configured human author identity. Do not add AI/Codex/
Claude attribution or a `Co-Authored-By` trailer.

Return only a <=20-line digest containing selected id, exact answers,
correctness, `ptid`, candidate chosen and why, paths to PDF/PNG/manifest/
validator/candidates, validation and LaTeX results, final PDF hashes, commit,
push result, and one next-step recommendation.

If required capture would exceed 6 GPU-hours, stop before running it and report
the measured estimate. Otherwise proceed autonomously.
