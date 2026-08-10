# Claude Code server task: rebuild Fig. 1 as a real qualitative comparison

You are working in `/media/disk2/YZX/research/vla` on the A40 server. Work
end-to-end and do not stop at a plan. Read `STATE.md` first, then only the
relevant sections/files named below. Do not read the whole repository.

## Objective

Replace `drafts/figs/fig1.pdf` with a publication-quality, full-width
qualitative figure that makes the paper's real OCR/text-dense advantage visible.
Each example must show:

1. the original benchmark image and its question as the shared input;
2. RBM (ours) first, followed by several real baselines;
3. the visual tokens retained by each method as aligned pixel-block overlays;
4. a red evidence box and a readable local zoom when the answer text is small;
5. each method's exact short answer, correctness, and the ground-truth answer.

This is a qualitative, case-level comparison. Never claim that RBM is overall
SOTA or universally better. `STATE.md` explicitly forbids that claim, and the
aggregate results show that FastV is often stronger on TextVQA/DocVQA. Use the
wording "selected cases where RBM preserves text that baselines miss".

## Known audited anchors

Start from `drafts/qualitative_examples.md`. Two strong cases are already
audited and should be used unless a required same-sample baseline cannot be
recovered:

- TextVQA `35014`: question "what is the date on the right page?"; GT and RBM
  `07/10/2012`; post-L2 says no date is visible.
- DocVQA `58439`: question asks the 1998 promotional-meeting spend; GT and RBM
  `$1.3 BILLION`; post-L2 and VisionZip-style answer `$1.3 million` (1000x unit
  error).

Also inspect these already audited alternatives if their images/masks compose
better: TextVQA `34886`, `35174`, `34646`, `34863`; DocVQA `433`, `17005`.
Mine one OCRBench example if possible where RBM is correct and at least two
same-budget baselines are wrong. Do not invent an OCRBench example just to fill
a dataset slot.

The server-side mining pass in `drafts/figures/camera_ready/` has already found
9 strict OCRBench three-arm flips among 200 common samples. Prefer `ocr0804`:
question asks for `CIRCULATION DATES`; GT/RBM is `OCTOBER 1999`, FastV-k3 says
`3/3/2013`, and Post-L2 says `4/14/99` at the same final token count. Other
strict options are `ocr0422`, `ocr0168`, `ocr0933`, `ocr0329`, `ocr0599`,
`ocr0552`, `ocr0479`, and `ocr0389`. Read
`drafts/figures/camera_ready/contact_sheet_manifest.json`; do not repeat the
candidate mining unless its referenced run files are stale or missing.

## Methods and fairness contract

Use Qwen3-VL-8B-Instruct, greedy decoding, the exact stored question, and 25%
final visual-token retention for every displayed row. Preferred rows:

1. `RBM (ours)` -- merger-input L2 ranking;
2. `FastV` -- the project's validated layer-2 query-conditioned baseline;
3. `VisionZip-style` -- the validated post-merger dominant/contextual port;
4. `Post-L2` -- the iso-selector post-merger control.

Only display a row when all of the following are available for that exact
sample: exact generated answer, authoritative correctness, final retained-token
indices/mask, processor geometry, and final visual-token count. If FastV or
VisionZip-style lacks per-sample data, run only the selected candidates to
capture it. Do not substitute aggregate accuracy. Do not show PyramidDrop in
the main figure because its canonical schedule has a different effective token
budget; it may appear only in an optional supplement with the budget printed.

All rows for an example must have the same final visual-token count. Assert this
in code and abort rendering on mismatch. Where methods select tokens at
different stages, map their retained tokens back to the shared processor image
grid and document the mapping. Call the overlays `retained visual tokens`, not
`attention`, unless the array is literally an attention value. Do not derive or
fake masks from image pixels.

## Data discovery and candidate selection

Prefer existing evaluated outputs under `runs/` over rerunning. Search the
full-split/n=500/n=200 per-sample JSONs referenced by:

- `drafts/qualitative_examples.md`
- `experiments/j7hf_baselines_n500.md`
- `experiments/j7_main_table.md`
- `src/v3_premerger/baselines_hf.py`
- `runs/v3_premerger_cells/`
- `runs/full_matrix/`

Join records by benchmark and canonical sample id. Report the candidate-pool
counts before choosing examples. Selection rule:

- RBM answer is correct under the authoritative benchmark scorer;
- at least two displayed baseline answers are wrong;
- the answer evidence is visibly localized and readable in the source image;
- prefer distinct failure types: missing text, character/unit corruption, and
  dense numeric OCR;
- no answer/mask may be hand-edited after seeing the figure.

Use OCRBench `ocr0804` in the three-dataset candidate layout unless visual
inspection shows that its evidence crop is unreadable. If it fails that visual
gate, choose another `strict_three_arm` OCRBench candidate from the manifest.
Never use a `padded_two_arm` candidate to claim a FastV failure.

## Capture bundle

Create `drafts/figures/qualitative_fig1/data/` with one directory per selected
sample and a root `manifest.json`. Keep weight files and raw run logs out of
Git. The manifest must contain, at minimum:

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "revision": "resolved commit",
  "keep_ratio": 0.25,
  "selection_rule": "RBM correct and >=2 displayed baselines wrong",
  "candidate_pool_counts": {},
  "examples": [
    {
      "benchmark": "TextVQA",
      "id": "35014",
      "image": "textvqa_35014/source.jpg",
      "question": "...",
      "ground_truth": ["..."],
      "evidence_bbox_source_px": [0, 0, 0, 0],
      "processor_size_wh": [0, 0],
      "unit_grid_hw": [0, 0],
      "methods": [
        {
          "key": "rbm",
          "label": "RBM (ours)",
          "answer": "...",
          "correct": true,
          "ptid": 0,
          "mask": "textvqa_35014/rbm_mask.npz",
          "mask_key": "keep",
          "capture_source": "path/to/run.json"
        }
      ]
    }
  ]
}
```

Each NPZ must contain a boolean 2-D `keep` array on the declared unit grid and
the raw integer kept indices. Include SHA-256 hashes for source images, JSONs,
and NPZs. Save exact commands, git commit, package versions, model revision,
seed, pixel cap, processor geometry, and method hook definitions in
`provenance.json`. Evidence boxes may be manually annotated from the original
image, but must be verified by opening the crop and confirming that it contains
the GT text. They are explanatory annotations, never inputs to selection.

## Figure design

Write a deterministic renderer at
`drafts/figures/qualitative_fig1/render_fig1.py`. Use Matplotlib/Pillow and
embed fonts in the PDF. The main output is full-width (`figure*`) and must fit a
normal ACM page without microscopic text.

Target three datasets in the main figure: TextVQA, DocVQA, and OCRBench. First
render a three-block candidate as `fig1_three_examples.pdf`. If it cannot pass
the 6.5-pt/readability gates at ACM print size, use the clearest two examples in
the main `fig1.pdf` and place the third in a supplementary companion figure;
record that layout decision in the validation report.

For each block:

- Left: original image, red evidence rectangle, benchmark + id, and the exact
  question directly below it. The question is part of the input, not a caption.
- Right: one aligned row per method, with RBM first. Each row has the method
  label, retained-token overlay, a magnified evidence crop, and a compact answer
  cell. Use visible square pixel blocks/borders for kept tokens and fade dropped
  areas. Preserve the underlying image enough to identify the evidence.
- Answer cell: `Answer: ...` followed by an icon/text marker `Correct` or
  `Incorrect`. Do not rely on green/red alone. Print `GT: ...` once per example.
- Use a consistent color-blind-safe style. RBM may use amber/gold emphasis;
  baselines should be neutral blue/gray. Reserve red only for evidence boxes and
  incorrect-answer emphasis.
- The zoom crop must show the same retained-token overlay as its method row;
  connect it to the red source box with a thin leader or an unambiguous inset
  border. Do not fabricate enlarged pixels or OCR text.
- Print `25% retained; identical final token count` once per example, with the
  actual `ptid` value.
- No nested decorative cards, gradients, screenshots of tables, or paragraph-
  length claims inside the figure.

The visual takeaway should be immediate: RBM keeps the small answer-bearing
text and answers correctly while the displayed baselines drop/corrupt it.

## Validation gates

Implement `validate_bundle.py` and fail closed. It must check:

1. all images, masks, answers, GTs, hashes, and provenance fields exist;
2. every mask shape matches `unit_grid_hw` and every kept count matches the
   declared budget/hook mapping;
3. all displayed methods have identical final `ptid` for each example;
4. correctness is recomputed with the repository's authoritative scorer and
   matches the manifest;
5. the evidence crop is inside image bounds and has at least 80 px on its
   shorter rendered side;
6. PDF and 300-dpi PNG are nonblank and no text/artists extend outside the
   canvas;
7. all font sizes are at least 6.5 pt at final embedded size;
8. source image aspect ratios are preserved and mask-to-image alignment is
   tested at all four corners.

Render PNG previews and inspect them at both full size and ACM two-column print
size. Iterate if labels, questions, evidence text, or answer cells are hard to
read. Save a machine-readable validation report next to the data bundle.

## Paper integration

After all gates pass:

- write `drafts/figs/fig1.pdf` and `drafts/figs/fig1.png`;
- update the Fig. 1 caption in `drafts/overleaf_submission/main.tex` so it
  describes the selected qualitative cases, exact methods, shared 25% budget,
  `ptid`, red evidence boxes/zooms, and the honest case-level scope;
- remove the old caption's claim that Fig. 1 is the single-image stage-axis
  pipeline;
- keep the method pipeline explanation in the surrounding prose;
- update `drafts/figs/README.md`, `STATE.md` (<=30 lines), and `DECISIONS.md`.

Do not alter numerical tables or claims. Do not commit weights, datasets beyond
the selected source images, raw logs, caches, or unrelated files.

## Completion report and Git

Run the validator and any relevant LaTeX build/smoke test. Then commit and push
to `origin main` using the repository's configured human author identity. Do
not add AI/Codex/Claude attribution or a `Co-Authored-By` trailer.

Return only a <=20-line digest containing:

- selected benchmark ids and why they passed the rule;
- exact per-method answers/correctness/ptid;
- candidate-pool counts, including the OCRBench result;
- paths to the PDF, PNG, manifest, validation report, and renderer;
- validation/LaTeX result;
- commit hash and push result;
- one recommended next step.

If a required capture would exceed 6 GPU-hours, stop before running it and
report the measured estimate. Otherwise proceed autonomously.
