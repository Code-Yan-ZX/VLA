# Server task: export auditable Fig.1 assets for `ocr0422`

This is a data-export task only. Do **not** design or render the final Fig.1.
Do not modify `drafts/figs/fig1.pdf`, `drafts/figs/fig1.png`, or the Overleaf
caption. The final composition will be rebuilt locally after these assets are
transferred.

## Locked sample and protocol

- Sample: OCRBench `ocr0422`.
- Source image: the original `/media/disk2/YZX/research/vla/runs/data/ocrbench/ocr0422.jpg`.
- Model: Qwen3-VL-8B-Instruct, exact revision used by the audited runs.
- Question: `which counter is boarding?`
- Ground truth: `A105-108` (include all scorer aliases used by the harness).
- Greedy decoding, the stored short-answer prompt, seed used by the audited run.
- Final visual-token retention: 25%.
- Required arms: `RBM` (pre-merger L2), `Post-L2` (same L2 selector after the
  native merger), and `FastV-k3` (query-conditioned baseline).

Use the existing verified harness and hooks. Do not invent a selector, derive a
mask from image pixels, or use an aggregate result as a substitute for a
sample-level mask. If a required arm has no saved mask, run only this sample to
capture it. Preserve the exact method-specific mask; never reuse the RBM mask
for another arm.

## Required transfer bundle

Create exactly this directory:

`drafts/figures/frontispiece_fig1/transfer/`

It must contain:

```text
ocr0422.jpg
manifest.json
provenance.json
rbm.npz
post_l2.npz
fastv_k3.npz
evidence_crop.png
debug_overlays_rbm.png
debug_overlays_post_l2.png
debug_overlays_fastv_k3.png
SHA256SUMS
README.md
```

### `manifest.json`

Include only measured values and explicit provenance:

- benchmark, sample id, source image path, exact question;
- GT string and normalized/alias forms;
- exact generated answer for each arm, authoritative scorer name and correctness;
- `ptid`, final visual-token count, final visual-token budget, keep ratio;
- original source width/height and processor width/height;
- processor `grid_thw`, merge factor, unit grid `(rows, cols)`, row-major index
  convention;
- evidence rectangle `(left, top, right, bottom)` in **original source pixels**;
- source run JSON paths and the command used for each arm;
- SHA-256 for every transferred file.

The evidence rectangle is an annotation for display only. It must be tight
around the `A105-108 ... boarding` text and must be inside source bounds.

### Method NPZ schema

Each NPZ (`rbm.npz`, `post_l2.npz`, `fastv_k3.npz`) must include:

- `keep`: boolean 2-D array shaped exactly `(rows, cols)`;
- `kept_indices`: raw integer row-major indices before reshaping;
- `unit_grid_hw`, `processor_hw`, `source_hw`;
- `n_full_units`, `n_kept_units`, `final_visual_tokens`, `ptid`;
- `stage` and `method` strings;
- `mask_space` describing whether indices are merger-input units or post-merger
  visual rows, plus the exact mapping back to the shared processor-image grid.

Abort if the displayed arms do not have identical `ptid` and identical final
visual-token count. Do not silently pad or truncate masks.

### Debug images

The three debug overlays are for alignment QA only, not the paper figure. Each
must use the original image with the corresponding **real** mask overlaid using
transparent square blocks and a visible grid boundary. Also export the tight
`evidence_crop.png` at native pixels (no text or arrows painted into the crop).

## Validation and provenance

Run a fail-closed validator that checks:

1. all required files exist and SHA-256 hashes match;
2. mask shapes and kept counts match `manifest.json` and 25% budget;
3. all three `ptid` and final visual-token counts are equal;
4. answers and correctness match the authoritative scorer;
5. evidence box is in bounds and the crop is non-empty/readable;
6. mask-to-image mapping agrees at all four grid corners;
7. no method-specific mask is byte-identical to another method unless the
   measured run truly proves identity, in which case report the identity and
   why it is expected.

Record in `provenance.json`: repo commit, model revision, package versions,
CUDA/device, seed, pixel cap, exact commands, hook definitions, capture time,
and validator output. Do not copy weights, caches, raw logs, or full datasets.

## Transfer command

After validation passes, keep the bundle below 20 MB, force-add only this
transfer directory, and commit it with no AI/Codex attribution:

```bash
git add -f drafts/figures/frontispiece_fig1/transfer
git diff --cached --stat
git commit -m "Export auditable OCRBench Fig.1 assets"
git push origin main
```

Return a short digest containing: commit hash, total bundle size, SHA-256 of
`ocr0422.jpg`, processor/grid geometry, each method's kept count, final visual
token count, `ptid`, evidence box, and validator result. Stop if any required
mask or the original image is unavailable; do not fabricate a substitute.
