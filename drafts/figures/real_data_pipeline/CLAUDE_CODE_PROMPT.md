# Claude Code task: build the real-data paper-figure pipeline

You are working in the VLA repository on the GPU server. Complete this task
end to end. Do not stop at a plan.

## 0. Restore project context

1. Read `AGENTS.md` and `STATE.md` first.
2. Read only the relevant parts of:
   - `drafts/figs_spec_for_user.md`
   - `drafts/research/method_figure_survey.md`
   - `drafts/figures/gen_fig1_rbm.py`
   - `scripts/mechanism_token_survival.py`
   - the scoring helpers in
     `src/v3_premerger/v3_premerger_runner.py`
3. Preserve unrelated worktree changes. Do not commit model weights, input
   images, raw benchmark data, logs, caches, or credentials.

## 1. Goal

Build a reproducible pipeline under
`drafts/figures/real_data_pipeline/`:

```text
inputs/   -> target-VLM feature capture -> data/*.npz + *.json
data/     -> validation                 -> data/*_validated.json
data/     -> Matplotlib render          -> outputs/fig1.{pdf,png}
audited experiment summaries            -> outputs/fig2.{pdf,png}
                                         -> outputs/fig3.{pdf,png}
```

The user will place several `.png`, `.jpg`, or `.jpeg` files in `inputs/`.
Process every supported image deterministically. Use the filename stem as the
sample id, and record a SHA-256 hash of the original file.

## 2. Non-negotiable scientific rule

Do not use CLIP, SigLIP, a generic vision encoder, fabricated saliency, or LLM
attention to produce the main FIG:1 score maps. RBM is defined by the target
merger-equipped VLM's own feature L2 scores at two hook points. Attention is a
different selector and would misrepresent the method.

For the headline Qwen3-VL-8B-Instruct path, reproduce the existing experiment
code exactly:

- Load `Qwen/Qwen3-VL-8B-Instruct` with the same vLLM/HF processor settings as
  the paper's Qwen3 cells. Default `max_pixels=1500000`; expose it as a CLI
  argument and record the value.
- Run one image per capture pass, no pruning, no stochastic decoding, and at
  most one generated token. Capturing must not change model numerics.
- PRE score source: the input to the first merger used by the real RBM ranking
  path, currently `visual.deepstack_merger_list[0]`. Reshape its hidden states
  from `[N*4, ..., C]` to `[N,4,C]`, then call the repository's exact
  `_score_units(feats, "l2")`. The current definition is the mean over the four
  patch-feature L2 norms, not CLIP similarity and not an attention map.
- POST score source: the same post-merger visual row used by the headline post
  arm, currently `cat([main, ds0, ds1, ds2], dim=1)`, then call the exact
  `_score_tokens(post_feat, "l2")`.
- Import these two scoring helpers from
  `src/v3_premerger/v3_premerger_runner.py`; do not duplicate or approximate
  their formulas.
- Reuse/refactor `wrap_capture()` and `scores_from_cap()` from
  `scripts/mechanism_token_survival.py`. That script is the verified reference
  and already documents the spatial mapping and processor geometry.

Implement Qwen3 first and make it the only required backend for FIG:1. Add a
clean backend interface for later Qwen2.5-VL and InternVL3 support, but do not
pretend their hook points are identical. A backend is enabled only after its
pre/post hook definitions and coordinate mapping are verified against the
corresponding runner path.

## 3. Capture outputs and provenance

Create `scripts/capture_real_l2.py`. It must accept at least:

```bash
python drafts/figures/real_data_pipeline/scripts/capture_real_l2.py \
  --input-dir drafts/figures/real_data_pipeline/inputs \
  --output-dir drafts/figures/real_data_pipeline/data \
  --model-family qwen3vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --keep-ratio 0.25 \
  --max-pixels 1500000 \
  --seed 0
```

For each image, write a compact NPZ containing only publication-relevant
arrays, never giant hidden-state tensors:

- `pre_l2`: float32 `[N]`
- `post_l2`: float32 `[N]`
- `pre_keep`: bool `[N]`
- `post_keep`: bool `[N]`
- `pre_rank`: int32 `[N]`, rank 1 is largest score
- `post_rank`: int32 `[N]`
- `grid_thw`: int32 `[3]`
- `unit_grid_hw`: int32 `[2]`

Write a sidecar JSON with:

- sample id, original filename, SHA-256
- model family/id and resolved revision/commit when available
- processor/model classes and package versions
- repository Git commit
- dtype/device, seed, `max_pixels`, keep ratio and exact `k`
- patch size, spatial merge size and processor-resized image size
- exact pre/post hook module names
- exact score definitions in plain text
- timestamp and capture command

Do not store secrets or absolute private server paths in committed manifests.

## 4. Validation gates

Create `scripts/validate_real_l2.py`. Fail loudly unless every sample satisfies:

1. `pre_l2` and `post_l2` are finite, one-dimensional, nonempty, and the same
   length.
2. `N == unit_grid_h * unit_grid_w` for a still image.
3. `k == max(1, round(N * keep_ratio))` and each keep mask contains exactly
   `k` entries.
4. Keep masks exactly equal a deterministic descending top-k of their own
   score arrays. Define and record the tie-breaking rule.
5. Rank arrays are permutations of `1..N`.
6. The input and post arrays refer to the same unit identity and coordinate
   order. Assert this from the verified Qwen3 geometry rather than assuming it.
7. Repeating one capture with the same seed produces numerically identical
   arrays within an explicitly recorded tolerance.
8. No capture tensor or hook remains installed across samples in a way that
   duplicates data or changes the forward result.

Write a machine-readable validation report and return nonzero on failure.

## 5. FIG:1 renderer

Extend `drafts/figures/gen_fig1_rbm.py` instead of replacing its layout. Add:

```text
--scores-npz PATH
--metadata-json PATH
--allow-layout-proxy
```

Behavior:

- The default must require measured NPZ data. Proxy generation is allowed only
  when `--allow-layout-proxy` is explicitly passed.
- With measured data, use `pre_l2`, `post_l2`, `pre_keep`, and `post_keep`
  directly. Never recompute scores from image pixels.
- Resize the displayed image to the processor-resized geometry in metadata,
  preserving aspect ratio and exact unit-grid alignment.
- Normalize pre and post scores separately only for color display; store and
  report the normalization rule. Selection always uses raw measured L2.
- Keep the accepted layout: one shared input on the left; upper RBM row
  `L2 rank -> keep top-k -> native merger`; lower control row
  `native merger -> L2 rank -> keep top-k`.
- Use the same amber L2-rank visual language in both rows. Orange may identify
  the post-control row, but must not imply a different selector.
- Replace `L2 RANK (PROXY)` with `MEASURED L2 RANK` and remove the layout-proxy
  warning only when both the NPZ and validation report pass.
- Add one small provenance label such as
  `Qwen3-VL-8B | measured merger features | 25% kept`; do not add claims or
  answer text.
- Do not call the native merger an average. It is a learned nonlinear merger.
- Do not display attention, question-conditioned signals, or fabricated
  values.

Render each selected input separately, for example
`outputs/fig1_<sample_id>.{pdf,png}`. Do not silently choose the visually nicest
sample. Produce a contact sheet so the author can choose an example while
seeing all candidates.

## 6. FIG:2 and FIG:3 renderers

Create canonical data files and plotting scripts rather than hard-coding
numbers into drawing functions:

- `data/fig2_values.json` from the audited full-split values in
  `drafts/figs_spec_for_user.md`, cross-checked against
  `experiments/j7_main_table.md` and
  `experiments/internvl3_main_matrix.md`.
- `data/fig3_values.json` from the audited n=200 retention values in
  `drafts/figs_spec_for_user.md`, cross-checked against
  `experiments/j8_ablations.md` and the authoritative summary named there.
- `scripts/render_fig2.py` and `scripts/render_fig3.py` must read those files,
  validate the expected model/benchmark keys and units, and then render.

Follow the exact palette, units, panel order, axis rules, and labels in
`drafts/figs_spec_for_user.md`. In particular, never put OCRBench points and
percentage points on one continuous axis, never hide negative GQA values, and
never invent missing uncertainty bars.

## 7. Reproducibility and execution

Add one orchestrator script, `scripts/run_all.sh`, with independent stages:

```text
capture -> validate -> render_fig1 -> render_fig2 -> render_fig3 -> verify_pdf
```

It must support resuming completed captures and must not rerun the model when
only plotting code changes. Record commands and versions in a manifest. Keep
GPU work serial because the project has one A40.

Add a CPU-only smoke test using a tiny synthetic NPZ to test schema, validation,
and plotting without model weights. Also run the existing runner dry checks
relevant to `_score_units` and `_score_tokens`.

## 8. Visual and PDF verification

For every final PDF:

1. Confirm it is one page with the intended physical size.
2. Render it back to PNG with Poppler at 200-300 dpi.
3. Inspect the rendered pixels for blank output, clipping, overlap, unreadable
   text, broken math glyphs, and misaligned masks.
4. Check the 7.09-inch ACM double-column view, not only a zoomed screenshot.
5. Save the verification report under `outputs/verification.json`.

Do not copy anything into `drafts/figs/fig1.pdf`, `fig2.pdf`, or `fig3.pdf`
until all data and visual gates pass. The current FIG:1 is an internal proxy and
must remain identifiable as such until replacement.

## 9. Completion report

At completion, return a digest of at most 20 lines containing:

- changed files and exact commands
- model/revision and input ids captured
- NPZ schema and validation result
- final output paths
- PDF visual-QA result
- remaining limitations
- Git commit hash

Commit only source code, small canonical JSON/CSV data, manifests without
private paths, and documentation. Do not commit user input images, raw feature
tensors, model weights, caches, logs, credentials, or unapproved copyrighted
previews. Use the repository's configured human Git identity and add no AI
attribution.
