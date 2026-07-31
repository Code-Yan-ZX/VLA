# Real-data figure pipeline

This directory is the hand-off boundary between GPU measurement and
publication rendering.

## Directory contract

- `inputs/`: user-provided source images. The current candidate set is tracked
  because the user explicitly requested server hand-off through Git. Do not add
  private or unlicensed images without the same explicit authorization.
- `data/`: measured per-image score arrays and provenance manifests. Large or
  private captures are ignored by Git.
- `scripts/`: capture, validation, and rendering programs created on the GPU
  server.
- `outputs/`: rendered previews and PDFs. Copy a figure to `drafts/figs/` only
  after its data provenance and visual QA pass.
- `CLAUDE_CODE_PROMPT.md`: task prompt for Claude Code on the GPU server.

## Scientific contract

FIG:1 must use the target VLM's measured merger-input and merger-output L2
scores. CLIP/SigLIP embeddings and LLM attention are different signals and
must not be substituted for the RBM selector. The existing verified reference
implementation is `scripts/mechanism_token_survival.py`, which imports the
exact `_score_units` and `_score_tokens` functions from the experiment runner.

FIG:2 and FIG:3 must read versioned JSON/CSV values derived from the audited
experiment digests. Plotting code must never contain unverified or generated
numbers.

For same-image qualitative comparison, RBM versus post-merger L2 is the
mandatory iso-selector/iso-budget pair. FastV is optional and requires a real
question supplied by the user; it must be labeled as query-conditioned and
must not be presented as an iso-selector stage control.
