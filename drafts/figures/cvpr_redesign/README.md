# CVPR-style local figure candidates

These candidates are rendered from the transferred measured bundle at
`../server_exports/cvpr_figure_data_v1/`. They do not replace the authoritative
Overleaf figures yet.

## Outputs

- `outputs/fig1_cvpr_candidate.{pdf,svg,png}`: two stable qualitative rows
  (OCRBench `ocr0804`, TextVQA `34982`) plus full-split aggregate anchor.
- `outputs/fig3_mechanism_candidate.{pdf,svg,png}`: per-image rank reshuffle,
  edge-unit demotion, and Qwen3-VL ranking-swap recovery.
- The former Fig. 4 regime/qualitative candidate was rejected during the
  2026-08-14 figure review and removed. Table 2 and the main-text scope
  discussion remain authoritative; no replacement Fig. 4 is planned.

## Selection and audit constraints

- `ocr0422` is retained in the bundle for provenance but is not used by the
  current Fig.1 candidate: its audited RBM answer is correct while an
  independent regeneration returned `A5`.
- `textvqa_35164` is also retained but not used for the flagship candidate:
  the independent Post-L2 regeneration changed the audited correctness.
- `ocr0804` and `textvqa_34982` have stable qualitative directions and are used
  in Fig.1. `gqa_201056134` remains in the audited bundle as a scope-boundary
  case but is not promoted into a standalone body figure.
- The NPZ `keep_*` arrays are square-packed display masks. Renderers rebuild
  spatial maps from raw `kept_indices` and the native unit grid; they never
  paste the padded square directly onto the image.
- Evidence boxes are empty in the server manifest. The candidates therefore
  show measured score maps and masks without an unrecorded evidence rectangle.

## Reproduction

```text
C:\Python314\python.exe drafts\figures\cvpr_redesign\validate_bundle.py
C:\Python314\python.exe drafts\figures\cvpr_redesign\render_fig1.py
C:\Python314\python.exe drafts\figures\cvpr_redesign\render_fig3.py
```

The validator must pass before any candidate is copied into
`drafts/overleaf_submission/figs/`.
