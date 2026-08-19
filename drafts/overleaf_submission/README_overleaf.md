# Overleaf submission package — RBM (Rank-Before-Merge)

## Upload

1. Build a clean allowlist package containing only `main.tex`, `supp.tex`,
   `references.bib`, and the ten referenced PDFs listed below. Do **not** zip
   this working directory: it also contains build files and internal reviews.
2. In Overleaf, set **Main document** = `main.tex`.
3. Compiler: **pdfLaTeX** (the `acmart` class with `sigconf` option; no
   XeLaTeX/LuaLaTeX-specific packages are used).
4. Compiling `main.tex` produces one continuous PDF containing the main paper,
   references, and S1--S10 supplementary material. `supp.tex` remains
   independently compilable when a separate supplementary PDF is required.

## File map

| file | role |
|------|------|
| `main.tex` | main paper plus appended S1--S10 supplement |
| `supp.tex` | dual-mode supplementary source (included by `main.tex` or compiled alone) |
| `references.bib` | verified bibliography used by the paper |
| `figs/fig1.pdf` | FIG 1 — method-specific measured masks on one OCRBench case |
| `figs/fig2.pdf` | FIG 2 — paired pre−post forest plot with 95% CIs |
| `figs/fig3.pdf` | FIG 3 — shared-panel retention-vs-gap curves (n=200) |
| `figs/token_survival_combined.pdf` | S2 representative TextVQA/DocVQA survival maps |
| `figs/token_survival_m1_rank_overlap.pdf` | S3 rank-agreement distributions |
| `figs/retention_curves.pdf` | S3 Qwen3-VL retention curves |
| `figs/compare_df7282e1.pdf` | S10 qualitative compare, sample df7282e1 |
| `figs/compare_img69.pdf` | S10 compare, WeChat-img-69 (alias of `微信图片_…69_123`) |
| `figs/compare_img72.pdf` | S10 compare, WeChat-img-72 |
| `figs/compare_img74.pdf` | S10 compare, WeChat-img-74 |

## Chinese-filename alias mapping

The 9 WeChat-source inputs have Chinese filenames. The four compare PDFs used
in the supplement were copied under ASCII aliases to avoid any Overleaf / git
path-encoding issues:

| alias | original filename stem |
|-------|------------------------|
| `compare_img69.pdf` | `微信图片_20260731144845_69_123` |
| `compare_img72.pdf` | `微信图片_20260731144900_72_123` |
| `compare_img74.pdf` | `微信图片_20260731144908_74_123` |
| `compare_df7282e1.pdf` | `df7282e1-f246-4955-9fdb-f65bea9844f1` |

Only the four panels listed above are part of the qualitative supplement. Any
additional internal panels should be released only after they are recovered,
sanitized, and added to the verified anonymous artifact manifest.

## Self-containment

All `\includegraphics` paths are `figs/...` (relative to the `.tex` files).
There are **no** `../` parent-relative paths — verified by
`grep -rn '\.\./' *.tex` returning empty.

## Known caveats

- `acmart.cls` is **not** bundled; Overleaf provides it natively (TeX Live
  ≥ 2020). If compiling locally, install `texlive-publishers`.
- This directory does **not yet** contain the run-level anonymous artifact
  indexed by Supplementary S9. Restore, sanitize, checksum, and verify that
  archive before submission; do not represent this source package alone as the
  reproducibility artifact.
- The compiled review paper uses 8 pages for the body and the references begin
  after the body; the integrated supplementary material follows after a forced
  page break. Page accounting for the separate supplement depends on the final
  submission portal and should be checked against the official ACM MM 2027
  instructions.
- `figs/fig1.pdf` uses method-specific measured masks; no mask is reused across
  RBM, post-merger L2, and FastV.
- The current PDF is a generic two-column author draft for supervisor review.
  It names Zhengxing Yan and is not tied to any journal or conference. The
  release artifact and internal provenance mapping remain separate concerns.

## Submission-time fields

After selecting a target venue, replace the generic `sigconf,nonacm` mode with
that venue's official template and verify its page limits, supplementary policy,
anonymity rules, and required metadata. Add affiliation and contact information
only after the author confirms them. The actual upload and submission require
human confirmation.
