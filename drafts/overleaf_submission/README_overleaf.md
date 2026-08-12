# Overleaf submission package — RBM (Rank-Before-Merge)

## Upload

1. Zip **this entire directory** (or upload the directory tree as-is).
2. In Overleaf, set **Main document** = `main.tex`.
3. Compiler: **pdfLaTeX** (the `acmart` class with `sigconf` option; no
   XeLaTeX/LuaLaTeX-specific packages are used).
4. Compiling `main.tex` produces one continuous PDF containing the main paper,
   references, and S1--S11 supplementary material. `supp.tex` remains
   independently compilable when a separate supplementary PDF is required.

## File map

| file | role |
|------|------|
| `main.tex` | main paper plus appended S1--S11 supplement |
| `supp.tex` | dual-mode supplementary source (included by `main.tex` or compiled alone) |
| `references.bib` | verified bibliography used by the paper |
| `figs/fig1.pdf` | FIG 1 — method-specific measured masks on one OCRBench case |
| `figs/fig2.pdf` | FIG 2 — paired pre−post forest plot with 95% CIs |
| `figs/fig3.pdf` | FIG 3 — shared-panel retention-vs-gap curves (n=200) |
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

The remaining six compare panels (`img70,71,73,75,76,77`) ship in the code
release under `drafts/figures/real_data_pipeline/outputs/`.

## Self-containment

All `\includegraphics` paths are `figs/...` (relative to the `.tex` files).
There are **no** `../` parent-relative paths — verified by
`grep -rn '\.\./' *.tex` returning empty.

## Known caveats

- `acmart.cls` is **not** bundled; Overleaf provides it natively (TeX Live
  ≥ 2020). If compiling locally, install `texlive-publishers`.
- The compiled review paper uses 8 pages for the body and 1 page for references;
  the integrated supplementary material follows after a forced page break.
- `figs/fig1.pdf` uses method-specific measured masks; no mask is reused across
  RBM, post-merger L2, and FastV.
- The paper is double-blind: no author/institution/repo identifiers appear in
  either `.tex` or the figure captions.

## Submission-time fields

Before submission, verify the official ACM MM 2027 call and replace the
provisional `\acmConference` date/location in `main.tex`. Keep `anonymous` review
mode and the anonymous author block for double-blind review. For the
camera-ready version, remove `anonymous`, replace the author/affiliation stub,
and insert the DOI/ISBN/copyright metadata supplied by ACM. The actual upload
and submission require human confirmation.
