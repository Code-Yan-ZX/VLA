# IEEE TCSVT submission package updated 2026 09 04

This folder is the updated IEEE Transactions on Circuits and Systems for Video
Technology (TCSVT) submission package assembled from the latest Chinese method
draft and the previous LaTeX final.

The main-text Figure 1 is the exact method-overview image selected in the
latest Word draft and is stored as `figs/fig1_word.png`. The measured result and
mechanism figures remain from the previous LaTeX final.

The package also incorporates the corrected full-split Qwen3-VL matched-boundary
control: pre-final minus post is $-5.64$ pp on GQA (95% CI $[-6.42,-4.86]$),
while the text-dense control remains positive. The manuscript therefore uses
workload-conditioned wording and does not claim universal pre-stage superiority.

## Files

- `main.tex`: IEEEtran two-column Transactions Paper manuscript.
- `supplement.tex`: standalone supplementary-material manuscript with the
  full-split matched-boundary control in S7b.
- `references.bib`: shared BibTeX database.
- `figs/`: figures used by the manuscript and supplement, including the Word
  selected `fig1_word.png`.

## Compile

```text
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

The sources use the standard `IEEEtran` class and `IEEEtran.bst`. They compile
with TeX Live 2026 and are suitable for upload to Overleaf without local style
files.

## TCSVT checks

- TCSVT Transactions Papers must not exceed 14 pages in IEEE two-column format,
  including figures; check the current author guide again at submission time.
- Upload `supplement.pdf` separately. It is intentionally not appended to
  `main.pdf`.
- Add every author's complete affiliation, country, corresponding-author e-mail,
  and ORCID before submission. The migrated source currently preserves only the
  author name available in the authoritative draft.
- Restore and verify the anonymous artifact referenced in Supplement S9 before
  claiming that the artifact package is complete.
- Reconfirm whether the active ScholarOne article type or special issue imposes
  any additional anonymity, keyword, graphical-abstract, or source-file rules.

Official guidance checked on 2026-08-27:

- https://ieee-cas.org/tcsvt-submission-manuscript
- https://ieee-cas.org/publication/tcsvt/guidelines-authors
- https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/
