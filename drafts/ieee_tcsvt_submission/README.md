# IEEE TCSVT submission package

This folder is the IEEE Transactions on Circuits and Systems for Video
Technology (TCSVT) LaTeX migration of the authoritative manuscript in
`../overleaf_submission/`.

## Files

- `main.tex`: IEEEtran two-column Transactions Paper manuscript.
- `supplement.tex`: standalone supplementary-material manuscript.
- `references.bib`: shared BibTeX database.
- `figs/`: vector figures used by the manuscript and supplement.

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
- Resolve the project-level GQA claim issue recorded in `../../STATE.md` before
  submission; this packaging pass does not alter scientific claims or results.
- Restore and verify the anonymous artifact referenced in Supplement S9 before
  claiming that the artifact package is complete.
- Reconfirm whether the active ScholarOne article type or special issue imposes
  any additional anonymity, keyword, graphical-abstract, or source-file rules.

Official guidance checked on 2026-08-27:

- https://ieee-cas.org/tcsvt-submission-manuscript
- https://ieee-cas.org/publication/tcsvt/guidelines-authors
- https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/

