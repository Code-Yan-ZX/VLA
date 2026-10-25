# CVM 2027 anonymous revision review draft

This directory contains a review draft migrated from the September 16 manuscript
to the official CVM 2027 anonymous review template. It is a revision review draft,
not a submission-approved manuscript. All authors must review the revised
scientific interpretation before upload.

The official review layout uses 10-point Times text, two columns, US Letter paper,
line rulers, and numbered pages. The official style and bibliography files are
included unchanged. The template's `****` paper-ID placeholder is retained until
the submission portal assigns an actual ID. Do not guess a paper number.

Build from this directory with TeX Live and `latexmk`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For a clean project working tree, redirect intermediate files to a temporary
directory with `-outdir=...`. The main source, bibliography with 51 cited
references, three used figure assets, and two figure-layout source files are
sufficient to rebuild. The compiled manuscript contains eight pages including
references.

`figure1_review.tex` uses rectangular crops of the unchanged original PNG to
omit unsupported explanatory labels and replace them with neutral descriptions.
`figure4_review.tex` shows the six
unchanged spatial data panels from the original PDF, omitting historical
aggregate-score headings and causal interpretations. The displayed maps retain
their historical diagnostic provenance. Neither edit changes heatmap values,
selected units, answer examples, or model results.

Names, email addresses, affiliations, funding, acknowledgments, the identifiable
repository link, and PDF author metadata are absent from the review manuscript.
Internal audits and research logs are excluded from the source archive.

The revision includes the main-only Qwen3-VL output control and its DocVQA
direction reversal. All reported TextVQA scores and intervals have been
recalculated from the original predictions using the pinned lmms-eval official
answer-normalization and leave-one-annotator-out metric. DocVQA scores and control
intervals have likewise been recalculated with the pinned lmms-eval ANLS
implementation. GQA scores have been recalculated with the benchmark's official
case/punctuation-insensitive exact-match rule; the pinned OCRBench scorer
including fullwidth normalization reproduces all 15 checked cells exactly.
The matched-harness
FastV comparison presents both its TextVQA advantage and RBM's OCRBench advantage.
Shared failures receive zero, and the TextVQA point estimates and interval use
the same complete 5,000-example denominator. No new model inference was run.

See `CHANGELOG.md` for changes and `SUBMISSION_CHECKLIST.md` for unresolved steps.
