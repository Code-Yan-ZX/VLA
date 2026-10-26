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
references, five used figure assets in `figs/`, and two figure-layout source
files rebuild the 11-page manuscript, including references. The source hashes,
font checks and independent clean-build verification are recorded in the
October 26 build QA report supplied with the project artifacts.

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
The October 26 supplement adds fresh complete-split inference for None, RBM,
FastV and an official Hi-Lo adaptation under shared vision-SDPA/text-eager
attention. It reports all attempted IDs, zero scores for failures, paired
intervals and common-completion sensitivity. Hi-Lo retains the authors' vendored
model path, which remains an implementation boundary. The earlier fixed
retention curves and DocVQA-500 stress subset retain their mixed-attention
protocol and are explicitly separated from the new main table. Corrected
complete-request measurements cover ten configurations on fixed TextVQA-100 and
OCRBench-100 requests with three repetitions. Quality and actual request cost
are paired on those same questions; the invalid first timing implementation is
excluded. A 95% quality target on these subsets is a point-estimate rule, not a
claim of full-split quality equivalence.

The project checkout separately provides `CHANGELOG.md` and
`SUBMISSION_CHECKLIST.md` for revision history and author-side review. These
internal documents are excluded from the minimal anonymous source archive.

The October 26 experimental report and technical audits are maintained outside
this anonymous source directory in `reports/`. Raw predictions, timing repeats,
annotations, model weights and machine-specific logs are excluded from the
review source archive. No conference upload has been made.
