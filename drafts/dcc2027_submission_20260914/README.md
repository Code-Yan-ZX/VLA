# DCC 2027 submission draft

This directory contains the DCC 2027 single-column manuscript derived from the
authoritative TCSVT draft and the figures embedded in
`drafts/RBM_merger_aware_中文_Fig1更新版.docx`. Figure 1 uses the
author-supplied replacement image `fig1_user_20260915_v2.png`. Figures 2, 4, and 5 use the
project's original PDF vector sources corresponding to the Word figures;
Figure 3 adds a task rate--distortion view of the frozen Qwen3-VL retention
sweep. All 51 research references are cited in the text without `\nocite`.

Build with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The PDF is a review draft. Before submission, all authors should verify the
spelling of their names, affiliations, email addresses, and corresponding-author
designation in `main.tex`.

The rate--distortion generation script is tracked at
`scripts/plot_dcc_rate_distortion.py`.
