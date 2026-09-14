# DCC 2027 submission draft

This directory contains the DCC 2027 single-column manuscript derived from the
authoritative TCSVT draft and the four figures embedded in
`drafts/RBM_merger_aware_中文_Fig1更新版.docx`. Figures 2, 4, and 5 use the
project's original PDF vector sources corresponding exactly to the Word
figures. Figure 1 keeps the exact updated Word image because the repository
contains only content-different vector predecessors for that version. Figure 3
adds a task rate--distortion view of the frozen Qwen3-VL retention sweep.

Build with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The PDF is a review draft. Before submission, replace the visible affiliation
and email placeholders in `main.tex` with the author's confirmed information.

The rate--distortion generation script is tracked at
`scripts/plot_dcc_rate_distortion.py`.
