# DCC 2027 submission draft

This directory contains the DCC 2027 single-column manuscript derived from the
authoritative TCSVT draft and the four figures embedded in
`drafts/RBM_merger_aware_中文_Fig1更新版.docx`. Figures 1 and 4 preserve the
selected Word content in vector form; Figures 2 and 5 retain the image-based
Word assets. Figure 3 adds a task rate--distortion view of the frozen Qwen3-VL
retention sweep.

Build with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The PDF is a review draft. Before submission, replace the visible affiliation
and email placeholders in `main.tex` with the author's confirmed information.

Figure-generation scripts are tracked at
`scripts/render_dcc_fig1_vector.py` and `scripts/plot_dcc_rate_distortion.py`.
