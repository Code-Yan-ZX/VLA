# DCC 2027 rewritten submission draft

This directory contains the evidence-audited rewrite requested on 2026-09-16.
The authoritative source is `main.tex`; `main.pdf` is its compiled 10-page US
Letter rendering with the official 12pt `dccpaper` class.

The rewrite separates the operational RBM comparisons from the Qwen3-VL
final-input control, reports model-specific implementation differences, removes
the unsupported RankBridge significance claim, reports OCRBench's effective
hybrid sample size as 181, and bounds the mechanism and efficiency conclusions.
The final manuscript contains 51 references, each cited substantively in the
body; it does not use `\nocite`.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`EVIDENCE_AUDIT.md` and `FIGURE_AUDIT.md` are internal traceability documents;
they are excluded from the external submission source archive. Before external
submission, all authors must confirm their displayed names, affiliations,
email addresses, corresponding-author designation, conflicts, and submission
metadata.
