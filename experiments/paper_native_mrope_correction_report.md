# Native-mRoPE paper correction report

Date: 2026-08-25

Audit authority: `origin/exp/deferred-rbm-n200`, commit `2ec9d15`

Authority report: `experiments/mrope_scorer_provenance_audit.md` at that commit

Official-rescore implementation: `scripts/audit_mrope_gateC_official_rescore.py` at that commit

## Scope and files

- Canonical submission: `drafts/overleaf_submission/main.tex` and `drafts/overleaf_submission/supp.tex`.
- Maintained LaTeX mirrors updated at the same semantic locations: `drafts/latex/paper_acmmm.tex` and `drafts/latex/supp_acmmm.tex`.
- `drafts/paper_acmmm.md` was not updated: `README.md` declares `drafts/overleaf_submission/` the submission authority, and this Markdown file is an older (last updated 2026-07-30) historical working draft rather than a synchronized submission mirror.
- No experiment code, raw output, archived draft, or unrelated user file was changed.

## Numeric correction

The only formal main-table numeric change is the Qwen3-VL-8B DocVQA same-scope regime-map row:

```text
DocVQA (dev; 600k cap) | n=200 | FastV-k3 0.5863 | RBM 0.5924 | +0.6 pp RBM (inconclusive)
```

- Old RBM value: `0.4239`, sourced from the positional-mismatched `vllm-mimic` cascade parent.
- Correct RBM value: official ANLS `0.5924`, native immediate RBM, `--mrope native`, 600k cap, 25% retention.
- Native run named by the audit: `runs/malt_goal_mode/h0n_docvqa_n200.json`.
- All other formal Table 1/2/3 numbers were left unchanged.

## Paired uncertainty audit

The exact desired raw pair is:

- `runs/malt_goal_mode/h0n_docvqa_n200.json`
- `runs/r2_same_scope/r2b_qwen3vl_fastv_k3_docvqa_r0.75_n500.json` (metadata scope `n=200`)

Both files are gitignored and absent from this checkout, so the exact native-immediate-RBM vs FastV paired bootstrap interval cannot be reconstructed locally. No interval was invented. The formal paper therefore reports only the audit-supported conclusion that RBM is numerically higher by about 0.6 pp and the paired difference is inconclusive/statistically indistinguishable.

As a tracked cross-check, the audited native-deferred K1 arm (official ANLS `0.5959`) joined to the tracked FastV per-sample answers gives K1-FastV `+0.96 pp`, paired-bootstrap 95% CI `[-6.96,+8.90] pp`, sign-flip `p≈0.811`, `n=200`; H0n differs from K1 by `-0.35 pp`, CI `[-1.85,+1.06] pp`. This supports conservative inconclusive wording but is not presented as the exact H0n-FastV CI.

Exact command to run after restoring both raw files:

```bash
python -c "from src.v3_premerger.paired_stats import load_cell_scores,run_pair; a,*_=load_cell_scores('malt_goal_mode/h0n_docvqa_n200.json','docvqa',False); b,*_=load_cell_scores('r2_same_scope/r2b_qwen3vl_fastv_k3_docvqa_r0.75_n500.json','docvqa',False); print(run_pair(a,b,'docvqa','RBM-native','FastV-k3',20000,0,n_nominal=200,is_table3=True))"
```

## Claim and accounting corrections

- Replaced “FastV leads all three Qwen3 non-OCR cells” with: FastV leads TextVQA and GQA; Qwen3 DocVQA is statistically indistinguishable (`0.5924` RBM vs `0.5863` FastV); RBM retains its clear OCRBench lead.
- Kept the native-position provenance statement: after replacing the one polluted cell, all formal `n=200` RBM cells use family-correct native positions; the Qwen2.5 `vllm-mimic` degeneration is not in the corrected table.
- Corrected OCRBench skip accounting: table values use nominal `n=200` denominators and count the identical skipped samples as zero (19 Qwen3; 20 Qwen2.5), rather than scoring only attempted samples.
- Removed the positional-mismatched RBM-to-FastV cascade numbers, tables, artifact-index entries, and derived conclusions from the body/supplement and maintained LaTeX mirrors. The remaining prespecified-extension count and supplement subsection references were adjusted accordingly. Raw cascade code/reports were not deleted.
- Formal-source searches found no `MALT`, `Deferred-RBM`, deferred-contextualization, or native-coordinate-immediate method naming.

## Compilation and checks

Commands, run from `drafts/overleaf_submission/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supp.tex
```

Results:

- Integrated PDF: `drafts/overleaf_submission/main.pdf`, 21 pages. Core body remains 8 pages, references 2 pages, supplement starts on page 11.
- Standalone supplement: `drafts/overleaf_submission/supp.pdf`, 11 pages.
- Final integrated log: 0 undefined references, 0 missing/undefined citations, 0 overfull boxes, 0 LaTeX/fatal errors.
- Fonts: 0 Type 3; all listed fonts embedded.
- Visual inspection: integrated page 6 (corrected regime-map table) and pages 14-15 (supplement flow after cascade removal) show no clipping, overlap, table overflow, or abnormal blank region.
- PDF text contains `0.5924`, the nominal-denominator/skip-as-zero note, and the inconclusive DocVQA wording.
- PDF text contains none of `0.4239`, `+16.2 F`, `all three Qwen3`, `over attempted`, `cascade`, `MALT`, or `Deferred-RBM`.
- Source audit over the four formal `.tex` files contains none of those stale strings or cascade terms. A broad directory search still finds “all three Qwen3” only in an untracked historical reviewer note (`drafts/overleaf_submission/review_back_half_20260819.md`) that documents the old defect; it is not submission source and was preserved.
- Non-blocking warnings: existing underfull box diagnostics and 95 BibTeX metadata-completeness warnings (mostly missing publisher/address/pages). No new blocking warning was introduced.

No venue-specific template/page-limit claim is made because the repository state still records that the venue is not selected. The generic submission layout remains at the established 8 body pages plus 2 reference pages.
