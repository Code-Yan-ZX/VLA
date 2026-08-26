# Final paper synchronization report

Date: 2026-08-26

## 1. Branch and base

- Working branch: `paper/submission-hardening-final`
- Exact base: `cad68f902fa4606d4c41f9d7b64e0dcdd86ccb82` (`origin/paper/native-mrope-audit-fix`)
- Work was performed in an isolated worktree; no user changes in the original worktree were overwritten.
- The experimental branches were not merged or cherry-picked. Only the requested reports were read with `git show`.

## 2. Source hierarchy used

1. `drafts/overleaf_submission/{main.tex,supp.tex}` and their compiled PDFs at the base commit.
2. `2ec9d15:experiments/mrope_scorer_provenance_audit.md`.
3. `74f15f1:experiments/merger_representation_method_discovery.md` and `74f15f1:experiments/merger_representation_goal_log.md`.
4. Aggregate control evidence in `results/acmmm_final_controls/analysis.json` and `reports/acmmm_final_controls.md`.
5. `drafts/paper_acmmm.md` was treated as a historical draft and did not overwrite the LaTeX source.

The academic-search workflow was restricted to the user-named near neighbours. Primary arXiv records were checked for HTC-VLM, IVC-Prune, Reroute, Nüwa, DUET-VLM, GOTS, OccamToken, and GMC; no open-ended method search was performed. FastV and VisionZip were already covered. `RBM-OT` is an internal failed representation candidate, not a verified external paper, and was not cited as prior art.

## 3. Modified paper and evidence files

- `drafts/overleaf_submission/main.tex`
- `drafts/overleaf_submission/supp.tex`
- `drafts/overleaf_submission/references.bib`
- `drafts/overleaf_submission/figs/fig2.pdf`
- `drafts/overleaf_submission/main.pdf`
- `drafts/figures/real_data_pipeline/data/fig2_values.json`
- `experiments/paired_metric_statistics.json`
- `experiments/paired_metric_statistics.md`
- `STATE.md`, `DECISIONS.md`, and this report

## 4. Substantive claim changes and reasons

- Abstract, introduction, results, limitations, and conclusion now scope the finding to tested text-dense workloads, shared query-blind L2-magnitude scoring, fixed model/budget, and merger-equipped architectures. This removes any universal-SOTA or all-task implication.
- The matched-depth full-split GQA result is reported as a significant counter-direction (`pre-final 0.4207`, `post 0.4771`, `-5.64 pp`, 95% CI `[-6.42,-4.86]`) rather than a tie. It is used as a workload-conditioned boundary, not hidden or generalized away.
- The original `n=200` GQA `0.450/0.450` estimate is retained only as an explicitly superseded small-subset estimate.
- The full matched-depth controls now report all four full splits: TextVQA `+27.68 pp`, DocVQA `+4.59 pp`, OCRBench `+235/1000`, and GQA `-5.64 pp`, with their paired intervals and Holm-adjusted p-values.
- Qwen2.5-VL OCRBench was changed from `476/183` to the matched 4M-pixel pair `480/182` (`+298/1000`); the main table, supplement, Figure 2, JSON statistics, and Markdown statistics mirror were synchronized.
- Qwen3-VL DocVQA remains `RBM 0.5924` versus `FastV 0.5863`, described only as numerical parity / paired inconclusive. No paired CI was invented.
- Related work was shortened and focused on direct novelty boundaries: prompt-aware scoring, discarded-message transport, subset complementarity, two-stage reduction, learned bottlenecks, and token lifetime. It does not claim those methods ran the same controlled stage experiment.
- Deferred/MALT/cascade/adaptive/residual-representation candidates were not restored as methods or contributions. The formal story remains finding-driven and mechanism-centered, with RBM as the minimal operationalization.
- S9 no longer claims that a completed anonymous manifest exists; it now describes an expected per-cell index that must be assembled and verified before upload.

## 5. Key-result provenance checklist

| Result | Raw or compressed evidence | Official scorer / analysis | Configuration and status |
|---|---|---|---|
| Main full-split Qwen stage matrix | historical `runs/full_matrix/j7_*` paths | `src/v3_premerger/official_scorers.py`; `experiments/paired_metric_statistics.{json,md}` | vLLM main matrix; raw run JSONs are absent locally, so the aggregate is not end-to-end locally auditable |
| Full matched-depth Qwen3 control, including GQA | 8 committed gzip payloads under `artifacts/acmmm_final_controls/p0_1/` | `scripts/analyze_acmmm_final_controls.py`; `results/acmmm_final_controls/analysis.json` | full splits, 25% retention, pre-final vs post, identical IDs/tokens, zero skips; 8/8 compressed payloads present |
| Qwen2.5 OCRBench `480/182` | gzip payloads under `artifacts/acmmm_final_controls/p0_2/` | same official scorer and analysis | Qwen2.5-VL-7B, full 1000, vLLM, L2, 25% retention, common 4M cap, greedy/max32/seed0, identical IDs and 96.8 mean visual tokens, zero skips, skip-as-zero denominator 1000 |
| Qwen3 DocVQA `0.5924/0.5863` | `runs/malt_goal_mode/h0n_docvqa_n200.json`; `runs/r2_same_scope/r2b_qwen3vl_fastv_k3_docvqa_r0.75_n500.json` | official ANLS and `src/v3_premerger/paired_stats.py`; procedure in `experiments/paper_native_mrope_correction_report.md` | native coordinate, 600k cap, nominal n=200; both exact raw files are absent, so parity/inconclusive is retained |
| InternVL3 full matrix | 16 `internvl3_*.json` names in S9 | official scorer; `experiments/internvl3_main_matrix.md` | aggregate and scripts present, 0/16 raw JSONs local |
| FastV/RBM regime table | 8 `r2b_*` and 5 `r2c_*` S9 names | official scorer and paired-stats code | reports/harness present; all 13 raw JSONs absent locally |

Formal accuracy never uses a harness `correct` field as the official metric. The nine prespecified 25%-retention text-dense contrasts retain their Holm family; the four matched-depth full-split controls use a separate four-test Holm family. Small unpaired or untested differences remain descriptive.

## 6. GQA contradiction resolution

The old `0.0 pp` value comes from Qwen3-VL `n=200`, 25%-retention pre-final/post, same positions/scorer/budget, recorded in `experiments/p0-3_prefinal_control.md`. The authoritative later result is the same pure-stage comparison on the full GQA split (`n=12578`), recorded in `reports/acmmm_final_controls.md` and `results/acmmm_final_controls/analysis.json`: `0.4207` vs `0.4771`, `-5.637 pp`, paired 95% CI `[-6.416,-4.857]`, Holm `p=5e-5`, zero skips, identical token counts. The paper now treats the small-subset tie as sampling error and the full split as the reported result.

## 7. DocVQA 0.5924 synchronization

Searches of the final TeX and PDF find `0.5924` and `0.5863` in the intended regime comparison and no `0.4239`. The table note, main discussion, supplement, and conclusion do not claim significance. Exact H0n-versus-FastV paired uncertainty remains blocked on the two missing raw files listed above.

## 8. Qwen2.5 OCRBench 480/182 decision

Adopted. All requested compatibility checks passed from the P0-2 report and committed gzip payloads: same Qwen2.5-VL-7B model, official `/1000` scorer, full 1000-item manifest, common 4M pixel cap, 25% retention and identical per-item token counts, identical IDs, zero skips, and denominator 1000. The native-resolution none arm is not iso-configuration and remains a descriptive upper-bound anchor; the replacement affects the matched pre/post contrast only.

## 9. Anonymous artifact integrity

Artifact complete: **no**.

- Historical contract: `0/53` staged JSONs. The 53-file gate is recoverable from S9 at commits `0829e97` through `3c24aa7`.
- Current `cad68f9` S9 contract: `0/29` staged JSONs. The build script dynamically parses current S9, so `STATE.md` and the older recovery plan's fixed 53 count had drifted from the manuscript.
- No `artifact_anonymous/`, anonymous `manifests/SHA256SUMS`, anonymity TSV, recovery report, or machine-readable audit-class metadata manifest exists locally.
- `results/acmmm_final_controls/MANIFEST.sha256` is a separate campaign manifest: 67 rows = 29 hash/size OK, 28 missing paths, 10 stale hash/size entries. The 28 committed gzip byte hashes pass; 24 decompressed payloads match corresponding raw hashes. Four none anchors have no recoverable local raw payload.

Exact historical 53 expected staging paths (first 24 are historical-only; last 29 are current S9):

```text
artifact_anonymous/results/gate_cas12_docvqa.json
artifact_anonymous/results/gate_cas12_gqa.json
artifact_anonymous/results/gate_cas12_ocrbench.json
artifact_anonymous/results/gate_cas12_textvqa.json
artifact_anonymous/results/gate_cas25_docvqa.json
artifact_anonymous/results/gate_cas25_gqa.json
artifact_anonymous/results/gate_cas25_ocrbench.json
artifact_anonymous/results/gate_cas25_textvqa.json
artifact_anonymous/results/gate_fst12_docvqa.json
artifact_anonymous/results/gate_fst12_gqa.json
artifact_anonymous/results/gate_fst12_ocrbench.json
artifact_anonymous/results/gate_fst12_textvqa.json
artifact_anonymous/results/gate_fst25_docvqa.json
artifact_anonymous/results/gate_fst25_gqa.json
artifact_anonymous/results/gate_fst25_ocrbench.json
artifact_anonymous/results/gate_fst25_textvqa.json
artifact_anonymous/results/gate_pre12_docvqa.json
artifact_anonymous/results/gate_pre12_gqa.json
artifact_anonymous/results/gate_pre12_ocrbench.json
artifact_anonymous/results/gate_pre12_textvqa.json
artifact_anonymous/results/gate_pre25_docvqa.json
artifact_anonymous/results/gate_pre25_gqa.json
artifact_anonymous/results/gate_pre25_ocrbench.json
artifact_anonymous/results/gate_pre25_textvqa.json
artifact_anonymous/results/internvl3_none_docvqa_r0.000_full.json
artifact_anonymous/results/internvl3_none_gqa_r0.000_full.json
artifact_anonymous/results/internvl3_none_ocrbench_r0.000_full.json
artifact_anonymous/results/internvl3_none_textvqa_r0.000_full.json
artifact_anonymous/results/internvl3_post_docvqa_r0.750_full.json
artifact_anonymous/results/internvl3_post_docvqa_r0.875_full.json
artifact_anonymous/results/internvl3_post_gqa_r0.750_full.json
artifact_anonymous/results/internvl3_post_ocrbench_r0.750_full.json
artifact_anonymous/results/internvl3_post_textvqa_r0.750_full.json
artifact_anonymous/results/internvl3_post_textvqa_r0.875_full.json
artifact_anonymous/results/internvl3_pre_docvqa_r0.750_full.json
artifact_anonymous/results/internvl3_pre_docvqa_r0.875_full.json
artifact_anonymous/results/internvl3_pre_gqa_r0.750_full.json
artifact_anonymous/results/internvl3_pre_ocrbench_r0.750_full.json
artifact_anonymous/results/internvl3_pre_textvqa_r0.750_full.json
artifact_anonymous/results/internvl3_pre_textvqa_r0.875_full.json
artifact_anonymous/results/r2b_qwen2vl_fastv_k3_docvqa_r0.75_n200.json
artifact_anonymous/results/r2b_qwen2vl_fastv_k3_gqa_r0.75_n200.json
artifact_anonymous/results/r2b_qwen2vl_fastv_k3_ocrbench_r0.75_n200.json
artifact_anonymous/results/r2b_qwen2vl_fastv_k3_textvqa_r0.75_n200.json
artifact_anonymous/results/r2b_qwen3vl_fastv_k3_docvqa_r0.75_n500.json
artifact_anonymous/results/r2b_qwen3vl_fastv_k3_gqa_r0.75_full12578.json
artifact_anonymous/results/r2b_qwen3vl_fastv_k3_ocrbench_r0.75_n500.json
artifact_anonymous/results/r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000.json
artifact_anonymous/results/r2c_qwen2vl_pre_r0.75_docvqa_n200.json
artifact_anonymous/results/r2c_qwen2vl_pre_r0.75_gqa_n200.json
artifact_anonymous/results/r2c_qwen2vl_pre_r0.75_ocrbench_n200.json
artifact_anonymous/results/r2c_qwen2vl_pre_r0.75_textvqa_n200.json
artifact_anonymous/results/r2c_qwen3vl_pre_r0.75_ocrbench_n200.json
```

## 10. Representation NO-GO

Not added to the supplement. The locked candidate loses to native RBM on all four reported workloads and is approximately `-2.14 pp` macro against the actual stronger parent `max(native RBM, FastV)`; `Full` is only the uncompressed upper bound. Its residual constructions are either badly scaled or highly redundant and trade spatial coverage for duplicate detail. Adding the failed candidate would blur the paper's already sufficient minimality evidence from five prespecified extension failures. It remains an internal NO-GO, not a proposed contribution.

## 11. Build and visual QA

- Commands: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` and the same for `supp.tex` from `drafts/overleaf_submission`.
- Outputs: integrated `main.pdf` = 22 pages; standalone `supp.pdf` = 12 pages. The body occupies approximately eight pages, with final discussion/conclusion sharing page 9 with references; references occupy pages 9--10; integrated supplement begins on page 11.
- Logs: 0 undefined citations, 0 undefined references, 0 overfull boxes, and no fatal LaTeX errors. Only non-blocking underfull warnings remain.
- Fonts: `pdffonts` reports no Type 3 fonts.
- Rendered pages covering the abstract, related work, main tables, Figure 2, limitations/conclusion, supplement start, and S7b/S9 were inspected. Figures, captions, page numbers, and the supplement title render normally. S7b is forced behind the earlier float queue so its heading, setup, complete full-width table, and interpretation remain together on page 18.
- Final PDF/TeX risk scan finds no `0.4239`, `+16.2`, `all three Qwen3`, `MALT`, `Deferred`, `MALT-C`, or `cascade`. Remaining `0.0` occurrences are legitimate: a TextVQA swap-control zero, negative-gate thresholds/results, and the explicitly superseded `n=200` GQA tie.

## 12. Remaining blockers and author decisions

Blockers before any external submission:

1. Restore the server export and choose one authoritative artifact contract (historical 53, current 29, or a new complete manifest), then stage, anonymity-scan, hash, and verify the exact upload bytes.
2. Restore the two DocVQA raw files and recompute the exact paired H0n-versus-FastV interval, or retain the present conservative parity wording.
3. Restore/replace the four missing none-anchor raw JSONs and regenerate the stale control manifest for the final repository state.

Author decisions:

1. Select the target venue/template and verify its body/reference/supplement limits; no venue page limit is asserted here.
2. Decide anonymous versus identified author metadata and supply affiliation/email if identified. The current generic draft names Zhengxing Yan.
3. Approve the final paper and artifact package before any external upload. No submission was performed.
