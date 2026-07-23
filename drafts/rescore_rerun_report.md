# Rescore Rerun — OFFICIAL metrics under short-answer prompting (keep=25%)

_Generated 2026-07-23 · offline CPU rescore of freshly re-run cells · scorer: `src/v3_premerger/official_scorers.py`_

## VERDICT (does pre>post hold under official metrics?)

> **✅ HOLD — pre>post is preserved on both text-dense benches under official metrics + proper short-answer prompting at keep=25%.**

- **textvqa** (VQA-acc): **HOLD** · pre−post gap = **+38.3 pp** (±4.5 pp combined stderr)
- **docvqa** (anls): **HOLD** · pre−post gap = **+26.5 pp** (±4.5 pp combined stderr)

**Prompt fix:** each TextVQA/DocVQA question now carries the canonical lmms-eval short-answer instruction `\nAnswer the question using a single word or phrase.` (subsets rebuilt by `scripts/fix_shortanswer_subsets.py`; originals backed up to `eval/subsets/_backup/`). Cells re-run at r=0.75 (keep=25%), n=200, --selector l2, seed=0 (== router_probe default), enforce_eager. DocVQA used the canonical big-doc config (--max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4).

## Official-metric results (mean ± binomial stderr)

| Bench | Metric | Baseline (none) | Post | Pre | Pre−Post gap (pp) | Retention pre/base | Retention post/base | Verdict |
|---|---|---|---|---|---|---|---|---|
| textvqa | VQA-acc | 85.8 ± 2.5 | 21.5 ± 2.9 | 59.8 ± 3.5 | +38.3 ± 4.5 | 70% | 25% | HOLD |
| docvqa | anls | 97.6 ± 1.1 | 20.0 ± 2.8 | 46.5 ± 3.5 | +26.5 ± 4.5 | 48% | 21% | HOLD |

_All values in %. stderr = sqrt(p(1−p)/n), n=200. Gap stderr = quadrature of the two independent cell stderrs._

## Per-cell sanity (n_answered, answer length under short-answer prompt)

| Cell | n | n_answered | n_skipped | median ans len | mean ans len | stored acc |
|---|---|---|---|---|---|---|
| textvqa/baseline (none_textvqa_r0.000_l2_n200.json) | 200 | 200 | 0 | 6 | 8 | 0.905 |
| textvqa/post (post_textvqa_r0.750_l2_n200.json) | 200 | 200 | 0 | 3 | 5 | 0.250 |
| textvqa/pre (pre_textvqa_r0.750_l2_n200.json) | 200 | 200 | 0 | 6 | 8 | 0.680 |
| docvqa/baseline (none_docvqa_r0.000_l2_n200.json) | 200 | 200 | 0 | 10 | 14 | 0.980 |
| docvqa/post (post_docvqa_r0.750_l2_n200.json) | 200 | 200 | 0 | 3 | 13 | 0.150 |
| docvqa/pre (pre_docvqa_r0.750_l2_n200.json) | 200 | 200 | 0 | 7 | 13 | 0.430 |

Median answer length should now be a single word/phrase (≪ the ~132-char verbose median seen under the raw-question prompt), confirming the prompt fix took effect.

## Artifacts

- Cells: `runs/v3_merger_aware/rescore_rerun/*.json` (per_sample saved)
- Summary: `runs/v3_merger_aware/rescore_rerun/rescore_summary.json`
- Subset fix: `scripts/fix_shortanswer_subsets.py`; backups `eval/subsets/_backup/`
- Rerun driver: `src/v3_premerger/v3_rescore_rerun.sh`
