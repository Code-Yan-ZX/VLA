# DocVQA image-cluster sensitivity (2026-10-26)

The frozen 500-question DocVQA subset contains **417 document-page images**. Original parquet metadata maps questionId to docId and the tuple (ucsf_document_id, ucsf_document_page_no); those two image identifiers form a verified bijection on all 5,349 validation questions. QID-based image filenames were not used as cluster IDs. Only metadata columns were read; no image decode or image-content hashing was needed.

Cluster sizes are 348 single-question pages, 57 two-question pages, 10 three-question pages, and 2 four-question pages. All 500 saved questions and joined GT strings match their original metadata. Scoring reuses the pinned ANLS function on the existing saved predictions; the primary scores and question-level CIs are unchanged.

For each of 20,000 paired bootstrap replicates, sample 417 images with replacement, carry all sampled image questions with their multiplicities, and divide by the number of questions in that replicate. This preserves the question-weighted estimand while allowing within-image dependence. Fixed seeds are 20261400 and 20261401.

| Pair | ANLS difference (pp) | Primary question-bootstrap 95% CI | Image-cluster 95% CI |
|---|---:|---:|---:|
| RBM − Hi-Lo | +6.39 | [+3.04, +9.78] | [+2.91, +9.87] |
| RBM − FastV-k3 | +12.32 | [+8.15, +16.47] | [+8.05, +16.61] |

Both intervals remain above zero. This is a sensitivity analysis for the fixed subset and does not remove the Hi-Lo attention/model-code confound or establish full-split generalization. No new equivalence or p-value claim is made.

Provenance, metadata mapping digest, prediction SHA256s, exact seeds, and unrounded results are preserved in `cvm2027_docvqa_cluster_sensitivity_20261026.json`.

Reproduce from the worktree root:

```bash
/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_doc_cluster.py
```

ANLS source: lmms-eval commit `8f40097532a7bf4db6123713f1a905e02c8f5971`, `lmms_eval/api/metrics.py`, SHA256 `eaf48195af80a46d22e545a03a1bd7e4cad4fcbadeb3d2d817ac7d148390d8f4`.

Original staging metadata has no independently verified Hub revision. Its canonical metadata-column projection SHA256 is `32d44390dbf3bef851e09497a93ad74ab428f6e3f3bbc5ec21cc93f866be1930`; the selected question-to-page mapping SHA256 is `0b32bff9473aebb0fdc00750bcebe71d8d3c8e24f60fde7ebaa43576560434ec`. The JSON records per-shard metadata digests, sizes, prediction SHA256s, seeds, and unrounded results.
