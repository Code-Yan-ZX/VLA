# Independent matched-main CPU audit (2026-10-26)

**PASS.** All eight full cells and all twelve planned pairs independently verified; each primary and common-answered interval uses 20,000 resamples with its prescribed seed (24 intervals total).

No GPU work, source prediction changes, manuscript edits, or figure rebuilds. The audit directly executes the pinned upstream TextVQA and OCRBench task functions; it does not call the main scorer or its metric wrappers. Bootstrap and sign-flip calculations are independently implemented.

| Cell | Independent official score | Attempts | Failures |
|---|---:|---:|---:|
| none_textvqa | 83.4500 | 5000 | 0 |
| pre_textvqa | 75.5820 | 5000 | 0 |
| fastv_textvqa | 76.8980 | 5000 | 0 |
| hilo_textvqa | 66.7480 | 5000 | 0 |
| none_ocrbench | 765.0000 | 1000 | 0 |
| pre_ocrbench | 610.0000 | 1000 | 0 |
| fastv_ocrbench | 453.0000 | 1000 | 0 |
| hilo_ocrbench | 612.0000 | 1000 | 0 |

TextVQA units are percentage points; OCRBench scores use points per 1000. Every official cell score, pair mean/delta/95% CI/p-value, fixed seed, count and common-answered result agrees within 1e-12 relative/absolute tolerance.

All IDs, official metadata, per-request visual N/K, kept indices, image grids, original input hashes, shared protocol and current dependency SHA256s pass. Saved first-repeat P3 parity is independently rechecked on 800 overlapping requests. The three warmup comparisons per cell are verified as recorded evidence; no GPU rerun is claimed.

Source hashes and all 24 reproduced interval values are recorded in the JSON. Question-paired 95% CIs do not prove equivalence or isolate algorithm differences beyond the declared implementation controls.

Reproduce from the worktree root (CPU only):

```bash
/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_audit_matched.py
```

Archived audit script: `/media/disk2/YZX/research/vla/.claude/worktrees/cvm-oneday-20261025/scripts/cvm24h_audit_matched.py`; SHA256 `56e72feea0adbb35aad2817833d6adf9cd58346d3a74b56c1224b485dd1d3881`.
