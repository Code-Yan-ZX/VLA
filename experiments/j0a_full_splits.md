# J0a — Full official splits landed (4 benchmarks)

**Date**: 2026-07-23 · **Builder**: `scripts/build_full_splits.py` (mirrors `build_subsets*.py`) · **Env**: conda `base` (datasets 3.6.0 / PIL 11.2.1 / pyarrow 21.0.0), CPU/network only, no GPU.

All 4 official complete splits are on disk under `eval/full_splits/` with images under `runs/data/`. Schema is byte-for-byte the subset schema (see per-benchmark notes). All HF cache forced to `/media/disk2/.../runs/data/hf_cache`; **`~/.cache/huggingface` stayed 0 bytes (no home writes).**

## Per-benchmark

| benchmark | jsonl | rows (got/expect) | images | disk | schema vs subset |
|---|---|---|---|---|---|
| TextVQA val | `eval/full_splits/textvqa_val.jsonl` | **5000 / 5000** | 5000 | 875M | identical `{id,image,question,gt}` |
| DocVQA val | `eval/full_splits/docvqa_val.jsonl` | **5349 / 5349** | 5349 | 3.1G | identical `{id,image,question,gt}` |
| OCRBench | `eval/full_splits/ocrbench.jsonl` | **1000 / 1000** | 1000 | 117M | subset `{…,question_type,dataset,choices}` **+ `category`** |
| GQA testdev | `eval/full_splits/gqa_testdev.jsonl` | **12578 / 12578** | 12578 | 1.3G | identical `{id,image,question,gt}` |

Cross-validation (all 4): 0 missing/non-absolute image paths, 0 empty q/gt, every image re-opened + verified on a 200-sample.

## Conventions mirrored from subsets
- **Short-answer instruction baked** verbatim into `question` for TextVQA/DocVQA/GQA: `"<q>\nAnswer the question using a single word or phrase."` (5000/5000, 5349/5349, 12578/12578 — matches latest `*_200.jsonl`). **OCRBench keeps the raw question** (no suffix), matching `ocrbench_200.jsonl`.
- **gt rules**: TextVQA = all 10 answers `;`-joined (official VQA-acc `min(#match/3,1)`); DocVQA = answer(s) `;`-joined (usually 1); GQA = single answer (exact-match); OCRBench = acceptable answers `;`-joined.
- Images saved as JPEG q92 to `runs/data/<bm>/<id>.jpg` (one file per question id, same as subsets).

## OCRBench 5-way `category` (task requirement)
`category` added on top of subset fields; derived from fine `question_type`. Verified **exactly 200 per skill** on the full 1000:
- Text Recognition ← Regular+Irregular+Artistic+Digit String (4×50)
- Scene Text-centric VQA ← Scene Text-centric VQA (200)
- Document Text-centric VQA ← Doc-oriented VQA (200)
- Handwriting Text Recognition ← Handwriting Recognition (IAM,50) + Handwritten Math Expression (HME100k,100) + **Non-Semantic Text Recognition (50)** = 200
- Key Information Extraction ← Key Information Extraction (200)

Non-Semantic→HTR is the only assignment yielding 200/skill (TR would otherwise be 250 / HTR 150). `question_type` (fine, 10 values) + `dataset` also carried. HME100k rows (100) keep the scorer flag `choices=["__nospace__"]`.

## GQA — no 20GB pull needed
12578 testdev questions reference only **398 unique images**, all present in `lmms-lab/gqa` `testdev_balanced_images` (~66MB curated config). **Coverage 398/398, 0 missing** → no full-GQA-image download. Each unique source image encoded once and written per question id (mirrors subset; 12578 files / 1.3G).

## Deviations / notes
1. **OCRBench** full = subset fields + `category` (the required 5-way skill field). No subset field dropped.
2. **TextVQA**: 13/5000 rows have 11 `;`-fields instead of 10 (one answer literally contains `;`). Same `;`-join behavior as the subset — not a new deviation; VQA-acc unaffected in practice.
3. **Env**: used `base`, not `qwen3vl_clean` — the latter has no `datasets`, and every existing `build_*.py` runs on `base` for the same reason (keeps the clean inference env clean).
4. **Network**: the HF python client threw intermittent `SSL UNEXPECTED_EOF` on large transfers (small OCRBench/GQA files survived). Fixed by pulling the exact parquet shards directly with **resumable `curl -C - --retry`** into `runs/data/staging/` (TextVQA 3 shards ≈920MB, DocVQA 6 shards ≈1.06GB) and loading via `load_dataset("parquet", data_files=…)`. Hub was reachable via proxy; `hf-mirror.com` returned empty for these repos so direct `huggingface.co` was used.
5. Staging parquets (1.9G) and HF cache (6.4G, incl. processed arrow) retained on `/media/disk2` for reproducibility; total `runs/data` = 14G.

## Next step
Point the full-matrix runner (`src/v3_premerger/v3_premerger_runner.py` / `serve_bench`) at these 4 jsonl for the full evaluation; scorers already key on `question_type`/`choices` (OCRBench) and the `;`-joined gt (TextVQA/DocVQA), so no scorer change is needed. Optional: add a 5-way OCRBench rollup reading `category`.
