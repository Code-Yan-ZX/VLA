# ACM MM Final Controls — Verification Report

**Date:** 2026-08-19
**Branch:** `exp/acmmm-final-controls`
**Scope:** Final experiment verification for submission. No LaTeX / paper text was modified. This report audits results and flags claim-level events for the author's decision.

---

## Executive summary

| Experiment | Result | Status |
|---|---|---|
| **P0-1** pure-stage control (Qwen3-VL-8B, full splits, pre-final vs post) | TextVQA **+27.7 pp**, DocVQA **+4.6 pp**, OCRBench **+235 pts**, GQA **−5.6 pp** (all Holm-significant) | ✅ verified — **CLAIM-LEVEL EVENT on GQA** |
| **P0-2** Qwen2.5-VL-7B OCRBench matched-config rerun (both arms 4 M-px, iso-token) | κ=0.25: pre **480** vs post **182** (**+298 pts**); κ=0.125: **328** vs **70** (+258 pts) | ✅ verified — **can replace Table 1 Qwen2.5 OCRBench cell: YES** |
| **P1** RBM vs FastV-k3, OCRBench full 1000, same HF eager harness (both arms 4 M-px) | Qwen3-VL: RBM **559** vs FastV **413** (**+146 pts**); Qwen2.5-VL: **382** vs **295** (**+87 pts**) | ✅ verified |

**Status legend:** `verified` = completed with official rescoring, iso-token/iso-sample checks passed; `failed-incomplete` = cell missing/aborted; `config-mismatch` = arms differ in configuration; `interpretation` = inference drawn from verified numbers (author decision required where flagged).

---

## P0-1 — Qwen3-VL-8B pure-stage control (pre-final vs post, FULL splits)

### Purpose
The headline RBM ("pre") ranks at the ViT layer-8 / deepstack[0]-input tap and slices all 4 mergers, while "post" ranks at the main-merger **output** — a confound in **both stage and feature space**. `pre-final` ranks L2 at the main-merger's **own input** (final ViT-block features) and selects top-κ merge-units from the visual output; deepstack mergers run FULL/untouched. The only difference vs `post` is *before vs after* the lossy 2×2 main merger — a pure stage variable.

### Protocol (iso-config with paper Table 1)
- Model: `Qwen/Qwen3-VL-8B-Instruct` (HF hub cache, offline), greedy (temp=0), `--selector l2`, `--max-tokens 32`, `--r 0.75` (keep 25%), `--max-pixels 0` (native).
- Flags: textvqa/ocrbench/gqa `--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9`; docvqa `--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9` (identical to `runs/full_matrix/j7_full_matrix_vllm.sh`).
- Engine: vLLM 0.19.0 inline (runner `src/v3_premerger/v3_premerger_runner.py`, mode `pre-final` / `post`), `enforce_eager=True`, `limit_mm_per_prompt={"image":1}`, V1 enabled.
- Deepstack behavior: **both** arms run all deepstack mergers untouched (full); `pre-final` selects the same k units from the merged visual output (unit equivalence), `post` prunes the merged splits.
- none arms: **not re-run**; reuse the verified full-split anchors cited by Table 1 (`runs/full_matrix/j7_qwen3vl_none_*`, `runs/r2_same_scope/r2_qwen3vl_none_docvqa_full5349.json`). Their official rescore reproduces the paper exactly (textvqa 0.8443, docvqa 0.9562, ocrbench 760, gqa 0.6165).
- Command: `python src/v3_premerger/v3_premerger_runner.py --model-family qwen3vl --benchmark <b> --subset eval/full_splits/<b>*.jsonl --n <N> --r 0.75 --mode {pre-final,post} --selector l2 --max-tokens 32 <FLAGS> --out results/acmmm_final_controls/p0_1/<tag>.json`

### Results (official rescoring, per-sample JSONs)

| bench | none (anchor) | pre-final | post | Δ (pre-final − post) | 95% CI | p (sign-flip) | Holm p | n paired | iso-token | mean tokens pre/post |
|---|---|---|---|---|---|---|---|---|---|---|
| TextVQA | 0.8443 | 0.4985 | 0.2217 | **+27.68 pp** | [+26.21, +29.19] | 5e-05 | 2e-04 | 5000 | ✓ (5000/5000) | 215.8 / 215.8 |
| DocVQA | 0.9562 | 0.2836 | 0.2377 | **+4.59 pp** | [+3.43, +5.78] | 5e-05 | 1.5e-04 | 5349 | ✓ (5349/5349) | 946.7 / 946.7 |
| OCRBench | 760/1000 | 419/1000 | 184/1000 | **+23.50 pp (+235 pts)** | [+19.90, +27.10] | 5e-05 | 1e-04 | 1000 | ✓ (1000/1000) | 228.9 / 228.9 |
| GQA | 0.6165 | 0.4207 | 0.4771 | **−5.64 pp** | [−6.42, −4.86] | 5e-05 | 5e-05 | 12578 | ✓ (12578/12578) | 96.8 / 96.8 |

- Attempted/completed/skipped: pre-final and post **0 skips on every benchmark** (5000/5000, 5349/5349, 1000/1000, 12578/12578). none anchors: textvqa 2/5000, ocrbench 18/1000 (context overrun, scored 0 — matches Table 1 note b), others 0.
- Sample-ID equality: ✓ (all benches). Per-sample token-count equality (prompt_token_ids): ✓ (all benches).
- Reproducibility: campaign `post` cells reproduce the paper's j7 `post` cells **bit-for-bit** (textvqa 0.2217=0.2217, docvqa 0.2377=0.2377, ocrbench 184=184, gqa 0.4771=0.4771; 0 answer strings differ).

### Interpretation (status: `verified` numbers + `interpretation`)
1. **Text-dense stage effect confirmed at full split.** Pre-final still beats post by a large margin on TextVQA (+27.7 pp) and OCRBench (+235 pts), and by a smaller but significant margin on DocVQA (+4.6 pp). Ranking *before* the lossy merger preserves text — a true stage effect, not a feature-depth artifact.
2. **Feature-depth contributes on top.** The headline layer-8 `pre` cells are higher than `pre-final` (TextVQA 0.605 vs 0.499; DocVQA 0.481 vs 0.284; OCRBench 547 vs 419), i.e. the deepstack[0]-input tap adds substantial value beyond the pure stage. The paper already discloses this (pre-final is the matched-depth control); the full-split numbers make the decomposition explicit.
3. **⚠️ CLAIM-LEVEL EVENT — GQA.** The paper (§sec:qwen, Table 1 caption, n=200 control) states *"pre-final gives 0.0 pp on GQA"* / *"no detected pure-stage difference"*. The **full-split** control shows pre-final **0.4207 vs post 0.4771, Δ = −5.64 pp, 95% CI [−6.42, −4.86], p = 5e-05, McNemar 941 vs 1650 (p≈0), n = 12578, iso-token verified**. The pure-stage effect on GQA is **significantly negative** (post-merger selection is *better* on object-QA), and the layer-8 tap does not dissolve it — it only partially compensates (headline pre 0.449 vs pre-final 0.421, +2.8 pp). The n=200 estimate (0.450=0.450) was a sampling artifact of that 200-item subset. **The statement "pre-final equals post on GQA (0.0 pp)" is not supported at full split** and must be revised before submission (e.g. "post-stage leads on GQA; magnitude 5.6 pp, 1/5 to 1/6 of the text-dense deltas"). No paper text was changed per instructions.
4. DocVQA: the n=200 pre-final estimate (+12.5 pp) **overstated** the full-split effect (+4.6 pp). All final numbers below use the full-split estimates.

---

## P0-2 — Qwen2.5-VL-7B OCRBench matched-configuration rerun

### Why
The paper Table 1 Qwen2.5-VL OCRBench primary cell was **not iso-configuration**: RBM (pre) ran at `max_pixels=4000000` (mean tokens 229.6) while post ran at native (mean tokens 282.0). Native resolution makes the qwen2vl `pre` path fail outright (verified: `j7_qwen2vl_pre_ocrbench_r0.750_full.json.broken_prev` = skip 1000/1000), so the matched rerun uses **4 M-px for BOTH arms** — the only cap where both complete, and the same cap the paper's RBM cell already used.

### Protocol (status: `verified`)
- Model `Qwen/Qwen2.5-VL-7B-Instruct` (HF hub cache; the directory the task calls "Qwen2.5-VL-8B-Instruct" is a misnamed Qwen2.5-VL-7B), OCRBench full 1000, `--max-pixels 4000000`, `--max-num-seqs 4 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9`, greedy, `--selector l2`, `--max-tokens 32`.
- κ = 0.25 (r=0.75) first, then κ = 0.125 (r=0.875). Both arms (pre / post) per κ.
- Engine: vLLM 0.19.0 inline runner. Logs next to each JSON + `results/acmmm_final_controls/p0_2/*.log`.

### Results (official rescoring)

| κ | pre (RBM) | post | Δ | paired Δ (pp) | 95% CI | p | Holm | n | iso-token | tokens pre/post |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.25** | **480/1000** | 182/1000 | **+298 pts** | +29.8 | [+26.4, +33.3] | 5e-05 | 1e-04 | 1000 | ✓ | 229.6 / 229.6 |
| 0.125 | **328/1000** | 70/1000 | **+258 pts** | +25.8 | [+22.7, +29.0] | 5e-05 | 5e-05 | 1000 | ✓ | 131.5 / 131.5 |

- Attempted/completed/skipped: 1000/1000/0 on all four cells. Context overrun: none at 4 M-px (the cap removes the native-resolution overruns). Skip handling is symmetric by construction (no skips).
- Sample-ID equality: ✓. Per-sample token-count equality: ✓ (every sample).
- **Verification of the +298 > +293 "conservative" claim:** the old mismatched pair (pre 476 @229.6 vs post 183 @282.0, +293) gave post a *larger* token budget. At matched iso-token budget (229.6 = 229.6) the gap is **+298** — the paper footnote (d) "conservative" characterization is **confirmed**.

### Answer to the required question
**"Can this result directly replace the paper Table 1 Qwen2.5-VL OCRBench primary cell: YES/NO?" → YES.**
- New κ=0.25 primary pair: **RBM 480 vs post 182 (Δ +298 pts)** at iso-configuration (same processor, pixel cap, input resolution, sample IDs, prompt, decoding, κ=0.25, identical per-sample token counts). The old pair (476/183) is superseded by this strictly more valid iso-configuration measurement.
- κ=0.125 secondary pair: **328 vs 70 (+258)**. Note: the old @12.5% cells (335/67) were already iso-config at native (both arms `max_pixels=0`, ptid 157.7); the new @12.5% pair runs at the campaign's unified 4 M-px cap, so it is a fresh matched-config measurement at a different cap — use it if the Table adopts the 4 M-px campaign; otherwise the old native @12.5% cells remain internally consistent.

---

## P1 — RBM vs FastV-k3, OCRBench full 1000, same HF harness

### Why
Give a same-harness RBM-vs-FastV comparison on OCRBench full (the paper Table 2 has n=200 native with ~10% skips). Both arms under `baselines_hf.py` (HF transformers eager attention), so **no vLLM number is mixed into this table**.

### Protocol (status: `verified`)
- Model Qwen3-VL-8B and Qwen2.5-VL-7B (HF hub cache, offline).
- RBM: `--mode pre --r-pre 0.25 --mrope native` (keep 25% of merger-input units; `--mrope native` REQUIRED — the vllm-mimic layout is the known-degenerate Qwen2.5-VL×pre path, Table 2 note iii; the paper's Table 3 RBM cells use native).
- FastV: `--mode fastv --fastv-k 3 --r 0.75` (keep 25% of image tokens after LLM layer 3; K=3 is the paper's best-K).
- Same subset (`eval/full_splits/ocrbench.jsonl`, n=1000), same prompt, greedy (max_tokens 32), `--max-pixels 4000000` on both arms (campaign-unified OCRBench cap), `--seed 0`.
- Smoke gates (n=8 per path) passed before the full runs.
- Command: `python src/v3_premerger/baselines_hf.py --mode {pre|fastv} --model <model> --benchmark ocrbench --subset eval/full_splits/ocrbench.jsonl --n 1000 [--r-pre 0.25 --mrope native | --r 0.75 --fastv-k 3] --max-pixels 4000000 --seed 0 --out results/acmmm_final_controls/p1/<tag>.json`

### Results (official rescoring; skips scored 0 in /1000)

| model | RBM /1000 | FastV-k3 /1000 | Δ (RBM−FastV) | paired Δ | 95% CI | p | Holm | n paired | skips (symmetric) | RBM-only / FastV-only / both | tokens RBM/FastV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | **559** | 413 | **+146 pts** | +15.9 pp | [+12.8, +18.9] | 5e-05 | 1e-04 | 921 | 79 = 79 (identical IDs) | 186 / 40 / 373 | 110.7 / 110.7 |
| Qwen2.5-VL-7B | **382** | 295 | **+87 pts** | +9.5 pp | [+5.9, +13.2] | 5e-05 | 5e-05 | 912 | 88 = 88 (identical IDs) | 190 / 103 / 192 | 134.7 / 134.7 |

- Skip symmetry: both arms skip the **exact same** hard long-OCR images (79 on Qwen3-VL, 88 on Qwen2.5-VL), overrun/OOM on HF eager attention; scores are over the common answered set and are conservative + symmetric. All other attempted samples answered (attempted = 1000 = 1000 both arms).
- Sample-ID equality: ✓. Per-sample token-count equality: ✓.
- McNemar (binary): Qwen3-VL 186 vs 40 (p≈0); Qwen2.5-VL 190 vs 103 (p≈0).

### Interpretation (status: `verified` numbers + `interpretation`)
- **RBM beats FastV-k3 on OCRBench under the identical HF harness at full n**, for both models (+146 / +87 pts, both Holm-significant). Direction matches the paper Table 2 OCRBench rows (n=200 native: RBM > FastV on both models); absolute values differ because P1 uses the 4 M-px campaign cap and the full 1000 sample set (Table 2 used native res on 200). The regime-map conclusion (RBM wins the OCR regime) is **strengthened** at full scale.
- These P1 cells are NOT a direct replacement for Table 2's OCRBench rows (different pixel cap / scope), but are a strictly stronger same-harness full-1000 confirmation.

---

## Config / environment / log paths (all cells traceable)

- **Environment:** conda `qwen3vl_clean`; python 3.10.20, torch 2.10.0+cu128, transformers 4.57.6, vLLM 0.19.0, PIL 12.2.0, numpy 2.2.6, scipy 1.15.3; 1× A40 46 GB; `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_MODELSCOPE=False`.
- **Models (HF hub cache, verified complete):** `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct` (17 GB, sha `0c351dd…`), `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` (16 GB, sha `cc594898…`). Note: the path `/media/disk2/models/` named in the task does not exist; the working copies are the HF hub cache above.
- **Result dir:** `results/acmmm_final_controls/{p0_1,p0_2,p1}/` (JSON + `.log` per cell; campaign logs `campaign.log`).
- **Per-cell config is stored inside each JSON** (`max_pixels`, `max_model_len`, `max_num_seqs`, `max_tokens`, `selector`, `vllm`, `wall_s`) and in `results/acmmm_final_controls/analysis.json` (`meta_*` fields).
- **Scoring:** `src/v3_premerger/official_scorers.py` (VQA-acc, ANLS, exact-match, OCRBench containment with HME branch via `question_type`), applied to per-sample predictions in `scripts/analyze_acmmm_final_controls.py`. No log numbers were hand-picked.
- **Analysis outputs:** `results/acmmm_final_controls/analysis.json` (machine-readable), `results/acmmm_final_controls/analysis/pXX_*_per_item.json` (per-item predictions), this report.
- **Reproducibility audit:** Table 1 Qwen3 cells rescored from the campaign post cells match the paper exactly; Table 1 audit block in `analysis.json` (`P0_1_table1_audit`) reproduces all 4 benches to the printed value.

## Known limitations
1. GQA claim-level event above requires the author's decision (paper text untouched per instructions).
2. P1 skips (79/88) are the known hard long-OCR images under HF eager attention; symmetric across arms, but the /1000 totals understate both arms equally.
3. P0-2 κ=0.125 and P1 use the campaign's 4 M-px cap, which differs from the old native-res cells; cross-config comparisons should be drawn with the cap noted.
4. none anchors for P0-1 are reused from earlier verified campaigns (not re-run this session); their official rescore reproduces the paper.

---

*End of report. Generated 2026-08-19 by the acmmm_final_controls campaign (`scripts/run_p0_1_full_split.sh`, `scripts/run_p0_2_ocrbench_matched.sh`, `scripts/run_p1_fastv_hf_ocrbench.sh`, `scripts/analyze_acmmm_final_controls.py`).*
