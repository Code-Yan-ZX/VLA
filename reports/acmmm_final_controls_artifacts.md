# ACM MM Final Controls — Artifacts & Reproducibility

**Companion to `reports/acmmm_final_controls.md`.** Captures what is committed to Git, what is not (and why), where each aggregate number comes from, and how to re-run/verify everything.

## 1. What is committed to Git (branch `exp/acmmm-final-controls`, merged into `main`)

| Path | Purpose |
|---|---|
| `reports/acmmm_final_controls.md` | Human verification report (protocol, results, claim-level GQA event) |
| `reports/acmmm_final_controls_artifacts.md` | This file |
| `results/acmmm_final_controls/analysis.json` | Machine-readable aggregate results (all official rescoring + paired stats + iso checks) |
| `results/acmmm_final_controls/cell_summary.csv` | One row per experiment cell: model/bench/mode/retention/pixel_cap/attempted/completed/skipped/official score/mean tokens/engine/source json/sha256 |
| `results/acmmm_final_controls/environment.txt` | Python/torch/transformers/vLLM/GPU/model revisions/git commit/env vars/scorer path |
| `results/acmmm_final_controls/MANIFEST.sha256` | sha256 + size of every source file (committed **and** non-committed raw per-sample JSONs) |
| `artifacts/acmmm_final_controls/**/*.json.gz` | gzip-compressed raw per-sample cell JSONs + per-item JSONs + P1 smokes (28 files, ~4.3 MiB) |
| `scripts/run_p0_1_full_split.sh` | P0-1 runner (Qwen3-VL-8B pre-final/post, full splits, native, r=0.75) |
| `scripts/run_p0_2_ocrbench_matched.sh` | P0-2 runner (Qwen2.5-VL OCRBench, both arms 4 M-px, r=0.75/0.875) |
| `scripts/run_p1_fastv_hf_ocrbench.sh` | P1 runner (RBM vs FastV-k3, same HF eager harness, `--mrope native`) |
| `scripts/run_acmmm_chain.sh` | Serial campaign chain (P0-1 → P0-2 → P1 → analysis) |
| `scripts/analyze_acmmm_final_controls.py` | Official rescore + paired bootstrap CI + sign-flip permutation + Holm + McNemar + iso-token/sample-id checks |
| `scripts/gen_acmmm_manifest.py` | Regenerates `cell_summary.csv` / `environment.txt` / `MANIFEST.sha256` |
| `STATE.md` | Project state update (claims the same GQA claim-level event) |

## 2. What is NOT committed to Git (and why)

* **Raw per-sample cell JSONs and per-item JSONs** — committed only in gzip form under `artifacts/acmmm_final_controls/` (~36 MiB raw → ~4.3 MiB gz). The uncompressed copies live on the server at `results/acmmm_final_controls/{p0_1,p0_2,p1}/` and `results/acmmm_final_controls/analysis/`.
* **`results/` and `runs/` directories** are `.gitignore`d by project convention (experiment data / logs are not versioned). Only the exact files in §1 were added with `git add -f`.
* **Cell `.log` files** (vLLM/HF engine logs) — not committed; they sit next to each JSON on the server (`results/acmmm_final_controls/{p0_1,p0_2,p1}/*.log`) and their paths are recorded in the main report.
* **The 4 P0-1 `none` anchor cells** (reused verified cells from `runs/full_matrix/` and `runs/r2_same_scope/`) — not re-run this session; their sha256/size are in `MANIFEST.sha256`; their official rescore reproduces the paper exactly (0.8443 / 0.9562 / 760 / 0.6165).
* **Model weights, dataset images, caches** — never committed.

## 3. Which aggregate number comes from which source JSON

Every aggregate in `analysis.json` / `cell_summary.csv` / the report is recomputed from the per-sample JSONs with `scripts/analyze_acmmm_final_controls.py` (official scorers; OCRBench uses `question_type` for the HME branch). Mapping:

| Number | Source JSON(s) |
|---|---|
| P0-1 pre-final/post official scores & paired stats | `results/acmmm_final_controls/p0_1/p0_1_qwen3_{pre-final,post}_{bench}_r0.750_full.json` |
| P0-1 none anchors | `runs/full_matrix/j7_qwen3vl_none_*.json`, `runs/r2_same_scope/r2_qwen3vl_none_docvqa_full5349.json` |
| P0-2 κ=0.25 | `results/acmmm_final_controls/p0_2/p0_2_qwen2_{pre,post}_ocrbench_r0.750_full.json` |
| P0-2 κ=0.125 | `results/acmmm_final_controls/p0_2/p0_2_qwen2_{pre,post}_ocrbench_r0.875_full.json` |
| P1 RBM / FastV-k3 | `results/acmmm_final_controls/p1/p1_{qwen3vl,qwen2vl}_{pre,fastv}_ocrbench_*_full.json` |
| per-item predictions | `results/acmmm_final_controls/analysis/pXX_*_per_item.json` (also gz in artifacts) |

Paired statistics (bootstrap CI, sign-flip p, Holm, McNemar, iso checks) live in `analysis.json` → `P0_1_pure_stage_control[bench]`, `P0_2_ocrbench_matched[k25/k125]`, `P1_rbm_vs_fastv_hf[fam]`.

## 4. How to re-run

```bash
# regenerate the aggregate analysis + report skeleton (CPU only)
python scripts/analyze_acmmm_final_controls.py

# regenerate cell_summary.csv / environment.txt / MANIFEST.sha256 (CPU only)
python scripts/gen_acmmm_manifest.py

# full re-run of every cell (GPU, single A40; ~4 h total, serial)
bash scripts/run_acmmm_chain.sh
```

`analyze_acmmm_final_controls.py` rescored from the committed (gz) raw JSONs must reproduce the six key numbers (see §6).

## 5. How to verify SHA256

`MANIFEST.sha256` lines are `<sha256> <bytes> <repo-relative-path>`. Verify against the committed artifacts:

```bash
sha256sum -c <(awk '!/^#/{print $3, $1}' results/acmmm_final_controls/MANIFEST.sha256) 2>/dev/null
# for the gz files, gunzip first then compare:
gunzip -c artifacts/acmmm_final_controls/p0_1/p0_1_qwen3_pre-final_textvqa_r0.750_full.json.gz | sha256sum
```

All 28 committed `.gz` files were sha256-verified against their uncompressed sources at creation (28/28 match).

## 6. Numbers that can directly enter Table 1 / Table 2 (verified)

* **Table 1 — Qwen2.5-VL OCRBench primary cell (κ=0.25): YES, replace with the matched-config pair** `RBM 480 vs post 182 (+298 pts)` (iso-token 229.6 = 229.6, iso-sample, p = 5e-05). Old mismatched pair (476/183, +293) is superseded; +298 > +293 confirms the old "conservative" footnote.
* **Table 1 — Qwen3-VL cells: unchanged.** The P0-1 `post` cells are bit-identical to the published cells (0.2217 / 0.2377 / 184 / 0.4771); the headline `pre` cells are not re-run this session.
* **Table 1 — Qwen2.5-VL OCRBench @12.5% (κ=0.125): optional.** New matched pair `328 vs 70 (+258)` at the campaign's 4 M-px cap; the old @12.5% cells (335/67) were already iso-config at native, so both are defensible — pick one config and state it.
* **Table 2 — OCRBench rows: not a direct replacement** (different scope/cap), but P1 is a strictly stronger same-harness full-1000 confirmation: Qwen3 `RBM 559 vs FastV-k3 413 (+146)`, Qwen2.5 `382 vs 295 (+87)`, symmetric skips, `--mrope native`.

## 7. ⚠️ Claim-level event — GQA must NOT use the old 0.0 pp framing

The full-split pure-stage control gives **pre-final 0.4207 vs post 0.4771, Δ = −5.64 pp, 95% CI [−6.42, −4.86], p = 5e-05, McNemar 941 vs 1650, n = 12578, iso-token verified** — a **significant negative** stage effect on object-QA. The paper's "pre-final gives 0.0 pp on GQA" (n = 200) is a sampling artifact and must be revised before submission. Do not overwrite this with the old n=200 result.
