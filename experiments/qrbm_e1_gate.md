# Q-RBM E1 — Causal-Importance Gate (live log)

> Pre-registered plan: `notes/qrbm_e1_plan.md` (FINAL 2026-09-01). Branch `exp/qrbm-e1`.
> This file is the gate's live log: pre-registered criteria first, results appended as they land.

## Pre-registered acceptance criteria (locked, from plan §4)
| # | Metric | GO | NO-GO |
|---|---|---|---|
| G1 (KEY) | held-out ΔnDCG@25%: predictor [pre-L2,post-L2,qsim] − [pre-L2,post-L2], fit dev_64 → eval held-out_64 | mean Δ ≥ +0.05 AND CI excl. 0 AND ≥3/4 benches positive | mean Δ ≤ +0.02 OR CI ∋ 0 |
| G2 | Answer-bearing Recall@25% (TextVQA/DocVQA/OCRBench), predictor-B vs RBM | B ≥ RBM on ≥2/3 benches and ≥ RBM−0.05 on 3rd | B < RBM on ≥2/3 |
| G3 | GQA causal ranking nDCG@25%, predictor-B vs RBM | B − RBM ≥ +0.05 | ≤ +0.02 |
| G4 | Per-sample win predictor (FastV vs RBM, logistic on Spearman(qsim,pre-L2)) | acc > 0.60, lower CI > 0.50 | ≤ 0.55 or CI ∋ 0.50 |
| GLOBAL | — | G1 GO is the precondition; G2–G4 are additional | **G1 fail → NO-GO, no scorer training** |

Statistical protocol: paired 95% bootstrap CI clustered by image + Wilcoxon (`src/v3_premerger/paired_stats.py`).

## Protocol snapshot (from plan)
- Model Qwen3-VL-8B-Instruct frozen; G=8 raster blocks; drop-out occlusion; teacher-forced GT-LL, a* chosen on full pass.
- Splits: dev_64 = runs/merger_repr/dev_{b}_64.jsonl (disjoint from gate_200); held-out_64 = new (build_e1_splits.py).
- Pass count: 4,608 (G=8). GPU est 0.5–1.1 A40·h (<6 ✓).

## J5-differentiation statement (pre-registered)
E1 ≠ J5: learned predictor (dev-fit) + causal-ΔLL supervision + held-out evaluation disjoint from all
prior gate sets. Fixed λ=0.5 qsim blend is diagnostic only. A G1 failure = "query does not help
causal-utility prediction pre-merger", not "J5 repeated".

---

## Run log

### 2026-09-01 — smoke test (pending)
- [ ] build_e1_splits.py: held-out_64 built, disjoint check PASS (vs dev_64 ∩ gate_200 = ∅)
- [ ] e1_causal_import.py n=4 × G=8: pass count OK, s/pass measured, ΔLL sanity (answer-bearing block drop ⇒ LL drops), a* selection OK
- [ ] budget reconfirm after smoke: estimated total A40·h < 6

### 2026-09-01 — gate run (pending)
- [ ] dev_64 × 4 benches, occlusion + signals → runs/e1/dev_*.json
- [ ] held-out_64 × 4 benches → runs/e1/heldout_*.json
- [ ] e1_metrics.py → runs/e1/metrics.json

## Verdict
**PENDING**

## Assets
- code: scripts/build_e1_splits.py, src/v3_premerger/e1_causal_import.py, scripts/e1_metrics.py
- data: runs/e1/ (gitignored)
