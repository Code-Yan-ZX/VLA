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

### 2026-09-01 — smoke test ✅
- **build_e1_splits.py**: held-out_64 built (4 benches × 64); disjoint audit PASS — ho∩gate_200 = ho∩dev_64 = dev_64∩gate_200 = ∅ (runs/e1/splits_audit.json).
- **e1_causal_import.py** (`py_compile` clean; CPU `--self-test` OK: block partition coverage/balance, occlusion slicing identity — survivor rows bit-identical, keep-all ≡ identity).
- **GPU smoke (8 cells × n=4 × G=8, --null-check) ALL GREEN**: model load 6.3s (HF cache /data/models/huggingface); null-check **|ΔLL|=0.00e+00 bit-exact on every sample**; ΔLL sane (docvqa "Paul" answer-bearing top-left block u=+26.7; GQA yes/no ≈ 0, "women" u=+1.5; docvqa pixel cap → ~570 units).
- **s/pass measured**: 0.06–0.21s typical (gqa 0.06–0.13; textvqa/docvqa 0.11–0.21); OCRBench giant docs (≤2754 units) up to 1.24s.
- **Budget reconfirmed**: 8 cells × 64 × (1+K+G) ≈ 4,608–8,704 passes × ~0.15s ≈ **25–50 min (0.4–0.9 A40·h) ≪ 6 GPU·h** → autonomous run authorized (per user rule 6).
- **Plan corrections (schema unchanged, no lock violation)**: pass count is 1+K+G (K = accepted-answer count, needed to pick a* on the full pass; GQA K=1, TextVQA K≈2–10); a* selected on the prefill path only (decode/prefill bf16 disagreement ~0.3–0.5 nats on long OCRBench answers would risk near-tied a* flips); fastv-k=3 (matches gateC FastV cells).

### 2026-09-01 — gate run ✅ (all cells, then metrics)
- **8/8 cells green**; total ≈ 0.7 A40·h (measured rate ~2 s/sample textvqa; budget reconfirmed <6 ✓).
- Coverage: textvqa/docvqa/gqa **64/64** dev + heldout; **OCRBench 54/64 dev, 60/64 heldout** — 14 native-res giant-doc samples OOM on 1×A40 (documented hardware skip; matches the paper's own ~10% OCRBench native-res skip handling). No other failures; no partial/corrupt data.
- `e1_metrics.py` → runs/e1/metrics.json (held-out only; predictors fit on dev only; resamples=4000).

## Results (held-out; nDCG@25%, Recall@25%, k=2 of G=8)
| bench | n | nDCG A | nDCG B | pre-L2 | post-L2 | FastV | R@25 B | RBM | FastV |
|---|---|---|---|---|---|---|---|---|---|
| textvqa | 64 | 0.249 | 0.249 | 0.203 | 0.083 | 0.229 | 0.297 | 0.328 | 0.445 |
| docvqa | 64 | 0.126 | 0.126 | 0.032 | −0.008 | −0.138 | 0.266 | 0.266 | 0.227 |
| gqa | 64 | 0.079 | 0.079 | 0.099 | −0.017 | 0.153 | 0.344 | 0.367 | 0.367 |
| ocrbench | 60 | 0.174 | 0.175 | −0.056 | −0.265 | −0.442 | 0.250 | 0.225 | 0.300 |

Per-sample signal-vs-causal Spearman (mean): pre-L2 +0.086/+0.074/+0.172/−0.038; post-L2 ≈ −0.01..−0.06; FastV +0.100/−0.137/+0.097/−0.027; qsim ≈ 0 (−0.014..+0.007) across benches.

## Verdict — **NO-GO** (pre-registered; no scorer training)
| # | Metric | Result | Verdict |
|---|---|---|---|
| G1 (KEY) | held-out ΔnDCG@25% (B−A), n=252 pooled | **+0.0003**, CI[−0.0000, +0.0008], p=0.74, pos 2/4 benches | **NO-GO** |
| G2 | R@25% B vs RBM (textvqa/docvqa/ocrbench) | −0.031 / 0.000 / +0.025 → B<RBM on 1/3, need ≥2/3 | NO-GO |
| G3 | GQA nDCG@25% B−RBM | −0.0207, CI[−0.106,+0.068] | NO-GO |
| G4 | per-sample win predictor (FastV vs RBM) | textvqa acc 0.74 (CI 0.60–0.89); docvqa 0.64, gqa 0.59, ocrbench 0.67 — only textvqa CI excludes 0.50 | NO-GO (1/4 benches clean) |
| GLOBAL | — | **G1 failed → NO-GO, do NOT enter E2 scorer training** | **NO-GO** |

**Bounded observations (not claims):**
- Adding the query feature (qsim) to the pre-merger predictor changes held-out causal-utility nDCG by **+0.000 to +0.001** — the query signal is predictively inert at the pre-merger stage. This closes the last untested cell of the design space (learned + causal-ΔLL-supervised + query-conditioned pre-merger ranking): pre-merger features are query-blind for *causal* (answer-dependent) unit utility, exactly as the paper's mechanism and the J5/D1 negatives predicted.
- No signal strongly predicts causal unit utility at the block level (best pre-L2 nDCG 0.203 textvqa / Spearman +0.172 GQA). post-L2 is worst (≤0.083, negative elsewhere) — consistent with M1/M2 (merger reshuffles saliency anti-text).
- FastV's layer-K attention is positively aligned with causal utility on TextVQA/GQA (0.229/0.153 nDCG) but **negative on DocVQA/OCRBench** (−0.138/−0.442) — the query-conditioned signal only helps where FastV wins empirically (scene-text) and misranks dense OCR, matching the paper's regime map.
- G4 textvqa (acc 0.74, CI excl. 0.50): on TextVQA, per-sample query-image alignment (Spearman(qsim, pre-L2)) predicts which method recovers more causal units — consistent with FastV's TextVQA edge being genuinely query-dependent at the sample level. Not a general result (1/4 benches).

## Outcome
Q-RBM (a query-conditioned pre-merger ranker supervised by causal utility) is **falsified at the gate** → NO-GO, no scorer training (E2 not entered). RBM submission stands as fallback; return to submission hardening (incl. the recorded GQA claim fix, notes/qrbm_e1_plan.md §8).

## Assets
- code: scripts/build_e1_splits.py, src/v3_premerger/e1_causal_import.py, scripts/e1_metrics.py, scripts/run_e1.sh
- data: runs/e1/ (gitignored): dev_*/heldout_*.jsonl, metrics.json, splits_audit.json, logs
