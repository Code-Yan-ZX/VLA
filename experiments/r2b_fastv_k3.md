# FastV k=3 same-scope supplement — digest (both Qwen models, official rescore)

## Purpose
Baseline fairness: the main-table FastV baseline used layer K=2; K-sensitivity
probes found k=3 best (0.751 vs k2 0.646 on earlier dev slice). Report FastV at
its BEST K (k=3) at the same scope so the RBM-vs-FastV comparison cannot be
attacked as "wrong-K handicap". HF harness (baselines_hf.py), eager attn,
official rescore. Run 2026-07-30 (after fixing a stale `--benchmark x`
placeholder in the batch script — derived from subset filename; DECISIONS
2026-07-30; src/ untouched).

## Official results (FastV k=3)
| model | TextVQA VQA-acc | GQA acc | DocVQA ANLS | OCRBench acc |
|---|---|---|---|---|
| Qwen3-VL-8B | 0.7771 (full 5000) | 0.5376 (full 12578) | 0.5863 (n200) | 0.415 (n200) |
| Qwen2.5-VL-7B | 0.7467 (n200) | 0.475 (n200) | 0.4852 (n200) | 0.285 (n200) |

## vs RBM (pre@25%) — the fairness comparison (refs: j7_main_table.md, cascade gate n200)
- TextVQA (full 5000): FastV-k3 0.7771 > RBM 0.605 — FastV wins by **+17.2pp**
  (query-conditioned attention's text-VQA edge persists, even wider at best K).
- GQA (full 12578): FastV-k3 0.5376 > RBM 0.449 — FastV wins by +8.9pp (consistent).
- DocVQA (n200, 600k cap): FastV-k3 0.5863 > RBM 0.4239 — FastV wins +16.2pp (consistent).
- OCRBench (n200): FastV-k3 0.415 < RBM 0.5801 — **RBM still wins OCR by ~16.5pp**
  even at FastV's best K (k3 does not rescue OCR; query-conditioned intra-LLM
  pruning still discards text patches). Core claim robust to K choice.

## Completeness caveats
- OCRBench n200 cells skip ~10% (Qwen3 19/200, Qwen2.5 20/200 — empty FastV
  outputs on some OCR prompts); reported acc over attempted, disclose in paper.
- Qwen2.5 cells are n200 (dev-scope); Qwen3 TextVQA/GQA full-split, DocVQA/OCR n200.
- k=3 ≈ 214 mean visual tokens (ptid 214) — comparable budget to Qwen3-VL
  RBM@25% (pre mean_ptid ≈213 on textvqa, gate cells); note InternVL3 pre@25%
  ptid is 690 (different merger/tokenization — never mix the two in one table).
  The comparison is "each method at its strong config", state budgets in the table.

## Claim impact
No change: "FastV (query-conditioned) wins TextVQA/GQA/DocVQA; RBM wins OCRBench
and never collapses" holds at FastV's best K on both Qwen families (margins
+17.2 / +8.9 / +16.2pp for FastV; −16.5pp OCR for RBM). Paper reports FastV-k3
as the baseline (strongest fair configuration).

## Assets
runs/r2_same_scope/r2b_*.json (8 cells) · runs/r2b_fastv_k3.out ·
fixed script runs/r2_same_scope/r2b_fastv_k3.sh (runs/ gitignored).
