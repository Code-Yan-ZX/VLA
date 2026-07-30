# RBM same-scope paired cells (r2c) — digest (Table 3 backing, n=200, official)

## Purpose
Fair FastV-k3-vs-RBM pairing at IDENTICAL n/scope (reviewer defense: no
mixed-n baseline comparisons). HF harness, `--mode pre --r-pre 0.25` (KEEP 25%
= r_eff 0.75, matched to FastV-k3's 25% retention), `--mrope native`,
docvqa 600k cap, seed 0, official rescore. Run 2026-07-30.

## Official results + same-scope Δ (pre − FastV-k3)
| model × bench (n) | RBM pre | FastV-k3 (r2b) | Δ | skip pre/FV |
|---|---|---|---|---|
| Qwen3-VL × OCRBench (200) | **0.575** | 0.415 | **+16.0pp RBM** | 19/19 |
| Qwen3-VL × TextVQA (full 5000) | 0.605 | **0.7771** | −17.2pp FastV | 0/5 |
| Qwen3-VL × GQA (full 12578) | 0.449 | **0.5376** | −8.9pp FastV | 0/0 |
| Qwen2.5-VL × TextVQA (200) | 0.6683 | **0.7467** | −7.8pp FastV | 0/0 |
| Qwen2.5-VL × GQA (200) | **0.520** | 0.475 | +4.5pp RBM | 0/0 |
| Qwen2.5-VL × DocVQA-600k (200) | **0.5062** | 0.4852 | +2.1pp RBM | 0/0 |
| Qwen2.5-VL × OCRBench (200) | **0.370** | 0.285 | +8.5pp RBM | 20/20 |
(Q3 TextVQA/GQA pairs use main-table full-split RBM values — same full scope.)

## Pairing integrity
Skip counts match FastV-k3 EXACTLY in every cell (19/19, 20/20, 0/0…) —
same hard samples skipped, mode-independent → fair paired comparison.
OCRBench skips = native-resolution giant-image OOM (`--max-pixels 0`),
identical across both methods; reported over attempted, disclosed.

## Interventions logged (DECISIONS.md)
1. Budget convention intercept: `--r-pre` = KEEP fraction (0.75→0.25 fix;
   probe-verified total_keep=0.25, ptid 208≈ref 213).
2. Harness bug: Qwen2.5-VL × pre × default `--mrope vllm-mimic` degenerate
   (repeated "addCriterion…" output → pseudo-zero scores); fixed `--mrope
   native` (family-agnostic correct positions); uniform native for all 5
   cells. Q3 native vs mimic: 0.575 vs 0.525 (same order). NO paper number
   anywhere comes from the degenerate path.

## Claim impact (paper Table 3)
No universal winner: FastV-k3 (query-conditioned) wins Qwen3-VL text-dense
three (+17.2/+8.9/+16.2pp); RBM wins OCRBench on BOTH families (+16.0/+8.5)
and edges Qwen2.5 n200 GQA/DocVQA (+4.5/+2.1). FastV's OCR collapse is
structural (best K doesn't rescue it). Framing: RBM = robust default, not
SOTA (red-line consistent).

## Assets
runs/r2_same_scope/r2c_rbm_scope.sh · r2c_*.json (5) · r2c_rbm_scope.out
