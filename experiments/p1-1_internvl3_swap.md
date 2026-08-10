# P1-1: InternVL3 M3 ranking-swap + kept-set identity (mechanism generalization)

**Date:** 2026-08-07 · **Verdict:** MECHANISM GENERALIZES to InternVL3. The M3 ranking-swap identity (swap≡pre, Jaccard(pre,swap)=1) holds on the third family.

## What was run

InternVL3-8B (pixel-shuffle + mlp1 merger, a different merger design from Qwen's PatchMerger). 3 arms: pre (rank at pixel-shuffle units, pre-mlp1), post (rank at merged tokens, post-mlp1), swap (POST forward path + PRE ranking - the M3 cross). n=200, greedy, keep 25% (r=0.75), L2, max_tokens=32. `--mask-ranking swap --mode post` (the swap arm). `src/v3_premerger/internvl3_swap_control.sh` (commit 603dd7f + swap-fix dd5ba62). Iso-token (ptid equal across arms). Jaccard(pre,swap) + answer-identity recorded via attach_kept_indices + --save-unit-scores.

## Result

| benchmark | pre | post | swap | Δ(pre−post) | swap−pre | Jaccard(pre,swap) | ans-id(pre==swap) | verdict |
|-----------|------|------|------|-------------|----------|-------------------|-------------------|---------|
| TextVQA (VQA-acc) | 0.7383 | 0.3450 | 0.7383 | +39.3pp | +0.00pp | 1.0 (n=197) | 1.00 | GENERALIZES |
| DocVQA (ANLS)     | 0.7442 | 0.4256 | 0.7453 | +31.9pp | +0.11pp | 1.0 (n=163) | 0.98 | GENERALIZES |
| GQA (exact-match) | 0.595  | 0.585  | 0.590  | +1.0pp  | −0.50pp | 1.0 (n=192) | 0.995 | GENERALIZES |

swap_diag: consumed=193-198, fallback=0, leftover=0 (clean swap, no fallback).

## Interpretation

- **Stage law holds on InternVL3**: pre≫post on text-dense (TextVQA +39.3pp, DocVQA +31.9pp), ~no effect on GQA (+1.0pp). Consistent with Qwen3-VL (+38.4/+24.3pp) and the workload-conditional law.
- **M3 mechanism generalizes**: swap (POST path + PRE ranking) recovers pre's accuracy exactly (swap−pre ≤ 0.11pp on text-dense), with Jaccard(pre,swap)=1.0 (kept-set identical) and answer-identity 0.98-1.0. So the result is determined by WHICH units are kept (the ranking), not the forward path - confirmed on a 2nd merger architecture (pixel-shuffle, not just Qwen's PatchMerger).
- This makes the M3 mechanism claim architecture-general (Qwen3-VL + Qwen2.5-VL + InternVL3), not Qwen-specific.

## Decision (per user P1-1 rule)

- swap≈pre (≤0.11pp on text-dense) AND Jaccard=1.0 AND answer-identity high (0.98-1.0) on ALL benchmarks -> **mechanism GENERALIZES to InternVL3**. Architecture-general mechanism claim OK -> write into paper.
- No flag. The mechanism (M3 ranking-swap identity) is now demonstrated on 3 families.

## Artifacts
- Per-sample JSONs: `runs/internvl3_swap_control/internvl3_{pre,post,swap}_{textvqa,docvqa,gqa}_r0.750_n200.json`
- Summary: `runs/internvl3_swap_control/internvl3_swap_summary.json`
- Verdict in `runs/gpu_chain/chain.log` (P1-1 DECISION section).
