# P0-3: Qwen3-VL pre-final pure-stage confound control

**Date:** 2026-08-06 · **Verdict:** text-dense stage effect CONFIRMED (pure-stage control valid); GQA "reversal" was a feature-space artifact (flagged, not self-packaged).

## What was run

New `--mode pre-final` (Qwen3-VL only): ranks L2 at `visual.merger`'s OWN input (final ViT-block features = main-merger INPUT), selects top-k merge-units from the visual output; deepstack mergers run untouched. By 2x2 merge-unit equivalence this is bit-identical to slicing the merger input. Compared to `post` (ranks at main-merger OUTPUT), the only difference is BEFORE vs AFTER the lossy 2x2 main merger -> isolates the pure STAGE effect from the original `pre` confound (which ranks at deepstack[0]-input, layer-8, far upstream, and slices all 4 mergers).

n=200, Qwen3-VL-8B, greedy (temp=0, max_tokens=32 = iso-config with existing Table-1 results), keep 25% (r=0.75), L2 selector, same subsets. Modes: none / pre-final / post. Commit `3df3df2` (runner + script). Outputs: `runs/qwen3_prefinal_control/`.

## Iso-token check (control validity) - PASS

pre-final and post prompt_token_ids match for every sample: TextVQA 200/200, DocVQA 200/200, GQA 200/200. The control is clean iso-token (both keep k_i=round(full_i*0.25) units).

## Result

| benchmark | none | pre-final | post | Δ(pre-final−post) | paired mean±SE | verdict |
|-----------|------|-----------|------|-------------------|----------------|---------|
| TextVQA (VQA-acc) | 0.858 | 0.515 | 0.215 | **+30.0pp** | 0.300±0.039 | STAGE EFFECT CONFIRMED |
| DocVQA (ANLS)     | 0.976 | 0.326 | 0.200 | **+12.5pp** | 0.125±0.032 | STAGE EFFECT CONFIRMED |
| GQA (exact-match) | 0.605 | 0.450 | 0.450 | **0.0pp**   | 0.000±0.032 | direction vanished |

## Interpretation

**Text-dense (main claim) - CONFIRMED and robust.** Even when ranking at the main-merger INPUT (pre-final, holding the feature space roughly constant with post) instead of deepstack[0]-input (original pre), pre-final still beats post (main-merger OUTPUT) by +30pp (TextVQA) / +12.5pp (DocVQA). The pre≫post text-dense claim is a TRUE stage effect (ranking before the lossy merger preserves text), NOT a deepstack[0]-input vs main-merger-output feature-space artifact. The pure-stage control is valid -> write into the paper (per user P0-3 rule branch 1).

**GQA - the "post>pre reversal" was a confound.** Original pre-vs-post on Qwen3 GQA = -2.8pp (post>pre, P0-2 McNemar z=-8.1). But pre-final (main-merger-input ranking) vs post = 0.0pp (tie). So the GQA "reversal" was driven entirely by the deepstack[0]-input (layer-8) feature space being bad for GQA ranking, NOT by a true stage reversal. With main-merger-input ranking, GQA shows NO stage effect (pre-final == post).

This REFINES (does not overturn) the workload-conditional story:
- The stage effect is text-dense-SPECIFIC and ROBUST across feature spaces (pre-final confirms pre>post).
- GQA has NO true stage reversal; the original post>pre was an artifact of layer-8 ranking. On GQA, ranking before vs after the lossy merger makes no difference (the merger doesn't hurt object features).

## Decision (per user P0-3 rule)

- TextVQA/DocVQA still significantly pre>post -> pure-stage control valid -> **write into paper** (branch 1).
- GQA direction vanished -> **CLAIM-LEVEL flag: report, do NOT self-package** (branch 2). The GQA "reversal" claim in the paper needs reframing (drop "post>pre reversal on GQA"; reframe as "no stage effect on GQA; the apparent reversal was a layer-8 ranking artifact"). Awaiting user decision on the exact GQA framing.

## Scope / caveats

- pre-final tested on Qwen3-VL only (the confound is Qwen3/deepstack-specific). Qwen2.5-VL has the same deepstack architecture (its GQA -2.6pp reversal is likely the same artifact) but was not re-tested with pre-final.
- pre-final is not a perfectly clean control (feature space still differs slightly: main-merger-input ctx-dim vs main-merger-output hidden-dim; post ranks the full [main+deepstack] concat while pre-final ranks main-merger input). But it is far cleaner than pre-vs-post, and the text-dense signal is large (+30/+12.5pp) so the conclusion is robust.
- The mechanism (lossy merger destroys text) is consistent: text-dense needs pre-merger ranking; object-QA doesn't.

## Artifacts
- Per-sample JSONs: `runs/qwen3_prefinal_control/qwen3_{none,pre-final,post}_{textvqa,docvqa,gqa}_r0.750_n200.json`
- Summary: `runs/qwen3_prefinal_control/prefinal_summary.json`
- Runner: `--mode pre-final` (commit `3df3df2`); script `src/v3_premerger/qwen3_prefinal_control.sh`.
