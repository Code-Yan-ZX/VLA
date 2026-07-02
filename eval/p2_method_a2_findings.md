# P2 Method A'' (CLIP contrastive boundary selector) — NEGATIVE (2026-07-02)

> Goal: fix v2's OCR failure (LLM embed_tokens space not contrastively aligned) by
> relocating SparseVLM's text↔patch signal to CLIP's CONTRASTIVE space (CLIP
> trained so text-embeds align with ViT-patch-features). **Outcome: A'' is
> CATASTROPHICALLY worse than v2** — the thesis's per-patch CLIP projection does
> NOT localize text. Third boundary selector to fail on OCR.

## Implementation (correct, working)
- **CLIP text tower** loaded from `openai/clip-vit-large-patch14-336` (the SAME
  checkpoint LLaVA-1.5's vision tower is built from). 12 layers, hidden=768,
  ~0.5GB. `visual_proj (768,1024)` matches the vision tower's hidden=1024 →
  confirmed aligned with the patches' training-time contrastive head.
- **Patch features**: vision-tower `last_hidden_state` patch tokens (pre-projector,
  `(B,576,1024)`) captured via a forward-hook, projected to CLIP's 768-d
  contrastive space via `visual_projection.weight.T`.
- **Question features**: `CLIPTextModel(question) @ text_projection.weight.T` →
  `(B,T,768)` contrastive.
- **Score**: `cos(clip_text_feat, clip_patch_feat)` per patch, max/mean over T.
- **Compaction**: CLIP-score top-k indices applied to PROJECTOR OUTPUT
  (B,576,4096) → v1-of-A'' reuses the v2 contiguous-compaction + placeholder-shrink
  plumbing. Selector wired as `--selector clip_query`.
- **Memory**: score matrix is `(T×576)` only; no `(B,H,S,S)`. No OOM.

## Mechanism check (GQA, --limit 3, r0.5): PASSED
- CLIP text encoder loads; visual_proj dims verified against vision tower.
- Scores computed: 576 patches, 12 question tokens, score range [0.016, 0.102, 0.214].
- kept=288, forward OK, no OOM. r=0 control keeps 576 (byte-identical no-op). ✔

## ★ OCR directional test (TextVQA, --limit 50, r0.5)
| selector | acc (n=50) |
|---|---|
| proxy (matched control) | **0.500** |
| v1 (true CLS-attn) | 0.445 |
| v2 query_aware (cosine, max) | 0.380 |
| **A'' clip_query (max)** | **0.180** ❌ |
| **A'' clip_query (mean)** | **0.140** ❌ |
| FastV (intra-LLM, n=200 ref) | 0.555 |

**A'' is WORSE than v2, not better** — the opposite of the thesis. Mean < max
(same pattern as v2: stopwords dilute the signal).

## Root cause (verified empirically, not just hypothesized)
**CLIP's contrastive loss aligns ONLY the pooled [CLS] token, NOT per-patch
features.** Per-patch visual_projection lands features in the 768-d space but
they are NOT contrastively trained against text there. Verified on:
1. **Synthetic text image** ("STOP 123" rendered): per-patch projection + CLIP-text
   gives **0/10 overlap** with the text patch region (top-10 patches scattered).
2. **Same image, alternative CLIP-based localizers** — ALL fail:
   - Last-layer CLS→patch attention: 0/10 overlap.
   - Attention rollout (CLS→patch): 0/10.
   - MaskCLIP (out_proj(V) at last layer): 0/10.
   Only CLIP's intended pooled-CLS image-text similarity works (cos=0.20), but
   that gives ONE global score per image, not per-patch — useless for selection.

This is the known CLIP-segmentation literature result: naive CLIP features are
NOT spatially text-discriminative; zero-shot segmentation needs MaskCLIP/CLIP-
Surgery tricks, and even those are coarse. The A'' thesis assumed per-patch
contrastive alignment that CLIP does not provide.

## Pattern (3 boundary selectors failed on OCR)
- v1 (boundary CLS-attn): TextVQA r50 0.445.
- v2 (boundary LLM-embed cosine): TextVQA r50 ~0.38.
- **A'' (boundary CLIP contrastive): TextVQA r50 ~0.18.**
- proxy (hidden-state deviation): TextVQA r50 0.530 — still the best boundary.
- FastV (intra-LLM): TextVQA r50 0.555 — best overall, NOT vLLM-integrable.

**Three distinct training-free boundary signals (vision-saliency, LLM-cosine,
CLIP-contrastive) all underperform the proxy and FastV on OCR.** The boundary-
training-free-OCR problem is genuinely hard — consistent with why the literature
has 5+ OCR-specific methods using learned components.

## Implication for the strategic fork
- The serving-throughput contribution (0/37 papers measure it) + the 3 serving-
  specific findings remain the strong, novel, validated core. Provisional GO
  unchanged (compression→served-speedup is selector-independent, established by
  the proxy probe).
- **Recommend: stop chasing the boundary selector.** Build the serving-aware
  method (early-prune + KV-cache/load-adaptive budget) on the **proxy** selector
  (best boundary accuracy, real serving speedup), with served-throughput as the
  core novelty. A'' confirmed there is no cheap training-free text-region
  localizer at the boundary.
- If a learned component is acceptable later: a small trained projection head on
  CLIP/ViT features (FlashVLM-style) or SparseVLM's full attention aggregation
  (drifts toward intra-LLM) are the remaining open angles — both break training-free.

## Artifacts (gitignored)
- `runs/p2_a2/gqa_r50_mech.json` (n=3 mechanism check)
- `runs/p2_a2/textvqa_r50_max.json` (n=50, max pool) · `textvqa_r50_mean.json` (mean)
- `runs/p2_a2/textvqa_r0_control.json` (n=20, r=0 byte-identical no-op control)
