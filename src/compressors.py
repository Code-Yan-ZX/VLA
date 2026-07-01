"""Visual-token compressors for the P2 go/no-go probe.

Boundary-level, training-free compressors that operate on vision-tower / projector
outputs *before* LLM fusion. Designed to hook into vLLM's
`LlavaMultiModalProjector.forward` (see notes/method-design.md §1b).

The probe compressor is `ClsAttnSelector` (CLS/attention-score selection, the
VisionZip / FasterVLM / VTC-CLS family). Kept minimal and dependency-light so it
can be CPU-tested on a dummy tensor without GPU/vLLM.

All selectors:
    input  : image_features  (B, N, D)   -- projector output (or vision-tower output)
             scores          (B, N)      -- per-token importance in [0,1] (e.g. CLS attn)
             pruning_rate    float in [0,1)  -- fraction of N tokens to DROP
    output : kept            (B, K, D)   with K = round(N * (1 - pruning_rate))
             keep_idx        (B, K) long -- indices kept (for placeholder reconciliation)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate(features: torch.Tensor, scores: torch.Tensor) -> None:
    if features.dim() != 3:
        raise ValueError(f"features must be (B,N,D), got {tuple(features.shape)}")
    if scores.dim() != 2:
        raise ValueError(f"scores must be (B,N), got {tuple(scores.shape)}")
    if features.shape[:2] != scores.shape:
        raise ValueError(
            f"features/scores batch/seq mismatch: {tuple(features.shape[:2])} "
            f"vs {tuple(scores.shape)}"
        )


def keep_count(n: int, pruning_rate: float) -> int:
    if not 0.0 <= pruning_rate < 1.0:
        raise ValueError(f"pruning_rate must be in [0,1), got {pruning_rate}")
    return max(1, round(n * (1.0 - pruning_rate)))


# --------------------------------------------------------------------------- #
# Probe compressor: CLS / attention-score selection
# --------------------------------------------------------------------------- #
@dataclass
class ClsAttnSelector:
    """Top-k token selection by an external importance score.

    Score source is decoupled from the selector so we can feed CLS-attention
    (captured by a forward hook on the vision tower) without coupling to the
    vision model here. Mirrors FasterVLM/VTC-CLS (last-layer [CLS]->patch attn,
    mean over heads).

    No learnable params, no extra forward pass (score is read from the encoder).
    """

    pruning_rate: float = 0.0

    def select(
        self,
        image_features: torch.Tensor,   # (B, N, D)
        scores: torch.Tensor,           # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate(image_features, scores)
        b, n, d = image_features.shape
        k = keep_count(n, self.pruning_rate)
        # top-k by score (descending). argsort is deterministic & GPU/CPU-portable;
        # for a tie-break we rely on sort stability (PyTorch stable sort by index).
        keep_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices  # (B,K)
        # gather rows
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)            # (B,K,D)
        kept = torch.gather(image_features, dim=1, index=gather_idx)     # (B,K,D)
        return kept, keep_idx

    # convenience: alias used by the vLLM hook wrapper
    def __call__(self, image_features: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        kept, _ = self.select(image_features, scores)
        return kept


# --------------------------------------------------------------------------- #
# Score utilities (CLS-attention extraction from a CLIP/SigLIP vision tower)
# --------------------------------------------------------------------------- #
def cls_attention_scores(
    attn_weights: torch.Tensor,
    cls_token_is_query: bool = True,
) -> torch.Tensor:
    """Convert raw vision-encoder attention weights into per-patch scores.

    `attn_weights` shape conventions:
        (B, H, Q, K)  -- standard multi-head attention (H heads, Q queries, K keys)
    For CLIP/SigLIP: query index 0 is the [CLS] token; the patches are keys 1..N.
    Score_i = mean over heads of attn[*, *, cls_idx, 1+i]  (how much CLS attends
    to patch i). Returns (B, N) in [0,1]-ish (softmax rows sum to 1).

    This runs inside a forward_hook on the vision tower, NOT inside the LLM, so
    it does not need to fight FlashAttention fusion (the survey §6.5.3 hurdle).
    """
    if attn_weights.dim() != 4:
        raise ValueError(f"expected (B,H,Q,K), got {tuple(attn_weights.shape)}")
    if cls_token_is_query:
        # CLS is query 0; patches are keys 1..N
        cls_to_patches = attn_weights[:, :, 0, 1:]      # (B, H, N)
    else:
        # fall back: mean attention received by each key (excluding CLS key 0)
        cls_to_patches = attn_weights[:, :, :, 1:].mean(dim=2)  # (B, H, N)
    return cls_to_patches.mean(dim=1)                    # (B, N)


# --------------------------------------------------------------------------- #
# vLLM hook helper (engine-side; imported only when integrating, not for CPU test)
# --------------------------------------------------------------------------- #
def make_projector_post_hook(pruning_rate: float, score_provider):
    """Build a forward-hook callable for `LlavaMultiModalProjector`.

    `score_provider` is a closure that returns the (B,N) CLS-attention scores
    captured from the vision tower (via its own hook). The returned hook prunes
    the projector output rows in-place-by-replacement at the boundary
    (post-projector, pre-LLM-fusion). Placeholder-count reconciliation is handled
    in the multimodal processor patch (see method-design.md §1b step 2).

    Kept here for reference; the actual vLLM monkeypatch lives in serve_bench.py
    so the CPU test path never imports vLLM.
    """
    selector = ClsAttnSelector(pruning_rate=pruning_rate)

    def hook(module, inputs, output):  # noqa: ANN001
        scores = score_provider()               # (B,N), from vision-tower hook
        kept, keep_idx = selector.select(output, scores)
        # stash keep_idx on module for the processor to read (placeholder count)
        module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
        module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
        return kept

    return hook


# --------------------------------------------------------------------------- #
# CPU self-test (run via `python -m src.compressors` or pytest)
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    torch.manual_seed(0)
    b, n, d = 2, 576, 4096           # LLaVA-1.5-7B: N=576, projector out D=4096
    feats = torch.randn(b, n, d)
    # synthetic CLS-attn: make patch 0..K clearly more important for batch 0
    scores = torch.rand(b, n)
    scores[0, :64] += 5.0

    for r in (0.0, 0.25, 0.50, 0.75):
        sel = ClsAttnSelector(pruning_rate=r)
        kept, idx = sel.select(feats, scores)
        k = keep_count(n, r)
        assert kept.shape == (b, k, d), f"r={r}: bad kept shape {kept.shape}"
        assert idx.shape == (b, k), f"r={r}: bad idx shape {idx.shape}"
        assert idx.dtype == torch.long
        # row integrity: kept[b,j] == feats[b, idx[b,j]]
        check = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        assert torch.equal(kept, check), f"r={r}: gather mismatch"
        # at r=0 we keep everything (order may differ but set must match)
        if r == 0.0:
            assert k == n
        # monotonic: higher k at lower r
    assert keep_count(n, 0.0) > keep_count(n, 0.25) > keep_count(n, 0.50) > keep_count(n, 0.75)
    # cls_attention_scores shape
    aw = torch.softmax(torch.randn(1, 4, 577, 577), dim=-1)   # 1 CLS + 576 patches, 4 heads
    s = cls_attention_scores(aw)
    assert s.shape == (1, 576), f"cls_attn bad shape {s.shape}"
    print("compressors self-test OK:",
          [keep_count(n, r) for r in (0.0, .25, .50, .75)], "scores", tuple(s.shape))


if __name__ == "__main__":
    _self_test()
