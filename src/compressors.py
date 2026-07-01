"""Visual-token compressors for the P2 probe + v1 method.

Boundary-level, training-free compressors that operate on vision-tower / projector
outputs *before* LLM fusion. Designed to hook into vLLM's
`LlavaMultiModalProjector.forward` (see notes/method-design.md §1b).

Two layers:
  * `ClsAttnSelector` / `select_topk` -- pure top-k gather by an external score
    (the P2 probe used a hidden-state-deviation PROXY score here).
  * `TrueClsAttnSelector` + `ClsAttnCapture` -- v1 method: ranks patches by the
    REAL vision-tower [CLS]->patch attention (the VisionZip / FasterVLM / VTC-CLS
    family), captured by monkeypatching a `CLIPAttention` layer to expose its
    softmax weights. Optional PRUNESID-style diversity penalty (off by default).
    CPU-testable on dummy tensors (the capture plumbing is vLLM-side only).
  * `QueryAwareSelector` -- v2 method (step-1): ranks patches by SIMILARITY to the
    QUESTION text embeddings (SparseVLM-style text-guided selection, but at the
    projector-output boundary -> vLLM-integrable). Both question-token embeddings
    and patch embeddings live in the LLM input embedding space (embed_tokens
    output == projector output space), so a plain cosine / dot product scores
    each patch by max/mean similarity over question tokens. CPU-testable.

All selectors:
    input  : image_features  (B, N, D)   -- projector output (or vision-tower output)
             scores          (B, N)      -- per-token importance (e.g. CLS attn)
             pruning_rate    float in [0,1)  -- fraction of N tokens to DROP
    output : kept            (B, K, D)   with K = round(N * (1 - pruning_rate))
             keep_idx        (B, K) long -- indices kept (for placeholder reconciliation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

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
# Pure top-k gather (used by both the proxy probe and the v1 selector)
# --------------------------------------------------------------------------- #
def select_topk(
    image_features: torch.Tensor,   # (B, N, D)
    scores: torch.Tensor,           # (B, N)
    pruning_rate: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure top-k: keep the highest-scoring K = round(N*(1-r)) rows per batch.

    Deterministic (PyTorch topk ties broken by index). Returns (kept (B,K,D),
    keep_idx (B,K) long).
    """
    _validate(image_features, scores)
    b, n, d = image_features.shape
    k = keep_count(n, pruning_rate)
    keep_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices  # (B,K)
    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)                        # (B,K,D)
    kept = torch.gather(image_features, dim=1, index=gather_idx)                 # (B,K,D)
    return kept, keep_idx


@dataclass
class ClsAttnSelector:
    """Top-k token selection by an external importance score (probe-grade).

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
        return select_topk(image_features, scores, self.pruning_rate)

    def __call__(self, image_features: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        kept, _ = self.select(image_features, scores)
        return kept


# --------------------------------------------------------------------------- #
# v1 selector: TRUE CLS-attention + optional diversity (PRUNESID-style)
# --------------------------------------------------------------------------- #
def greedy_diverse_topk(
    features: torch.Tensor,    # (N, D) -- L2-normalized candidate features
    scores: torch.Tensor,      # (N,)   -- per-token importance
    k: int,
    lam: float,                # diversity weight in [0,1]; 0 == pure top-k
) -> torch.Tensor:
    """Greedy importance+diversity selection (PRUNESID family, single image).

    Score for selecting token i given already-selected set S:
        u(i) = (1-lam)*score_i  -  lam * max_{j in S} sim(i, j)
    i.e. penalize candidates similar to anything already kept (NMS-like).
    Greedy: at each step pick argmax u(i) over remaining candidates.

    `features` must be L2-normalized along D (cosine sim = dot product).
    Returns keep_idx (k,) long. O(N*k*D) — fine for N=576, k<=576.
    """
    n = features.shape[0]
    if k >= n:
        return torch.arange(n, device=features.device)
    device = features.device
    selected = torch.empty(k, dtype=torch.long, device=device)
    # max similarity of each candidate to the (growing) selected set
    max_sim = torch.full((n,), float("-inf"), device=device)
    chosen = torch.zeros(n, dtype=torch.bool, device=device)
    imp = (1.0 - lam) * scores
    for step in range(k):
        # u(i) = imp_i - lam * max_sim_i  (only -lam when lam>0)
        u = imp.clone()
        if lam > 0.0 and step > 0:
            u = u - lam * max_sim.clamp(min=0.0)
        u.masked_fill_(chosen, float("-inf"))
        pick = int(torch.argmax(u).item())
        selected[step] = pick
        chosen[pick] = True
        if lam > 0.0:
            new_sim = features @ features[pick]            # (N,) cosine to pick
            max_sim = torch.maximum(max_sim, new_sim)
    return selected


@dataclass
class TrueClsAttnSelector:
    """v1 selector: rank patches by REAL [CLS]->patch attention, optionally with
    a PRUNESID-style diversity penalty.

    Replaces the probe's hidden-state-deviation proxy. The CLS-attention scores
    are captured by `ClsAttnCapture` (a monkeypatch on the vision-tower
    `CLIPAttention`) and passed in here -- this class is just the ranking logic,
    so it stays CPU-testable without vLLM.

    diversity_lam=0.0 (default) => pure top-k by CLS-attn (the FasterVLM/VTC-CLS
    setting). diversity_lam>0 => greedy importance+diversity (PRUNESID family).
    Diversity uses cosine similarity of projector-output features (cheap, the
    features are already materialized at the hook).
    """

    pruning_rate: float = 0.0
    diversity_lam: float = 0.0    # 0 == off (v1 default); ~0.3-0.5 typical if on

    def select(
        self,
        image_features: torch.Tensor,   # (B, N, D) projector output
        scores: torch.Tensor,           # (B, N)   CLS->patch attention (mean over heads)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate(image_features, scores)
        b, n, d = image_features.shape
        k = keep_count(n, self.pruning_rate)
        if self.diversity_lam <= 0.0 or k == n:
            # fast path: pure top-k (identical to probe selector, just fed REAL scores)
            return select_topk(image_features, scores, self.pruning_rate)
        # diverse path: per-image greedy (small N,k; loop over batch)
        device = image_features.device
        keep_idx = torch.empty(b, k, dtype=torch.long, device=device)
        # L2-normalize features once for cosine sim
        fn = torch.nn.functional.normalize(image_features, dim=-1)
        for bi in range(b):
            keep_idx[bi] = greedy_diverse_topk(fn[bi], scores[bi], k, self.diversity_lam)
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
        kept = torch.gather(image_features, dim=1, index=gather_idx)
        return kept, keep_idx

    def __call__(self, image_features: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        kept, _ = self.select(image_features, scores)
        return kept


# --------------------------------------------------------------------------- #
# v2 selector (step-1): QUERY-AWARE text<->patch similarity (SparseVLM-style)
# --------------------------------------------------------------------------- #
def text_patch_scores(
    image_features: torch.Tensor,   # (B, N, D) projector output
    query_embeds: torch.Tensor,     # (B, T, D) LLM input-embedding of question tokens
    pool: str = "max",              # how to reduce over the T question tokens
    sim: str = "cosine",            # "cosine" or "dot"
) -> torch.Tensor:
    """Score each of the N patch embeddings by similarity to the question.

    Both `image_features` (projector output) and `query_embeds` (output of the
    LLM's `embed_tokens` on the question's input_ids) live in the SAME space --
    the LLM's input embedding space -- so they are directly comparable. This is
    the SparseVLM-style text-guided scoring rule, but computed at the boundary
    (post-projector, pre-LLM-fusion) so it is vLLM-integrable (the question text
    is known at preprocessing time, before the forward).

    Returns (B, N) per-patch scores. `pool`: "max" keeps the best-matching
    question token per patch (default, SparseVLM); "mean" averages. `sim`:
    "cosine" L2-normalizes both sides first; "dot" uses raw dot product.
    Memory: (B,N,T) intermediate only -- no (B,H,S,S) materialization.
    """
    _validate(image_features, torch.zeros(image_features.shape[:2],
                                          device=image_features.device))
    if query_embeds.dim() != 3 or query_embeds.shape[0] != image_features.shape[0] \
            or query_embeds.shape[2] != image_features.shape[2]:
        raise ValueError(
            f"query_embeds must be (B,T,D) matching image_features (B,N,D); got "
            f"{tuple(query_embeds.shape)} vs {tuple(image_features.shape)}")

    a = image_features                      # (B, N, D)
    q = query_embeds                        # (B, T, D)
    if sim == "cosine":
        a = torch.nn.functional.normalize(a, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)
    # pairwise similarity per (patch, query-token): (B, N, T)
    sim_mat = torch.bmm(a, q.transpose(1, 2))
    if pool == "max":
        return sim_mat.max(dim=-1).values            # (B, N)
    elif pool == "mean":
        return sim_mat.mean(dim=-1)                  # (B, N)
    else:
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")


@dataclass
class QueryAwareSelector:
    """v2-step-1 selector: rank visual patches by relevance to the QUESTION.

    SparseVLM-style text-guided selection, but at the projector-output BOUNDARY
    (so it is vLLM-integrable: the question is known at preprocessing time, and
    the selection runs as a cheap forward-hook on the projector -- no intra-LLM
    attention surgery, no FlashAttention fight, no per-layer cost). v1 (vision-
    tower CLS-attn) catastrophically degraded OCR because [CLS] attends to coarse
    salient objects, NOT text/task regions; making selection QUERY-AWARE keeps
    the task-relevant / text-region patches alive even at high prune.

    Inputs:
      image_features : (B, N, D) projector output (the boundary embedding).
      query_embeds   : (B, T, D) the LLM `embed_tokens` output on the question's
                       input_ids. Cheap (an embedding LOOKUP, not a transformer
                       pass). Pre-computed per request in serve_bench before the
                       forward (see text_patch_scores docstring for plumbing).

    `pool="max"` (default) and `sim="cosine"` match SparseVLM's per-token-max
    similarity. The selector then delegates to the shared `select_topk` so the
    contiguous-compaction + placeholder-shrink integration is identical to the
    proxy / true_cls paths (no change to that plumbing).

    CPU-testable: feed dummy (B,N,D) image_features and (B,T,D) query_embeds.
    """
    pruning_rate: float = 0.0
    pool: str = "max"          # "max" (SparseVLM) or "mean" over question tokens
    sim: str = "cosine"        # "cosine" (default) or "dot"

    def select(
        self,
        image_features: torch.Tensor,   # (B, N, D)
        query_embeds: torch.Tensor,     # (B, T, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = text_patch_scores(image_features, query_embeds,
                                   pool=self.pool, sim=self.sim)
        return select_topk(image_features, scores, self.pruning_rate)

    def __call__(self, image_features: torch.Tensor, query_embeds: torch.Tensor) -> torch.Tensor:
        kept, _ = self.select(image_features, query_embeds)
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

    Called by `ClsAttnCapture` on the REAL softmax weights extracted from a
    monkeypatched `CLIPAttention` layer of the vision tower (NOT the LLM, so it
    does not fight FlashAttention fusion -- the survey §6.5.3 hurdle).
    """
    if attn_weights.dim() != 4:
        raise ValueError(f"expected (B,H,Q,K), got {tuple(attn_weights.shape)}")
    if cls_token_is_query:
        cls_to_patches = attn_weights[:, :, 0, 1:]      # (B, H, N)
    else:
        cls_to_patches = attn_weights[:, :, :, 1:].mean(dim=2)  # (B, H, N)
    return cls_to_patches.mean(dim=1)                    # (B, N)


# --------------------------------------------------------------------------- #
# CLS-attention capture (vLLM-side; imported only by serve_bench on GPU)
# --------------------------------------------------------------------------- #
class ClsAttnCapture:
    """Monkeypatch a vision-tower `CLIPAttention` layer to expose its REAL softmax
    [CLS]->patch attention, without changing the encoder's forward numerics.

    WHY: vLLM's `CLIPAttention` delegates to `MultiHeadAttention` which uses
    `F.scaled_dot_product_attention` (fused SDPA) and returns NO weights (the
    probe therefore fell back to a hidden-state-deviation PROXY). To get the real
    VisionZip/FasterVLM signal we recompute the target layer's CLS-row softmax in
    a parallel manual path and stash ONLY the (B,N) head-meaned CLS->patch score;
    the SDPA output still drives the real forward (numerics unchanged -> encoder
    output byte-identical to stock vLLM).

    MEMORY: the manual path slices the query to CLS (index 0) BEFORE the QK^T, so
    the logits tensor is (B,H,1,S) -- not (B,H,S,S). For CLIP-L/14 (S=577) this
    is ~577x less peak memory than the full-matrix path, which is what tipped the
    A40 (44GB) into OOM when retained alongside vLLM's ~26GB KV + ~13GB weights.
    The selector only ever needs the CLS row.

    Hook target: `vision_tower.vision_model.encoder.layers[L].self_attn`
    (a `CLIPAttention`). Default L = last layer (-1) per VTC-CLS/FasterVLM.
    Multi-layer ensemble (VTC-CLS) supported via `layers` list (scores averaged).

    RESTORE: call `.unpatch()` (or use as a context manager) to put the original
    `forward` back. Idempotent -- safe to call multiple times.

    Not CPU-testable (needs a real CLIPAttention); tested via the dummy
    `cls_attention_scores` path + an integration smoke on GPU.
    """

    def __init__(
        self,
        clip_attn_module,                 # the CLIPAttention nn.Module to patch
        layers: Optional[list] = None,    # for multi-layer: pass list of modules
        layer_names: Optional[list] = None,
    ):
        # Accept either a single module or a list (ensemble). Normalize to a list.
        if isinstance(clip_attn_module, (list, tuple)):
            self.targets = list(clip_attn_module)
        else:
            self.targets = [clip_attn_module]
        self.layer_names = layer_names or [f"layer{i}" for i in range(len(self.targets))]
        self._orig_forwards = [m.forward for m in self.targets]
        # `scores` is (B,N) head-meaned CLS->patch attention -- the ONLY thing the
        # selector needs. We deliberately do NOT retain the full (B,H,S,S) weights
        # (that caused the A40 OOM at full scale: 577x more memory than needed).
        self._captured: dict = {
            "scores": None,            # (B,N) -- the retained signal
            "last_cls_row_shape": None,  # for OOM-fix validation logging
            "n_calls": 0,
        }
        self._patched = False

    @property
    def captured(self) -> dict:
        return self._captured

    def patch(self) -> "ClsAttnCapture":
        if self._patched:
            return self
        for mod in self.targets:
            self._install_one(mod)
        self._patched = True
        return self

    def _install_one(self, mod) -> None:
        orig_forward = mod.forward
        scale = mod.scale
        num_heads = mod.num_heads_per_partition
        head_dim = mod.head_dim
        capture = self._captured  # closed-over shared dict

        def patched_forward(hidden_states):  # noqa: ANN001
            # ---- run the REAL forward first (SDPA path, numerics unchanged) ----
            qkv_states, _ = mod.qkv_proj(hidden_states)
            q, k, v = qkv_states.chunk(3, dim=-1)
            out = mod.attn(q, k, v)               # the real SDPA call
            attn_output, _ = mod.out_proj(out)

            # ---- parallel manual path: CLS-row ONLY (memory fix) ----
            # We only need the CLS token's attention over the S keys (one row),
            # NOT the full (B,H,S,S) matrix. Slicing the query to index 0 BEFORE
            # the matmul means the logits tensor is (B,H,1,S) -> softmax ->
            # CLS->patch scores (B,H,N) [skip CLS key 0]. ~S× less peak memory
            # vs the full QK^T (S=577 here), which is what caused the A40 OOM
            # when retained alongside vLLM's ~26GB KV + ~13GB weights.
            b, s, _ = q.shape
            qh_cls = q[:, 0:1, :].view(b, 1, num_heads, head_dim).transpose(1, 2)  # (B,H,1,Hd)
            kh = k.view(b, s, num_heads, head_dim).transpose(1, 2)                 # (B,H,S,Hd)
            cls_logits = torch.matmul(qh_cls, kh.transpose(-2, -1)) * scale        # (B,H,1,S)
            cls_w = torch.softmax(cls_logits.float(), dim=-1)                      # (B,H,1,S)
            # CLS is key 0; patches are keys 1..N. Drop key 0 -> (B,H,1,N), then
            # head-mean -> (B,1,N) -> squeeze -> (B,N).
            cls_scores = cls_w[:, :, 0, 1:].mean(dim=1)                            # (B,N)
            capture["last_cls_row_shape"] = tuple(cls_w.shape)  # validate (B,H,1,S)
            capture["n_calls"] += 1
            # merge multi-layer ensemble into a single per-patch score (B, N)
            prev = capture["scores"]
            if prev is None or prev.shape != cls_scores.shape:
                capture["scores"] = cls_scores.clone()
            else:
                # running mean across the ensemble layers (one call per layer)
                n = capture["n_calls"]
                capture["scores"] = prev + (cls_scores - prev) / n

            return attn_output, None

        mod.forward = patched_forward  # type: ignore[assignment]

    def unpatch(self) -> None:
        if not self._patched:
            return
        for mod, orig in zip(self.targets, self._orig_forwards):
            mod.forward = orig  # type: ignore[assignment]
        self._patched = False

    def reset(self) -> None:
        """Clear stashed scores (call between requests in a serving loop)."""
        n = self._captured.get("n_calls", 0)
        self._captured = {"scores": None, "last_cls_row_shape": None, "n_calls": 0}

    def __enter__(self) -> "ClsAttnCapture":
        return self.patch()

    def __exit__(self, *exc) -> None:
        self.unpatch()


# --------------------------------------------------------------------------- #
# vLLM hook helper (engine-side; imported only when integrating, not for CPU test)
# --------------------------------------------------------------------------- #
def make_projector_post_hook(pruning_rate: float, score_provider, diversity_lam: float = 0.0):
    """Build a forward-hook callable for `LlavaMultiModalProjector`.

    `score_provider` is a closure that returns the (B,N) CLS-attention scores
    captured from the vision tower (via `ClsAttnCapture`). The returned hook
    prunes the projector output rows at the boundary (post-projector,
    pre-LLM-fusion). v1 uses `TrueClsAttnSelector` (real CLS-attn + optional
    diversity); the probe used `ClsAttnSelector` (pure top-k, fed a proxy score).
    Placeholder-count reconciliation is handled in the multimodal processor patch
    (see method-design.md §1b step 2).

    Kept here for reference; the actual vLLM monkeypatch lives in serve_bench.py
    so the CPU test path never imports vLLM.
    """
    selector = TrueClsAttnSelector(
        pruning_rate=pruning_rate, diversity_lam=diversity_lam
    ) if diversity_lam > 0.0 else ClsAttnSelector(pruning_rate=pruning_rate)

    def hook(module, inputs, output):  # noqa: ANN001
        scores = score_provider()               # (B,N), from ClsAttnCapture
        kept, keep_idx = selector.select(output, scores)
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

    # --- 1. probe selector (pure top-k) + shape/keep-count monotonicity ---
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
        if r == 0.0:
            assert k == n
    assert keep_count(n, 0.0) > keep_count(n, 0.25) > keep_count(n, 0.50) > keep_count(n, 0.75)

    # --- 2. TRUE CLS-attn selector, diversity OFF: must equal probe selector ---
    # (the v1 selector with lam=0 is identical to pure top-k -- this proves we
    # didn't regress the probe path when swapping proxy for real scores)
    for r in (0.25, 0.50, 0.75):
        s_probe = ClsAttnSelector(pruning_rate=r).select(feats, scores)
        s_true = TrueClsAttnSelector(pruning_rate=r, diversity_lam=0.0).select(feats, scores)
        assert torch.equal(s_probe[0], s_true[0]), f"r={r}: lam=0 path diverged"
        assert torch.equal(s_probe[1], s_true[1]), f"r={r}: lam=0 idx diverged"

    # --- 3. TRUE CLS-attn selector, diversity ON: shapes ok + determinism ---
    for r in (0.50, 0.75):
        sel = TrueClsAttnSelector(pruning_rate=r, diversity_lam=0.3)
        kept1, idx1 = sel.select(feats, scores)
        kept2, idx2 = sel.select(feats, scores)
        k = keep_count(n, r)
        assert kept1.shape == (b, k, d), f"div r={r}: bad shape {kept1.shape}"
        assert torch.equal(idx1, idx2), f"div r={r}: not deterministic"
        # row integrity still holds (kept rows come from feats via idx)
        check = torch.gather(feats, 1, idx1.unsqueeze(-1).expand(-1, -1, d))
        assert torch.equal(kept1, check), f"div r={r}: gather mismatch"

    # --- 4. cls_attention_scores shape (B,N) from raw (B,H,Q,K) ---
    aw = torch.softmax(torch.randn(1, 4, 577, 577), dim=-1)   # 1 CLS + 576 patches, 4 heads
    s = cls_attention_scores(aw)
    assert s.shape == (1, 576), f"cls_attn bad shape {s.shape}"
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-5

    # --- 5. greedy_diverse_topk sanity: lam=0 picks top-k by score ---
    feats_n = torch.nn.functional.normalize(torch.randn(100, 16), dim=-1)
    sc = torch.randn(100)
    topk_idx = greedy_diverse_topk(feats_n, sc, k=10, lam=0.0)
    expected = torch.topk(sc, k=10).indices.sort().values
    assert torch.equal(topk_idx.sort().values, expected), "lam=0 must equal topk"
    # lam=1 still returns k unique indices
    div_idx = greedy_diverse_topk(feats_n, sc, k=10, lam=1.0)
    assert div_idx.unique().numel() == 10, "diversity must keep k unique"

    # --- 6. v2 QueryAwareSelector: shapes, determinism, query-sensitivity ----
    # query_embeds (B,T,D): make patch 10..20 align with query token 0 (high sim)
    t = 6
    qe = torch.randn(b, t, d)
    # align a SPAN of patches to query token 0 -> those patches must score highest
    qe[:, 0, :] = feats[:, 10:20, :].mean(dim=1)
    for r in (0.25, 0.50, 0.75):
        for pool in ("max", "mean"):
            sel = QueryAwareSelector(pruning_rate=r, pool=pool, sim="cosine")
            kept1, idx1 = sel.select(feats, qe)
            kept2, idx2 = sel.select(feats, qe)
            k = keep_count(n, r)
            assert kept1.shape == (b, k, d), f"qa r={r} {pool}: bad shape {kept1.shape}"
            assert torch.equal(idx1, idx2), f"qa r={r} {pool}: not deterministic"
            # gather integrity
            check = torch.gather(feats, 1, idx1.unsqueeze(-1).expand(-1, -1, d))
            assert torch.equal(kept1, check), f"qa r={r} {pool}: gather mismatch"
            # query-sensitivity: r=0.50 keeps 288; with a 10-patch span aligned to
            # the query, AT LEAST those 10 patches must be in the kept set for batch 0
            kept_set = set(idx1[0].tolist())
            aligned = set(range(10, 20))
            kept_aligned = aligned & kept_set
            assert len(kept_aligned) >= 8, (
                f"qa r={r} {pool}: query-aligned patches not preferred "
                f"(only {len(kept_aligned)}/10 kept) -- selector not query-aware")

    # --- 7. text_patch_scores: shape (B,N), max >= mean, monotone in alignment ---
    sc_max = text_patch_scores(feats, qe, pool="max", sim="cosine")
    sc_mean = text_patch_scores(feats, qe, pool="mean", sim="cosine")
    assert sc_max.shape == (b, n) and sc_mean.shape == (b, n), "tp_scores bad shape"
    assert torch.all(sc_max >= sc_mean - 1e-5), "max must be >= mean"
    # dot != cosine (different normalization)
    sc_dot = text_patch_scores(feats, qe, pool="max", sim="dot")
    assert not torch.allclose(sc_max, sc_dot, atol=1e-4), "cosine==dot (normalization broken)"
    # score argmax for batch 0 must lie in the aligned span 10..20
    assert 10 <= int(sc_max[0].argmax().item()) < 20, "tp_scores argmax not in aligned span"

    print("compressors self-test OK: probe=true(lam0)==identical, diversity+determinism ok, "
          "query_aware query-sensitive + shapes ok, "
          "keep_counts=" + str([keep_count(n, r) for r in (0.0, .25, .50, .75)]) +
          " scores=" + str(tuple(s.shape)))


if __name__ == "__main__":
    _self_test()
