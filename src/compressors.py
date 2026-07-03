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
# A'' selector (step-2): CLIP CONTRASTIVE text<->patch (the v2 fix)
# --------------------------------------------------------------------------- #
# WHY A'': v2 scored in the LLM embed_tokens space -- NOT contrastively aligned,
# so word-semantics vs glyph-pixels don't match in cosine -> OCR collapsed (r50
# ~0.38 vs proxy 0.530). CLIP was trained (contrastive) so that CLIP-text embeds
# ALIGN with CLIP-ViT features. So scoring CLIP-patch-features by
# CLIP_text_encoder(question) . CLIP_patch_feature should rank text/OCR-relevant
# patches HIGH, training-free. This is the SparseVLM signal relocated to the
# boundary via CLIP (contrastive) features.
#
# MECHANISM (vLLM-side, see serve_bench.build_engine):
#   * Load a CLIPTextModel + text_projection + visual_projection from the SAME
#     CLIP checkpoint that LLaVA-1.5's vision tower is built from
#     (openai/clip-vit-large-patch14-336). ~125M params, ~0.5GB.
#   * Capture the vision tower's last_hidden_state patch tokens (pre-projector,
#     (B,N,1024)) via the existing vision-tower forward-hook. Apply CLIP's
#     visual_projection (1024->768) to land them in the CONTRASTIVE space.
#   * Embed the question via CLIPTextModel + text_projection (768->768 contrastive).
#   * Score = cos(clip_text_feat, clip_patch_feat) per patch, max/mean over the T
#     question tokens (SparseVLM-style).
#   * Selection indices are then applied to the PROJECTOR OUTPUT (B,N,4096) ->
#     v1-of-A'' keeps contiguous-compaction + placeholder-shrink identical to
#     proxy/v1/v2 plumbing (no pre-projector early-prune yet -- follow-on).
#
# Memory: score is (T_text x N=576) only -- never (B,H,S,S). visual_projection of
# (B,576,1024)->(B,576,768) is ~3.5MB fp16. Cheap.

def clip_text_patch_scores(
    clip_patch_feats: torch.Tensor,   # (B, N, D_clip=768) ALREADY in contrastive space
    clip_text_feats: torch.Tensor,    # (B, T, D_clip=768) ALREADY in contrastive space
    pool: str = "max",                # "max" (SparseVLM default) or "mean" over T tokens
    sim: str = "cosine",              # "cosine" (default) or "dot"
) -> torch.Tensor:
    """Score each of the N CLIP patch features by similarity to the CLIP text features.

    Both inputs MUST live in CLIP's contrastive space (i.e. CLIPVisionModel output
    @ visual_projection for patches, CLIPTextModel output @ text_projection for the
    question). CLIP was trained so this dot product is a meaningful cross-modal
    similarity; this is the v2 fix (v2 used LLM embed_tokens, which is NOT
    contrastively aligned -> OCR failed).

    Returns (B, N) per-patch scores. Memory: (B, N, T) intermediate only.
    """
    _validate(clip_patch_feats,
              torch.zeros(clip_patch_feats.shape[:2], device=clip_patch_feats.device))
    if clip_text_feats.dim() != 3 or clip_text_feats.shape[0] != clip_patch_feats.shape[0] \
            or clip_text_feats.shape[2] != clip_patch_feats.shape[2]:
        raise ValueError(
            f"clip_text_feats must be (B,T,D_clip) matching clip_patch_feats (B,N,D_clip); "
            f"got {tuple(clip_text_feats.shape)} vs {tuple(clip_patch_feats.shape)}")

    a = clip_patch_feats                       # (B, N, D_clip)
    q = clip_text_feats                        # (B, T, D_clip)
    if sim == "cosine":
        a = torch.nn.functional.normalize(a, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)
    sim_mat = torch.bmm(a, q.transpose(1, 2))  # (B, N, T)
    if pool == "max":
        return sim_mat.max(dim=-1).values       # (B, N)
    elif pool == "mean":
        return sim_mat.mean(dim=-1)             # (B, N)
    else:
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")


@dataclass
class ClipQuerySelector:
    """A'' selector: rank patches by CLIP CONTRASTIVE text<->patch similarity.

    The v2 fix: v2 used LLM embed_tokens (word-semantics space) vs post-projector
    patches (LLM space) -- NOT contrastively aligned, so OCR failed (r50 ~0.38).
    A'' uses CLIP's contrastive alignment (CLIP trained so text-embeds align with
    ViT-patch-features): score CLIP-projected patches by CLIP-text(question) and
    select top-k. Training-free.

    Inputs (both pre-computed by serve_bench from CLIP, NOT the LLM):
      projector_output : (B, N, D_llm=4096) -- the boundary embeddings we COMPACT.
                         (kept for signature symmetry; the SELECT happens on the
                         CLIP-space scores, but the GATHER is on this tensor so
                         v1-of-A'' applies CLIP-score indices to projector output,
                         identical contiguous-compaction plumbing to proxy/v1/v2.)
      clip_patch_feats : (B, N, D_clip=768) -- vision-tower last_hidden_state patch
                         tokens @ CLIP visual_projection (contrastive space).
      clip_text_feats  : (B, T, D_clip=768) -- CLIPTextModel(question) @ text_proj.

    `pool="max"` (default) matches SparseVLM's per-token-max similarity. The
    selector delegates to the shared `select_topk` so the contiguous-compaction +
    placeholder-shrink integration is identical to proxy/v1/v2 plumbing.

    CPU-testable: feed dummy (B,N,D_clip) patch feats + (B,T,D_clip) text feats.
    """
    pruning_rate: float = 0.0
    pool: str = "max"          # "max" (SparseVLM) or "mean" over question tokens
    sim: str = "cosine"        # "cosine" (default) or "dot"

    def select(
        self,
        projector_output: torch.Tensor,   # (B, N, D_llm) -- the tensor we compact
        clip_patch_feats: torch.Tensor,   # (B, N, D_clip) -- for SCORING
        clip_text_feats: torch.Tensor,    # (B, T, D_clip) -- for SCORING
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate(projector_output,
                  torch.zeros(projector_output.shape[:2], device=projector_output.device))
        if projector_output.shape[:2] != clip_patch_feats.shape[:2]:
            raise ValueError(
                f"projector_output and clip_patch_feats must share (B,N); got "
                f"{tuple(projector_output.shape[:2])} vs {tuple(clip_patch_feats.shape[:2])}")
        scores = clip_text_patch_scores(clip_patch_feats, clip_text_feats,
                                        pool=self.pool, sim=self.sim)
        # apply the CLIP-score top-k indices to the PROJECTOR OUTPUT (v1 of A'').
        return select_topk(projector_output, scores, self.pruning_rate)

    def __call__(self, projector_output, clip_patch_feats, clip_text_feats) -> torch.Tensor:
        kept, _ = self.select(projector_output, clip_patch_feats, clip_text_feats)
        return kept


# --------------------------------------------------------------------------- #
# P3 cross-compressor panel: ToMe merge (published, different reduction MODE)
# --------------------------------------------------------------------------- #
# WHY: the v1 paper measured served throughput only for OUR proxy selector ->
# "only proxy" + the "0/37 measure served throughput" claim looks self-serving.
# P3 shows the FRAMEWORK measures served throughput across multiple compressors
# with DIFFERENT reduction modes. ToMe (Bolya et al. ICLR'23) is the key NEW
# compressor: a PUBLISHED method whose reduction mode is MERGE (average similar
# tokens -> preserves info) vs the prune family's DISCARD. This shows the
# framework's served-throughput measurement is compressor-agnostic across both
# selection-signal (proxy/cls/query) AND reduction-mode (prune/merge) axes.
#
# ALGORITHM (Bolya et al. ICLR'23, "Token Merging: ToMe", section 3.2):
#   1. Bipartite soft matching: split the N tokens into two sets A (even idx)
#      and B (odd idx) by alternating. This bipartition guarantees each merge
#      pair is disjoint (a token is matched at most once per step).
#   2. Similarity: cos(A_i, B_j) -- the most similar B for each A, and the most
#      similar A for each B (mutual matching, ToMe's "linked" pairs).
#   3. Select the r most-similar MUTUAL pairs and merge each by averaging.
#   4. Output = remaining A + remaining B + merged -> N - r tokens.
# ToMe applies this per transformer layer; at the projector-output BOUNDARY
# (post-projector, pre-LLM-fusion) we apply it iteratively in ONE shot to
# reduce N=576 -> k (multiple bipartite rounds because each round yields at
# most floor(N/2) merges). The signal + average-merge rule are ToMe-exact;
# only the application site (single boundary vs every layer) differs, which is
# what vLLM-integrability at the boundary requires (no intra-LLM surgery).
#
# Compared to PRUNE (proxy/true_cls/query): merge PRESERVES info (averaged
# token carries both sources' signal) vs prune DISCARDS the dropped tokens.
# Throughput is identical at iso-k (placeholder-shrink makes both k-short);
# the question P3 measures is whether merge's info-preservation shows up as
# better accuracy at iso-throughput (hypothesis) -- and whether the merge
# compute itself adds visible overhead vs the O(1) gather of prune.


def _tome_bipartite_step(
    features: torch.Tensor,   # (B, N, D)
    max_pairs: int,
) -> torch.Tensor:
    """One ToMe bipartite soft-matching + average-merge step.

    Splits features into A (even idx) and B (odd idx), finds mutual most-similar
    pairs by cosine, and merges up to `max_pairs` of them (the most-similar
    first). Returns (B, N - r_eff, D) where r_eff = #pairs actually merged
    (capped by the number of mutual matches found; if zero mutual, falls back
    to the global most-similar pairs so progress is guaranteed).

    ToMe-exact for the merge rule; "one step" here is one application of ToMe's
    bipartite matching (ToMe applies one step per transformer layer; we apply
    iteratively at the boundary, see `tome_merge`).
    """
    b, n, d = features.shape
    if n <= 1 or max_pairs <= 0:
        return features
    a = features[:, 0::2, :]                        # (B, nA, D)
    bb = features[:, 1::2, :]                       # (B, nB, D)
    na, nb = a.shape[1], bb.shape[1]
    # cosine similarity (ToMe uses plain dot product on L2-normalized features)
    an = torch.nn.functional.normalize(a, dim=-1)
    bn = torch.nn.functional.normalize(bb, dim=-1)
    sim = torch.bmm(an, bn.transpose(1, 2))         # (B, na, nb)
    # for each a, best b (most similar)
    a_vals, a_to_b = sim.max(dim=2)                  # (B, na)
    # for each b, best a (mutual check)
    _, b_to_a = sim.max(dim=1)                       # (B, nb) argmax along a
    a_grid = torch.arange(na, device=features.device).unsqueeze(0).expand(b, -1)
    b_back = torch.gather(b_to_a, 1, a_to_b)         # (B, na): a that each a's best-b points back to
    mutual = (b_back == a_grid)                      # (B, na) bool
    masked_vals = a_vals.masked_fill(~mutual, float("-inf"))
    # fallback: if a batch has zero mutual matches, use raw most-similar (progress)
    n_mut = mutual.sum(dim=1)                        # (B,)
    no_mut = (n_mut == 0).view(b, 1)
    masked_vals = torch.where(no_mut, a_vals, masked_vals)
    # select top-r per batch (r capped by available finite-valued candidates)
    finite_per_b = (masked_vals > float("-inf")).sum(dim=1)   # (B,)
    r_eff = int(finite_per_b.min().clamp(min=1).item())
    r_eff = max(1, min(max_pairs, r_eff, na, nb))
    _, top_a = masked_vals.topk(k=r_eff, dim=1)       # (B, r_eff) -- a-side local indices
    top_b = torch.gather(a_to_b, 1, top_a)            # (B, r_eff) -- b-side local indices
    # merged = average (ToMe's weight=0.5 default; could be size-weighted for repeats)
    ga = top_a.unsqueeze(-1).expand(-1, -1, d)
    gb = top_b.unsqueeze(-1).expand(-1, -1, d)
    merged = (torch.gather(a, 1, ga) + torch.gather(bb, 1, gb)) * 0.5
    # build output: drop the selected a's and b's, append merged
    a_mask = torch.ones(b, na, dtype=torch.bool, device=features.device)
    a_mask.scatter_(1, top_a, False)
    b_mask = torch.ones(b, nb, dtype=torch.bool, device=features.device)
    b_mask.scatter_(1, top_b, False)
    # if any duplicate top_b (non-mutual fallback can collide), the b_mask just
    # drops both -> b-count decreases by |unique(top_b)|. Pad/truncate to r_eff
    # so output shape is deterministic (N - r_eff). We re-pad merged if needed.
    out_list = []
    for bi in range(b):
        rem_a = a[bi][a_mask[bi]]
        rem_b = bb[bi][b_mask[bi]]
        cat = torch.cat([rem_a, rem_b, merged[bi]], dim=0)
        out_list.append(cat)
    out = torch.stack(out_list, dim=0)               # (B, na+nb-r_eff', D)
    # ensure exact length N - r_eff (pad by repeating last row or truncate)
    target_len = n - r_eff
    if out.shape[1] != target_len:
        if out.shape[1] > target_len:
            out = out[:, :target_len, :]
        else:
            pad = target_len - out.shape[1]
            rep = out[:, -1:, :].expand(-1, pad, -1)
            out = torch.cat([out, rep], dim=1)
    return out


def tome_merge(
    image_features: torch.Tensor,    # (B, N, D) -- projector output
    pruning_rate: float,
    max_iters: int = 32,
) -> torch.Tensor:
    """ToMe-style bipartite soft-matching + average-merge at the projector-output
    boundary. Iteratively applies `_tome_bipartite_step` until the sequence is
    exactly k = round(N * (1 - pruning_rate)) tokens long.

    This is a DIFFERENT REDUCTION MODE from the prune family: instead of
    discarding (1-r)*N tokens, it MERGES them into the kept tokens by averaging.
    The output sequence is genuinely k-shorter (placeholder-shrink makes the LLM
    forward identical to a prune at iso-k -> throughput is comparable), but each
    output token may carry signal from 1, 2, or more original patches.

    Published method: ToMe, Bolya et al. ICLR'23. ToMe applies one bipartite
    step per transformer layer; we apply it iteratively at one boundary (the
    only vLLM-integrable site without intra-LLM surgery). The merge rule is
    ToMe-exact (cosine similarity, mutual most-similar pairs, average merge).

    Returns (B, k, D) merged features (NOT a (kept, keep_idx) tuple -- merge has
    no keep_idx; the output token at position j may be an average of several
    originals). Callers use the returned tensor directly as the projector output.
    """
    _validate(image_features,
              torch.zeros(image_features.shape[:2], device=image_features.device))
    b, n, d = image_features.shape
    k = keep_count(n, pruning_rate)
    if k >= n:
        return image_features
    cur = image_features
    it = 0
    while cur.shape[1] > k and it < max_iters:
        need = cur.shape[1] - k
        cur = _tome_bipartite_step(cur, max_pairs=need)
        it += 1
    # final exact-k guard (rounding): truncate or pad
    if cur.shape[1] > k:
        cur = cur[:, :k, :]
    elif cur.shape[1] < k:
        pad = k - cur.shape[1]
        rep = cur[:, -1:, :].expand(-1, pad, -1)
        cur = torch.cat([cur, rep], dim=1)
    return cur.contiguous()


@dataclass
class TomeMergeSelector:
    """ToMe (Bolya et al. ICLR'23) token-merging compressor at the projector-output
    boundary. The P3 cross-compressor panel's MERGE member (vs the prune family).

    Select() returns (kept, keep_idx) for signature compatibility with the prune
    selectors, but keep_idx is a dummy arange (merge has no per-token index -- the
    output token at position j may be an average of several originals). The
    placeholder-shrink integration in serve_bench only needs kept.shape[1] == k.
    """
    pruning_rate: float = 0.0

    def select(
        self,
        image_features: torch.Tensor,   # (B, N, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, d = image_features.shape
        k = keep_count(n, self.pruning_rate)
        merged = tome_merge(image_features, self.pruning_rate)
        # dummy keep_idx (merge has no real per-token index; arange satisfies the
        # placeholder-shrink plumbing which only reads shape[1])
        keep_idx = torch.arange(k, device=image_features.device).unsqueeze(0).expand(b, -1)
        return merged, keep_idx

    def __call__(self, image_features: torch.Tensor) -> torch.Tensor:
        merged, _ = self.select(image_features)
        return merged


# --------------------------------------------------------------------------- #
# P3 cross-compressor panel: RANDOM prune (trivial baseline / sanity floor)
# --------------------------------------------------------------------------- #
def random_prune(
    image_features: torch.Tensor,   # (B, N, D)
    pruning_rate: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random uniform prune: keep K = round(N*(1-r)) tokens chosen uniformly at
    random. The trivial baseline / sanity floor -- if a real compressor doesn't
    beat random at iso-throughput, the selection signal is worthless.

    Deterministic given the same generator seed (so runs are reproducible). The
    generator is constructed per-call from a fixed seed in serve_bench (the
    default torch.Generator() with manual_seed) -> reproducible across runs.

    Returns (kept (B,K,D), keep_idx (B,K) long).
    """
    _validate(image_features,
              torch.zeros(image_features.shape[:2], device=image_features.device))
    b, n, d = image_features.shape
    k = keep_count(n, pruning_rate)
    if k >= n:
        idx = torch.arange(n, device=image_features.device).unsqueeze(0).expand(b, -1)
        return image_features, idx
    # per-batch random permutation, take first k (CPU/GPU agnostic; uses the
    # generator's seed for reproducibility)
    keep_idx = torch.empty(b, k, dtype=torch.long, device=image_features.device)
    for bi in range(b):
        perm = torch.randperm(n, generator=generator, device=image_features.device)
        keep_idx[bi] = perm[:k]
    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
    kept = torch.gather(image_features, dim=1, index=gather_idx)
    return kept, keep_idx


@dataclass
class RandomPruneSelector:
    """Random uniform prune baseline (the P3 panel's sanity floor).

    Same signature as the prune selectors so the projector-hook plumbing is
    identical (just fed random indices instead of a scored top-k).
    """
    pruning_rate: float = 0.0
    seed: int = 0

    def select(
        self,
        image_features: torch.Tensor,   # (B, N, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, d = image_features.shape
        device = image_features.device
        # build a seeded generator on the features' device (CPU-testable)
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return random_prune(image_features, self.pruning_rate, generator=gen)

    def __call__(self, image_features: torch.Tensor) -> torch.Tensor:
        kept, _ = self.select(image_features)
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

    # --- 8. A'' ClipQuerySelector: shapes, determinism, query-sensitivity -------
    # CLIP space is D_clip=768 (LLaVA's CLIP-L/14-336 visual+text projections).
    d_clip = 768
    clip_patch = torch.randn(b, n, d_clip)
    clip_text = torch.randn(b, t, d_clip)
    # align a SPAN of CLIP patches to CLIP text token 0 -> those must score highest
    clip_patch[:, 30:40, :] = clip_text[:, 0:1, :].expand(-1, 10, -1)
    for r in (0.25, 0.50, 0.75):
        for pool in ("max", "mean"):
            sel = ClipQuerySelector(pruning_rate=r, pool=pool, sim="cosine")
            kept1, idx1 = sel.select(feats, clip_patch, clip_text)
            kept2, idx2 = sel.select(feats, clip_patch, clip_text)
            k = keep_count(n, r)
            assert kept1.shape == (b, k, d), f"clip r={r} {pool}: bad shape {kept1.shape}"
            assert torch.equal(idx1, idx2), f"clip r={r} {pool}: not deterministic"
            # gather integrity: kept comes from feats via idx (CLIP-scored selection
            # but projector-output gather)
            check = torch.gather(feats, 1, idx1.unsqueeze(-1).expand(-1, -1, d))
            assert torch.equal(kept1, check), f"clip r={r} {pool}: gather mismatch"
            # query-sensitivity: aligned span 30..40 must be in the kept set
            kept_set = set(idx1[0].tolist())
            aligned = set(range(30, 40))
            kept_aligned = aligned & kept_set
            assert len(kept_aligned) >= 8, (
                f"clip r={r} {pool}: CLIP-aligned patches not preferred "
                f"(only {len(kept_aligned)}/10 kept) -- selector not query-aware")

    # --- 9. clip_text_patch_scores: shape (B,N), max>=mean, argmax in aligned span -
    cs_max = clip_text_patch_scores(clip_patch, clip_text, pool="max", sim="cosine")
    cs_mean = clip_text_patch_scores(clip_patch, clip_text, pool="mean", sim="cosine")
    assert cs_max.shape == (b, n) and cs_mean.shape == (b, n), "clip tp_scores bad shape"
    assert torch.all(cs_max >= cs_mean - 1e-5), "clip max must be >= mean"
    # dot != cosine
    cs_dot = clip_text_patch_scores(clip_patch, clip_text, pool="max", sim="dot")
    assert not torch.allclose(cs_max, cs_dot, atol=1e-4), "clip cosine==dot (norm broken)"
    # argmax for batch 0 must lie in the aligned span 30..40
    assert 30 <= int(cs_max[0].argmax().item()) < 40, "clip tp_scores argmax not in aligned span"

    # --- 10. P3 ToMe merge: shapes, exact-k, determinism, merge-semantics -------
    # ToMe must (a) emit exactly k tokens, (b) be deterministic at fixed seed (no
    # generator, but cosine + topk are deterministic), and (c) actually MERGE
    # (output != a pure subset of input rows -- at least one output row is an
    # average of two inputs).
    for r in (0.25, 0.50, 0.75):
        merged = tome_merge(feats, r)
        k = keep_count(n, r)
        assert merged.shape == (b, k, d), f"tome r={r}: bad shape {merged.shape}, want (2,{k},{d})"
        m2 = tome_merge(feats, r)
        assert torch.equal(merged, m2), f"tome r={r}: not deterministic"
        # r=0 returns input unchanged
    assert torch.equal(tome_merge(feats, 0.0), feats), "tome r=0 should be identity"
    # merge-semantics: at least one output row is NOT byte-identical to any input
    # row (it's an average). Check for batch 0: each merged row's nearest input
    # row should have cos-sim < 0.999 for at least one merged row (the merged ones).
    merged = tome_merge(feats, 0.50)
    feats0n = torch.nn.functional.normalize(feats[0], dim=-1)
    merged0n = torch.nn.functional.normalize(merged[0], dim=-1)
    cos_to_nearest = (merged0n @ feats0n.T).max(dim=1).values  # (k,)
    # at least 10% of merged rows should be "new" (not byte-identical to an input)
    n_merged_rows = int((cos_to_nearest < 0.999).sum().item())
    assert n_merged_rows >= k // 10, (
        f"tome r=0.50: only {n_merged_rows}/{k} merged rows are non-identical to "
        f"any input -- merge not happening (cos_to_nearest min={cos_to_nearest.min().item():.4f})")
    # high-similarity clusters actually DO merge: build two tight clusters of
    # tokens, ToMe should merge within clusters first (output preserves the
    # cluster centers' info). Cluster A = copies of token 0, Cluster B = copies
    # of token 100. After r=0.5 merge, output should be ~half A + half B (still
    # two distinct cluster means), not a random mix.
    cluster_feats = torch.randn(1, 40, 32)
    cluster_feats[:, :20, :] = cluster_feats[:, 0:1, :] + 0.01 * torch.randn(1, 20, 32)
    cluster_feats[:, 20:, :] = cluster_feats[:, 20:21, :] + 0.01 * torch.randn(1, 20, 32)
    merged_cluster = tome_merge(cluster_feats, 0.50)   # 40 -> 20
    assert merged_cluster.shape == (1, 20, 32), "tome cluster shape"
    # the 20 outputs should form TWO clusters (low-rank structure), not one blob
    mc = merged_cluster[0]
    cent = mc.mean(dim=0, keepdim=True)
    dists = (mc - cent).norm(dim=-1)
    # if ToMe merged across clusters, we'd see a single blob (small dists). If it
    # merged WITHIN clusters, we keep the two distinct centers (large dists).
    assert dists.std().item() > 1e-3, (
        f"tome cluster: merged output is a blob (std={dists.std().item():.4f}) -- "
        f"not merging within clusters as expected")

    # --- 11. P3 RandomPruneSelector: shapes, determinism, randomness ------------
    for r in (0.25, 0.50, 0.75):
        sel = RandomPruneSelector(pruning_rate=r, seed=42)
        k1, idx1 = sel.select(feats)
        k2, idx2 = sel.select(feats)
        assert k1.shape == (b, keep_count(n, r), d), f"rand r={r}: bad shape {k1.shape}"
        assert torch.equal(idx1, idx2), f"rand r={r}: not deterministic at fixed seed"
        # gather integrity: kept rows come from feats via idx
        check = torch.gather(feats, 1, idx1.unsqueeze(-1).expand(-1, -1, d))
        assert torch.equal(k1, check), f"rand r={r}: gather mismatch"
        # different seed -> (very likely) different selection
        sel_b = RandomPruneSelector(pruning_rate=r, seed=99)
        _, idx_b = sel_b.select(feats)
        assert not torch.equal(idx1, idx_b), f"rand r={r}: seed has no effect"
        # all indices in [0, n) and unique per batch row
        for bi in range(b):
            assert idx1[bi].min() >= 0 and idx1[bi].max() < n, f"rand r={r}: idx OOB"
            assert idx1[bi].unique().numel() == keep_count(n, r), f"rand r={r}: dup idx"

    print("compressors self-test OK: probe=true(lam0)==identical, diversity+determinism ok, "
          "query_aware+clip_query query-sensitive + shapes ok, "
          "tome-merge exact-k+deterministic+merge-semantics ok, "
          "random-prune deterministic+seed-sensitive ok, "
          "keep_counts=" + str([keep_count(n, r) for r in (0.0, .25, .50, .75)]) +
          " scores=" + str(tuple(s.shape)))


if __name__ == "__main__":
    _self_test()
