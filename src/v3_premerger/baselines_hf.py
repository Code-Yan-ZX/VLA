"""HF-transformers baseline harness for FastV and PyramidDrop, SAME model /
SAME weights / SAME prompt / SAME sampling / SAME scorer as the vLLM runner
(v3_premerger_runner.py).  See experiments/j4_baselines_hf_design.md.

WHY HF (not vLLM): vLLM V1 fixes the attention metadata / sequence length for
the whole LLM stack, so it cannot CHANGE the token count BETWEEN decoder layers
-- which is exactly what FastV (one-shot prune after layer K) and PyramidDrop
(staged pyramid drop across 4 layer bands) do.  Both official implementations
(github.com/DL-Prism/FastV, github.com/Cooperx521/PyramidDrop) are therefore
HF-transformers modeling patches.  We mirror them here against the SAME Qwen
weights the runner uses; the engine difference (HF eager vs vLLM flash-attn) is
disclosed in the paper, and EFFICIENCY numbers are still measured under vLLM.

Generation protocol replicated from the runner (v3_premerger_runner.py:main):
  * prompt   : the `question` field verbatim (the short-answer instruction is
               baked into eval/subsets/*.jsonl), wrapped in the Qwen chat
               template with one image, add_generation_prompt=True (== the
               runner's llm.chat make_msgs + chat template).
  * sampling : greedy (runner temperature=0.0) -> HF do_sample=False / argmax;
               max_tokens=32 (runner default).
  * pixels   : --max-pixels>0 -> processor max_pixels; ==0 -> processor default
               (runner passes mm_processor_kwargs={"max_pixels":...} only when
               --max-pixels>0; same min/max-pixels calibration per family).
  * output   : JSON with the SAME fields as the runner so official_scorers.py
               offline re-scoring is seamless: model / mode / benchmark / r /
               acc / n_skipped / mean_ptid_len / per_sample[{id,question,gt,
               answer,correct,prompt_token_ids, ...}].

Method correspondence (line-by-line, URLs in the design digest):
  FastV -- at LLM layer K (default 2, --fastv-k) take that layer's softmaxed
    attention [1,H,L,L], AVERAGE OVER HEADS, read the LAST query token's row
    (last_layer_attention_avg[-1]), restrict to IMAGE-token columns, keep the
    top round(n_img*(1-r)); subsequent layers see only kept tokens.  This is
    DL-Prism/FastV src/transformers/.../modeling_llama.py INPLACE branch:
        last_layer_attention_avg = mean(attn, dim=1)[0]
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
        ..._image = ...[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
        top_attention_rank_index = ....topk(ATTENTION_RANK).indices + SYS_LENGTH
        keep_indexs = cat(text_before, top, text_after).sort()
        hidden_states = hidden_states[:, keep_indexs, :]
    (We generalise SYS/text to "all non-image positions are always kept"; the
    official reads layer K-1's attention and prunes BEFORE layer K -- an
    off-by-one we document; the paper's "prune after layer 2" is what we run.)
  PyramidDrop -- split the LLM into 4 equal layer bands (0-25/25-50/50-75/
    75-100%); after bands 0,1,2 rank the CURRENT image tokens by attention
    (same text-query->image-key score as FastV) and keep round(n_img0*ratio)
    with ratios [1.0,0.75,0.5,0.25] (official default is lambda=0.5 ->
    [1.0,0.5,0.25,0.125]; we use the fair-budget schedule mandated here and
    expose --pyramid-ratios).  Official: Cooperx521/PyramidDrop
    llava/model/modeling_llama_pdrop.py pdrop_rank_drop:
        image_tokens = int(cur_image_token * ratio_list[cur_num])
        keep_length  = int(cur_image_token * ratio_list[cur_num+1])
        # rank by attention from the last instruction token to image keys,
        # mean over heads:
        attention_avg_head = mean(attn, dim=0)[:, image_index:+image_tokens]
        attention_avg_text = mean(attention_avg_head, dim=0)
        top_rank_index     = attention_avg_text.topk(keep_length).indices
    (Official recomputes Q/K of the NEXT layer for the ranking score; we reuse
    the just-finished layer's attention -- same text-query->image-key semantics,
    documented simplification.  Official uses floor (int); we use round to match
    the runner's keep=round(full*(1-r)) convention, documented.)
  Equivalent keep for PyramidDrop (the fairness number vs a uniform-r method):
    keep_equiv = sum_s(ratio_s * L_s) / sum_s(L_s),  L_s = #layers in band s.
    For 4 EQUAL bands and ratios [1.0,0.75,0.5,0.25]:
        keep_equiv = mean(ratios) = 2.5/4 = 0.625  ->  r_equiv = 1-0.625 = 0.375.
    i.e. PyramidDrop's default schedule spends the same LLM token-budget as a
    uniform method at keep=62.5% (r=0.375).  r_equiv is what we store in "r".

NO GPU is touched at import; the model is loaded only in main().  --dry-check
builds a TINY random-init Qwen2.5-VL on CPU (no weights download, no GPU) and
verifies the manual layer loop reproduces the native forward exactly at r=0 and
runs FastV/Pyramid end-to-end with correct keep counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# --------------------------------------------------------------------------- #
# Model registry (same ids/family logic as the runner).
# --------------------------------------------------------------------------- #
MODELS = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def detect_family(model_id: str) -> str:
    """Qwen2*/Qwen2.5-VL -> qwen2vl (patch14); else qwen3vl (patch16)."""
    return "qwen2vl" if "qwen2" in model_id.lower() else "qwen3vl"


# --------------------------------------------------------------------------- #
# Data + scoring (verbatim copies from v3_premerger_runner.py so the ONLINE
# `correct` matches the runner; the AUTHORITATIVE metric is the offline re-score
# by official_scorers.py over per_sample[].answer/gt -- unchanged contract).
# We COPY rather than import the runner because its module top does `import
# vllm` (heavy / engine-specific); these pure scoring functions are stable.
# --------------------------------------------------------------------------- #
from dataclasses import dataclass
from typing import Optional


@dataclass
class Sample:
    id: str
    image: str
    question: str
    gt: str
    extra: dict


def load_subset(path: str) -> list[Sample]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(Sample(
                id=str(o["id"]), image=o["image"], question=o["question"],
                gt=str(o["gt"]),
                extra={k: v for k, v in o.items()
                       if k not in {"id", "image", "question", "gt"}}))
    return out


def _norm_words(s: str) -> list[str]:
    return [tok for tok in "".join(
        c if (c.isalnum() or c.isspace()) else " " for c in s.strip().lower()
    ).split()]


def _singular(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 2 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 1 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def score_gqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    p_words = _norm_words(pred)
    g_norm = "".join(c for c in gt.strip().lower()
                     if c.isalnum() or c.isspace()).strip()
    g_words = g_norm.split()
    if not g_words:
        return 0
    if g_norm in {"yes", "no"}:
        lead = None
        for w in p_words:
            if w not in {"a", "an", "the"}:
                lead = w
                break
        return 1 if (lead in {"yes", "no"} and lead == g_norm) else 0
    syns = {g_norm, _singular(g_norm) if len(g_words) == 1 else g_norm}
    if choices:
        for c in choices:
            cn = "".join(ch for ch in c.strip().lower()
                         if ch.isalnum() or c.isspace()).strip()
            if cn:
                syns.add(cn)
                if len(cn.split()) == 1:
                    syns.add(_singular(cn))
    p_text = " ".join(p_words)
    for s in syns:
        s_words = s.split()
        if len(s_words) == 1:
            sg = _singular(s)
            if any(w == s or _singular(w) == sg for w in p_words):
                return 1
        elif s in p_text:
            return 1
    return 0


def score_textvqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    p_words = _norm_words(pred)
    p_text = " ".join(p_words)
    for gt_i in [x.strip() for x in gt.split(";") if x.strip()]:
        g_words = _norm_words(gt_i)
        if not g_words:
            continue
        gi = " ".join(g_words)
        if len(g_words) == 1:
            sg = _singular(g_words[0])
            if any(w == g_words[0] or _singular(w) == sg for w in p_words):
                return 1
        elif gi in p_text:
            return 1
    return 0


score_docvqa = score_textvqa


def score_yesno(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    g = gt.strip().lower()
    if g not in {"yes", "no"}:
        return 0
    for w in _norm_words(pred):
        if w in {"a", "an", "the"}:
            continue
        return 1 if w == g else 0
    return 0


def score_mc_letter(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    g = gt.strip().upper()
    if not (len(g) == 1 and g.isalpha()):
        return 0
    p = pred.strip().upper()
    if not p:
        return 0
    for tok in p.split():
        core = tok.rstrip(".,:;)\"'")
        if len(core) == 1 and core.isalpha():
            return 1 if core == g else 0
    return 1 if (p[0].isalpha() and p[0] == g) else 0


def score_chartqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    p, g = pred.strip(), gt.strip()

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None

    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf:
        return int(abs(pf - gf) / abs(gf) <= 0.05)
    return int(" ".join(_norm_words(p)) == " ".join(_norm_words(g)))


def score_ocrbench(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    nospace = bool(choices) and "__nospace__" in choices
    p = pred.lower().strip().replace("\n", " ")
    for g in gt.split(";"):
        g = g.lower().strip().replace("\n", " ")
        if not g:
            continue
        if nospace:
            if g.replace(" ", "") in p.replace(" ", ""):
                return 1
        elif g in p:
            return 1
    return 0


SCORERS = {
    "gqa": score_gqa,
    "textvqa": score_textvqa,
    "docvqa": score_docvqa,
    "mme": score_yesno,
    "mmbench": score_mc_letter,
    "scienceqa": score_mc_letter,
    "chartqa": score_chartqa,
    "ocrbench": score_ocrbench,
}


# --------------------------------------------------------------------------- #
# Core pruning primitives (engine-independent; unit-tested in run_dry_check).
# --------------------------------------------------------------------------- #
def make_causal_mask(n: int, device, dtype) -> torch.Tensor:
    """Plain additive causal mask [1,1,n,n]: 0 below+on diagonal, -inf above.
    Correct for BOTH target families: Qwen2.5-VL-7B (use_sliding_window=False,
    all 'full_attention') and Qwen3-VL-8B (no sliding layers) -- verified via
    text_config.layer_types.  A sliding-window family would need the sliding
    mask; we assert it away at setup (documented limitation)."""
    m = torch.full((n, n), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None, :, :]


def build_prune_plan(mode: str, n_layers: int, n_image: int, r: float,
                     fastv_k: int, ratios: list[float]) -> dict[int, int]:
    """layer_idx -> number of IMAGE tokens to KEEP after that layer.
    FastV    : single entry at layer K (keep round(n_image*(1-r))).
    Pyramid  : entries at the last layer of bands 0,1,2 (keep round(n_image*
               ratios[1/2/3])); band boundaries = round({.25,.5,.75}*n_layers),
               matching the official layer_list=[L/4, L/2, 3L/4]."""
    plan: dict[int, int] = {}
    if mode == "fastv":
        k = min(max(0, fastv_k), n_layers - 1)
        plan[k] = max(1, int(round(n_image * (1.0 - r))))
    elif mode == "pyramid":
        bounds = [round(f * n_layers) for f in (0.25, 0.50, 0.75)]
        for s in range(3):                       # drops after bands 0,1,2
            plan[max(0, bounds[s] - 1)] = max(1, int(round(n_image * ratios[s + 1])))
    return plan


def pyramid_band_layers(n_layers: int) -> list[int]:
    """#layers in each of the 4 equal bands (sums to n_layers)."""
    b = [round(f * n_layers) for f in (0.25, 0.50, 0.75)]
    edges = [0] + b + [n_layers]
    return [edges[i + 1] - edges[i] for i in range(4)]


def pyramid_keep_equiv(ratios: list[float], n_layers: int) -> float:
    """Token-budget-equivalent UNIFORM keep ratio of a pyramid schedule:
    keep_equiv = sum_s(ratio_s * L_s) / sum_s(L_s),  L_s = #layers in band s.
    Equal bands -> mean(ratios).  For [1.0,0.75,0.5,0.25] -> 0.625 (r=0.375)."""
    Ls = pyramid_band_layers(n_layers)
    return float(sum(ratios[s] * Ls[s] for s in range(4)) / sum(Ls))


def rank_keep_indices(attn_w: torch.Tensor, image_mask: torch.Tensor,
                      k_keep: int) -> torch.Tensor:
    """Official FastV/PyramidDrop ranking: average attention OVER HEADS, take the
    LAST query token's row, read IMAGE-token columns, keep the top k_keep; all
    NON-image positions are always kept; return the sorted full-sequence keep
    index set.  attn_w: [1, heads, L, L] (softmaxed).  image_mask: bool[L]."""
    L = attn_w.shape[-1]
    dev = attn_w.device
    img_pos = image_mask.nonzero(as_tuple=False).squeeze(-1)     # [n_img]
    if img_pos.numel() == 0:
        return torch.arange(L, device=dev)
    a = attn_w[0].mean(dim=0)                                    # [L, L] over heads
    qrow = a[-1]                                                 # last query row [L]
    scores = qrow.index_select(0, img_pos)                       # [n_img]
    k = min(max(1, int(k_keep)), int(img_pos.numel()))
    keep_img = img_pos.index_select(0, scores.topk(k).indices)
    non_img = (~image_mask).nonzero(as_tuple=False).squeeze(-1)
    return torch.cat([non_img, keep_img]).sort().values


# --------------------------------------------------------------------------- #
# Prefill with layer-wise pruning (manual layer loop over the NATIVE decoder
# layers + a DynamicCache we slice at each drop).  Reuses the model's own
# rotary_emb / layernorms / MLP / lm_head -- the ONLY thing we drive manually is
# the layer iteration order and the token pruning between layers (which vLLM V1
# cannot express).
#
# TRANSFORMERS 4.57 LAYER-API COMPAT (root cause of the J4 STEP2 crash):
#   * Qwen2.5-VL decoder layers use the LEGACY API: forward(...,
#     output_attentions=...) -> (hidden, [attn_weights]);
#   * Qwen3-VL decoder layers use the MODERN API: forward(...) returns a BARE
#     hidden tensor and SILENTLY DROPS output_attentions (the layer does
#     `hidden, _ = self.self_attn(...)` and never surfaces weights).  Indexing
#     that bare tensor with out[0]/out[1] silently takes sequence rows, loses the
#     batch dim, and the next layer's RoPE broadcast fails with
#     "size of tensor a (32) must match b (128)" (num_heads vs head_dim).
#   => at prune layers we ALWAYS replicate the pre-norm block ourselves and call
#   `layer.self_attn(..., output_attentions=True, use_cache=True)` directly:
#   the ATTENTION module (both families, eager impl) returns (output, weights)
#   unconditionally.  Identical math: standard pre-norm residual block,
#   attention_dropout=0 at eval.  Verified by the dry-check r=0 equivalence.
#
# Qwen3-VL DEEPSTACK: the native TextModel.forward ADDS deepstack_visual_embeds
# (a list of [n_img, H] features tapped from vision-encoder layers) to the LLM
# hidden states at the IMAGE positions right after the first
# len(deepstack_visual_embeds) decoder layers (8B: after layers 0,1,2).  The
# capture stub therefore also grabs visual_pos_masks + deepstack_visual_embeds
# and the manual loop replays the addition (through the img_ord map, so an
# already-pruned set of image tokens gets exactly its own rows).  Qwen2.5-VL
# has no deepstack -> None -> skipped.
# --------------------------------------------------------------------------- #
def _cache_kv(cache, li: int):
    """(keys, values) tensors of cache layer li -- portable across transformers
    versions (4.57+: cache.layers[li].keys/.values; older: key/value_cache[li])."""
    if hasattr(cache, "layers"):
        return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


def _cache_set_kv(cache, li: int, k: torch.Tensor, v: torch.Tensor):
    if hasattr(cache, "layers"):
        cache.layers[li].keys = k
        cache.layers[li].values = v
    else:
        cache.key_cache[li] = k
        cache.value_cache[li] = v


def _cache_len(cache) -> int:
    return int(_cache_kv(cache, 0)[0].shape[-2])


def _split_mrope_pos(position_ids: torch.Tensor) -> torch.Tensor:
    """get_rope_index returns [3,bs,L]; a packed [4,bs,L] carries text positions
    in row 0 (only used for FA2 packed masking, which we don't use) -- drop it,
    exactly as the native TextModel.forward does before rotary."""
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:]
    return position_ids


def _layer_step(layer, hidden, attn_mask, pos_emb, cache, need_attn: bool):
    """Run ONE decoder layer; returns (hidden_out, attn_weights_or_None).

    need_attn=False -> native layer call; normalises both return APIs (legacy
    tuple (Qwen2.5-VL) / modern bare tensor (Qwen3-VL 4.57, which drops
    output_attentions entirely)).
    need_attn=True  -> replicate the pre-norm block and call self_attn directly
    so the softmaxed weights are available under BOTH families (the attention
    module always returns (output, weights); the Qwen3-VL LAYER does not)."""
    if not need_attn:
        out = layer(
            hidden,
            attention_mask=attn_mask,
            position_ids=None,                # eager RoPE uses position_embeddings
            past_key_values=cache,
            use_cache=True,
            cache_position=None,
            position_embeddings=pos_emb,
        )
        return (out[0] if isinstance(out, tuple) else out), None
    residual = hidden
    h_norm = layer.input_layernorm(hidden)
    attn_out, attn_w = layer.self_attn(
        hidden_states=h_norm,
        attention_mask=attn_mask,
        position_embeddings=pos_emb,
        past_key_values=cache,
        cache_position=None,
        output_attentions=True,               # Qwen2.5-VL named kwarg; Qwen3-VL
        use_cache=True,                       # absorbed by **kwargs (harmless)
    )
    hidden = residual + attn_out
    residual = hidden
    hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, attn_w


def prefill_pruned(model, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
                   image_mask_1d: torch.Tensor, mode: str, cfg: dict,
                   deepstack=None):
    """Run the pruned prefill.  Returns (hidden_normed [1,L',H],
    position_ids_reduced [3,1,L'], cache, image_mask_reduced [L'], diag dict).
    L' == L when no effective pruning (r=0 / keep-all).
    deepstack: optional list of [n_img_full, H] Qwen3-VL visual features to ADD
    at image positions after the first len(deepstack) layers (native parity)."""
    from transformers import DynamicCache

    LM = model.model.language_model
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    assert not getattr(LM, "has_sliding_layers", False), \
        "sliding-window layers are not supported by the plain-causal manual mask"

    position_ids = _split_mrope_pos(position_ids)
    hidden = inputs_embeds
    L0 = int(hidden.shape[1])
    n_image0 = int(image_mask_1d.sum())
    n_text = L0 - n_image0

    cache = DynamicCache(config=LM.config)
    plan = build_prune_plan(mode, len(LM.layers), n_image0, cfg["r"],
                            cfg["fastv_k"], cfg["ratios"])
    image_mask = image_mask_1d.clone()
    # img_ord[i] = index of current position i in the ORIGINAL image-token list
    # (valid at image positions); lets deepstack replay survive pruning.
    img_ord = image_mask.cumsum(0) - 1
    pos_emb = LM.rotary_emb(hidden, position_ids)               # (cos,sin) [1,L,hd]
    n_image_kept = n_image0
    fired = []
    n_deepstack = len(deepstack) if deepstack is not None else 0
    attn_mask = make_causal_mask(int(hidden.shape[1]), device, dtype)
    for idx, layer in enumerate(LM.layers):
        need = idx in plan
        hidden, attn_w = _layer_step(layer, hidden, attn_mask, pos_emb, cache, need)
        # Qwen3-VL deepstack: native adds visual features after the first
        # n_deepstack layers (at image positions) -- replay before any prune at
        # this layer so the ranking sees the same hidden the next layer would.
        if idx < n_deepstack:
            sel = img_ord[image_mask]
            emb = deepstack[idx].to(device=device, dtype=hidden.dtype)
            emb = emb.index_select(0, sel)
            hidden[:, image_mask] = hidden[:, image_mask] + emb
        if need and hidden.shape[1] > 1:
            keep = rank_keep_indices(attn_w, image_mask, plan[idx])
            hidden = hidden.index_select(1, keep)
            position_ids = position_ids.index_select(2, keep)
            image_mask = image_mask.index_select(0, keep)
            img_ord = img_ord.index_select(0, keep)
            pos_emb = LM.rotary_emb(hidden, position_ids)       # recompute (== index_select)
            attn_mask = make_causal_mask(int(hidden.shape[1]), device, dtype)
            # crop the KV already written by layers 0..idx to the kept positions
            for li in range(idx + 1):
                k, v = _cache_kv(cache, li)
                _cache_set_kv(cache, li, k.index_select(2, keep),
                              v.index_select(2, keep))
            n_image_kept = int(image_mask.sum())
            fired.append((idx, int(keep.numel())))
    hidden = LM.norm(hidden)
    diag = {"n_image_full": n_image0, "n_image_kept": n_image_kept,
            "n_text": n_text, "L0": L0, "L_after": int(image_mask.numel()),
            "prune_plan": {str(k): v for k, v in plan.items()}, "fired": fired,
            "n_deepstack": n_deepstack}
    return hidden, position_ids, cache, image_mask, diag


@torch.no_grad()
def generate_pruned(model, inputs_embeds, position_ids, image_mask_1d, mode,
                    cfg, max_new_tokens, eos_ids, deepstack=None):
    """Pruned prefill + greedy autoregressive decode (KV-cache reused; the cache
    is already cropped to L', so decode just appends one token/step)."""
    LM = model.model.language_model
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    hidden, position_ids, cache, image_mask, diag = prefill_pruned(
        model, inputs_embeds, position_ids, image_mask_1d, mode, cfg,
        deepstack=deepstack)
    logits = model.lm_head(hidden)
    next_tok = int(logits[0, -1].argmax(-1))
    gen = [next_tok]
    cur_pos = int(position_ids.max())
    kv_len = _cache_len(cache)
    embed = model.get_input_embeddings()
    for _ in range(max_new_tokens - 1):
        if next_tok in eos_ids:
            break
        tok_emb = embed(torch.tensor([[next_tok]], device=device))
        cur_pos += 1
        pos_new = torch.full((3, 1, 1), cur_pos, device=device,
                             dtype=position_ids.dtype)
        pe = LM.rotary_emb(tok_emb, pos_new)
        dec_mask = torch.zeros(1, 1, 1, kv_len + 1, device=device, dtype=dtype)
        h = tok_emb
        for layer in LM.layers:
            o = layer(h, attention_mask=dec_mask, position_ids=None,
                      past_key_values=cache, use_cache=True, cache_position=None,
                      position_embeddings=pe)
            h = o[0] if isinstance(o, tuple) else o   # legacy tuple / modern bare
        kv_len += 1
        h = LM.norm(h)
        next_tok = int(model.lm_head(h)[0, -1].argmax(-1))
        gen.append(next_tok)
    return gen, diag


# --------------------------------------------------------------------------- #
# Prepared-input capture: let the NATIVE outer forward (vision encode + merger +
# get_rope_index -- all family-specific, incl. Qwen3-VL deepstack) build
# inputs_embeds + position_ids, intercepted at the language-model boundary so we
# don't reimplement the family-specific embedding stage.
# --------------------------------------------------------------------------- #
class _Captured(Exception):
    pass


def capture_prepared_inputs(model, model_inputs: dict):
    """Returns (inputs_embeds [1,L,H], position_ids [3or4,1,L], deepstack).
    deepstack: list of [n_img,H] Qwen3-VL deepstack visual features (added at
    image positions after the first decoder layers) or None (Qwen2.5-VL).
    Stubs language_model.forward to grab the prepared tensors, then restores it.
    No forward compute is wasted (we raise immediately on capture)."""
    LM = model.model.language_model
    orig = LM.forward
    box = {}

    def stub(*args, **kwargs):
        box["inputs_embeds"] = kwargs.get(
            "inputs_embeds", args[0] if args else None)
        box["position_ids"] = kwargs.get("position_ids")
        box["deepstack_visual_embeds"] = kwargs.get("deepstack_visual_embeds")
        box["visual_pos_masks"] = kwargs.get("visual_pos_masks")
        raise _Captured()

    LM.forward = stub
    try:
        with torch.no_grad():
            model.model(**model_inputs)
    except _Captured:
        pass
    finally:
        LM.forward = orig
    if box.get("inputs_embeds") is None:
        raise RuntimeError("capture failed: language_model.forward never called")
    return (box["inputs_embeds"], box["position_ids"],
            box.get("deepstack_visual_embeds"))


# --------------------------------------------------------------------------- #
# Input construction -- SAME prompt/pixels as the runner's llm.chat.
# --------------------------------------------------------------------------- #
def _cap_image_pixels(image, max_pixels):
    """PIL pre-resize enforcing the pixel budget BEFORE the processor:
    transformers 4.57's Qwen3-VL processor SILENTLY IGNORES the per-call
    max_pixels kwarg (verified: identical image_grid_thw with and without the
    kwarg, both via image_processor(...) and via images_kwargs). Edges are
    rounded to multiples of 32 (= patch 16 x merge 2), aspect preserved."""
    if not max_pixels or max_pixels <= 0:
        return image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    import math
    scale = math.sqrt(max_pixels / float(w * h))
    nw = max(32, round(w * scale / 32) * 32)
    nh = max(32, round(h * scale / 32) * 32)
    return image.resize((nw, nh))


def build_inputs(processor, image, question: str, max_pixels: int, device):
    """One image + the verbatim question through the Qwen chat template
    (add_generation_prompt=True == the runner's generation setup).  max_pixels>0
    -> PIL pre-resize to the pixel budget (processor kwargs are ignored by
    transformers 4.57, see _cap_image_pixels); ==0 -> native resolution."""
    image = _cap_image_pixels(image, max_pixels)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], return_tensors="pt",
                       padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


# --------------------------------------------------------------------------- #
# CLI (runner-like).
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none",
                    choices=["none", "fastv", "pyramid"])
    ap.add_argument("--model", default=None,
                    help="HF id (Qwen/Qwen3-VL-8B-Instruct or "
                         "Qwen/Qwen2.5-VL-7B-Instruct). Family auto-detected.")
    ap.add_argument("--model-family", default="qwen3vl",
                    choices=["qwen3vl", "qwen2vl"],
                    help="used only if --model is not given.")
    ap.add_argument("--benchmark", default=None,
                    choices=list(SCORERS.keys()))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--r", type=float, default=0.0,
                    help="FastV drop ratio (keep=round(n_img*(1-r))). IGNORED for "
                         "pyramid schedule (see --pyramid-ratios); for pyramid we "
                         "STORE r_equiv=1-keep_equiv in the output 'r' field, where "
                         "keep_equiv=sum(ratio_s*L_s)/sum(L_s) (=mean for equal "
                         "bands; [1.0,0.75,0.5,0.25] -> 0.625 -> r=0.375).")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--fastv-k", type=int, default=2,
                    help="FastV prune layer (paper default K=2; attention of this "
                         "layer ranks the image tokens).")
    ap.add_argument("--pyramid-ratios", default="1.0,0.75,0.5,0.25",
                    help="per-band KEEP ratios (4 bands). Official default is "
                         "1.0,0.5,0.25,0.125 (lambda=0.5); we use the fair-budget "
                         "schedule by default.")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help=">0 -> processor max_pixels (iso-token calibration per "
                         "family); 0 -> processor default (runner parity).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-check", action="store_true",
                    help="CPU-only: build a TINY random-init Qwen2.5-VL (no "
                         "weights, no GPU) and verify the manual loop == native "
                         "forward at r=0 + FastV/Pyramid run with correct keeps.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# No-GPU self test (TINY random model on CPU; no weights download).
# --------------------------------------------------------------------------- #
def _tiny_model():
    """A minimal random-init Qwen2.5-VL on CPU for logic verification only."""
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    cfg = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=200, hidden_size=32, intermediate_size=64,
            num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256, rms_norm_eps=1e-6,
            use_sliding_window=False, sliding_window=None,
            # mrope_section is doubled internally -> must sum to head_dim//2 (=4)
            rope_scaling={"mrope_section": [2, 1, 1], "rope_type": "default"},
        ),
        vision_config=dict(
            depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
            in_channels=3, out_hidden_size=32,      # merger out == text hidden
            fullatt_block_indexes=[1], window_size=112,  # realistic window math
        ),
        image_token_id=151, vision_start_token_id=150, vision_end_token_id=149,
    )
    torch.manual_seed(0)
    # eager attention is REQUIRED for output_attentions (the ranking signal);
    # mirrors the real-model load (attn_implementation="eager").
    m = Qwen2_5_VLForConditionalGeneration._from_config(
        cfg, attn_implementation="eager").to(torch.float32).eval()
    return m, cfg


def run_dry_check():
    import math
    print("[dry-check] (A) pure-function unit tests")
    # rank_keep_indices: mean over heads, last query row, image columns, top-k.
    torch.manual_seed(0)
    L, H = 10, 3
    attn = torch.rand(1, H, L, L)
    img = torch.zeros(L, dtype=torch.bool)
    img[2:6] = True                                  # image at 2,3,4,5
    keep = rank_keep_indices(attn, img, 2)
    a = attn[0].mean(0)[-1]
    want_img = (torch.tensor([2, 3, 4, 5])[a[2:6].topk(2).indices]).sort().values
    want = torch.cat([torch.tensor([0, 1, 6, 7, 8, 9]), want_img]).sort().values
    assert torch.equal(keep, want), (keep, want)
    print("[dry-check]   OK rank_keep_indices (heads-avg, last-query, image cols)")

    # build_prune_plan + pyramid schedule + keep_equiv.
    p = build_prune_plan("fastv", 28, 100, 0.75, 2, [1, .75, .5, .25])
    assert p == {2: 25}, p
    pp = build_prune_plan("pyramid", 28, 100, 0.0, 2, [1.0, 0.75, 0.5, 0.25])
    assert pp == {6: 75, 13: 50, 20: 25}, pp        # bands [7,7,7,7]->last 6,13,20
    assert pyramid_band_layers(28) == [7, 7, 7, 7], pyramid_band_layers(28)
    assert pyramid_band_layers(36) == [9, 9, 9, 9], pyramid_band_layers(36)
    ke = pyramid_keep_equiv([1.0, 0.75, 0.5, 0.25], 28)
    assert abs(ke - 0.625) < 1e-9, ke              # equal bands -> mean
    print(f"[dry-check]   OK prune plan + keep_equiv={ke:.3f} (r_equiv={1-ke:.3f})")

    print("[dry-check] (B) tiny-model equivalence + end-to-end (CPU)")
    try:
        m, cfg = _tiny_model()
    except Exception as e:
        print(f"[dry-check]   SKIP tiny-model build ({type(e).__name__}: {str(e)[:120]})")
        print("[dry-check] (A) PASS; (B) skipped")
        return
    LM = m.model.language_model
    hidden_size = cfg.text_config.hidden_size
    L = 12
    X = torch.randn(1, L, hidden_size)
    P = torch.arange(L).view(1, 1, L).expand(3, 1, L).contiguous()
    img = torch.zeros(L, dtype=torch.bool)
    img[2:8] = True                                  # 6 image tokens

    # (B1) equivalence: manual loop @ r=0 (keep-all) == native forward.
    with torch.no_grad():
        nat = LM(inputs_embeds=X, position_ids=P.clone(), use_cache=False,
                 output_attentions=False, return_dict=True)
        logits_nat = m.lm_head(nat.last_hidden_state)
        hid0, _, _, _, d0 = prefill_pruned(
            m, X.clone(), P.clone(), img.clone(), "fastv",
            {"r": 0.0, "fastv_k": 1, "ratios": [1, .75, .5, .25]})
        logits_man = m.lm_head(hid0)
    assert d0["n_image_kept"] == d0["n_image_full"] == 6, d0
    maxdiff = (logits_nat - logits_man).abs().max().item()
    assert logits_nat.shape == logits_man.shape == (1, L, cfg.text_config.vocab_size)
    assert maxdiff < 1e-3, f"manual!=native maxdiff={maxdiff}"
    agree = int((logits_nat.argmax(-1) == logits_man.argmax(-1)).all())
    print(f"[dry-check]   OK manual==native @r=0 (maxdiff={maxdiff:.2e}, "
          f"argmax_all_equal={agree})")

    # (B2) FastV r=0.5: keep round(6*0.5)=3 image tokens -> L' = 6 text + 3 = 9.
    hid, pos, cache, im, df = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "fastv",
        {"r": 0.5, "fastv_k": 2, "ratios": [1, .75, .5, .25]})
    assert df["n_image_kept"] == 3 and hid.shape[1] == 9, df
    n_layers = len(m.model.language_model.layers)
    assert _cache_len(cache) == 9 and _cache_kv(cache, n_layers - 1)[0].shape[-2] == 9
    print(f"[dry-check]   OK FastV r=0.5: img 6->3, L 12->9, cache cropped at ALL "
          f"layers (fired={df['fired']})")

    # (B3) Pyramid: keep 75/50/25% of 6 -> after 3 drops: 5,3,2 image; final L=8.
    hid, pos, cache, im, dp = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "pyramid",
        {"r": 0.0, "fastv_k": 2, "ratios": [1.0, 0.75, 0.5, 0.25]})
    # n_image=6: round(6*.75)=5? round(4.5)=4 (banker's) -> int(round(4.5))=4 in py3
    exp = [max(1, int(round(6 * q))) for q in (0.75, 0.5, 0.25)]
    assert dp["n_image_kept"] == exp[-1], (dp, exp)
    assert hid.shape[1] == 6 + exp[-1], (dp, exp)
    print(f"[dry-check]   OK Pyramid keeps {exp} (final img {dp['n_image_kept']}, "
          f"L 12->{hid.shape[1]}, fired={dp['fired']})")

    # (B6) deepstack replay is a no-op for zero features (index plumbing sane).
    hid_ref, _, _, _, _ = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "fastv",
        {"r": 0.5, "fastv_k": 2, "ratios": [1, .75, .5, .25]})
    hid_ds, _, _, _, dd = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "fastv",
        {"r": 0.5, "fastv_k": 2, "ratios": [1, .75, .5, .25]},
        deepstack=[torch.zeros(6, hidden_size), torch.zeros(6, hidden_size)])
    assert dd["n_deepstack"] == 2, dd
    assert torch.equal(hid_ds, hid_ref), "zero deepstack must not change hidden"
    print("[dry-check]   OK deepstack replay (n_deepstack=2, zero-add identity, "
          "img_ord survives pruning)")

    # (B4) end-to-end greedy generation runs and terminates.
    eos = {cfg.text_config.eos_token_id if hasattr(cfg.text_config, 'eos_token_id')
           else 1}
    gen, dg = generate_pruned(
        m, X.clone(), P.clone(), img.clone(), "fastv",
        {"r": 0.5, "fastv_k": 2, "ratios": [1, .75, .5, .25]}, 4, eos)
    assert 1 <= len(gen) <= 4 and all(isinstance(t, int) for t in gen)
    print(f"[dry-check]   OK end-to-end greedy decode ({len(gen)} tokens)")

    # (B5) capture path: native vision encode + merger + get_rope_index, grabbed
    # at the language-model boundary by the stub.
    pv = torch.randn(16, 3 * 2 * 14 * 14)            # t=1,h=4,w=4 -> 16 patches
    grid = torch.tensor([[1, 4, 4]])                 # -> 16/4 = 4 image tokens
    ids = torch.tensor([[5, 6, 150, 151, 151, 151, 151, 149, 7, 8, 9]])
    ie, pos, ds = capture_prepared_inputs(
        m, {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": pv, "image_grid_thw": grid})
    assert ie.shape == (1, 11, hidden_size) and pos.shape[0] == 3, (ie.shape, pos.shape)
    assert ds is None, "Qwen2.5-VL has no deepstack"
    emb0 = m.get_input_embeddings()(ids)
    assert (ie - emb0).abs().sum().item() > 0        # vision embeds were scattered
    print("[dry-check]   OK capture: native vision+merger+get_rope_index "
          f"-> inputs_embeds{tuple(ie.shape)}, pos{tuple(pos.shape)}, scatter=on")
    print("[dry-check] ALL PASS")


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.dry_check:
        run_dry_check()
        return

    missing = [n for n, v in (("--benchmark", args.benchmark),
                              ("--subset", args.subset),
                              ("--out", args.out)) if not v]
    if missing:
        raise SystemExit("required (unless --dry-check): " + ", ".join(missing))

    model_id = args.model or MODELS[args.model_family]
    family = detect_family(model_id)
    ratios = [float(x) for x in args.pyramid_ratios.split(",") if x.strip()]
    if len(ratios) != 4 or ratios[0] != 1.0:
        raise SystemExit("--pyramid-ratios must be 4 comma values starting at "
                         "1.0 (e.g. 1.0,0.75,0.5,0.25)")

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16,
        attn_implementation="eager",            # REQUIRED: output_attentions
    ).to(device).eval()
    load_s = time.perf_counter() - t0
    image_token_id = model.config.image_token_id

    # eos set for manual decode
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    if processor.tokenizer.pad_token_id is not None:
        eos_ids.add(processor.tokenizer.pad_token_id)

    # effective r stored in the output (pyramid -> r_equiv for fair comparison)
    if args.mode == "pyramid":
        n_layers = len(model.model.language_model.layers)
        keep_equiv = pyramid_keep_equiv(ratios, n_layers)
        r_eff = round(1.0 - keep_equiv, 4)
    elif args.mode == "fastv":
        keep_equiv, r_eff = None, args.r
    else:
        keep_equiv, r_eff = None, 0.0
    cfg = {"r": args.r, "fastv_k": args.fastv_k, "ratios": ratios}

    samples = load_subset(args.subset)[:args.n]
    scorer = SCORERS[args.benchmark]

    per_sample = []
    n_skip = n_ok = correct = 0
    ptid_counts = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            image = Image.open(s.image).convert("RGB")
            inputs = build_inputs(processor, image, s.question,
                                  args.max_pixels, device)
            n_prompt_full = int(inputs["input_ids"].shape[1])
            if args.mode == "none":
                # NATIVE HF generation (standard path; the vLLM-equivalence target).
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=args.max_tokens,
                        do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
                ans = processor.decode(
                    out[0][n_prompt_full:], skip_special_tokens=True).strip()
                n_img_full = int((inputs["input_ids"][0] == image_token_id).sum())
                diag = {"n_image_full": n_img_full, "n_image_kept": n_img_full,
                        "n_text": n_prompt_full - n_img_full, "L0": n_prompt_full,
                        "L_after": n_prompt_full}
                ptid = n_prompt_full
            else:
                # MANUAL pruned prefill + greedy decode.
                image_mask = (inputs["input_ids"][0] == image_token_id)
                inputs_embeds, position_ids, deepstack = capture_prepared_inputs(
                    model, {k: v for k, v in inputs.items()})
                gen, diag = generate_pruned(
                    model, inputs_embeds, position_ids, image_mask, args.mode,
                    cfg, args.max_tokens, eos_ids, deepstack=deepstack)
                ans = processor.decode(gen, skip_special_tokens=True).strip()
                # effective tokens through the LLM: fastv -> post-prune length;
                # pyramid -> band-weighted average (text + image*keep_equiv).
                if args.mode == "fastv":
                    ptid = int(diag["L_after"])
                else:
                    ptid = int(round(diag["n_text"]
                                     + diag["n_image_full"] * keep_equiv))
        except Exception as e:
            print(f"[j4] sample {s.id} FAILED ({type(e).__name__}: {str(e)[:160]})",
                  file=sys.stderr, flush=True)
            per_sample.append({"id": s.id, "correct": 0, "skipped": True,
                               "answer": "", "gt": s.gt, "question": s.question})
            n_skip += 1
            continue
        if ans:
            n_ok += 1
        c = scorer(ans, s.gt, s.extra.get("choices"))
        correct += int(c)
        ptid_counts.append(ptid)
        rec = {"id": s.id, "correct": int(c), "skipped": False, "answer": ans,
               "gt": s.gt, "question": s.question, "prompt_token_ids": ptid,
               "n_image_full": diag["n_image_full"],
               "n_image_kept": diag["n_image_kept"], "n_text": diag["n_text"]}
        if args.benchmark == "ocrbench" and "question_type" in s.extra:
            rec["question_type"] = s.extra["question_type"]
        per_sample.append(rec)
        if (i + 1) % 10 == 0:
            print(f"[j4] {i+1}/{len(samples)} running_acc="
                  f"{correct / max(1, len(samples) - n_skip):.3f}", flush=True)
    wall = time.perf_counter() - t0

    n_scored = len(samples) - n_skip
    acc = correct / n_scored if n_scored else 0.0
    result = {
        "model": model_id, "model_family": family,
        "mode": args.mode, "benchmark": args.benchmark, "r": r_eff,
        "n": len(samples), "max_tokens": args.max_tokens, "max_pixels": args.max_pixels,
        "seed": args.seed,
        "fastv_k": args.fastv_k if args.mode == "fastv" else None,
        "pyramid_ratios": ratios if args.mode == "pyramid" else None,
        "pyramid_keep_equiv": (round(keep_equiv, 4)
                               if args.mode == "pyramid" else None),
        "wall_s": round(wall, 3), "req_per_s": round(n_scored / wall, 4) if wall else 0.0,
        "acc": round(acc, 4), "n_answered": n_ok, "n_skipped": n_skip,
        "mean_ptid_len": (round(sum(ptid_counts) / len(ptid_counts), 1)
                          if ptid_counts else 0),
        "load_s": round(load_s, 1),
        "engine": "hf-transformers (eager)", "vllm_note":
            "engine differs from runner (vLLM); efficiency numbers use vLLM",
        "per_sample": per_sample,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[j4] mode={args.mode} r_eff={r_eff} {args.benchmark}: acc={acc:.3f} "
          f"mean_ptid={result['mean_ptid_len']:.0f} wall={wall:.1f}s "
          f"skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
