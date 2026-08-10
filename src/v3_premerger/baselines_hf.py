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
  * pixels   : --max-pixels>0 -> PIL pre-resize each image to <= that pixel
               budget (aspect preserved, edges to multiples of 32) BEFORE the
               processor; ==0 -> native resolution. (Processor kwargs would be
               ignored anyway: transformers 4.57 drops the max_pixels kwarg and
               vLLM V1 ignores engine-level mm_processor_kwargs -- PIL
               pre-resize is the effective enforcement on both harnesses, same
               pixel-budget calibration per family.)
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

Cascade (two-stage, single budget) -- NEW modes for the pre-registered cascade
gate (DECISIONS 2026-07-28):
  --mode pre --r-pre X      : Stage-1 ONLY.  PRE-merger L2 top-kappa on merger-
    input UNITS (X = KEEP fraction of units; query-blind).  The native Patch-
    Merger is per-unit (4 consecutive ViT patches -> 1 token; LayerNorm+MLP act
    on each unit independently), so a kept unit's merged token is BIT-IDENTICAL
    whether it is selected before or after the merger (the runner's M3 swap==pre
    identity) -- we therefore run the FULL native vision stage (captured at the
    language-model boundary, same stub as FastV) and index-select the survivors
    out of the prepared inputs_embeds / position_ids / deepstack rows.  The
    keep-MASK itself is computed exactly like the vLLM runner's pre mode
    (v3_premerger_runner.PreMergerPruner): unit = spatial_merge_size**2 = 4
    consecutive patches, score = feats.float().norm(-1).mean(-1) (mean per-patch
    L2 of the unit), k_i = max(1, round(f_i*X)) per image, top-k per image, ONE
    mask computed from the FIRST merger called (Qwen3-VL: deepstack[0] input,
    i.e. layer-8 features -- the runner's mask_computed_at='deepstack_0';
    Qwen2.5-VL: the single main merger) and applied consistently to the main
    merged tokens AND every deepstack row set (all mergers consume the same
    block-major patch order, verified in Qwen3VLVisionModel.forward).  Each
    surviving image token keeps its NATIVE mrope 3-D coordinate (index_select on
    the native get_rope_index positions).  No FastV.
  --mode cascade --r-pre X --r Y --fastv-k K : Stage-1 pre (keep fraction X)
    -> native merger -> Stage-2 FastV pruning the SURVIVING image tokens after
    LLM layer K, where --r Y is the DROP fraction OF THE STAGE-1 REMAINDER
    (documented convention; NOT of the full grid).  TOTAL image-token keep =
    X*(1-Y)  (modulo per-image rounding): e.g. X=0.5, Y=0.5 -> 25% total;
    X=0.5, Y=0.75 -> 12.5% total.  Degenerate identities (self-tested in
    --dry-check): X=1.0 -> cascade == FastV-alone at Y (stage 1 is the
    identity); Y=0.0 -> cascade == pre-alone at X (stage 2 keeps everything).
    The output "r" field stores the TOTAL drop 1-X*(1-Y) (runner convention);
    "r_pre"/"r_fastv"/"total_keep" carry the stage-wise definition.

RankBridge (cross-stage rank FUSION, single budget) -- the user-directed
2026-08-07 gate; structurally distinct from cascade (fusion over the FULL
candidate set instead of serial pruning):
  --mode rankbridge --r 0.75 --fastv-k 3 [--rb-fuse quota|rrf]
    ALL visual tokens stay in the sequence through LLM layer K (no prior
    pruning).  Per native merger UNIT we cache the pre-merger L2 RANK
    (1 = highest mean-patch L2 within its image; same score as --mode pre,
    first merger called = mask source).  At layer K we read the FastV
    query-conditioned attention score (mean over heads, last query row, image
    columns) -> per-image query rank, and FUSE the two ranks into the final
    keep set with the per-image RBM budget k_i = max(1, round(f_i*(1-r))):
      quota (A): rho*k_i seats reserved for the best pre-ranks, the remaining
        k_i - rho*k_i filled by attention rank among the non-protected units
        (rho=0 -> bit-identical to FastV; rho=1 -> pure pre-rank top-k).
      rrf   (B): reciprocal-rank fusion score
        s_i = 1/(c + r_pre_i) + lambda/(c + r_query_i), keep top k_i.
    Rationale: cascade starves FastV of candidates (query-blind stage-1
    discards degrade stage-2's token set; NO-GO 2026-07-29); RankBridge lets
    FastV rank the FULL set while the protected quota shields dense/OCR units
    whose raw-patch information the merger-distorted attention rank misses.
    Survivors keep their NATIVE mrope coordinates (pruning happens inside the
    LLM, exactly like FastV -- no vllm-mimic needed).  Final post-vision token
    budget == RBM's per-image budget formula, so mean_ptid matches --mode pre
    cells sample-for-sample.

NO GPU is touched at import; the model is loaded only in main().  --dry-check
builds a TINY random-init Qwen2.5-VL on CPU (no weights download, no GPU) and
verifies the manual layer loop reproduces the native forward exactly at r=0,
runs FastV/Pyramid end-to-end with correct keep counts, and validates the
pre-merger mask + cascade degeneracies (X=1 == FastV, Y=0 == pre) + RankBridge
degeneracies (rho=0 == FastV keep set AND hidden, rho=1 == pre-rank top-k).
"""
from __future__ import annotations

import argparse
import json
import math
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
    RankBridge: single entry at layer K; the value is the GLOBAL keep count
               (used only to fire the layer and as a diag reference) -- the
               actual keep set is the per-image fused budget
               sum_i max(1, round(f_i*(1-r))) computed in
               rankbridge_keep_indices.
    Pyramid  : entries at the last layer of bands 0,1,2 (keep round(n_image*
               ratios[1/2/3])); band boundaries = round({.25,.5,.75}*n_layers),
               matching the official layer_list=[L/4, L/2, 3L/4]."""
    plan: dict[int, int] = {}
    if mode in ("fastv", "rankbridge"):
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
# PRE-merger stage (cascade Stage-1 / --mode pre).  Mirrors the vLLM runner's
# PreMergerPruner (v3_premerger_runner.py) EXACTLY on the mask semantics:
#   * units  : `unit` = spatial_merge_size**2 (=4) CONSECUTIVE ViT patches
#              (both PatchMerger families group rows by .view(-1, hidden*unit));
#   * score  : feats.float().norm(-1).mean(-1) -- mean per-patch L2 of the unit
#              (the runner's _score_units(feats, "l2"));
#   * k_i    : max(1, round(full_i * keep_frac)) per image, top-k per image;
#   * source : ONE mask from the FIRST merger called (qwen3vl: deepstack[0]
#              input == layer deepstack_visual_indexes[0] features; qwen2vl:
#              the single main merger), applied to every merger's output rows
#              (all mergers share the same block-major patch sequence, so the
#              same unit mask is consistent across main + deepstack -- verified
#              in transformers Qwen3VLVisionModel.forward and the runner diag
#              mask_computed_at='deepstack_0').
# The merge is per-unit (norm+MLP over each 4-patch group independently) so a
# SURVIVOR'S merged token is identical whether selected pre or post merge
# (runner M3 swap==pre identity) -- hence the implementation runs the FULL
# native vision stage and index-selects survivors out of the captured
# inputs_embeds / position_ids / deepstack (no merger patching needed).
# --------------------------------------------------------------------------- #
def premerger_keep_units(hs: torch.Tensor, grid_thw: torch.Tensor,
                         keep_frac: float, unit: int):
    """hs: [seq, ctx] first-called merger input; grid_thw: [n_img, 3].
    Returns (kept_idx [K] sorted global unit indices, diag dict)."""
    keep_frac = float(keep_frac)
    assert 0.0 < keep_frac <= 1.0, f"--r-pre must be in (0,1] (got {keep_frac})"
    seq = hs.shape[0]
    ctx = hs.shape[-1]
    num_units = seq // unit
    full = (grid_thw.prod(-1) // unit).tolist()
    assert sum(full) == num_units, \
        f"unit count mismatch: grid_thw->{sum(full)} vs hs->{num_units}"
    if keep_frac >= 1.0:
        kept = torch.arange(num_units, device=hs.device)
        return kept, {"n_units_full": num_units, "n_units_kept": num_units,
                      "full_per_image": full, "k_per_image": list(full),
                      "kept_per_image": [list(range(f)) for f in full],
                      "degraded_subunit": False}
    if num_units == 0:
        return (torch.zeros(0, dtype=torch.long, device=hs.device),
                {"n_units_full": 0, "n_units_kept": 0, "full_per_image": full,
                 "k_per_image": [], "degraded_subunit": True})
    feats = hs.reshape(num_units, unit, ctx)
    scores = feats.float().norm(dim=-1).mean(dim=-1)           # runner _score_units l2
    k_per = [max(1, int(round(f * keep_frac))) for f in full]
    keep = torch.zeros(num_units, dtype=torch.bool, device=hs.device)
    kept_per_image = []                                        # runner kept_log format
    off = 0
    for f, k in zip(full, k_per):
        s_i = scores[off:off + f]
        idx = torch.topk(s_i, min(k, f)).indices
        keep[off + idx] = True
        kept_per_image.append(sorted(idx.tolist()))            # LOCAL, sorted
        off += f
    kept = keep.nonzero(as_tuple=False).squeeze(-1)            # original order
    diag = {"n_units_full": num_units, "n_units_kept": int(kept.numel()),
            "full_per_image": full, "k_per_image": k_per,
            "degraded_subunit": False}
    if num_units <= 4096:                                      # diagnostic cap
        diag["kept_per_image"] = kept_per_image                # == runner --save-unit-scores kept_log
    return kept, diag


def apply_premerger(inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
                    deepstack, image_mask_1d: torch.Tensor,
                    kept_units: torch.Tensor):
    """Index-select the FULL prepared tensors down to the surviving pre-merger
    units.  Each surviving image token keeps its native mrope coordinate
    (index_select along the seq axis of the native get_rope_index positions);
    every deepstack row set [n_units_full, H] is sliced by the SAME kept unit
    indices (shared block-major patch order).  Returns (inputs_embeds',
    position_ids', deepstack', image_mask')."""
    img_pos = image_mask_1d.nonzero(as_tuple=False).view(-1)      # [n_units_full]
    assert int(kept_units.numel()) <= int(img_pos.numel()) and (
        kept_units.numel() == 0 or int(kept_units.max()) < int(img_pos.numel())
    ), f"kept units {int(kept_units.numel())} exceed image tokens {int(img_pos.numel())}"
    keep_img = img_pos.index_select(0, kept_units)
    keep_text = (~image_mask_1d).nonzero(as_tuple=False).view(-1)
    keep = torch.cat([keep_text, keep_img]).sort().values
    ie = inputs_embeds.index_select(1, keep)
    pos = position_ids.index_select(-1, keep)
    ds = (None if deepstack is None
          else [e.index_select(0, kept_units.to(e.device)) for e in deepstack])
    return ie, pos, ds, image_mask_1d.index_select(0, keep)


# --------------------------------------------------------------------------- #
# RankBridge (2026-08-07 gate): cross-stage rank FUSION at FastV layer K over
# the FULL candidate set (no prior pruning).  Pre-merger L2 RANKS (query-blind,
# protect dense/OCR detail) are fused with FastV's query-conditioned attention
# RANK (true query relevance) per image; final budget = RBM's per-image formula.
# --------------------------------------------------------------------------- #
def premerger_unit_ranks(hs: torch.Tensor, grid_thw: torch.Tensor, unit: int):
    """Per-unit WITHIN-IMAGE pre-merger L2 rank (1 = highest score).  Same
    score formula as premerger_keep_units (runner _score_units l2: mean
    per-patch L2 of the unit), but returns RANKS instead of a thresholded mask
    so the fusion can use them.  hs: [seq, ctx] first-called merger input;
    grid_thw: [n_img, 3].  Returns (ranks [num_units] long, diag)."""
    seq = hs.shape[0]
    ctx = hs.shape[-1]
    num_units = seq // unit
    full = (grid_thw.prod(-1) // unit).tolist()
    assert sum(full) == num_units, \
        f"unit count mismatch: grid_thw->{sum(full)} vs hs->{num_units}"
    if num_units == 0:
        return (torch.zeros(0, dtype=torch.long, device=hs.device),
                {"full_per_image": []})
    feats = hs.reshape(num_units, unit, ctx)
    scores = feats.float().norm(dim=-1).mean(dim=-1)           # runner _score_units l2
    ranks = torch.empty(num_units, dtype=torch.long, device=hs.device)
    off = 0
    for f in full:
        f = int(f)
        # rank 1 = largest score; deterministic tie-break = lower index first
        order = scores[off:off + f].argsort(descending=True)
        ranks[off:off + f][order] = torch.arange(1, f + 1, device=hs.device)
        off += f
    return ranks, {"full_per_image": full}


def rankbridge_keep_indices(attn_w: torch.Tensor, image_mask: torch.Tensor,
                            pre_ranks: torch.Tensor, units_per_image: list,
                            keep_frac: float, fuse: str = "quota",
                            rho: float = 0.2, rrf_lambda: float = 1.0,
                            rrf_c: float = 60.0):
    """RankBridge final keep mask, computed at FastV layer K over the FULL
    candidate set.  Attention score semantics are FastV's exactly (mean over
    heads, LAST query row, image columns); per-image budget
    k_i = max(1, round(f_i*keep_frac)) (the RBM convention, so the final
    token budget matches --mode pre cells sample-for-sample).
      quota: q_i = min(k_i, round(rho*k_i)) seats reserved for the LOWEST
        pre-ranks (best pre-merger L2), remaining seats filled by attention
        rank among the non-protected.  rho=0 -> identical keep set to FastV
        (self-tested); rho=1 -> pure pre-rank top-k.
      rrf:   score_i = 1/(c + r_pre_i) + lambda/(c + r_query_i); top k_i.
    pre_ranks: [n_image_tokens] within-image pre-ranks aligned with the
    image-token order (identity here: unit j == image token j, no prior
    pruning).  Returns (sorted full-sequence keep indices, diag dict)."""
    L = attn_w.shape[-1]
    dev = attn_w.device
    img_pos = image_mask.nonzero(as_tuple=False).squeeze(-1)     # [n_img]
    n_img = int(img_pos.numel())
    assert pre_ranks.numel() == n_img, \
        f"pre_ranks {pre_ranks.numel()} != image tokens {n_img}"
    assert sum(int(f) for f in units_per_image) == n_img, \
        f"units_per_image {units_per_image} != image tokens {n_img}"
    if n_img == 0:
        return torch.arange(L, device=dev), \
            {"k_per_image": [], "n_protected": 0, "keep_total": 0}
    a = attn_w[0].mean(dim=0)                                    # [L, L] over heads
    qrow = a[-1]                                                 # last query row [L]
    qscores = qrow.index_select(0, img_pos)                      # [n_img]
    keep_img = []
    k_per_image = []
    n_protected = 0
    off = 0
    for f in units_per_image:
        f = int(f)
        k_i = max(1, int(round(f * float(keep_frac))))
        rq = qscores[off:off + f]
        rp = pre_ranks[off:off + f]
        if fuse == "quota":
            q_i = min(k_i, int(round(float(rho) * k_i)))
            if q_i > 0:
                prot = rp.topk(q_i, largest=False).indices       # best pre-ranks
            else:
                prot = torch.zeros(0, dtype=torch.long, device=dev)
            is_prot = torch.zeros(f, dtype=torch.bool, device=dev)
            is_prot[prot] = True
            rest = (~is_prot).nonzero(as_tuple=False).squeeze(-1)
            k_rest = k_i - int(prot.numel())
            if k_rest > 0 and rest.numel() > 0:
                take = rest[rq.index_select(0, rest)
                            .topk(min(k_rest, int(rest.numel()))).indices]
            else:
                take = torch.zeros(0, dtype=torch.long, device=dev)
            sel = torch.cat([prot, take])
            n_protected += int(prot.numel())
        elif fuse == "rrf":
            r_q = rq.argsort(descending=True).argsort() + 1      # 1 = best attn
            score = (1.0 / (float(rrf_c) + rp.float())
                     + float(rrf_lambda) / (float(rrf_c) + r_q.float()))
            sel = score.topk(min(k_i, f)).indices
        else:
            raise ValueError(f"unknown --rb-fuse: {fuse}")
        keep_img.append(img_pos[off:off + f].index_select(0, sel))
        k_per_image.append(int(sel.numel()))
        off += f
    non_img = (~image_mask).nonzero(as_tuple=False).squeeze(-1)
    keep = torch.cat([non_img, *keep_img]).sort().values
    return keep, {"k_per_image": k_per_image, "n_protected": n_protected,
                  "keep_total": int(sum(k_per_image))}


def mimic_vllm_pre_positions(n_text_pre: int, grid_thw, spatial_merge_size: int,
                             n_units_kept: int, n_text_post: int,
                             ref_position_ids: torch.Tensor) -> torch.Tensor:
    """Reproduce the mrope positions the vLLM-0.19 runner assigns in pre mode
    (vllm qwen3_vl.py _get_mrope_input_positions + gpu_model_runner truncation):
    the FULL (t,H,W) grid block is laid down at the image location even
    though only k = n_units_kept placeholder tokens exist there, and the whole
    position array is truncated to the true token count N = T1+k+T2.  Hence the
    k surviving image tokens get the ROW-MAJOR FIRST-k grid coordinates (NOT
    their native sparse coordinates) and the post-image text tokens inherit the
    grid-block continuation (the 'Qwen3 interleaved self-consistent' unfixed
    path the J7 vLLM pre cells ran with -- STATE survey point 1).  Shape ==
    ref_position_ids' row convention ([3,1,N]; if the family's native ids carry
    a 4th packed-text row it is replicated and dropped later by
    _split_mrope_pos, exactly as the native path does)."""
    import numpy as np
    t = int(grid_thw[0, 0]); H = int(grid_thw[0, 1]) // spatial_merge_size
    W = int(grid_thw[0, 2]) // spatial_merge_size
    T1, k, T2 = n_text_pre, n_units_kept, n_text_post
    text_pre = np.broadcast_to(np.arange(T1), (3, T1)).copy()
    grid = np.indices((t, H, W)).reshape(3, -1) + T1
    text_post = (np.broadcast_to(np.arange(T2), (3, T2))
                 + int(grid.max()) + 1)
    arr = np.concatenate([text_pre, grid, text_post], axis=1)[:, :T1 + k + T2]
    pos = torch.from_numpy(arr.astype(np.int64))
    rows = int(ref_position_ids.shape[0]) if ref_position_ids.ndim == 3 else 3
    if rows == 4:
        pos = torch.cat([torch.arange(T1 + k + T2).view(1, -1), pos], dim=0)
    return pos.view(rows, 1, T1 + k + T2)


# --------------------------------------------------------------------------- #
# RBM-OT (Stage B of experiments/rbm_ot_server_task.md, pre-registered
# 2026-08-10): keep the plain-RBM anchor set BIT-IDENTICAL, but transport the
# DROPPED pre-merger units into the anchors via balanced Sinkhorn OT BEFORE
# the model's native nonlinear merger runs.  LOCKED constants: tau=0.05,
# exactly 20 Sinkhorn iterations, cosine cost, row mass 1, equal anchor
# capacity n_drop/n_anchor.  No parameter search permitted after results.
# --------------------------------------------------------------------------- #
RBOT_TAU = 0.05
RBOT_ITERS = 20


def rbot_descriptors(hs_unit: torch.Tensor) -> torch.Tensor:
    """hs_unit: [U, unit, D] -> per-unit descriptor = L2-normalized mean over
    the patch slots (fp32)."""
    x = hs_unit.float().mean(1)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sinkhorn_balanced_plan(cost: torch.Tensor, tau: float = RBOT_TAU,
                           iters: int = RBOT_ITERS) -> torch.Tensor:
    """Balanced entropic-OT plan, stable log-domain, fp32.  cost [n_drop,
    n_anchor] (cosine cost 1 - cos).  Marginals: every dropped row carries
    mass 1; every anchor has equal capacity n_drop/n_anchor.  Returns P
    [n_drop, n_anchor] after exactly `iters` alternating (u, v) updates."""
    n_drop, n_anchor = cost.shape
    dev = cost.device
    log_K = -cost.float() / tau
    log_u = torch.zeros(n_drop, device=dev)
    log_v = torch.zeros(n_anchor, device=dev)
    log_a = torch.zeros(n_drop, device=dev)
    log_b = torch.full((n_anchor,), math.log(n_drop / n_anchor), device=dev)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)
    return torch.exp(log_u[:, None] + log_K + log_v[None, :])


def rbot_plan(hs, grid_thw, keep_frac, unit, tau=RBOT_TAU, iters=RBOT_ITERS):
    """From the FIRST-called merger input (the SAME tensor plain RBM scores):
    RBM anchors (bit-identical to premerger_keep_units) + one balanced
    Sinkhorn plan per image (dropped descriptors -> anchor descriptors, cosine
    cost, NO cross-image transport).  Returns dict(kept, diag_pre, unit,
    imgs=[(drop_idx, anchor_idx, P)] per image, marg_res, tau, iters)."""
    kept, dp = premerger_keep_units(hs, grid_thw, keep_frac, unit)
    U = hs.shape[0] // unit
    desc = rbot_descriptors(hs.reshape(U, unit, hs.shape[-1]))
    full = dp["full_per_image"]
    k_per = dp["k_per_image"]
    kept_set = set(kept.tolist())
    imgs, marg_res = [], 0.0
    off = 0
    for f, k in zip(full, k_per):
        units_i = range(off, off + f)
        anchors = [u for u in units_i if u in kept_set]
        drops = [u for u in units_i if u not in kept_set]
        assert len(anchors) == k, (len(anchors), k)
        if drops:
            D = desc[torch.tensor(drops, device=hs.device)]
            A = desc[torch.tensor(anchors, device=hs.device)]
            P = sinkhorn_balanced_plan(1.0 - D @ A.t(), tau, iters)
            marg_res = max(
                marg_res, float((P.sum(1) - 1.0).abs().max()),
                float((P.sum(0) - len(drops) / len(anchors)).abs().max()))
        else:
            P = torch.zeros(0, len(anchors))
        imgs.append((torch.tensor(drops, dtype=torch.long, device=hs.device),
                     torch.tensor(anchors, dtype=torch.long, device=hs.device),
                     P))
        off += f
    return {"kept": kept, "diag_pre": dp, "imgs": imgs, "unit": unit,
            "marg_res": marg_res, "tau": tau, "iters": iters}


def rbot_apply(hs, plan, unit):
    """Enrich the anchor rows of a merger-input tensor [U*unit, D] with the
    cached plan: for each anchor j and each patch slot p INDEPENDENTLY,
    H'_j,p = (H_j,p + sum_i P_ij H_i,p) / (1 + sum_i P_ij).  Dropped/other
    rows untouched; length and dtype preserved; keep=100% -> bitwise identity.
    Applied on EVERY merger call (deepstack + main) with the SAME plan."""
    U = hs.shape[0] // unit
    H = hs.reshape(U, unit, hs.shape[-1])
    out = H.clone()
    for drops, anchors, P in plan["imgs"]:
        if drops.numel() == 0:
            continue
        Hd = H[drops].float()                       # [n_drop, unit, D]
        W = P.sum(0)                                # [n_anchor]
        contrib = torch.einsum("ia,iud->aud", P.float(), Hd)
        out[anchors] = ((H[anchors].float() + contrib)
                        / (1.0 + W)[:, None, None]).to(hs.dtype)
    return out.reshape(-1, hs.shape[-1])


class MergerTap:
    """Captures the FIRST merger call's input hidden_states (once per sample
    pass) across visual.merger + visual.deepstack_merger_list[*] -- the mask
    source that mirrors the runner's cached-once-per-pass mask.  Install once
    after model load; reset() per sample; .first_hs / .first_tag after the
    captured forward."""

    def __init__(self, visual):
        self.first_hs = None
        self.first_tag = None
        self.call_order = []
        self._handles = []
        targets = [("main", visual.merger)]
        dsl = getattr(visual, "deepstack_merger_list", None)
        if dsl is not None:
            targets += [(f"deepstack_{i}", m) for i, m in enumerate(dsl)]
        for tag, mod in targets:
            self._handles.append(mod.register_forward_pre_hook(
                self._make_hook(tag), with_kwargs=True))

    def _make_hook(self, tag):
        def hook(module, args, kwargs):
            self.call_order.append(tag)
            if self.first_hs is None:
                hs = kwargs.get("hidden_states", args[0] if args else None)
                if hs is not None:
                    self.first_hs = hs.detach()
                    self.first_tag = tag
        return hook

    def reset(self):
        self.first_hs = None
        self.first_tag = None
        self.call_order = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class MergerEnrichTap:
    """RBM-OT enrichment at the merger inputs.  Forward PRE-hooks on
    visual.merger + visual.deepstack_merger_list[*].  Per sample pass: the
    FIRST merger call computes the RBM anchor set + balanced Sinkhorn plans
    from its own input (plan source == plain-RBM mask source, runner parity)
    and the hook swaps in the enriched tensor (SAME shape/dtype, full length
    unchanged -- only anchor rows are rewritten); every subsequent merger call
    reuses the SAME plan on its OWN features (row alignment exact: unit j is
    the same spatial unit for all mergers, so main/deepstack stay aligned).
    Install once after model load; reset(grid_thw) per sample."""

    def __init__(self, visual, keep_frac, unit, tau=RBOT_TAU,
                 iters=RBOT_ITERS):
        self.keep_frac = keep_frac
        self.unit = unit
        self.tau = tau
        self.iters = iters
        self.grid_thw = None
        self.plan = None
        self.first_tag = None
        self.call_order = []
        self._handles = []
        targets = [("main", visual.merger)]
        dsl = getattr(visual, "deepstack_merger_list", None)
        if dsl is not None:
            targets += [(f"deepstack_{i}", m) for i, m in enumerate(dsl)]
        for tag, mod in targets:
            self._handles.append(mod.register_forward_pre_hook(
                self._make_hook(tag), with_kwargs=True))

    def _make_hook(self, tag):
        def hook(module, args, kwargs):
            self.call_order.append(tag)
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None or self.grid_thw is None:
                return None
            if self.plan is None:
                self.plan = rbot_plan(hs.detach(), self.grid_thw,
                                      self.keep_frac, self.unit,
                                      self.tau, self.iters)
                self.first_tag = tag
            new = rbot_apply(hs, self.plan, self.unit)
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": new}
            return (new,) + tuple(args[1:]), kwargs
        return hook

    def reset(self, grid_thw):
        self.grid_thw = grid_thw
        self.plan = None
        self.first_tag = None
        self.call_order = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


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
    rb_diag = None
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
            if mode == "rankbridge":
                rb = cfg["rb"]
                keep, rb_diag = rankbridge_keep_indices(
                    attn_w, image_mask, rb["pre_ranks"], rb["units_per_image"],
                    rb["keep_frac"], rb["fuse"], rb["rho"], rb["rrf_lambda"],
                    rb["rrf_c"])
            else:
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
    if rb_diag is not None:
        diag["rb"] = rb_diag
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
                    choices=["none", "fastv", "pyramid", "pre", "cascade",
                             "rankbridge", "rbmot"])
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
    ap.add_argument("--r-pre", type=float, default=None,
                    help="PRE-merger stage: KEEP fraction of merger-input units "
                         "(top-kappa by mean-patch L2, per image; query-blind). "
                         "REQUIRED for --mode pre/cascade. mode=pre: total keep "
                         "= r_pre. mode=cascade: total keep = r_pre*(1-r), where "
                         "--r is the FastV DROP fraction OF THE STAGE-1 REMAINDER "
                         "(documented convention, not of the full grid). "
                         "Degenerate: r_pre=1.0 -> cascade==FastV; r=0.0 -> "
                         "cascade==pre.")
    ap.add_argument("--mrope", default="vllm-mimic",
                    choices=["vllm-mimic", "native"],
                    help="pre/cascade ONLY: mrope positions of the pruned "
                         "sequence. vllm-mimic (DEFAULT) replicates the vLLM "
                         "runner's scaled-placeholder layout (full grid block "
                         "truncated to the token count -> survivors carry "
                         "row-major first-k grid coords), so HF answers match "
                         "the reference vLLM pre cells; native keeps each "
                         "survivor's original get_rope_index coordinate "
                         "(diagnostic).")
    ap.add_argument("--rb-fuse", default="quota", choices=["quota", "rrf"],
                    help="rankbridge ONLY: fusion of pre-merger L2 rank and "
                         "layer-K attention rank. quota: rho*k_i protected "
                         "seats for best pre-ranks, rest by attention; rrf: "
                         "score=1/(c+r_pre)+lambda/(c+r_query).")
    ap.add_argument("--rb-rho", type=float, default=0.2,
                    help="rankbridge quota ONLY: protected fraction of each "
                         "image's keep budget (0 -> identical to FastV; "
                         "1 -> pure pre-rank top-k).")
    ap.add_argument("--rb-lambda", type=float, default=1.0,
                    help="rankbridge rrf ONLY: weight of the query-rank term.")
    ap.add_argument("--rb-rrf-c", type=float, default=60.0,
                    help="rankbridge rrf ONLY: RRF constant c.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--fastv-k", type=int, default=2,
                    help="FastV prune layer (paper default K=2; attention of this "
                         "layer ranks the image tokens).")
    ap.add_argument("--pyramid-ratios", default="1.0,0.75,0.5,0.25",
                    help="per-band KEEP ratios (4 bands). Official default is "
                         "1.0,0.5,0.25,0.125 (lambda=0.5); we use the fair-budget "
                         "schedule by default.")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help=">0 -> PIL pre-resize each image to <= this pixel "
                         "budget before the processor (iso-token calibration "
                         "per family; processor kwargs are ignored by "
                         "transformers 4.57); 0 -> native resolution (runner "
                         "parity).")
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

    # premerger_keep_units: runner-exact mask semantics (unit = 4 consecutive
    # patches; score = mean per-patch L2; k_i = max(1, round(f_i*keep)); top-k
    # per image; keep_frac=1.0 identity; k floors at 1).
    torch.manual_seed(1)
    ctx = 6
    feats = torch.randn(6, 4, ctx)                 # 6 units
    hs = feats.reshape(24, ctx)
    grid = torch.tensor([[1, 4, 4], [1, 2, 4]])    # units/image = 4, 2
    kept, dp = premerger_keep_units(hs, grid, 0.5, 4)
    sc = feats.float().norm(dim=-1).mean(dim=-1)     # runner _score_units(l2)
    k0, k1 = max(1, int(round(4 * 0.5))), max(1, int(round(2 * 0.5)))   # 2, 1
    want = torch.cat([sc[:4].topk(k0).indices,
                      4 + sc[4:].topk(k1).indices]).sort().values
    assert torch.equal(kept, want), (kept, want)
    assert dp["k_per_image"] == [2, 1] and dp["n_units_kept"] == 3, dp
    kept1, dp1 = premerger_keep_units(hs, grid, 1.0, 4)
    assert kept1.numel() == 6 and dp1["n_units_kept"] == 6   # r_pre=1 identity
    _, dp2 = premerger_keep_units(hs, grid, 0.1, 4)
    assert dp2["k_per_image"] == [1, 1], dp2         # k floors at 1
    print(f"[dry-check]   OK premerger mask (runner semantics: per-img top-k "
          f"{dp['k_per_image']}, keep=1.0 identity, floor@1)")

    # RankBridge: premerger_unit_ranks + fused keep sets (pure functions).
    torch.manual_seed(2)
    hs2 = torch.randn(24, 6)                                   # 6 units x 4 patches
    grid2 = torch.tensor([[1, 4, 4], [1, 2, 4]])               # 4 + 2 units
    ranks2, dr2 = premerger_unit_ranks(hs2, grid2, 4)
    sc2 = hs2.reshape(6, 4, 6).float().norm(dim=-1).mean(dim=-1)  # runner l2 score
    assert dr2["full_per_image"] == [4, 2], dr2
    assert int(ranks2[:4][int(sc2[:4].argmax().item())]) == 1  # rank 1 = best L2
    assert sorted(ranks2[:4].tolist()) == [1, 2, 3, 4] and \
        sorted(ranks2[4:].tolist()) == [1, 2], ranks2
    print("[dry-check]   OK premerger_unit_ranks (within-image, 1=best L2)")

    L3 = 12
    attn3 = torch.rand(1, 3, L3, L3)
    img3 = torch.zeros(L3, dtype=torch.bool); img3[2:8] = True  # 6 image tokens
    pr3 = torch.tensor([1, 2, 3, 4, 5, 6])                     # unit j == token j
    keep_rb0, d_rb0 = rankbridge_keep_indices(
        attn3, img3, pr3, [6], 0.5, fuse="quota", rho=0.0)
    keep_fv3 = rank_keep_indices(attn3, img3, 3)
    assert torch.equal(keep_rb0, keep_fv3), (keep_rb0, keep_fv3)
    assert d_rb0["n_protected"] == 0 and d_rb0["keep_total"] == 3, d_rb0
    keep_rb1, d_rb1 = rankbridge_keep_indices(
        attn3, img3, pr3, [6], 0.5, fuse="quota", rho=1.0)
    got_img = torch.masked_select(keep_rb1, img3.index_select(0, keep_rb1))
    assert torch.equal(got_img, torch.tensor([2, 3, 4])), got_img  # pre top-3
    assert d_rb1["n_protected"] == 3, d_rb1
    keep_rrf, d_rrf = rankbridge_keep_indices(
        attn3, img3, pr3, [6], 0.5, fuse="rrf", rrf_lambda=1.0, rrf_c=60.0)
    assert d_rrf["keep_total"] == 3 and keep_rrf.numel() == 3 + (L3 - 6), d_rrf
    # per-image budget with 2 images: k = [max(1,round(4*.25)), max(1,round(2*.25))]
    img3b = torch.zeros(L3, dtype=torch.bool); img3b[2:8] = True
    pr3b = torch.tensor([3, 1, 2, 6, 1, 2])                    # img1: 4, img2: 2
    keep_q, d_q = rankbridge_keep_indices(
        attn3, img3b, pr3b, [4, 2], 0.25, fuse="quota", rho=0.5)
    assert d_q["k_per_image"] == [1, 1], d_q                   # floor@1 per image
    print("[dry-check]   OK rankbridge fusion (quota rho=0 == FastV keep set; "
          "rho=1 == pre top-k; rrf counts; per-image budget floor@1)")

    # apply_premerger: text untouched, survivors keep their seq positions,
    # deepstack rows sliced by the SAME kept unit indices (row j == unit j).
    Lp, Hp = 10, 5
    Xp = torch.randn(1, Lp, Hp)
    Pp = torch.arange(Lp).view(1, 1, Lp).expand(3, 1, Lp).contiguous().clone()
    imp = torch.zeros(Lp, dtype=torch.bool); imp[2:8] = True    # 6 image tokens
    dsp = [torch.arange(6).float().view(6, 1).expand(6, Hp).clone()]
    keptp = torch.tensor([0, 3, 5])
    iep, posp, ds2p, im2p = apply_premerger(Xp, Pp, dsp, imp, keptp)
    want_cols = torch.cat([torch.tensor([0, 1, 8, 9]),
                           torch.tensor([2, 5, 7])]).sort().values
    assert torch.equal(iep[0], Xp[0].index_select(0, want_cols))
    assert torch.equal(posp[0, 0], want_cols)        # positions preserved
    assert torch.equal(ds2p[0][:, 0], torch.tensor([0., 3., 5.]))
    assert int(im2p.sum()) == 3 and iep.shape[1] == 7
    print("[dry-check]   OK apply_premerger (survivor positions + deepstack "
          "row alignment)")

    # mimic_vllm_pre_positions: full grid block + truncation to T1+k+T2.
    # T1=2 text, grid (t=1,h=4,w=4)->H=W=2 (4 grid cols), k=2, T2=3 -> N=7:
    # text_pre cols (0,1); survivors = row-major first-2 grid coords
    # (2,2,2),(2,2,3); text_post inherits grid continuation (2,3,2),(2,3,3)
    # then its own base max+1=4 -> (4,4,4).
    refp = torch.arange(9).view(1, 1, 9).expand(3, 1, 9).contiguous()
    pm = mimic_vllm_pre_positions(
        2, torch.tensor([[1, 4, 4]]), 2, 2, 3, refp)
    assert pm.shape == (3, 1, 7), pm.shape
    want_cols = torch.tensor([[0, 1, 2, 2, 2, 2, 4],
                              [0, 1, 2, 2, 3, 3, 4],
                              [0, 1, 2, 3, 2, 3, 4]])
    assert torch.equal(pm.view(3, 7), want_cols), pm.view(3, 7)
    pm4 = mimic_vllm_pre_positions(
        2, torch.tensor([[1, 4, 4]]), 2, 2, 3,
        torch.arange(9).view(1, 1, 9).expand(4, 1, 9).contiguous())
    assert pm4.shape == (4, 1, 7) and torch.equal(pm4[1:], want_cols.view(3, 1, 7))
    print("[dry-check]   OK vllm-mimic mrope (grid-block + truncation, "
          "3-row and 4-row conventions)")

    # RBM-OT (Stage B) pure-function gates ----------------------------------
    # (C1) balanced Sinkhorn plan: marginals (row 1, col n_drop/n_anchor)
    # residual < 1e-3, finite.  Uses the locked input distribution: cosine
    # cost 1 - D.A^T over L2-normalized unit descriptors (arbitrary cost
    # matrices are NOT the method's input contract).
    torch.manual_seed(11)
    desc_c1 = torch.nn.functional.normalize(torch.randn(13, 16), dim=-1)
    cost_c1 = 1.0 - desc_c1[4:] @ desc_c1[:4].t()  # 9 dropped x 4 anchors
    P_c1 = sinkhorn_balanced_plan(cost_c1)
    assert torch.isfinite(P_c1).all()
    res_row = float((P_c1.sum(1) - 1.0).abs().max())
    res_col = float((P_c1.sum(0) - 9.0 / 4.0).abs().max())
    assert max(res_row, res_col) < 1e-3, (res_row, res_col)
    print(f"[dry-check]   OK rbmot Sinkhorn marginals (row res {res_row:.2e}, "
          f"col res {res_col:.2e} < 1e-3; tau={RBOT_TAU}, iters={RBOT_ITERS})")

    # (C2) anchors == plain-RBM kept BIT-EXACT; per-image budget = k_i units
    # (= sum(k_i) merged tokens, sum(k_i)*unit pre-merger patch rows).
    torch.manual_seed(12)
    hs_c2 = torch.randn(28, 8)                     # 7 units x 4 patches
    grid_c2 = torch.tensor([[1, 4, 4], [1, 3, 4]])  # 4 + 3 units
    plan_c2 = rbot_plan(hs_c2, grid_c2, 0.5, 4)
    kept_ref, dp_ref = premerger_keep_units(hs_c2, grid_c2, 0.5, 4)
    assert torch.equal(plan_c2["kept"], kept_ref)  # RBM bit-identity
    assert dp_ref["k_per_image"] == [2, 2], dp_ref
    n_anchor_c2 = int(plan_c2["kept"].numel())
    assert n_anchor_c2 == sum(dp_ref["k_per_image"]) == 4, n_anchor_c2
    assert n_anchor_c2 * 4 == 16                   # kept pre-merger patch rows
    print("[dry-check]   OK rbmot anchors == plain-RBM kept bitwise; budget "
          "sum(k_i)=4 merged tokens / 16 pre-merger rows")

    # (C3) each enriched patch slot == specified weighted barycenter; dropped
    # rows untouched.
    H_c2 = hs_c2.reshape(7, 4, 8)
    out_c2 = rbot_apply(hs_c2, plan_c2, 4).reshape(7, 4, 8)
    for drops, anchors, P in plan_c2["imgs"]:
        if drops.numel() == 0:
            continue
        W = P.sum(0)
        for jj, a in enumerate(anchors.tolist()):
            for p in range(4):
                want = ((H_c2[a, p] + (P[:, jj].unsqueeze(-1)
                                       * H_c2[drops, p]).sum(0))
                        / (1.0 + W[jj]))
                assert torch.allclose(out_c2[a, p], want, atol=1e-5), (a, p)
    unit_kept_c2 = torch.zeros(7, dtype=torch.bool)
    unit_kept_c2[plan_c2["kept"]] = True
    rows_drop_c2 = (~unit_kept_c2).repeat_interleave(4)
    assert torch.equal(out_c2.reshape(28, 8)[rows_drop_c2],
                       hs_c2[rows_drop_c2])
    print("[dry-check]   OK rbmot barycenter per patch slot (all 4 slots) + "
          "dropped rows untouched")

    # (C4) keep_frac=1.0 -> bitwise identity; (C5) NO cross-image transport,
    # deterministic repeated output, finite.
    plan_id = rbot_plan(hs_c2, grid_c2, 1.0, 4)
    assert torch.equal(rbot_apply(hs_c2, plan_id, 4), hs_c2)
    assert plan_id["marg_res"] == 0.0
    hs_c3 = hs_c2.clone()
    hs_c3[16:] += 10.0                             # perturb image-2 units only
    plan_c3 = rbot_plan(hs_c3, grid_c2, 0.5, 4)
    out_c3 = rbot_apply(hs_c3, plan_c3, 4)
    out_c2b = rbot_apply(hs_c2, plan_c2, 4)
    assert torch.equal(out_c3[:16], out_c2b[:16])  # image-1 rows unaffected
    assert torch.isfinite(out_c3).all()
    assert torch.equal(rbot_apply(hs_c3, plan_c3, 4), out_c3)
    print("[dry-check]   OK rbmot keep=1 bitwise identity, no cross-image "
          "transport, deterministic, finite")

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

    # (B7) cascade DEGENERATE IDENTITIES (the pre-registered self test).
    cfg0 = {"r": 0.0, "fastv_k": 2, "ratios": [1, .75, .5, .25]}
    cfg5 = {"r": 0.5, "fastv_k": 2, "ratios": [1, .75, .5, .25]}
    # (B7a) Y=0.0 -> cascade == pre-alone: FastV with r=0 keeps EVERY survivor
    # (plan fires but is a no-op) -> hidden identical to the empty plan.
    hid_pre, _, _, _, _ = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "pre", cfg0)
    hid_c0, _, _, _, dc0 = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "fastv", cfg0)
    assert dc0["n_image_kept"] == 6, dc0
    assert torch.allclose(hid_pre, hid_c0, atol=1e-5), \
        f"Y=0 cascade != pre (maxdiff={(hid_pre-hid_c0).abs().max():.2e})"
    print("[dry-check]   OK degenerate Y=0.0: cascade(r_fastv=0) == pre-alone")
    # (B7b) X=1.0 -> apply_premerger is the identity -> cascade == FastV-alone.
    ie1, pos1, _, im1 = apply_premerger(
        X.clone(), P.clone(), [torch.zeros(6, hidden_size)], img.clone(),
        torch.arange(6))
    assert torch.equal(ie1, X) and torch.equal(im1, img), "X=1 must be identity"
    hid_fv, _, _, _, _ = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "fastv", cfg5)
    hid_c1, _, _, _, dc1 = prefill_pruned(m, ie1, pos1, im1, "fastv", cfg5)
    assert dc1["n_image_kept"] == 3, dc1
    assert torch.allclose(hid_fv, hid_c1, atol=1e-5), \
        f"X=1 cascade != FastV (maxdiff={(hid_fv-hid_c1).abs().max():.2e})"
    print("[dry-check]   OK degenerate X=1.0: cascade(r_pre=1) == FastV-alone")
    # (B7c) a real two-stage cut: keep 3 units -> FastV r=0.5 keeps
    # round(3*.5)=2 -> L' = 6 text + 2 = 8, cache cropped, deepstack rows right.
    kept3 = torch.tensor([0, 2, 5])
    ds_emb = [torch.arange(6).float().view(6, 1).expand(6, hidden_size).clone()]
    ie3, pos3, ds3, im3 = apply_premerger(
        X.clone(), P.clone(), ds_emb, img.clone(), kept3)
    assert ie3.shape[1] == 9 and int(im3.sum()) == 3, (ie3.shape, im3.sum())
    assert torch.equal(ds3[0][:, 0], torch.tensor([0., 2., 5.])), "ds rows"
    hid_c3, _, cache3, _, dc3 = prefill_pruned(m, ie3, pos3, im3, "fastv", cfg5)
    assert dc3["n_image_kept"] == 2 and hid_c3.shape[1] == 8, dc3
    assert _cache_len(cache3) == 8, _cache_len(cache3)
    print("[dry-check]   OK cascade two-stage counts: 6 units -> pre 3 -> "
          f"FastV 2 (L 12->9->8, cache@8, fired={dc3['fired']})")

    # (B8) rankbridge DEGENERATE IDENTITY: quota rho=0 == FastV end-to-end
    # (same keep set at layer K -> identical hidden), and rho=0.5 fires the
    # protected-quota path with the right counts (single 6-token image:
    # k=max(1,round(6*.5))=3, protected=round(.5*3)=2).
    cfg_rb = {"r": 0.5, "fastv_k": 2, "ratios": [1, .75, .5, .25],
              "rb": {"pre_ranks": torch.arange(1, 7), "units_per_image": [6],
                     "keep_frac": 0.5, "fuse": "quota", "rho": 0.0,
                     "rrf_lambda": 1.0, "rrf_c": 60.0}}
    hid_rb, _, _, _, drb = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "rankbridge", cfg_rb)
    assert drb["n_image_kept"] == 3, drb
    assert drb["rb"]["n_protected"] == 0, drb
    assert torch.allclose(hid_rb, hid_fv, atol=1e-5), \
        f"rankbridge rho=0 != FastV (maxdiff={(hid_rb-hid_fv).abs().max():.2e})"
    print("[dry-check]   OK rankbridge rho=0 == FastV (hidden identical, "
          f"fired={drb['fired']})")
    cfg_rb["rb"] = {**cfg_rb["rb"], "rho": 0.5}
    hid_rb2, _, _, _, drb2 = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "rankbridge", cfg_rb)
    assert drb2["n_image_kept"] == 3 and drb2["rb"]["n_protected"] == 2, drb2
    cfg_rb["rb"] = {**cfg_rb["rb"], "fuse": "rrf"}
    hid_rb3, _, _, _, drb3 = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "rankbridge", cfg_rb)
    assert drb3["n_image_kept"] == 3, drb3
    print("[dry-check]   OK rankbridge quota rho=0.5 (protected=2/3) + rrf "
          "(kept=3) end-to-end")

    # (B9) DEFERRED RBM (rho=1.0) vs plain RBM (pre): per-image kept indices
    # bit-identical (survivor positions equal), but survivor hidden states
    # NON-identical -- all tokens live through layers 0..K, so anchors are
    # contextualized with tokens that are deleted only at layer K.  Stage-A
    # token-lifetime arm of experiments/rbm_ot_server_task.md.
    keep_frac_def = 0.25                                   # locked keep 25%
    k_def = max(1, int(round(6 * keep_frac_def)))          # 6 units -> 2
    sc_def = X[0, 2:8].float().norm(dim=-1)                # unit==token l2 score
    order_def = sc_def.argsort(descending=True)
    ranks_def = torch.empty(6, dtype=torch.long)
    ranks_def[order_def] = torch.arange(1, 7)
    kept_def = order_def[:k_def].sort().values             # plain-RBM keep set
    X_pre, pos_pre, _, im_pre = apply_premerger(
        X.clone(), P.clone(), None, img.clone(), kept_def)
    hid_pre, pos_pre2, _, _, dpre = prefill_pruned(
        m, X_pre, pos_pre, im_pre, "pre",
        {"r": 0.0, "fastv_k": 2, "ratios": [1, .75, .5, .25]})
    assert dpre["n_image_kept"] == k_def, dpre
    cfg_def = {"r": 0.75, "fastv_k": 2, "ratios": [1, .75, .5, .25],
               "rb": {"pre_ranks": ranks_def, "units_per_image": [6],
                      "keep_frac": keep_frac_def, "fuse": "quota", "rho": 1.0,
                      "rrf_lambda": 1.0, "rrf_c": 60.0}}
    hid_def, pos_def, _, _, ddef = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "rankbridge", cfg_def)
    assert ddef["n_image_kept"] == k_def and \
        ddef["rb"]["n_protected"] == k_def, ddef
    assert ddef["rb"]["k_per_image"] == [k_def], ddef
    assert hid_def.shape == hid_pre.shape == (1, 6 + k_def, hidden_size), \
        (hid_def.shape, hid_pre.shape)
    assert torch.equal(pos_def, pos_pre2), \
        "kept positions (per-image kept indices) must be bit-identical"
    diff_def = (hid_def - hid_pre).abs().max().item()
    assert diff_def > 1e-4, \
        f"deferred survivor hidden must differ from pre (maxdiff={diff_def:.2e})"
    print(f"[dry-check]   OK deferred rho=1 == plain-RBM kept indices "
          f"(k={k_def}, positions equal), survivor hidden NON-identical "
          f"(maxdiff={diff_def:.2e})")

    # (B10) RBM-OT keep=100% end-to-end through the NATIVE merger hooks
    # (tiny model): enriched capture == plain capture bitwise.
    et10 = MergerEnrichTap(m.visual, 1.0, 4)
    pv10 = torch.randn(16, 3 * 2 * 14 * 14)
    grid10 = torch.tensor([[1, 4, 4]])
    ids10 = torch.tensor([[5, 6, 150, 151, 151, 151, 151, 149, 7, 8, 9]])
    mi10 = {"input_ids": ids10, "attention_mask": torch.ones_like(ids10),
            "pixel_values": pv10, "image_grid_thw": grid10}
    ie_nat, pos_nat, _ = capture_prepared_inputs(
        m, {k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in mi10.items()})
    et10.reset(grid10)
    ie_ot, pos_ot, _ = capture_prepared_inputs(
        m, {k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in mi10.items()})
    assert torch.equal(ie_ot, ie_nat) and torch.equal(pos_ot, pos_nat), \
        "keep=100% rbmot must be bitwise identical to native"
    assert et10.plan is not None and int(et10.plan["kept"].numel()) == 4
    assert et10.plan["marg_res"] == 0.0 and et10.call_order == ["main"]
    et10.remove()
    print("[dry-check]   OK rbmot keep=100% == native bitwise (through real "
          "merger pre-hooks)")

    # (B11) MergerEnrichTap plan REUSE across mergers: computed ONCE from the
    # first-called merger input, applied with exact row alignment on every
    # subsequent merger's own features (deepstack/main alignment mechanism).
    import torch.nn as nn
    from types import SimpleNamespace
    torch.manual_seed(13)
    hs11 = torch.randn(28, 8)                      # 7 units x 4 patches
    grid11 = torch.tensor([[1, 4, 4], [1, 3, 4]])
    stub = SimpleNamespace(merger=nn.Identity(),
                           deepstack_merger_list=[nn.Identity()])
    et11 = MergerEnrichTap(stub, 0.5, 4)
    et11.reset(grid11)
    out_ds = stub.deepstack_merger_list[0](hs11)   # first-called merger
    out_main = stub.merger(hs11)                   # reuses the cached plan
    assert et11.call_order == ["deepstack_0", "main"], et11.call_order
    assert et11.first_tag == "deepstack_0"
    assert torch.equal(out_ds, out_main)           # same plan, same input
    unit_kept11 = torch.zeros(7, dtype=torch.bool)
    unit_kept11[et11.plan["kept"]] = True
    rows_drop11 = (~unit_kept11).repeat_interleave(4)
    rows_anch11 = unit_kept11.repeat_interleave(4)
    assert torch.equal(out_main[rows_drop11], hs11[rows_drop11])
    assert (out_main[rows_anch11] - hs11[rows_anch11]).abs().max() > 1e-5
    et11.remove()
    print("[dry-check]   OK rbmot plan reuse: deepstack_0-first, one plan, "
          "both mergers row-aligned (anchors enriched, drops untouched)")

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

    # cascade/pre/rankbridge: tap the merger inputs for the pre-merger L2
    # mask/ranks (first merger called wins: qwen3vl deepstack_0 / qwen2vl main
    # -- runner parity).
    spatial_merge = int(getattr(model.visual, "spatial_merge_size", 2))
    spatial_unit = spatial_merge ** 2
    tap = (MergerTap(model.visual)
           if args.mode in ("pre", "cascade", "rankbridge") else
           MergerEnrichTap(model.visual, 1.0 - args.r, spatial_unit)
           if args.mode == "rbmot" else None)

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
    elif args.mode == "pre":
        if args.r_pre is None:
            raise SystemExit("--mode pre requires --r-pre (unit KEEP fraction; "
                             "e.g. --r-pre 0.25 for pre-alone@25%)")
        keep_equiv, r_eff = None, round(1.0 - args.r_pre, 4)
    elif args.mode == "cascade":
        if args.r_pre is None:
            raise SystemExit("--mode cascade requires --r-pre (Stage-1 unit KEEP "
                             "fraction); --r is the FastV drop of the STAGE-1 "
                             "REMAINDER -> total keep = r_pre*(1-r)")
        keep_equiv = None
        r_eff = round(1.0 - args.r_pre * (1.0 - args.r), 4)
    elif args.mode == "rankbridge":
        if not (0.0 < args.r < 1.0):
            raise SystemExit("--mode rankbridge requires 0 < --r < 1 (TOTAL drop "
                             "fraction; keep=1-r, e.g. --r 0.75 for 25% keep)")
        keep_equiv, r_eff = None, args.r
    elif args.mode == "rbmot":
        if not (0.0 < args.r < 1.0):
            raise SystemExit("--mode rbmot requires 0 < --r < 1 (TOTAL drop "
                             "fraction; keep=1-r, e.g. --r 0.75 for 25% keep)")
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
            elif args.mode in ("pre", "cascade"):
                # STAGE 1: pre-merger L2 top-kappa over merger-input units
                # (mask from the first merger called -- runner parity), then
                # index-select survivors out of the FULL native prepared
                # tensors (kept merged tokens are bit-identical pre/post merge).
                image_mask = (inputs["input_ids"][0] == image_token_id)
                tap.reset()
                inputs_embeds, position_ids, deepstack = capture_prepared_inputs(
                    model, {k: v for k, v in inputs.items()})
                if tap.first_hs is None:
                    raise RuntimeError("merger tap captured nothing (no image?)")
                kept, dpre = premerger_keep_units(
                    tap.first_hs, inputs["image_grid_thw"], args.r_pre,
                    spatial_unit)
                ie, pos, ds, im2 = apply_premerger(
                    inputs_embeds, position_ids, deepstack, image_mask, kept)
                # mrope: default replicates the vLLM runner's scaled-placeholder
                # positions (full grid block + truncation to token count); the
                # survivors then carry row-major first-k grid coords, exactly
                # as the reference vLLM pre cells did.
                if args.mrope == "vllm-mimic":
                    img_run = image_mask.nonzero(as_tuple=False).view(-1)
                    n_full = int(img_run.numel())
                    t1 = int(img_run[0]) if n_full else 0
                    single_span = bool(n_full == 0 or (
                        int(img_run[-1]) - int(img_run[0]) + 1 == n_full))
                    multi_img = (inputs["image_grid_thw"].shape[0] > 1
                                 if "image_grid_thw" in inputs else False)
                    if single_span and not multi_img:
                        t2 = int(image_mask.numel()) - t1 - n_full
                        pos = mimic_vllm_pre_positions(
                            t1, inputs["image_grid_thw"], spatial_merge,
                            int(kept.numel()), t2,
                            position_ids).to(device=position_ids.device)
                        dpre["mrope"] = "vllm-mimic"
                    else:
                        dpre["mrope"] = "native (multi-image/split span)"
                else:
                    dpre["mrope"] = "native"
                # STAGE 2: pre -> no further pruning (empty plan); cascade ->
                # FastV at layer K over the SURVIVING image tokens (--r = drop
                # fraction of the stage-1 remainder).
                stage2 = "pre" if args.mode == "pre" else "fastv"
                gen, diag = generate_pruned(
                    model, ie, pos, im2, stage2, cfg, args.max_tokens,
                    eos_ids, deepstack=ds)
                diag["pre"] = {**dpre, "mask_source": tap.first_tag,
                               "merger_call_order": tap.call_order[:6],
                               "r_pre_keep_frac": args.r_pre,
                               "r_fastv_drop_of_remainder": (
                                   args.r if args.mode == "cascade" else None),
                               "total_keep_target": (
                                   round(args.r_pre, 4) if args.mode == "pre"
                                   else round(args.r_pre * (1.0 - args.r), 4))}
                ans = processor.decode(gen, skip_special_tokens=True).strip()
                ptid = int(diag["L_after"])
            elif args.mode == "rankbridge":
                # FULL visual tokens through layer K; per-unit pre-merger L2 RANKS
                # cached from the first-called merger input; at layer K the
                # fused rank (quota/RRF) picks the final keep set.  Survivors
                # keep NATIVE mrope coordinates (prune inside the LLM == FastV).
                image_mask = (inputs["input_ids"][0] == image_token_id)
                tap.reset()
                inputs_embeds, position_ids, deepstack = capture_prepared_inputs(
                    model, {k: v for k, v in inputs.items()})
                if tap.first_hs is None:
                    raise RuntimeError("merger tap captured nothing (no image?)")
                pre_ranks, drk = premerger_unit_ranks(
                    tap.first_hs, inputs["image_grid_thw"], spatial_unit)
                n_img_tok = int(image_mask.sum())
                assert pre_ranks.numel() == n_img_tok, \
                    f"unit/image-token mismatch: {pre_ranks.numel()} vs {n_img_tok}"
                cfg_rb = dict(cfg)
                cfg_rb["rb"] = {"pre_ranks": pre_ranks,
                                "units_per_image": drk["full_per_image"],
                                "keep_frac": 1.0 - args.r,
                                "fuse": args.rb_fuse, "rho": args.rb_rho,
                                "rrf_lambda": args.rb_lambda,
                                "rrf_c": args.rb_rrf_c}
                gen, diag = generate_pruned(
                    model, inputs_embeds, position_ids, image_mask, "rankbridge",
                    cfg_rb, args.max_tokens, eos_ids, deepstack=deepstack)
                diag["rb"] = {**diag.get("rb", {}), "fuse": args.rb_fuse,
                              "rho": (args.rb_rho
                                      if args.rb_fuse == "quota" else None),
                              "rrf_lambda": (args.rb_lambda
                                             if args.rb_fuse == "rrf" else None),
                              "rrf_c": (args.rb_rrf_c
                                        if args.rb_fuse == "rrf" else None),
                              "keep_frac": round(1.0 - args.r, 4),
                              "mask_source": tap.first_tag}
                ans = processor.decode(gen, skip_special_tokens=True).strip()
                ptid = int(diag["L_after"])
            elif args.mode == "rbmot":
                # RBM-OT: anchors == plain RBM bit-identical (same first-called
                # merger input, same L2 keep rule); dropped units transported
                # into the anchors as balanced Sinkhorn barycenters BEFORE the
                # native merger (MergerEnrichTap pre-hooks; plan cached once
                # per pass, reused on every merger's own features).  Locked
                # tau/iters live in the constants -- no CLI knobs by design.
                image_mask = (inputs["input_ids"][0] == image_token_id)
                tap.reset(inputs["image_grid_thw"])
                inputs_embeds, position_ids, deepstack = capture_prepared_inputs(
                    model, {k: v for k, v in inputs.items()})
                if tap.plan is None:
                    raise RuntimeError("rbmot tap saw no merger call (no image?)")
                plan = tap.plan
                kept, dpre = plan["kept"], plan["diag_pre"]
                ie, pos, ds, im2 = apply_premerger(
                    inputs_embeds, position_ids, deepstack, image_mask, kept)
                if args.mrope == "vllm-mimic":
                    img_run = image_mask.nonzero(as_tuple=False).view(-1)
                    n_full = int(img_run.numel())
                    t1 = int(img_run[0]) if n_full else 0
                    single_span = bool(n_full == 0 or (
                        int(img_run[-1]) - int(img_run[0]) + 1 == n_full))
                    multi_img = (inputs["image_grid_thw"].shape[0] > 1
                                 if "image_grid_thw" in inputs else False)
                    if single_span and not multi_img:
                        t2 = int(image_mask.numel()) - t1 - n_full
                        pos = mimic_vllm_pre_positions(
                            t1, inputs["image_grid_thw"], spatial_merge,
                            int(kept.numel()), t2,
                            position_ids).to(device=position_ids.device)
                        dpre["mrope"] = "vllm-mimic"
                    else:
                        dpre["mrope"] = "native (multi-image/split span)"
                else:
                    dpre["mrope"] = "native"
                gen, diag = generate_pruned(
                    model, ie, pos, im2, "pre", cfg, args.max_tokens,
                    eos_ids, deepstack=ds)
                diag["rbmot"] = {"tau": plan["tau"], "iters": plan["iters"],
                                 "marg_res": round(plan["marg_res"], 6),
                                 "keep_frac": round(1.0 - args.r, 4),
                                 "n_units_full": dpre["n_units_full"],
                                 "n_anchor": int(kept.numel()),
                                 "n_drop": dpre["n_units_full"] - int(kept.numel()),
                                 "mask_source": tap.first_tag,
                                 "merger_call_order": tap.call_order[:6],
                                 "mrope": dpre["mrope"]}
                ans = processor.decode(gen, skip_special_tokens=True).strip()
                ptid = int(diag["L_after"])
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
        if "pre" in diag:                                      # pre/cascade diag
            rec["pre"] = diag["pre"]                           # incl. kept_per_image
        if "rb" in diag:                                       # rankbridge diag
            rec["rb"] = diag["rb"]
        if "rbmot" in diag:                                    # rbmot diag
            rec["rbmot"] = diag["rbmot"]
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
        "fastv_k": (args.fastv_k if args.mode in ("fastv", "cascade",
                                                  "rankbridge") else None),
        "r_pre": args.r_pre if args.mode in ("pre", "cascade") else None,
        "r_fastv": args.r if args.mode == "cascade" else None,
        "total_keep": (round(args.r_pre, 4) if args.mode == "pre" else
                       round(args.r_pre * (1.0 - args.r), 4)
                       if args.mode == "cascade" else
                       round(1.0 - args.r, 4)
                       if args.mode in ("rankbridge", "rbmot") else None),
        "rb_fuse": args.rb_fuse if args.mode == "rankbridge" else None,
        "rb_rho": (args.rb_rho if args.mode == "rankbridge"
                   and args.rb_fuse == "quota" else None),
        "rb_lambda": (args.rb_lambda if args.mode == "rankbridge"
                      and args.rb_fuse == "rrf" else None),
        "rb_rrf_c": (args.rb_rrf_c if args.mode == "rankbridge"
                     and args.rb_fuse == "rrf" else None),
        "rbmot_tau": RBOT_TAU if args.mode == "rbmot" else None,
        "rbmot_iters": RBOT_ITERS if args.mode == "rbmot" else None,
        "r_convention": ("r = TOTAL drop (runner convention); cascade --r is "
                         "the FastV drop of the STAGE-1 remainder, total keep "
                         "= r_pre*(1-r); rankbridge total keep = 1-r with the "
                         "per-image budget max(1, round(f_i*(1-r)))"),
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
