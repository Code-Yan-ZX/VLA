#!/usr/bin/env python
"""Q-RBM E1 -- Causal-Importance Gate driver (pre-registered, notes/qrbm_e1_plan.md).

For each sample (image + question + GT-answer set, frozen Qwen3-VL) this driver
measures a CAUSAL utility per raster-order contiguous spatial BLOCK of
PRE-merger merge units: how much removing the block's units at the merger input
(drop-out -- slice the unit's 4 rows, bit-identical to how RBM prunes; NOT
zero-embedding) lowers the teacher-forced log-likelihood of the canonical GT
answer a* (chosen ONCE on the full pass as the accepted GT answer with max LL).

Per block it also records the mean of four signals captured on the FULL pass:
  pre_l2 : per-unit mean-patch L2 of the merger input (_score_units("l2"))
  post_l2: per-unit L2 of the POST-merge tokens (postmerger_keep_tokens rows)
  fastv  : layer-K attention, mean over heads, LAST-PROMPT-query row over the
           image columns (rank_keep_indices score semantics)
  qsim   : max_t cos(merger(unit), embed(q_t)) (the runner J5 query feature)

Reuses baselines_hf.py wholesale (Sample/load_subset/build_inputs,
capture_prepared_inputs, MergerTap, apply_premerger, the manual causal prefill
machinery _layer_step/_cache_* /_split_mrope_pos, and the native decoder/lm_head
forward path) -- the harness is NOT modified.  The qsim computation is ported
from v3_premerger_runner.py (_find_embed_tokens/_qa_tokenize_question +
_qa_blend_scores semantics) so no new architecture is needed.

CLI (mirrors the repo argparse style):
  python src/v3_premerger/e1_causal_import.py --benchmark {textvqa|docvqa|gqa|ocrbench}
      --split {dev|heldout} [--n N] [--g G] [--fastv-k K] [--out DIR] [--seed 0]

Splits:
  dev     : runs/merger_repr/dev_{bench}_64.jsonl
  heldout : eval/subsets/e1_heldout_{bench}_64.jsonl

Output: one JSON line per sample -> {out}/{split}_{bench}.jsonl
  {"id": str, "bench": str, "n_units": int, "g": int, "a_star": str,
   "ll_full": float,
   "blocks": [{"g": int, "unit_idx": [int,...], "pre_l2": float, "post_l2": float,
               "fastv": float, "qsim": float, "causal_util": float}, ...]}
  causal_util = u_g = ll_full(a*) - ll_without-g(a*) (positive = block removal
  lowers the GT-answer LL = causally important).

Resumable: samples whose id is already in the out file are skipped; every line
is flushed after each sample (appendable across OOM-restarts).

Per-sample forward accounting (for the smoke s/pass measurement):
  1 + K + G LLM prefills -- 1 prompt-only full pass (layer-K attention + all
  signals) + K accepted-answer prefill passes over [prompt + answer] (a* =
  argmax LL; ll_full = LL(a*), the SAME code path as every occlusion) + G
  occlusion prefills over [pruned prompt + a*], +1 if --null-check (drop
  nothing -> u=0 verification).  Every LL is computed by ll_from_prefill, so
  a* / ll_full / u_g are fully consistent (no cross-path bf16 ambiguity).
  The plan's "G+1" counted only the occlusion budget; K is the accepted-answer
  count (TextVQA multi-answer K~2-10, GQA K=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import baselines_hf as BH  # harness reuse (no modification of baselines_hf.py)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BENCHES = ["textvqa", "docvqa", "gqa", "ocrbench"]
DEFAULT_G = 8
DEFAULT_FASTV_K = 3          # matches the existing FastV gate cells (gateC --fastv-k 3)
# gateC calibration: docvqa capped at 600k px (HF-constrained); others native (0).
BENCH_MP = {"docvqa": 600000}
NULL_TOL = 1e-3              # null-occlusion |ll_full - ll_with-all| must be below this


# --------------------------------------------------------------------------- #
# Pure helpers (self-testable on CPU, no model / GPU required)
# --------------------------------------------------------------------------- #
def block_partition(n_units: int, g: int) -> list[list[int]]:
    """Split [0..n_units) into g contiguous raster-order chunks (blocks), as
    even as possible (sizes differ by <= 1).  Returns [] for n_units<=0; the
    block count is capped at n_units (a unit is the atomic occlusion item)."""
    if n_units <= 0:
        return []
    g = max(1, int(g))
    if g > n_units:
        g = n_units
    bounds = sorted({int(round(i * n_units / g)) for i in range(g + 1)})
    return [list(range(bounds[i], bounds[i + 1]))
            for i in range(len(bounds) - 1)]


def accepted_answers(gt: str) -> list[str]:
    """Unique non-empty GT answers, split on ';' (TextVQA/DocVQA/OCRBench are
    semicolon-joined multi-answer; GQA is single -> naturally one element)."""
    out = []
    for a in str(gt).split(";"):
        a = a.strip()
        if a and a not in out:
            out.append(a)
    return out


def _answer_positions(pos_base: int, m: int, dtype, device) -> torch.Tensor:
    """mrope positions [3,1,m] for m answer text tokens: all three dims carry
    the continuing text position (pos_base + 1 + j), matching how the harness
    feeds generated text tokens (baselines_hf.generate_pruned)."""
    cols = torch.arange(pos_base + 1, pos_base + 1 + m,
                        dtype=dtype, device=device).unsqueeze(0)  # [1,m]
    return cols.repeat(3, 1, 1)


# --------------------------------------------------------------------------- #
# qsim port from v3_premerger_runner.py (module top imports vllm -> cannot
# import the runner; these two helpers + the qsim score are copied verbatim
# in behavior, adjusted for the HF model object).
# --------------------------------------------------------------------------- #
def _find_embed_tokens(model):
    """Locate the LLM word embedding (port of runner._find_embed_tokens)."""
    import torch.nn as nn
    candidates = [
        "model.language_model.model.embed_tokens",   # qwen3vl HF top-level wrapper
        "language_model.model.embed_tokens",         # qwen2.5vl
        "model.language_model.embed_tokens",
        "language_model.embed_tokens",
        "model.model.embed_tokens",
        "model.embed_tokens",
        "embed_tokens",
    ]

    def _ok(m):
        w = getattr(m, "weight", None)
        return (w is not None and getattr(w, "ndim", 0) == 2
                and w.shape[0] >= 100000)            # vocab-scale rows

    for path in candidates:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and _ok(obj):
            return obj, path
    best, best_path = None, None
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "embed_tokens" and _ok(mod):
            if best is None or mod.weight.shape[0] > best.weight.shape[0]:
                best, best_path = mod, name
    return (best, f"named_modules:{best_path}") if best is not None else (None, "NOT_FOUND")


def _qa_tokenize_question(tokenizer, question: str) -> list:
    try:
        return list(tokenizer(question, add_special_tokens=False)["input_ids"])
    except Exception:
        return list(tokenizer.encode(question, add_special_tokens=False))


def question_embed(embed, tokenizer, question):
    """Normalized [nqt, H] question word-embedding tokens (runner._qa_precompute
    _q_embed).  Returns None when the question tokenizes to nothing."""
    ids = _qa_tokenize_question(tokenizer, question)
    nv = int(embed.weight.shape[0])
    if ids:
        ids = [i for i in ids if 0 <= i < nv]
    if not ids:
        return None
    with torch.no_grad():
        e = embed(torch.as_tensor(ids, dtype=torch.long,
                                  device=embed.weight.device)).float()
    return F.normalize(e, dim=-1).contiguous()


# --------------------------------------------------------------------------- #
# LLM forward path (reuses baselines_hf's manual causal prefill machinery)
# --------------------------------------------------------------------------- #
def full_prefill(model, inputs_embeds, position_ids, image_mask_1d,
                 deepstack=None, fastv_k=None):
    """Full causal prefill, NO pruning -- mirrors baselines_hf.prefill_pruned
    minus the rank/prune steps.  Returns (hidden [1,L,H] post-norm, cache,
    image_mask, attn_w_or_None).  When fastv_k is given, captures the softmaxed
    attention [1,heads,L,L] of that decoder layer (eager)."""
    from transformers import DynamicCache
    LM = model.model.language_model
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    position_ids = BH._split_mrope_pos(position_ids)
    hidden = inputs_embeds
    image_mask = image_mask_1d.clone()
    img_ord = image_mask.cumsum(0) - 1
    pos_emb = LM.rotary_emb(hidden, position_ids)
    n_deepstack = len(deepstack) if deepstack is not None else 0
    attn_mask = BH.make_causal_mask(int(hidden.shape[1]), device, dtype)
    cache = DynamicCache(config=LM.config)
    attn_w = None
    for idx, layer in enumerate(LM.layers):
        need = (fastv_k is not None and idx == fastv_k)
        hidden, a = BH._layer_step(layer, hidden, attn_mask, pos_emb, cache, need)
        # Qwen3-VL deepstack: native adds visual features after the first
        # n_deepstack layers at the image positions (replay, as prefill_pruned).
        if idx < n_deepstack:
            sel = img_ord[image_mask]
            emb = deepstack[idx].to(device=device, dtype=hidden.dtype)
            emb = emb.index_select(0, sel)
            hidden[:, image_mask] = hidden[:, image_mask] + emb
        if need and a is not None:
            attn_w = a
    hidden = LM.norm(hidden)
    return hidden, cache, image_mask, attn_w


def answer_token_ids(tokenizer, answer: str) -> list[int]:
    try:
        return list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    except Exception:
        return list(tokenizer.encode(answer, add_special_tokens=False))


def _scored_ll(logits_rows: torch.Tensor, ans_ids: list[int], device) -> float:
    """logits_rows [m, V] at the m positions predicting ans_ids[0..m-1];
    teacher-forced log-likelihood = sum of log_softmax scores (nats)."""
    ans_t = torch.as_tensor(ans_ids, dtype=torch.long, device=device)
    return float(F.log_softmax(logits_rows, dim=-1)
                 .gather(1, ans_t[:, None]).sum().item())


def ll_from_prefill(model, inputs_embeds, position_ids, image_mask, deepstack,
                    ans_ids, pos_base, device, dtype, kept_units=None) -> float:
    """Teacher-forced LL of an answer via ONE full causal prefill over
    [prompt + answer] -- the exact path used by the occlusion passes, so every
    causal utility compares like-for-like.  Occlusion (kept_units given) drops
    the complement at the merger input via apply_premerger (drop-out, the same
    slicing RBM pruning uses).  pos_base is the FULL-prompt last-text position,
    passed in so full + occlusion passes are position-identical.  Returns
    (ll, n_prompt_after)."""
    if kept_units is not None:
        inputs_embeds, position_ids, deepstack, image_mask = BH.apply_premerger(
            inputs_embeds, position_ids, deepstack, image_mask, kept_units)
    m = len(ans_ids)
    ans_emb = model.get_input_embeddings()(
        torch.as_tensor(ans_ids, dtype=torch.long, device=device)).unsqueeze(0)
    ie_full = torch.cat([inputs_embeds, ans_emb], dim=1)
    pos_full = torch.cat([position_ids,
                          _answer_positions(pos_base, m,
                                            position_ids.dtype, device)],
                         dim=-1)
    im_full = torch.cat([image_mask,
                         torch.zeros(m, dtype=torch.bool, device=image_mask.device)])
    hidden, _, _, _ = full_prefill(model, ie_full, pos_full, im_full,
                                   deepstack=deepstack, fastv_k=None)
    logits = model.lm_head(hidden)
    n_prompt = int(inputs_embeds.shape[1])
    scores = logits[0, n_prompt - 1:n_prompt + m - 1]     # [m, V]
    return _scored_ll(scores, ans_ids, device), n_prompt


def _pos_base(position_ids: torch.Tensor, image_mask: torch.Tensor) -> int:
    """Last prompt TEXT position (mrope): the answer text tokens continue the
    text counter at pos_base+1.  Computed once per sample on the FULL prompt so
    full + occlusion passes are position-identical (n_text never changes)."""
    return int(position_ids[:, :, ~image_mask].max())


# --------------------------------------------------------------------------- #
# Per-sample causal-importance computation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def process_sample(model, processor, embed, tap, sample, bench, g, fastv_k,
                   max_pixels, device, dtype, null_check):
    from PIL import Image
    t0 = time.perf_counter()
    image = Image.open(sample.image).convert("RGB")
    inputs = BH.build_inputs(processor, image, sample.question, max_pixels, device)
    image_mask = (inputs["input_ids"][0] == image_token_id(model))

    # -- capture the native vision stage (merger input + post-merge embeds) ----
    tap.reset()
    inputs_embeds, position_ids, deepstack = BH.capture_prepared_inputs(
        model, {k: v for k, v in inputs.items()})
    if tap.first_hs is None:
        raise RuntimeError("merger tap captured nothing (no image?)")
    position_ids = BH._split_mrope_pos(position_ids)
    spatial_merge = int(getattr(model.visual, "spatial_merge_size", 2))
    unit = spatial_merge ** 2
    grid_thw = inputs["image_grid_thw"]
    num_units = int(grid_thw.prod(-1)) // unit
    if num_units != tap.first_hs.shape[0] // unit or num_units != int(image_mask.sum()):
        raise RuntimeError(
            f"unit-count mismatch: grid_thw->{num_units}, merger-input->"
            f"{tap.first_hs.shape[0] // unit}, image_mask->{int(image_mask.sum())}")
    img_pos = image_mask.nonzero(as_tuple=False).view(-1)       # [num_units]
    ctx = tap.first_hs.shape[-1]

    # -- per-unit signals (all from the full pass, no extra forwards) ----------
    feats = tap.first_hs.reshape(num_units, unit, ctx)
    pre_l2 = feats.float().norm(dim=-1).mean(dim=-1)            # _score_units l2
    post_l2 = inputs_embeds[0].index_select(0, img_pos).float().norm(dim=-1)
    # qsim: max_t cos(merger(unit), embed(q_t)) -- runner _qa_blend_scores.
    merger_out = model.visual.merger(tap.first_hs)              # [num_units, H]
    m_norm = F.normalize(merger_out.float(), dim=-1)
    qe = question_embed(embed, processor.tokenizer, sample.question)
    qsim = (m_norm @ qe.t()).max(dim=-1).values if qe is not None \
        else torch.zeros(num_units, device=device)

    # -- full pass: prompt-only prefill (layer-K attention for the fastv signal)
    _, _, _, attn_w = full_prefill(
        model, inputs_embeds, position_ids, image_mask, deepstack=deepstack,
        fastv_k=fastv_k)
    if attn_w is None:
        raise RuntimeError(f"layer-{fastv_k} attention not captured")
    a = attn_w[0].mean(dim=0)                                   # [L, L] over heads
    qrow = a[-1]                                                # last (prompt) row
    fastv = qrow.index_select(0, img_pos)                       # per-unit
    pos_base = _pos_base(position_ids, image_mask)

    # -- choose a* = accepted GT answer with max teacher-forced LL, ALL via the
    #    prefill path (ll_from_prefill = the EXACT code path the occlusion
    #    passes use).  This guarantees a* / ll_full / every u_g are computed
    #    identically -- no cross-path (decode vs prefill) bf16 ambiguity, which
    #    measured ~0.3-0.5 nats on long OCRBench answers and could flip a* for
    #    near-tied answers.  ll_full = LL(a*) from the SAME call.
    answers = accepted_answers(sample.gt)
    ll_per_answer = []
    ids_per_answer = []
    for ans in answers:
        ids = answer_token_ids(processor.tokenizer, ans)
        if not ids:
            ll_per_answer.append(float("-inf")); ids_per_answer.append([]); continue
        ll, _ = ll_from_prefill(
            model, inputs_embeds, position_ids, image_mask, deepstack,
            ids, pos_base, device, dtype)
        ll_per_answer.append(ll); ids_per_answer.append(ids)
    best = max(range(len(answers)), key=lambda i: ll_per_answer[i])
    a_star = answers[best]
    a_star_ids = ids_per_answer[best]
    if not a_star_ids:
        raise RuntimeError("a* tokenizes to empty")
    ll_full = ll_per_answer[best]

    # -- occlusion passes: drop one raster block per forward -------------------
    blocks = block_partition(num_units, g)
    g_eff = len(blocks)
    u_g = {}
    for gi, uidx in enumerate(blocks):
        drop = torch.zeros(num_units, dtype=torch.bool, device=device)
        drop[torch.as_tensor(uidx, dtype=torch.long, device=device)] = True
        kept = (~drop).nonzero(as_tuple=False).view(-1)
        ll_wo, _ = ll_from_prefill(
            model, inputs_embeds, position_ids, image_mask, deepstack,
            a_star_ids, pos_base, device, dtype, kept_units=kept)
        u_g[gi] = ll_full - ll_wo

    # -- null check: drop nothing -> u == 0, LL identical to full --------------
    null_u = None
    if null_check:
        kept_all = torch.arange(num_units, device=device)
        ll_null, _ = ll_from_prefill(
            model, inputs_embeds, position_ids, image_mask, deepstack,
            a_star_ids, pos_base, device, dtype, kept_units=kept_all)
        null_u = abs(ll_full - ll_null)
        if null_u > NULL_TOL:
            raise RuntimeError(
                f"null occlusion mismatch {sample.id}: ll_full={ll_full:.6f} "
                f"ll_with-all={ll_null:.6f} |d|={null_u:.2e}")

    # -- assemble the record ---------------------------------------------------
    rec = {"id": str(sample.id), "bench": bench, "n_units": num_units,
           "g": g_eff, "a_star": a_star, "ll_full": round(ll_full, 6),
           "blocks": []}
    for gi, uidx in enumerate(blocks):
        uidx_t = torch.as_tensor(uidx, dtype=torch.long, device=device)
        rec["blocks"].append({
            "g": gi,
            "unit_idx": uidx,
            "pre_l2": round(float(pre_l2[uidx_t].mean()), 6),
            "post_l2": round(float(post_l2[uidx_t].mean()), 6),
            "fastv": round(float(fastv[uidx_t].mean()), 6),
            "qsim": round(float(qsim[uidx_t].mean()), 6),
            "causal_util": round(u_g[gi], 6),
        })
    n_passes = 1 + len(answers) + g_eff + (1 if null_check else 0)
    wall = time.perf_counter() - t0
    print(f"[e1] {sample.id} n_units={num_units} g={g_eff} n_ans={len(answers)} "
          f"passes={n_passes} s/pass={wall / max(1, n_passes):.3f} "
          f"ll_full={ll_full:.4f} a_star={a_star!r}"
          + (f" null|d|={null_u:.2e}" if null_u is not None else ""), flush=True)
    return rec


def image_token_id(model) -> int:
    return int(model.config.image_token_id)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def split_source(bench: str, split: str) -> str:
    if split == "dev":
        return os.path.join(REPO, "runs", "merger_repr", f"dev_{bench}_64.jsonl")
    return os.path.join(REPO, "eval", "subsets", f"e1_heldout_{bench}_64.jsonl")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Q-RBM E1 causal-importance gate (per-block occlusion delta-LL)")
    ap.add_argument("--benchmark", default=None, choices=BENCHES)
    ap.add_argument("--split", default=None, choices=["dev", "heldout"])
    ap.add_argument("--n", type=int, default=None,
                    help="max samples (default: all in the split file)")
    ap.add_argument("--g", type=int, default=DEFAULT_G,
                    help="number of raster-order contiguous spatial blocks "
                         "(default %(default)s)")
    ap.add_argument("--fastv-k", type=int, default=DEFAULT_FASTV_K,
                    help="FastV layer whose attention ranks image tokens "
                         "(default %(default)s, matches the gate FastV cells)")
    ap.add_argument("--out", default=None,
                    help="output DIR (file: {out}/{split}_{bench}.jsonl; "
                         "default runs/e1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="override the per-benchmark pixel cap (docvqa 600000, "
                         "others native 0)")
    ap.add_argument("--null-check", action="store_true",
                    help="extra occlusion pass dropping NOTHING; asserts "
                         "u=0 / LL identical to full (plan smoke check)")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only structural self-test (pure functions + the "
                         "apply_premerger occlusion slicing identity); no model")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# CPU-only structural self-test (no GPU, no model weights)
# --------------------------------------------------------------------------- #
def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"[self-test][FAIL] {msg}", flush=True)

    # block_partition: coverage, contiguity, balance, count
    for n in (0, 1, 3, 8, 9, 10, 64, 1000):
        for g in (1, 2, 3, 7, 8, 12, 1000):
            bl = block_partition(n, g)
            if n == 0:
                check(bl == [], f"block_partition(0,{g}) empty")
                continue
            flat = [u for b in bl for u in b]
            check(flat == list(range(n)),
                  f"block_partition({n},{g}) covers all units in order")
            check(len(bl) == min(g, n), f"block_partition({n},{g}) block count")
            for b in bl:
                check(bool(b) and all(b[i] < b[i + 1] for i in range(len(b) - 1)),
                      f"block_partition({n},{g}) blocks contiguous increasing")
            sizes = sorted(len(b) for b in bl)
            check(sizes[-1] - sizes[0] <= 1, f"block_partition({n},{g}) balanced")

    # accepted_answers
    check(accepted_answers("a;b; a ;b;c") == ["a", "b", "c"], "accepted_answers dedupe")
    check(accepted_answers("single") == ["single"], "accepted_answers single")
    check(accepted_answers("  ;;  ") == [], "accepted_answers empty")

    # _answer_positions
    pa = _answer_positions(17, 3, torch.long, "cpu")
    check(tuple(pa.shape) == (3, 1, 3) and pa[0, 0].tolist() == [18, 19, 20],
          "_answer_positions values/shape")

    # apply_premerger occlusion slicing identity on random tensors (CPU):
    # dropping a block must remove exactly that block's post-merge rows and
    # leave every survivor's row BIT-IDENTICAL (per-unit merger independence --
    # the property that makes pre-merger drop-out == post-capture index-select).
    torch.manual_seed(0)
    H, ctx, unit = 16, 8, 4
    n_units = 8
    n_text = 6
    # image_mask marks the POST-MERGE image rows (1:1 with pre-merger units),
    # as in the real harness (baselines_hf main / apply_premerger).
    L = n_text + n_units
    ie = torch.randn(1, L, H)
    pos = torch.arange(L, dtype=torch.long).repeat(3, 1).unsqueeze(1)
    im = torch.zeros(L, dtype=torch.bool)
    im[n_text:] = True
    ds = [torch.randn(n_units, H) for _ in range(2)]
    bl = block_partition(n_units, 2)
    dropped = bl[0]
    kept = torch.tensor([u for u in range(n_units) if u not in set(dropped)])
    ie2, pos2, ds2, im2 = BH.apply_premerger(ie, pos, ds, im, kept)
    img_pos = im.nonzero(as_tuple=False).view(-1)
    kept_global = img_pos.index_select(0, kept)
    check(int(im2.sum()) == n_units - len(dropped), "occlusion row count")
    check(torch.equal(ie2[0, im2], ie[0, kept_global]), "survivor rows bit-identical")
    check(all(torch.equal(a.index_select(0, kept), b)
              for a, b in zip(ds, ds2)), "deepstack rows sliced by unit idx")
    check(torch.equal(pos2[:, 0, im2], pos[:, 0, kept_global]),
          "position_ids sliced with image rows")
    # dropping NOTHING == identity
    ie3, pos3, ds3, im3 = BH.apply_premerger(ie, pos, ds, im,
                                             torch.arange(n_units, dtype=torch.long))
    check(torch.equal(ie3, ie) and torch.equal(pos3, pos) and torch.equal(im3, im),
          "apply_premerger(keep all) is the identity")

    if ok:
        print("[self-test] OK -- pure functions + occlusion slicing identity pass",
              flush=True)
        return 0
    print("[self-test] FAIL", flush=True)
    return 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.benchmark or not args.split:
        raise SystemExit("--benchmark and --split are required (unless --self-test)")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise SystemExit("E1 requires a GPU (--self-test is the CPU-only path)")
    dtype = torch.bfloat16
    max_pixels = args.max_pixels if args.max_pixels is not None \
        else BENCH_MP.get(args.benchmark, 0)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    load_s = time.perf_counter() - t0
    embed, embed_path = _find_embed_tokens(model)
    if embed is None:
        raise SystemExit("LLM word-embedding not found -- qsim would be "
                         "disabled; refusing to run")

    src = split_source(args.benchmark, args.split)
    samples = BH.load_subset(src)
    if args.n is not None:
        samples = samples[:args.n]
    out_dir = args.out or os.path.join(REPO, "runs", "e1")
    out_path = os.path.join(out_dir, f"{args.split}_{args.benchmark}.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    # resumable: skip ids already present in the out file
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(str(json.loads(line)["id"]))
                    except Exception:
                        pass

    tap = BH.MergerTap(model.visual)
    n_skip = n_written = 0
    for i, s in enumerate(samples):
        if str(s.id) in done_ids:
            n_skip += 1
            continue
        try:
            rec = process_sample(model, processor, embed, tap, s, args.benchmark,
                                 args.g, args.fastv_k, max_pixels, device, dtype,
                                 args.null_check)
        except Exception as e:
            print(f"[e1] sample {s.id} FAILED ({type(e).__name__}: "
                  f"{str(e)[:200]})", file=sys.stderr, flush=True)
            continue
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        done_ids.add(str(s.id))
        n_written += 1
    tap.remove()
    print(f"[e1] {args.benchmark}/{args.split}: samples={len(samples)} "
          f"written={n_written} skipped_resume={n_skip} "
          f"-> {out_path}", flush=True)
    print(f"[e1] model load {load_s:.1f}s; embed_path={embed_path}; "
          f"max_pixels={max_pixels}; g={args.g}; fastv_k={args.fastv_k}; "
          f"null_check={args.null_check}", flush=True)


if __name__ == "__main__":
    main()
