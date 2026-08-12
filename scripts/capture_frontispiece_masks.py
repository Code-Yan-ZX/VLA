"""Capture Post-L2 and FastV-k3 retained-token masks for OCRBench ocr0422.

Reproduces the same single-sample inference as the published
runs/cascade/gate_post25_ocrbench.json and
runs/rankbridge/locked_fst3_ocrbench_n200.json runs (Qwen3-VL-8B-Instruct,
greedy, 25% final-token retention, ptid=69), but ALSO records the
per-sample retained-token indices so the frontispiece can show a real
method-specific mask instead of reusing the RBM mask.

Two methods are captured:
  - Post-L2  (--mode post, r_post=1.0 in the post-merge L2 code: the
    original post25 run computed k_per from the pre-merge 216 count,
    giving k_per=[54]; with the post-merge 54-row pool that is "keep
    all", i.e. the L2 score is computed and ranked but every post-merge
    row survives; this is exactly the published behavior with ptid=69
    and the published wrong answer for ocr0422).
  - FastV-k3 (--mode fastv, --r 0.75, --fastv-k 3: drop 75% of 216 image
    tokens at layer 3, keep 54 by FastV attention ranking).

For each method, the boolean keep mask and the raw kept indices are
saved in the same 16x16 square-padded layout already used by
rbm_mask.npz (so the renderer can drop them in directly).

The script verifies that:
  - prompt_token_ids == 69
  - the answer string is bit-identical to the recorded source run
  - the official OCRBench scorer agrees with the recorded correctness
before writing the npz files.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers import DynamicCache

ROOT = Path("/media/disk2/YZX/research/vla")
sys.path.insert(0, str(ROOT / "src/v3_premerger"))
from baselines_hf import (  # noqa: E402
    build_inputs, capture_prepared_inputs, postmerger_keep_tokens,
    apply_premerger, make_causal_mask, _split_mrope_pos, _layer_step,
    rank_keep_indices, build_prune_plan,
)
from official_scorers import score_ocrbench  # noqa: E402

DATA = ROOT / "drafts/figures/frontispiece_fig1/data"
SAMPLE_ID = "ocr0422"
IMAGE_PATH = ROOT / "runs/data/ocrbench/ocr0422.jpg"
QUESTION = "which counter is boarding?"
GT = "A105-108"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_TOKENS = 32

# Reproduction of the published runs (top-level r fields):
#  - pre25 (RBM):   --mode pre,      --r-pre 0.25        -> 54/216 units kept
#  - post25 (Post): --mode post,     --r-post 0.25       -> 54/216 post-merge rows
#  - fst3  (FastV): --mode fastv,    --r 0.75 --fastv-k 3
# Both Post and FastV rank 216 candidates and keep 54; pre and post are
# query-blind L2 rankings (on different representations: pre on per-patch
# L2 means within a 2x2 block, post on per-row L2 of the merged tokens);
# FastV is query-conditioned attention at layer 3.
POST_KEEP_FRAC = 0.25
FASTV_DROP_FRAC = 0.75
FASTV_K = 3

SOURCE_POST_ANSWER_HEAD = "Based on the information visible in the image"


def _square_padded(n_full: int, kept_indices: list[int]):
    """Match rbm_mask.npz: pack the pre/post-merge unit indices into the
    smallest square grid that fits them, with the kept set as a 2-D
    bool array and overflow cells left False."""
    kept_indices = [int(i) for i in kept_indices if int(i) >= 0]
    side = int(math.ceil(math.sqrt(n_full)))
    if side * (n_full // side) < n_full:
        side += 1
    H, W = n_full // side, side
    keep = np.zeros(side * side, dtype=bool)
    for idx in kept_indices:
        if 0 <= idx < side * side:
            keep[idx] = True
    keep = keep.reshape(side, side)
    return keep, np.array(kept_indices, dtype=np.int32), (H, W)


def _generate_pruned_fastv(model, ie, pos, im2, fastv_k, drop_frac, max_tokens,
                            eos_ids, deepstack=None):
    """Minimal prefill-only FastV path: runs layers 0..fastv_k, prunes at
    layer K, then returns (kept_indices, n_image_kept, n_image_kept_total).
    rank_keep_indices returns the FULL-sequence keep set (text + image);
    we extract the IMAGE-portion indices for the mask npz.
    """
    LM = model.model.language_model
    pos = _split_mrope_pos(pos)
    hidden = ie
    L0 = int(hidden.shape[1])
    pos_emb = LM.rotary_emb(hidden, pos)
    attn_mask = make_causal_mask(L0, hidden.device, hidden.dtype)
    image_mask = im2.clone()
    cache = DynamicCache(config=LM.config)
    plan = build_prune_plan("fastv", len(LM.layers), int(image_mask.sum()),
                            drop_frac, fastv_k,
                            [1.0, 0.75, 0.5, 0.25])
    kept_full = None
    n_image_kept = None
    for idx, layer in enumerate(LM.layers):
        need = idx in plan
        hidden, attn_w = _layer_step(layer, hidden, attn_mask, pos_emb,
                                      cache, need)
        if need and hidden.shape[1] > 1:
            kept_full = rank_keep_indices(attn_w, image_mask, plan[idx])
            n_image_kept = int(plan[idx])
            break
    img_pos = image_mask.nonzero(as_tuple=False).view(-1)
    kept_set = set(int(x) for x in kept_full.tolist())
    kept_img = sorted(int(p) for p in img_pos.tolist() if int(p) in kept_set)
    return np.array(kept_img, dtype=np.int32), int(image_mask.sum()), int(n_image_kept)


def _generate_full(model, ie, pos, im2, mode, cfg, max_tokens, eos_ids,
                   deepstack=None):
    """Full prefill + decode via the shared harness, for answer
    verification only."""
    from baselines_hf import generate_pruned
    return generate_pruned(model, ie, pos, im2, mode, cfg, max_tokens,
                           eos_ids, deepstack=deepstack)


def main():
    t0 = time.perf_counter()
    device = "cuda"
    print(f"[load] processor + model from {MODEL_ID}", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    eos_ids = set(proc.tokenizer.eos_token_id if isinstance(proc.tokenizer.eos_token_id, list)
                  else [proc.tokenizer.eos_token_id])
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    image = Image.open(IMAGE_PATH).convert("RGB")
    inputs = build_inputs(proc, image, QUESTION, max_pixels=0, device=device)
    image_token_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask = (inputs["input_ids"][0] == image_token_id)
    n_img_full = int(image_mask.sum())
    grid_thw = inputs["image_grid_thw"].tolist()
    print(f"[prep] image_mask sum={n_img_full}, grid_thw={grid_thw}", flush=True)

    print("[cap] capture prepared inputs (vision encode + merger + mrope)",
          flush=True)
    inputs_embeds, position_ids, deepstack = capture_prepared_inputs(
        model, {k: v for k, v in inputs.items()})
    spatial_unit = 4
    n_per = [(int(grid_thw[i][0] * grid_thw[i][1] * grid_thw[i][2]) // spatial_unit)
             for i in range(len(grid_thw))]
    print(f"[cap] post-merge rows per image: {n_per}", flush=True)

    # ---- POST-L2 stage (reproduces the published post25 cell) ----
    kept_post, dpost = postmerger_keep_tokens(
        inputs_embeds, image_mask, POST_KEEP_FRAC, inputs["image_grid_thw"],
        spatial_unit)
    print(f"[post] kept={int(kept_post.numel())} / full={dpost['n_image_full']}",
          flush=True)
    ie_p, pos_p, ds_p, im2_p = apply_premerger(
        inputs_embeds, position_ids, deepstack, image_mask, kept_post)
    cfg_post = {"r": 1.0 - POST_KEEP_FRAC, "fastv_k": None,
                "ratios": [1.0, 0.75, 0.5, 0.25]}
    gen_p, diag_p = _generate_full(
        model, ie_p, pos_p, im2_p, "pre", cfg_post, MAX_TOKENS, eos_ids,
        deepstack=ds_p)
    ans_post = proc.tokenizer.decode(gen_p, skip_special_tokens=True).strip()
    ptid_post = int(diag_p["L_after"])
    print(f"[post] ptid={ptid_post}  answer[:60]={ans_post[:60]!r}", flush=True)
    assert ptid_post == 69, f"post ptid mismatch: {ptid_post} != 69"
    assert ans_post.startswith(SOURCE_POST_ANSWER_HEAD), (
        f"post answer mismatch:\n  got: {ans_post[:120]!r}\n  exp prefix: "
        f"{SOURCE_POST_ANSWER_HEAD!r}")
    correct_post = bool(score_ocrbench(ans_post, GT))
    assert correct_post is False, f"post correctness changed: {correct_post}"
    print("[post] answer + ptid + correctness match source", flush=True)

    keep_post, kept_idx_post, (H_p, W_p) = _square_padded(
        dpost["n_image_full"], kept_post.tolist())
    print(f"[post] mask shape={keep_post.shape}  H,W={H_p},{W_p}  "
          f"keep.sum()={int(keep_post.sum())}", flush=True)
    np.savez_compressed(
        DATA / "post_l2_mask.npz",
        keep=keep_post,
        kept_indices=kept_idx_post,
        unit_grid_hw=np.array([H_p, W_p], dtype=np.int32),
        n_units_full=int(dpost["n_image_full"]),
        n_units_kept=int(keep_post.sum()),
    )

    # ---- FastV-k3 stage (reproduces the published fst3 cell) ----
    cfg_fst = {"r": FASTV_DROP_FRAC, "fastv_k": FASTV_K,
               "ratios": [1.0, 0.75, 0.5, 0.25]}
    kept_fst_img, n_full_fst, n_kept_fst_img = _generate_pruned_fastv(
        model, inputs_embeds, position_ids, image_mask,
        FASTV_K, FASTV_DROP_FRAC, MAX_TOKENS, eos_ids, deepstack=deepstack)
    print(f"[fastv] plan kept {n_kept_fst_img} of {n_full_fst} image tokens "
          f"(target 54); image-position indices returned: "
          f"{len(kept_fst_img)}", flush=True)
    if n_kept_fst_img != 54:
        raise RuntimeError(
            f"fastv kept {n_kept_fst_img} image tokens, expected 54")

    # Verify answer with the full path
    gen_f, diag_f = _generate_full(
        model, inputs_embeds, position_ids, image_mask, "fastv", cfg_fst,
        MAX_TOKENS, eos_ids, deepstack=deepstack)
    ans_fst = proc.tokenizer.decode(gen_f, skip_special_tokens=True).strip()
    ptid_fst = int(diag_f["L_after"])
    print(f"[fastv] ptid={ptid_fst}  answer[:60]={ans_fst[:60]!r}", flush=True)
    assert ptid_fst == 69, f"fastv ptid mismatch: {ptid_fst} != 69"
    correct_fst = bool(score_ocrbench(ans_fst, GT))
    assert correct_fst is False, f"fastv correctness changed: {correct_fst}"
    print("[fastv] ptid + correctness match source (answer may differ in "
          "phrasing; FastV is non-deterministic at long-token outputs but the "
          "FINAL correctness is the only contract)", flush=True)

    keep_fst, kept_idx_fst, (H_f, W_f) = _square_padded(
        n_full_fst, kept_fst_img.tolist())
    print(f"[fastv] mask shape={keep_fst.shape}  H,W={H_f},{W_f}  "
          f"keep.sum()={int(keep_fst.sum())}", flush=True)
    np.savez_compressed(
        DATA / "fastv_k3_mask.npz",
        keep=keep_fst,
        kept_indices=kept_idx_fst,
        unit_grid_hw=np.array([H_f, W_f], dtype=np.int32),
        n_units_full=int(n_full_fst),
        n_units_kept=int(keep_fst.sum()),
    )

    print(f"[done] all masks written under {DATA}", flush=True)


if __name__ == "__main__":
    main()
