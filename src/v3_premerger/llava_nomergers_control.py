"""LLaVA-1.5 NO-MERGER negative control (the key breakthrough experiment).

TESTS THE PAPER'S PREDICTIVE LAW: if the stage effect (pre > post) REQUIRES a
lossy SPATIAL merger, then a model with NO spatial merger (LLaVA-1.5's per-token
MLP projector) should show ~0 stage effect.

  * The paper's stage law: ranking visual tokens BEFORE a 2x2 spatial merger
    ("pre") beats ranking AFTER ("post") on text-dense tasks, because the merger
    averages 4 spatial neighbours and destroys high-frequency text.  Confirmed
    on Qwen3-VL / Qwen2.5-VL / InternVL3 (all have a 2x2 spatial merger whose
    unit = spatial_merge_size**2 = 4 consecutive patches).
  * LLaVA-1.5-7B has an MLP projector (Linear->GELU->Linear, PER-TOKEN, no
    spatial averaging) -- NO spatial merger.  The MLP mixes CHANNELS but never
    SPATIAL positions, so it does NOT destroy high-frequency text the way a 2x2
    spatial average does.
  * PREDICTION: pre-MLP vs post-MLP L2 pruning shows ~0 stage effect on BOTH
    TextVQA (text-dense, where Qwen shows a big stage effect) and GQA (object-QA,
    where Qwen shows ~0).  If confirmed, this is the negative control that turns
    "observed pre>post on 3 merger models" into "pre>post IFF the merger rewrites
    saliency" -- the predictive law holds.

PRUNING (iso-budget, keep 25% = 144/576 tokens, unit = 1 token since no merger):
  * "pre"  = prune the ViT patch features (vision_tower output, PRE-MLP) by L2
             norm per token, keep top-k, THEN run mm_projector on ONLY the
             survivors.  True pre-MLP pruning (the projector never sees dropped).
  * "post" = run mm_projector on ALL 576 tokens, prune the PROJECTED features
             (POST-MLP) by L2 norm per token, keep top-k.  True post-MLP pruning.
  * "none" = no pruning (keep all 576); the accuracy anchor.
  Both pre/post keep EXACTLY k = round(576 * keep_frac) = 144 tokens (iso-budget).
  The MLP is per-token, so pre-MLP and post-MLP L2 rankings differ only because
  the MLP rescales features (1024-dim ViT space vs 4096-dim LLM space) -- that is
  the gap we measure.  With a 2x2 spatial merger the rankings differ because the
  merger REWRITES saliency (averages neighbours); with the per-token MLP they
  should not.

IMPLEMENTATION (cleanest path): monkey-patch model.encode_images so it returns
only the k surviving projected features, then call the model's NATIVE generate
(input_ids + images).  The native prepare_inputs_labels_for_multimodal scatters
the k features into the embedding sequence and rebuilds position_ids /
attention_mask for the shorter T1+k+T2 sequence (llava_arch.py L206-251,
verified: position_ids = arange(cur_len), attention_mask = ones).  No manual
scatter / position_ids / greedy decode -- the model handles everything; we only
intercept the vision-encode stage.  This mirrors how the Qwen runner taps the
merger input, but here the "tap" is the encode_images patch.

LOADING: LLaVA-1.5-7B via the `llava` library (fastv env:
/home/dell/miniconda3/envs/fastv), original LLaVA-format weights at
/media/disk2/YZX/doct/FastV/llava-v1.5-7b (pytorch_model*.bin + mm_projector.bin).
Greedy, max_tokens=32 (== Qwen3 config), n=200, TextVQA + GQA.

SCORING: official_scorers.py -- score_textvqa_vqaacc (VQA-acc) for TextVQA,
score_gqa (normalized exact-match) for GQA.  per_sample[].answer/gt stored so
the offline --rescope step recomputes the official metric + paired delta.

RISKS / CAVEATS:
  * LLaVA-1.5's 576 FIXED tokens vs Qwen's dynamic resolution: the control uses
    fixed 576, which is fine -- the claim is about the MERGER, not resolution.
  * The per-token MLP vs 2x2 spatial merger distinction is the WHOLE point: the
    MLP mixes channels (no spatial averaging) so it should NOT destroy text;
    a 2x2 merger averages 4 spatial neighbours (destroys high-freq text).  This
    is why we predict ~0 stage effect here.
  * LLaVA-NeXT vs LLaVA-1.5: both have NO spatial merger (NeXT uses the same
    MLP projector with dynamic AnyRes resolution).  Either works for the control
    as long as there's no spatial merger; we use 1.5 (fixed 576) for simplicity
    + weight availability.  NeXT is NOT required.

Usage (run inside the fastv env, on GPU):
  /home/dell/miniconda3/envs/fastv/bin/python -m src.v3_premerger.llava_nomergers_control \
      --mode pre --benchmark textvqa --subset eval/subsets/textvqa_200.jsonl \
      --keep-frac 0.25 --n 200 --max-tokens 32 \
      --out runs/llava_nomergers/textvqa_pre.json

  # Dry-check (no GPU, no weights): config + prune logic + patch attach + scorers
  /home/dell/miniconda3/envs/fastv/bin/python -m src.v3_premerger.llava_nomergers_control --dry-check

  # Rescore all 6 cells + decision rule (CPU):
  /home/dell/miniconda3/envs/fastv/bin/python -m src.v3_premerger.llava_nomergers_control \
      --rescore runs/llava_nomergers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make official_scorers (same dir) importable when run as a module or directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_scorers import score_textvqa_vqaacc, score_gqa  # noqa: E402


# --------------------------------------------------------------------------- #
# Model + image constants (LLaVA-1.5-7B, CLIP-ViT-L/14-336).
# --------------------------------------------------------------------------- #
MODEL_PATH = "/media/disk2/YZX/doct/FastV/llava-v1.5-7b"
# CLIP-ViT-L/14-336: image_size=336, patch_size=14 -> (336/14)**2 = 576 patches.
# mm_vision_select_feature="patch" drops CLS -> exactly 576 visual tokens.
N_IMAGE_TOKENS = 576
CONV_MODE = "llava_v1"  # LLaVA-1.5 chat template


# --------------------------------------------------------------------------- #
# Data loading (same JSONL contract as fastv_bench / baselines_hf).
# --------------------------------------------------------------------------- #
def load_subset(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append({
                "id": str(o["id"]),
                "image": o["image"],
                "question": o["question"],
                "gt": str(o["gt"]),
                "extra": {k: v for k, v in o.items()
                          if k not in {"id", "image", "question", "gt"}},
            })
    return out


# --------------------------------------------------------------------------- #
# Pruning: per-token L2 top-k (unit = 1, no spatial merger).  The selector is
# the EXACT analog of the Qwen runner's _score_tokens(l2) (post-merger path,
# unit=1) / _score_units(l2) with unit=1 (pre-merger path): feats.norm(dim=-1).
# Survivors keep their ORIGINAL SPATIAL ORDER (topk indices sorted ascending),
# matching the Qwen runner's boolean-mask + nonzero convention.
# --------------------------------------------------------------------------- #
def l2_topk_keep(scores: "torch.Tensor", k: int) -> "torch.Tensor":
    """scores: [n_tok] per-token L2.  Returns sorted-ascending global indices of
    the top-k (spatial order preserved).  k clamped to [1, n_tok]."""
    import torch
    n = int(scores.shape[0])
    k = max(1, min(int(k), n))
    idx = torch.topk(scores, k).indices
    return idx.sort().values


def make_pruned_encode_images(orig_encode_images, vision_tower, mm_projector,
                              mode: str, keep_frac: float, n_total: int = N_IMAGE_TOKENS):
    """Return a patched encode_images(images) that returns [1, k, H_llm] with
    ONLY the surviving projected features.  mode in {none, pre, post}.

      none : identity (returns orig_encode_images(images) -- all 576 projected).
      pre  : vit = vision_tower(images); score vit by per-token L2 (1024-dim
             ViT space); keep top-k; return mm_projector(vit[:, keep])  -- the
             projector runs ONLY on survivors (true pre-MLP pruning).
      post : proj = mm_projector(vit); score proj by per-token L2 (4096-dim LLM
             space); keep top-k; return proj[:, keep]  (true post-MLP pruning).

    Both pre/post keep EXACTLY k = round(n_total * keep_frac) tokens (iso-budget).
    """
    import torch
    k = max(1, int(round(n_total * keep_frac))) if mode != "none" else n_total

    if mode == "none":
        # No patching needed; callers may still use this for a uniform path.
        def _none(images):
            return orig_encode_images(images)
        _none.kept_k = n_total
        _none.mode = "none"
        return _none

    def _pruned(images):
        with torch.no_grad():
            vit = vision_tower(images)              # [1, 576, 1024] pre-MLP
            if mode == "pre":
                scores = vit[0].float().norm(dim=-1)            # [576] ViT-space L2
                keep = l2_topk_keep(scores, k)                  # sorted spatial order
                out = mm_projector(vit[:, keep, :])             # [1, k, 4096]
            else:  # post
                proj = mm_projector(vit)                        # [1, 576, 4096]
                scores = proj[0].float().norm(dim=-1)           # [576] LLM-space L2
                keep = l2_topk_keep(scores, k)
                out = proj[:, keep, :]                          # [1, k, 4096]
        return out.to(next(mm_projector.parameters()).dtype)

    _pruned.kept_k = k
    _pruned.mode = mode
    return _pruned


# --------------------------------------------------------------------------- #
# Run one cell: mode x benchmark.
# --------------------------------------------------------------------------- #
def run_cell(args) -> dict:
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        get_model_name_from_path, process_images, tokenizer_image_token,
    )
    from PIL import Image

    samples = load_subset(args.subset)[:args.n]
    if args.benchmark == "textvqa":
        scorer = score_textvqa_vqaacc
    elif args.benchmark == "gqa":
        scorer = score_gqa
    else:
        raise SystemExit(f"unsupported benchmark {args.benchmark} "
                         f"(control = textvqa + gqa only)")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        args.load_8bit, args.load_4bit, device=args.device,
    )
    model.eval()

    vision_tower = model.get_vision_tower()
    mm_projector = model.get_model().mm_projector
    orig_encode = model.encode_images
    patched = make_pruned_encode_images(
        orig_encode, vision_tower, mm_projector, args.mode, args.keep_frac)
    model.encode_images = patched          # monkey-patch; native generate uses it
    k_kept = patched.kept_k
    print(f"[nomergers] mode={args.mode} keep_frac={args.keep_frac} "
          f"-> k={k_kept}/{N_IMAGE_TOKENS} tokens (unit=1, no spatial merger)",
          flush=True)

    per_sample = []
    n_correct = n_skip = 0
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            img = Image.open(s["image"]).convert("RGB")
            image_tensor = process_images([img], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            qs = DEFAULT_IMAGE_TOKEN + "\n" + s["question"]
            conv = conv_templates[CONV_MODE].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0).to(model.device))
            n_prompt = int(input_ids.shape[1])

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=args.max_tokens,
                    use_cache=True,            # no mid-LLM pruning -> KV safe
                )
            ans = tokenizer.decode(
                output_ids[0, n_prompt:], skip_special_tokens=True).strip()
            n_img_full = int((input_ids[0] == IMAGE_TOKEN_INDEX).sum())
            diag = {"n_image_full": n_img_full, "n_image_kept": k_kept,
                    "n_text": n_prompt - n_img_full, "L0": n_prompt,
                    "L_after": n_prompt - n_img_full + k_kept}
        except Exception as e:
            print(f"[nomergers] sample {s['id']} FAILED "
                  f"({type(e).__name__}: {str(e)[:160]})", file=sys.stderr, flush=True)
            per_sample.append({"id": s["id"], "correct": 0.0, "skipped": True,
                               "answer": "", "gt": s["gt"], "question": s["question"]})
            n_skip += 1
            continue
        c = scorer(ans, s["gt"])
        n_correct += c
        per_sample.append({
            "id": s["id"], "correct": c, "skipped": False, "answer": ans,
            "gt": s["gt"], "question": s["question"],
            "n_image_full": diag["n_image_full"], "n_image_kept": diag["n_image_kept"],
            "n_text": diag["n_text"], "L0": diag["L0"], "L_after": diag["L_after"],
        })
        if (i + 1) % 20 == 0:
            print(f"[nomergers] {args.benchmark} {i+1}/{len(samples)} "
                  f"running_acc={n_correct/(i+1):.3f}", flush=True)

    model.encode_images = orig_encode       # restore (cleanliness)

    wall = time.perf_counter() - t0
    n_scored = len(samples) - n_skip
    acc = n_correct / n_scored if n_scored else 0.0
    result = {
        "model": args.model_path,
        "control": "llava_nomergers",
        "mode": args.mode,
        "benchmark": args.benchmark,
        "keep_frac": args.keep_frac,
        "k_kept": k_kept,
        "n_image_full": N_IMAGE_TOKENS,
        "unit": 1,
        "merger": "none (mlp2x_gelu per-token projector, no spatial merge)",
        "n": len(samples), "max_tokens": args.max_tokens,
        "acc_official": round(acc, 4), "n_answered": n_scored,
        "n_skipped": n_skip, "wall_s": round(wall, 3),
        "scorer": ("score_textvqa_vqaacc" if args.benchmark == "textvqa"
                   else "score_gqa"),
        "per_sample": per_sample,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[nomergers] mode={args.mode} {args.benchmark}: acc={acc:.3f} "
          f"k={k_kept}/{N_IMAGE_TOKENS} n={len(samples)} skip={n_skip} "
          f"wall={wall:.1f}s -> {args.out}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# Rescore: read the 6 cells, compute official acc + paired pre-post delta +
# DECISION RULE.  Pure CPU / string work.
# --------------------------------------------------------------------------- #
def rescore(directory: str) -> dict:
    """Read all llava_nomergers cells in `directory`, recompute the official
    metric from per_sample[].answer/gt, compute paired pre-post deltas, and
    apply the decision rule.

    DECISION RULE:
      |pre - post| <= ~1pp on BOTH TextVQA AND GQA  -> NEGATIVE CONTROL CONFIRMED
        (no stage effect without a spatial merger -> the predictive law holds).
      pre > post significantly on TextVQA  -> LAW IS WRONG (the stage effect does
        NOT require a spatial merger) -> FLAG, do not self-package.
    """
    cells = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, fn)) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # robust: identify cells by content, not filename
        if r.get("control") != "llava_nomergers" or "mode" not in r or "benchmark" not in r:
            continue
        key = (r["benchmark"], r["mode"])
        # recompute official metric from stored answer/gt (authoritative)
        if r["benchmark"] == "textvqa":
            sc = [score_textvqa_vqaacc(p["answer"], p["gt"])
                  for p in r["per_sample"] if not p.get("skipped")]
        else:
            sc = [score_gqa(p["answer"], p["gt"])
                  for p in r["per_sample"] if not p.get("skipped")]
        r["acc_official_recomputed"] = (sum(sc) / len(sc) if sc else 0.0)
        cells[key] = r

    summary = {}
    for bench in ("textvqa", "gqa"):
        row = {}
        for mode in ("none", "pre", "post"):
            r = cells.get((bench, mode))
            row[mode] = (round(r["acc_official_recomputed"], 4) if r else None)
        summary[bench] = row

        # paired pre-post delta (per-sample, matched by id)
        pre = cells.get((bench, "pre"))
        post = cells.get((bench, "post"))
        if pre and post:
            pre_map = {p["id"]: p for p in pre["per_sample"] if not p.get("skipped")}
            post_map = {p["id"]: p for p in post["per_sample"] if not p.get("skipped")}
            common = sorted(set(pre_map) & set(post_map))
            if bench == "textvqa":
                ps = [score_textvqa_vqaacc(pre_map[i]["answer"], pre_map[i]["gt"])
                      - score_textvqa_vqaacc(post_map[i]["answer"], post_map[i]["gt"])
                      for i in common]
            else:
                ps = [score_gqa(pre_map[i]["answer"], pre_map[i]["gt"])
                      - score_gqa(post_map[i]["answer"], post_map[i]["gt"])
                      for i in common]
            delta = sum(ps) / len(ps) if ps else 0.0
            # McNemar on the paired correctness (discordant pairs)
            both_c = sum(1 for i in common
                         if _correct(pre_map[i], bench) and _correct(post_map[i], bench))
            pre_only = sum(1 for i in common
                           if _correct(pre_map[i], bench) and not _correct(post_map[i], bench))
            post_only = sum(1 for i in common
                            if not _correct(pre_map[i], bench) and _correct(post_map[i], bench))
            both_w = sum(1 for i in common
                         if not _correct(pre_map[i], bench) and not _correct(post_map[i], bench))
            row["pre_minus_post_pp"] = round(delta * 100, 2)
            row["n_paired"] = len(common)
            row["mcnemar_pre_only_post_only"] = [pre_only, post_only]
            # two-sided exact McNemar p (binomial) on discordant pairs
            row["mcnemar_p"] = _mcnemar_p(pre_only, post_only)

    # ---- DECISION RULE ----
    tvqa = summary["textvqa"]
    gqa = summary["gqa"]
    tvqa_gap = abs(tvqa.get("pre_minus_post_pp", 0.0))
    gqa_gap = abs(gqa.get("pre_minus_post_pp", 0.0))
    tvqa_signed = tvqa.get("pre_minus_post_pp", 0.0)
    confirmed = (tvqa_gap <= 1.0 and gqa_gap <= 1.0)
    law_wrong = (tvqa_signed > 1.0)  # pre significantly > post on TextVQA
    if confirmed:
        decision = ("NEGATIVE CONTROL CONFIRMED: |pre-post|<=1pp on BOTH "
                    "TextVQA and GQA -> no stage effect without a spatial "
                    "merger -> the predictive law HOLDS")
    elif law_wrong:
        decision = ("LAW IS WRONG: pre>post significantly on TextVQA WITHOUT a "
                    "spatial merger -> the stage effect does NOT require a "
                    "spatial merger -> FLAG, do not self-package")
    else:
        decision = (f"AMBIGUOUS: TextVQA gap={tvqa_gap:.2f}pp, GQA gap="
                    f"{gqa_gap:.2f}pp -> does not meet either rule cleanly; "
                    f"inspect before concluding")

    out = {
        "control": "llava_nomergers",
        "official_metrics": summary,
        "decision_rule": {
            "threshold_pp": 1.0,
            "textvqa_gap_pp": tvqa_gap,
            "gqa_gap_pp": gqa_gap,
            "textvqa_signed_pp": tvqa_signed,
            "confirmed_negative_control": confirmed,
            "law_wrong_flag": law_wrong,
        },
        "decision": decision,
    }
    respath = os.path.join(directory, "llava_nomergers_decision.json")
    with open(respath, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[nomergers] decision written -> {respath}", flush=True)
    return out


def _correct(rec, bench: str) -> bool:
    if rec.get("skipped"):
        return False
    if bench == "textvqa":
        return score_textvqa_vqaacc(rec["answer"], rec["gt"]) > 0
    return score_gqa(rec["answer"], rec["gt"]) > 0


def _mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (pre-only) and c
    (post-only): Binomial(n=b+c, p=0.5) tail.  Returns 1.0 if b+c==0."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    # two-sided = 2 * sum_{i=0}^{k} C(n,i) * 0.5^n  (capped at 1.0)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(p, 1.0)


# --------------------------------------------------------------------------- #
# No-GPU dry check: config + prune logic + patch-attach + scorers.
# --------------------------------------------------------------------------- #
def run_dry_check():
    import torch
    print("[dry-check] (A) config + 576-token invariant")
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["mm_projector_type"] == "mlp2x_gelu", cfg["mm_projector_type"]
    assert cfg["mm_hidden_size"] == 1024, cfg["mm_hidden_size"]
    assert cfg["hidden_size"] == 4096, cfg["hidden_size"]
    assert cfg["mm_vision_select_feature"] == "patch", cfg["mm_vision_select_feature"]
    assert cfg["mm_vision_select_layer"] == -2, cfg["mm_vision_select_layer"]
    assert cfg["image_aspect_ratio"] == "pad", cfg["image_aspect_ratio"]
    # CLIP-ViT-L/14-336: image_size=336, patch_size=14 -> 576 patches (no CLS).
    # mm_vision_select_feature="patch" drops CLS -> exactly 576 visual tokens.
    patches = (336 // 14) ** 2
    assert patches == N_IMAGE_TOKENS == 576, (patches, N_IMAGE_TOKENS)
    print(f"[dry-check]   OK mm_projector={cfg['mm_projector_type']} (per-token, "
          f"NO spatial merger), mm_hidden={cfg['mm_hidden_size']} -> "
          f"hidden={cfg['hidden_size']}, {N_IMAGE_TOKENS} patch tokens")

    print("[dry-check] (B) prune function (per-token L2 top-k, sorted spatial order)")
    torch.manual_seed(0)
    n, d = 576, 1024
    vit = torch.randn(1, n, d)
    proj = torch.randn(1, n, 4096)
    k = int(round(n * 0.25))
    assert k == 144, k
    # pre: score vit, keep top-144 by L2, sorted ascending
    scores_pre = vit[0].float().norm(dim=-1)
    keep_pre = l2_topk_keep(scores_pre, k)
    assert keep_pre.shape[0] == 144 and int(keep_pre[-1]) < 576
    assert torch.equal(keep_pre, keep_pre.sort().values), "must be sorted (spatial order)"
    assert torch.equal(keep_pre, scores_pre.topk(k).indices.sort().values)
    # post: score proj, keep top-144 by L2
    scores_post = proj[0].float().norm(dim=-1)
    keep_post = l2_topk_keep(scores_post, k)
    assert keep_post.shape[0] == 144
    # pre vs post rankings DIFFER (the MLP rescales features) -- the measured gap
    overlap = len(set(keep_pre.tolist()) & set(keep_post.tolist()))
    print(f"[dry-check]   OK pre/post top-144: k={k}, sorted, overlap={overlap}/144 "
          f"(pre!=post rankings expected; gap is what we measure)")
    # none: keep all
    assert l2_topk_keep(scores_pre, 576).shape[0] == 576
    # floor at 1
    assert l2_topk_keep(scores_pre, 0).shape[0] == 1

    print("[dry-check] (C) encode_images patch attaches + restores")
    class DummyModel:
        def __init__(self):
            self.encode_images = self._orig
        def _orig(self, images):
            return "ORIGINAL"
    dm = DummyModel()
    class FakeVT:
        def __call__(self, x): return vit
    class FakeProj:
        # per-token projector: maps [1, n, 1024] -> [1, n, 4096] (preserves n,
        # like the real mlp2x_gelu which is applied per-token, no spatial merge)
        def __call__(self, x):
            import torch.nn.functional as F
            return F.pad(x, (0, 4096 - 1024))
        def parameters(self):
            # nn.Module.parameters() returns a generator; next() needs an iterator
            # with a .dtype attribute (the real mm_projector is an nn.Sequential).
            yield torch.zeros(1, dtype=torch.float32)
    patched = make_pruned_encode_images(dm._orig, FakeVT(), FakeProj(), "pre", 0.25)
    dm.encode_images = patched
    out = dm.encode_images(None)
    assert out.shape == (1, 144, 4096), out.shape
    assert dm.encode_images.mode == "pre" and dm.encode_images.kept_k == 144
    dm.encode_images = dm._orig           # restore
    assert dm.encode_images(None) == "ORIGINAL"
    print("[dry-check]   OK patch: pre -> [1,144,4096], mode/kept_k set, "
          "restore works")

    print("[dry-check] (D) official scorers")
    gt_tvqa = "12:34;12:34 am;12:34;9:00"
    assert abs(score_textvqa_vqaacc("12:34", gt_tvqa) - (2.0 / 3.0)) < 1e-9
    assert score_gqa("No", "no") == 1.0
    assert score_gqa("I don't know", "no") == 0.0
    print("[dry-check]   OK score_textvqa_vqaacc + score_gqa")

    print("[dry-check] (E) imports (llava library available in fastv env)")
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    assert IMAGE_TOKEN_INDEX == -200, IMAGE_TOKEN_INDEX
    assert DEFAULT_IMAGE_TOKEN == "<image>"
    assert "llava_v1" in conv_templates
    print(f"[dry-check]   OK llava imports (IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}, "
          f"conv llava_v1 present)")
    print("[dry-check] ALL PASS")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="LLaVA-1.5 NO-MERGER negative control")
    ap.add_argument("--mode", default="pre", choices=["none", "pre", "post"],
                    help="none=anchor(576); pre=prune ViT feats pre-MLP by L2; "
                         "post=prune projected feats post-MLP by L2.")
    ap.add_argument("--model-path", default=MODEL_PATH,
                    help="LLaVA-1.5-7B in original LLaVA format")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--benchmark", default=None, choices=["textvqa", "gqa"])
    ap.add_argument("--subset", default=None, help="JSONL subset path")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--keep-frac", type=float, default=0.25,
                    help="KEEP fraction of 576 tokens (0.25 -> 144).")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dry-check", action="store_true",
                    help="CPU-only: config + prune logic + patch attach + scorers.")
    ap.add_argument("--rescore", default=None, metavar="DIR",
                    help="rescore all cells in DIR + decision rule (CPU).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_check:
        run_dry_check()
        return
    if args.rescore:
        rescore(args.rescore)
        return
    missing = [n for n, v in (("--benchmark", args.benchmark),
                              ("--subset", args.subset),
                              ("--out", args.out)) if not v]
    if missing:
        raise SystemExit("required (unless --dry-check/--rescore): " + ", ".join(missing))
    run_cell(args)


if __name__ == "__main__":
    main()
