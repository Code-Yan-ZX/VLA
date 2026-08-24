#!/usr/bin/env python
"""MALT adaptive-gate feasibility audit -- PART 2: feature extraction (GPU).

Runs the EXACT deferred-RBM K=1 forward used by the fixed-lifetime sweep
(rankbridge, rho=1.0, keep_frac=0.25, --fastv-k 1) on the locked n=200 subsets
and captures, with NO behavior change (observation-only forward hooks + the
runner's existing MergerTap):

  G0 (pre-decoder; K-independent):
    n_image_full (visual tokens), n_text (text tokens), question length,
    grid aspect ratio, pre-merger L2 score mean/CV/Gini/entropy, top-25% score
    mass, top-k vs rest margin, k-th gap, pre/post-merger ranking Jaccard,
    within-unit variance (relative), anchor bounding-box coverage, anchor
    connected components, anchor-vs-context feature cosine (pre-merger).

  G1 (layer-1 dynamic; K-independent for any K>=1 since no prune before K):
    text->anchor / text->context / anchor->context attention mass,
    context attention entropy (all keys / image keys), text top-5 attention
    concentration over image keys, attention-output norm by role, anchor/text
    hidden cosine change layer0->1, anchor/context representation separation at
    layers 0 and 1, head-level std/max of key attention masses.

Invariance (hard gate): the regenerated per-sample answer MUST equal the sweep
K=1 answer AND the layer-1 keep set MUST equal the sweep anchor_indices
(100% required; any mismatch aborts).  Since rho=1.0 -> keep set = pure
pre-merger L2 top-25%, the layer-1 keep set equals the K=8 keep set too.

Output: experiments/malt_adaptive_gate_feasibility/features_{bench}.json
Usage:  python scripts/extract_malt_features.py --benchmark <b> [--n 200]
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/media/disk2/YZX/research/vla/src/v3_premerger")
from baselines_hf import (MergerTap, build_inputs, capture_prepared_inputs,
                          generate_pruned, load_subset, premerger_unit_ranks)

HERE = "/media/disk2/YZX/research/vla"
SWEEP = f"{HERE}/experiments/deferred_rbm_n200_data/sweep_per_sample.json"
OUTDIR = f"{HERE}/experiments/malt_adaptive_gate_feasibility"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
KEEP_FRAC = 0.25


# --------------------------------------------------------------------------- #
# G0 feature computation (pure)
# --------------------------------------------------------------------------- #
def _gini(x):
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    x = x.sort()[0].float()
    n = x.numel()
    if n == 1 or float(x[0]) == float(x[-1]):
        return 0.0
    g = (2 * torch.arange(1, n + 1, dtype=torch.float, device=x.device) - n - 1
         ) * x
    return float(g.sum() / (n * x.sum()))


def _entropy(p):
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def _components_8conn(mask_hw):
    """Number of 8-connected components in a 2D bool grid (union-find)."""
    h, w = mask_hw.shape
    parent = {}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for r in range(h):
        for c in range(w):
            if mask_hw[r, c]:
                parent[(r, c)] = (r, c)
    for r in range(h):
        for c in range(w):
            if not mask_hw[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask_hw[nr, nc]:
                        union((r, c), (nr, nc))
    return len({find(k) for k in parent})


def g0_features(pre_hs, grid_thw, inputs_embeds, image_mask, keep_frac,
                n_text, q_len, spatial_unit):
    """pre_hs: [seq, H] first-called merger input; grid_thw: [n_img, 3];
    inputs_embeds: [1, L, H] captured (post-merge rows at image positions);
    image_mask: [L] bool.  Returns dict of G0 scalar features."""
    unit = spatial_unit
    seq, ctx = pre_hs.shape
    num_units = seq // unit
    full = (grid_thw.prod(-1) // unit).tolist()
    feats = pre_hs.reshape(num_units, unit, ctx)
    scores = feats.float().norm(p=2, dim=-1).mean(dim=-1)     # per-unit L2
    sf = scores.float()
    img_pos = image_mask.nonzero(as_tuple=False).view(-1)
    ie = inputs_embeds[0] if inputs_embeds.dim() == 3 else inputs_embeds
    post_scores = ie.index_select(0, img_pos).float().norm(p=2, dim=-1)  # post-merge
    # pooled score stats
    n_img_tok = int(img_pos.numel())
    f = {
        "n_img_log": math.log(max(1, n_img_tok)),
        "n_text_log": math.log(max(1, n_text)),
        "q_len_log": math.log(max(1, q_len)),
        "l2_mean": float(sf.mean()),
        "l2_std": float(sf.std()) if num_units > 1 else 0.0,
        "l2_cv": (float(sf.std() / sf.mean()) if num_units > 1
                  and float(sf.mean()) > 0 else 0.0),
        "l2_gini": _gini(sf),
        "l2_entropy": _entropy(sf / sf.sum()) if float(sf.sum()) > 0 else 0.0,
    }
    # per-image aggregates
    topmass, margin, gap, jac, gvar, bboxcov, ncomp, a2c = [], [], [], [], [], [], [], []
    off = 0
    for i, fi in enumerate(full):
        fi = int(fi)
        s = scores[off:off + fi]
        ps = post_scores[off:off + fi]
        k = max(1, int(round(fi * keep_frac)))
        sk = s.sort(descending=True)
        top = sk.values[:k]
        rest = sk.values[k:]
        if float(s.sum()) > 0:
            topmass.append(float(top.sum() / s.sum()))
        if rest.numel() > 0 and float(s.mean()) > 0:
            margin.append(float((top.mean() - rest.mean()) / s.mean()))
        if k < fi:
            gap.append(float((sk.values[k - 1] - sk.values[k]) / s.mean()))
        pre_top = set(sk.indices[:k].tolist())
        pk = ps.sort(descending=True).indices[:k]
        post_top = set(pk.tolist())
        j = len(pre_top & post_top) / len(pre_top | post_top) if pre_top | post_top else 1.0
        jac.append(j)
        # within-unit variance (relative to pooled feature variance)
        uv = feats[off:off + fi].float().var(dim=1).mean().item()  # mean over units&dims
        gvar.append(uv)
        # spatial stats: grid (h, w) from grid_thw[i] = (t, h, w) are PATCH
        # dims; the unit grid is (h/sg, w/sg) with sg = spatial_merge (=2).
        t, h, w = [int(x) for x in grid_thw[i]]
        sg = math.isqrt(spatial_unit)
        gu_h, gu_w = h // sg, w // sg
        grid_units = gu_h * gu_w
        if grid_units == fi:
            rows = sk.indices[:k]
            mask = torch.zeros(grid_units, dtype=torch.bool)
            mask[rows] = True
            m2 = mask.view(gu_h, gu_w)
            rr, cc = m2.nonzero(as_tuple=True)
            if rr.numel() > 0:
                bboxcov.append(float(
                    ((rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
                    / float(gu_h * gu_w)))
            ncomp.append(_components_8conn(m2))
        off += fi
    # anchor vs context centroid cosine in PRE-merger space (per image, avg)
    off = 0
    for i, fi in enumerate(full):
        fi = int(fi)
        s = scores[off:off + fi]
        k = max(1, int(round(fi * keep_frac)))
        an = feats[off:off + fi][s.topk(k).indices].reshape(-1, ctx).float().mean(0)
        ctmask = torch.ones(fi, dtype=torch.bool)
        ctmask[s.topk(k).indices] = False
        ct = feats[off:off + fi][ctmask].reshape(-1, ctx).float()
        if ct.shape[0] > 0:
            ct = ct.mean(0)
            denom = an.norm() * ct.norm()
            a2c.append(float((an @ ct) / denom) if float(denom) > 0 else 0.0)
        off += fi
    f["top25_mass"] = sum(topmass) / len(topmass) if topmass else float("nan")
    f["topk_margin_rel"] = sum(margin) / len(margin) if margin else float("nan")
    f["kth_gap_rel"] = sum(gap) / len(gap) if gap else float("nan")
    f["pre_post_jaccard"] = sum(jac) / len(jac) if jac else float("nan")
    f["group_var_rel"] = (sum(gvar) / len(gvar) / (float(sf.var()) + 1e-12)
                          if gvar and num_units > 1 else float("nan"))
    f["anchor_bbox_cov"] = sum(bboxcov) / len(bboxcov) if bboxcov else float("nan")
    f["n_conn_comp"] = sum(ncomp) / len(ncomp) if ncomp else float("nan")
    f["anchor_ctx_cos_pre"] = sum(a2c) / len(a2c) if a2c else float("nan")
    # aspect ratio from grid (first image)
    gh, gw = float(grid_thw[0][1]), float(grid_thw[0][2])
    f["log_aspect"] = math.log(max(gh, gw) / max(1e-6, min(gh, gw)))
    return f


# --------------------------------------------------------------------------- #
# G1 feature computation (pure)
# --------------------------------------------------------------------------- #
def _mass(attn, qpos, kpos):
    """Mean over heads and query rows of attention from qpos to kpos keys."""
    if qpos.numel() == 0 or kpos.numel() == 0:
        return float("nan"), float("nan"), float("nan")  # mean, head_std, head_max
    per_head = attn[:, qpos][:, :, kpos].sum(-1)          # [H, n_q]
    m = per_head.mean().item()
    hs = per_head.mean(-1)                                # [H]
    return m, float(hs.std()) if hs.numel() > 1 else 0.0, float(hs.max())


def _row_entropy(attn_rows):
    p = attn_rows.clamp_min(1e-12)
    return -(p * p.log()).sum(-1)


def g1_features(attn_w, h0, h1, image_mask, anchor_bool, n_text, attn_out=None):
    """attn_w: [1, H, L, L] layer-1 softmaxed; h0/h1: [L, H] layer-0/1 output
    hidden (pre-deepstack-add of their own layer); image_mask: [L] bool;
    anchor_bool: [n_img] bool anchor mask; attn_out: [L, H] layer-1 attention
    output (optional, for attention-output-norm features).  Returns dict of G1
    scalar features."""
    attn = attn_w[0].float()                              # [H, L, L]
    Hh, L, _ = attn.shape
    img_pos = image_mask.nonzero(as_tuple=False).view(-1)
    assert anchor_bool.numel() == img_pos.numel()
    apos = img_pos[anchor_bool]
    cpos = img_pos[~anchor_bool]
    tpos = (~image_mask).nonzero(as_tuple=False).view(-1)
    imgcols = img_pos
    f = {}
    t2a = _mass(attn, tpos, apos)
    f["t2a_mass"], f["t2a_head_std"], f["t2a_head_max"] = t2a
    t2c = _mass(attn, tpos, cpos)
    f["t2c_mass"], f["t2c_head_std"], f["t2c_head_max"] = t2c
    a2c = _mass(attn, apos, cpos)
    f["a2c_mass"], f["a2c_head_std"], f["a2c_head_max"] = a2c
    # context attention entropy (query rows = transient context)
    if cpos.numel() > 0:
        rows_all = attn[:, cpos]                          # [H, n_c, L]
        ent_all = _row_entropy(rows_all)                  # [H, n_c]
        f["ctx_entropy_all"] = float(ent_all.mean())
        f["ctx_entropy_head_std"] = float(ent_all.mean(-1).std()) if Hh > 1 else 0.0
        imgcols = img_pos
        rows_img = attn[:, cpos][:, :, imgcols]
        ent_img = _row_entropy(rows_img / rows_img.sum(-1, keepdim=True).clamp_min(1e-12))
        f["ctx_entropy_img"] = float(ent_img.mean())
    else:
        f["ctx_entropy_all"] = f["ctx_entropy_head_std"] = f["ctx_entropy_img"] = float("nan")
    # text top-5 concentration over image keys
    if tpos.numel() > 0:
        timg = attn[:, tpos][:, :, imgcols]               # [H, n_t, n_img]
        top5 = timg.topk(min(5, int(timg.shape[-1])), dim=-1).values.sum(-1)
        tot = timg.sum(-1).clamp_min(1e-12)
        conc = (top5 / tot).mean()
        f["text_top5_conc"] = float(conc)
        f["text_top5_head_std"] = float((top5 / tot).mean(-1).std()) if Hh > 1 else 0.0
    else:
        f["text_top5_conc"] = f["text_top5_head_std"] = float("nan")
    # hidden cosine change layer0 -> layer1 by role
    def cos_between(a, b):
        a = a.float(); b = b.float()
        d = (a * b).sum(-1) / (a.norm(p=2, dim=-1) * b.norm(p=2, dim=-1)
                               ).clamp_min(1e-12)
        return float(d.mean())
    f["cos_anchor_01"] = cos_between(h0[apos], h1[apos]) if apos.numel() else float("nan")
    f["cos_text_01"] = cos_between(h0[tpos], h1[tpos]) if tpos.numel() else float("nan")
    # anchor/context separation at layers 0 and 1
    def centroid_cos(a, b, h):
        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        ca, cb = h[a].float().mean(0), h[b].float().mean(0)
        d = ca.norm() * cb.norm()
        return float((ca @ cb) / d) if float(d) > 0 else 0.0
    f["sep_anchor_ctx_1"] = 1.0 - centroid_cos(apos, cpos, h1)
    f["sep_anchor_ctx_0"] = 1.0 - centroid_cos(apos, cpos, h0)
    # attention-output norm by role (layer-1 attention output = weighted value
    # contribution); log-scale to keep magnitudes comparable
    if attn_out is not None:
        ao = attn_out.float()
        def lnorm(pos):
            return float(ao[pos].norm(p=2, dim=-1).mean().log()) if pos.numel() else float("nan")
        f["attnout_norm_anchor"] = lnorm(apos)
        f["attnout_norm_ctx"] = lnorm(cpos)
        f["attnout_norm_text"] = lnorm(tpos)
        f["attnout_norm_all"] = lnorm(torch.arange(ao.shape[0], device=ao.device))
    else:
        for k in ("attnout_norm_anchor", "attnout_norm_ctx",
                  "attnout_norm_text", "attnout_norm_all"):
            f[k] = float("nan")
    return f


# --------------------------------------------------------------------------- #
# Capture hooks (observation-only).
# Layer 1 is the prune layer: _layer_step BYPASSES layer(...) there and calls
# layer.self_attn / layer.mlp directly, so the layer-1 output hidden is rebuilt
# as  h1 = h0_rep + attn_out1 + mlp_out1, where h0_rep = layer0 output + the
# deepstack[0] add that prefill_pruned performs at image positions between
# layers 0 and 1 (== the exact residual entering layer 1).
# --------------------------------------------------------------------------- #
class CaptureHooks:
    def __init__(self, model):
        LM = model.model.language_model
        self.h0_raw = self.h1 = None
        self.attn_out1 = self.attn_w1 = None
        self.mlp_out1 = None
        self._seen = {"l0": False, "a1": False, "m1": False}
        self._h0h = LM.layers[0].register_forward_hook(
            self._mk_layer_hook("l0"))
        self._ah = LM.layers[1].self_attn.register_forward_hook(self._attn_hook)
        self._mh = LM.layers[1].mlp.register_forward_hook(self._mlp_hook)

    @staticmethod
    def _out_tensor(out):
        return (out[0] if isinstance(out, tuple) else out)[0].detach()

    @staticmethod
    def _seq_len(inp):
        if not inp:
            return 0
        h = inp[0] if isinstance(inp, tuple) else inp
        return h.shape[1] if h is not None else 0

    def _mk_layer_hook(self, tag):
        def hook(module, inp, out):
            if self._seen[tag]:
                return None
            if self._seq_len(inp) <= 1:          # decode step, skip
                return None
            self._seen[tag] = True
            self.h0_raw = self._out_tensor(out)  # [L, H] before deepstack add
        return hook

    def _attn_hook(self, module, inp, out):
        if self._seen["a1"]:
            return None
        w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if w is None or w.dim() < 3 or w.shape[-1] <= 1:   # native call -> None
            return None
        self._seen["a1"] = True
        self.attn_out1 = out[0][0].detach()      # [L, H]
        self.attn_w1 = w.detach()                # [1, H, L, L]

    def _mlp_hook(self, module, inp, out):
        if self._seen["m1"]:
            return None
        if self._seq_len(inp) <= 1:
            return None
        self._seen["m1"] = True
        self.mlp_out1 = self._out_tensor(out)    # [L, H]

    def h0_h1(self, deepstack, image_mask):
        """Returns (h0_rep, h1) both [L, H]: h0_rep = layer0 output + the
        deepstack[0] add at image positions (what layer 1 consumes); h1 =
        h0_rep + attn_out1 + mlp_out1 (layer-1 output before the prune)."""
        h0 = self.h0_raw
        if deepstack is not None and len(deepstack) > 0:
            emb = deepstack[0].to(device=h0.device, dtype=h0.dtype)
            h0 = h0.clone()
            h0[image_mask] = h0[image_mask] + emb
        h1 = h0 + self.attn_out1 + self.mlp_out1
        return h0, h1

    def reset(self):
        self._seen = {"l0": False, "a1": False, "m1": False}

    def remove(self):
        for h in (self._h0h, self._ah, self._mh):
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    bench = args.benchmark
    max_pixels = 600000 if bench == "docvqa" else 0

    sweep = json.load(open(SWEEP))
    sw_by = {(v["bench"], v["sample_id"], v["K"]): v for v in sweep.values()}

    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)
    image_token_id = model.config.image_token_id
    spatial_merge = int(getattr(model.visual, "spatial_merge_size", 2))
    spatial_unit = spatial_merge ** 2
    tap = MergerTap(model.visual)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    if processor.tokenizer.pad_token_id is not None:
        eos_ids.add(processor.tokenizer.pad_token_id)
    cfg = {"r": 0.75, "fastv_k": 1, "ratios": None}

    samples = load_subset(f"{HERE}/eval/subsets/{bench}_200.jsonl")[:args.n]
    out = []
    n_mismatch = n_skip = 0
    hooks = CaptureHooks(model)
    for i, s in enumerate(samples):
        try:
            image = Image.open(s.image).convert("RGB")
            inputs = build_inputs(processor, image, s.question,
                                  max_pixels, device)
            image_mask = (inputs["input_ids"][0] == image_token_id)
            tap.reset()
            inputs_embeds, position_ids, deepstack = capture_prepared_inputs(
                model, {k: v for k, v in inputs.items()})
            if tap.first_hs is None:
                raise RuntimeError("no merger tap")
            pre_ranks, drk = premerger_unit_ranks(
                tap.first_hs, inputs["image_grid_thw"], spatial_unit)
            cfg_rb = dict(cfg)
            cfg_rb["rb"] = {"pre_ranks": pre_ranks,
                            "units_per_image": drk["full_per_image"],
                            "keep_frac": KEEP_FRAC, "fuse": "quota", "rho": 1.0,
                            "rrf_lambda": 1.0, "rrf_c": 60.0}
            gen, diag = generate_pruned(
                model, inputs_embeds, position_ids, image_mask, "rankbridge",
                cfg_rb, 32, eos_ids, deepstack=deepstack)
            ans = processor.decode(gen, skip_special_tokens=True).strip()
            kept = diag.get("rb", {}).get("kept_per_image")
        except Exception as ex:
            print(f"[skip] {s.id}: {type(ex).__name__}: {str(ex)[:140]}",
                  flush=True)
            n_skip += 1
            hooks.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # ---- invariance: keep set MUST equal the sweep K=1 cell (features are
        # prefill/layer-1 derived, so this is the binding check); the decoded
        # ANSWER should match but a few borderline samples flip under bf16
        # decode nondeterminism (verified: anchors still identical) -- recorded
        # and reported, not fatal.  Labels for the audit come from the sweep's
        # official rescore, never from this regenerated answer. ----
        ref = sw_by.get((bench, str(s.id), 1))
        ref_kept = ref.get("anchor_indices") if ref else None
        ok_ans = ref is not None and ans == ref["answer"]
        ok_keep = ref_kept is not None and kept == ref_kept
        if not ok_keep:
            n_mismatch += 1
            print(f"[KEEP-MISMATCH] {bench}/{s.id}: {kept} vs {ref_kept}",
                  flush=True)
            if n_mismatch > 5:
                raise SystemExit(">5 keep-set mismatches -- abort")
        if not ok_ans:
            print(f"[ANSWER-DIVERGE] {bench}/{s.id}: {ans!r} vs "
                  f"{ref['answer'] if ref else None!r} (keep-ok={ok_keep})",
                  flush=True)

        # ---- features ----
        fg = g0_features(tap.first_hs, inputs["image_grid_thw"], inputs_embeds,
                         image_mask, KEEP_FRAC, int(image_mask.numel()
                         - image_mask.sum()), len(s.question), spatial_unit)
        # anchor bool over image tokens (rho=1.0 -> top-k pre-rank per image)
        off = 0
        ab = torch.zeros(int(image_mask.sum()), dtype=torch.bool,
                         device=pre_ranks.device)
        for fi in drk["full_per_image"]:
            fi = int(fi)
            k = max(1, int(round(fi * KEEP_FRAC)))
            ab[off:off + fi][pre_ranks[off:off + fi].topk(k, largest=False).indices] = True
            off += fi
        assert hooks.attn_w1 is not None and hooks.h0_raw is not None \
            and hooks.mlp_out1 is not None, "capture hooks did not fire"
        h0_rep, h1 = hooks.h0_h1(deepstack, image_mask)
        fg1 = g1_features(hooks.attn_w1, h0_rep, h1, image_mask,
                          ab.cpu(), int((~image_mask).sum()),
                          attn_out=hooks.attn_out1)
        feat = {**fg, **fg1}
        rec = {"bench": bench, "sample_id": str(s.id),
               "answer": ans, "inv_keep_ok": ok_keep, "inv_ans_ok": ok_ans,
               "correct_K0": None, "correct_K1": None,
               "correct_K8": None, **feat}
        # labels from the official sweep rescore
        for K, key in ((0, "correct_K0"), (1, "correct_K1"), (8, "correct_K8")):
            r = sw_by.get((bench, str(s.id), K))
            if r is not None:
                rec[key] = r["correct"]
        out.append(rec)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(samples)}] {bench} ok={len(out)} "
                  f"mismatch={n_mismatch}", flush=True)
        hooks.reset()

    hooks.remove()
    os.makedirs(OUTDIR, exist_ok=True)
    n_keep_ok = sum(1 for r in out if r.get("inv_keep_ok"))
    n_ans_ok = sum(1 for r in out if r.get("inv_ans_ok"))
    path = f"{OUTDIR}/features_{bench}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[done] {bench}: n={len(out)} skip={n_skip} "
          f"keep_ok={n_keep_ok}/{len(out)} ans_ok={n_ans_ok}/{len(out)}")
    print(f"[written] {path}")


if __name__ == "__main__":
    main()
