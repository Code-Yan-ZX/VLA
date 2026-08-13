#!/usr/bin/env python3
"""Measure candidate workload discriminators on real Qwen3-VL features to
calibrate the Direction A router. Loads the model once, runs a handful of
images from TextVQA (text-dense --> should route PRE) and GQA (scene/object
--> should route POST), and reports per-image statistics for several candidate
features. No answers generated (visual.forward only).

Output: JSON with per-image {bench, id, unit stats + candidate discriminators}.
"""
from __future__ import annotations

import sys, json, argparse, time
import torch
sys.path.insert(0, 'src/v3_premerger')
import v3_premerger_runner as R

import vllm
from vllm import LLM

def grid_recompute(proc, image_path, max_pixels=0):
    from PIL import Image
    kw = {}
    if max_pixels and max_pixels > 0:
        kw["max_pixels"] = max_pixels
    g = proc.image_processor(Image.open(image_path), return_tensors="pt", **kw)
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-pixels", type=int, default=0)
    args = ap.parse_args()

    llm = LLM(model="Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.9,
              max_model_len=8192, enforce_eager=True,
              limit_mm_per_prompt={"image": 1})
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    visual = model.visual
    sm = visual.spatial_merge_size
    unit = sm ** 2

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    captured = []                      # (n_units, feats [n, unit, ctx]) FIFO

    def hook(module, args):
        hs = args[0]
        if not isinstance(hs, torch.Tensor) or hs.ndim not in (2, 3) \
                or hs.shape[0] == 0:
            return
        seq = hs.shape[0]
        ctx = hs.shape[-1]
        n_units = seq // unit
        if n_units == 0:
            return
        feats = hs.detach().float().reshape(n_units, unit, ctx)
        captured.append((n_units, feats))

    handle = visual.deepstack_merger_list[0].register_forward_pre_hook(hook)

    def measure(paths):
        out = []
        for p in paths:
            captured.clear()
            g = grid_recompute(proc, p, args.max_pixels)
            thw = g["image_grid_thw"][0].tolist()
            px = g["pixel_values"].to(dtype=model.visual.dtype)
            model.visual(px, grid_thw=torch.as_tensor([thw]))
            if not captured:
                out.append({"path": p, "error": "no capture"})
                continue
            n_u, feats = captured[0]
            # candidate variables
            v = feats.var(dim=1, unbiased=False).mean(dim=-1)   # [n_u]
            l2s = feats.norm(dim=-1).mean(dim=-1)               # [n_u]
            mu, sd = v.mean(), v.std()
            rec = {
                "path": p, "n_units": int(n_u),
                "mean_var": float(mu), "std_var": float(sd),
                "mean_l2": float(l2s.mean()),
                "hf_ratio_mean": float((v > mu).float().mean()),        # >mean
                "hf_ratio_p1sd": float((v > mu + 1.0 * sd).float().mean()),
                "hf_ratio_p2sd": float((v > mu + 2.0 * sd).float().mean()),
                "hf_ratio_2x": float((v > 2.0 * mu).float().mean()),
                "var_cv": float(sd / max(mu, 1e-9)),
                "skew_var": float((((v - mu) / max(sd, 1e-9)) ** 3).mean()),
                "l2_entropy": R._unit_workload_stats(feats)["l2_entropy"],
                "var_share_top10": float(
                    v.topk(max(1, n_u // 10)).values.sum() / max(v.sum(), 1e-9)),
            }
            out.append(rec)
        return out

    from collections import deque
    import random
    random.seed(0)
    tv_paths, gqa_paths = [], []
    for line in open('eval/subsets/textvqa_500.jsonl'):
        o = json.loads(line)
        tv_paths.append(o["image"])
    for line in open('eval/subsets/gqa_500.jsonl'):
        o = json.loads(line)
        gqa_paths.append(o["image"])
    tv_paths = tv_paths[:args.n]
    gqa_paths = gqa_paths[:args.n]

    result = {"textvqa": measure(tv_paths), "gqa": measure(gqa_paths)}
    print(json.dumps(result, indent=1, ensure_ascii=False))

    # separation diagnostic: mean of each discriminator per class
    import statistics as st
    def col(cls, key):
        vals = [r[key] for r in result[cls] if "error" not in r]
        return (sum(vals) / len(vals)) if vals else float("nan")
    keys = ["hf_ratio_mean", "hf_ratio_p1sd", "hf_ratio_p2sd", "hf_ratio_2x",
            "var_cv", "skew_var", "l2_entropy", "var_share_top10", "mean_var"]
    print("\n=== class means ===")
    print(f"{'feature':<16} {'TextVQA':>9} {'GQA':>9} {'sign':>6}")
    for k in keys:
        a, b = col("textvqa", k), col("gqa", k)
        sign = "+" if b > a else "-" if a > b else "="
        print(f"{k:<16} {a:9.4f} {b:9.4f} {sign:>6}")

if __name__ == "__main__":
    main()