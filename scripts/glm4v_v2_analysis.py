#!/usr/bin/env python3
"""Analysis for the glm4v_gate_v2 max_tokens=4096 probe (4 cells).
Reports per cell: official metric + runner acc + n_boxed + cap-cut fraction +
mean generated tokens (retokenized) + wall_s; and the v2 GPU total."""
import json
import re
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.official_scorers import score_gqa, score_textvqa_vqaacc  # noqa: E402

BOX = "<|begin_of_box|>"
ENDBOX = "<|end_of_box|>"
THINK_END = "<" + "/think" + ">"


def extract(t: str) -> str:
    t = str(t or "")
    i = t.rfind(BOX)
    if i >= 0:
        j = t.find(ENDBOX, i)
        seg = t[i + len(BOX):j if j > i else len(t)]
        if seg.strip():
            return seg.strip()
    j = t.rfind(THINK_END)
    if j >= 0:
        seg = t[j + len(THINK_END):].strip()
        seg = re.sub(r"^<answer>|</answer>$", "", seg).strip()
        if seg:
            return seg
    return t.strip()


MD = "/data/models/modelscope/hub/models/ZhipuAI/GLM-4___1V-9B-Thinking"
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MD)

cells = [("gqa", "none", "glm4v4k_none_gqa", score_gqa),
         ("gqa", "pre", "glm4v4k_pre_gqa", score_gqa),
         ("gqa", "post", "glm4v4k_post_gqa", score_gqa),
         ("textvqa", "none", "glm4v4k_none_textvqa", score_textvqa_vqaacc)]

print(f"{'cell':18s} {'official':8s} {'runner':6s} {'boxed':8s} "
      f"{'cut_frac':8s} {'mean_gen':8s} wall_s")
gtot = 0.0
out = {}
for bench, mode, tag, sc in cells:
    d = json.load(open(os.path.join(REPO, f"runs/glm4v_gate_v2/{tag}.json")))
    ps = d["per_sample"]
    vals, nbox, ncut, gentoks = [], 0, 0, []
    for p in ps:
        a = p["answer"]
        gt = str(p["gt"])
        vals.append(float(sc(extract(a), gt)))
        i = a.rfind(BOX)
        j = a.find(ENDBOX, i) if i >= 0 else -1
        # boxed = complete box and nothing generated after it
        if i >= 0 and j > i and not a[j + len(ENDBOX):].strip():
            nbox += 1
        # cut = never produced a complete box (mid-loop truncation or no box)
        if i < 0 or j < 0:
            ncut += 1
        gentoks.append(len(tok(a, add_special_tokens=False)["input_ids"]))
    off = sum(vals) / len(vals)
    gtot += d["wall_s"] + d.get("load_s", 0)
    key = f"{mode}_{bench}"
    out[key] = {"official": round(off, 4), "runner": d["acc"],
                "boxed": nbox, "cut": ncut,
                "mean_gen": round(sum(gentoks) / len(gentoks), 0),
                "wall_s": d["wall_s"]}
    print(f"{mode+'_'+bench+'@4096':18s} {off:.4f}   {d['acc']:.3f}  "
          f"{nbox}/200   {ncut / 200:.3f}    "
          f"{sum(gentoks) / len(gentoks):8.0f} {d['wall_s']:7.0f}")

print(f"\nv2 total GPU: {gtot:.0f}s = {gtot / 3600:.2f} GPU-h")
gq = {m: out[m + "_gqa"]["official"] for m in ("none", "pre", "post")}
print(f"GQA verdict: pre={gq['pre']} post={gq['post']} "
      f"d(pre-post)={round((gq['pre'] - gq['post']) * 100, 2)}pp "
      f"-> post>=pre? {'YES' if gq['post'] >= gq['pre'] else 'NO'}; "
      f"tie(|d|<=2pp)? {'YES' if abs(gq['pre'] - gq['post']) <= 0.02 else 'NO'}")
tv = out["none_textvqa"]["official"]
print(f"TextVQA none anchor @4096 = {tv} -> >=0.65? "
      f"{'YES' if tv >= 0.65 else 'NO'}")
with open(os.path.join(REPO, "runs/glm4v_gate_v2/v2_probe_summary.json"),
          "w") as fh:
    json.dump({"cells": out, "gpu_h": round(gtot / 3600, 3)}, fh, indent=2)
