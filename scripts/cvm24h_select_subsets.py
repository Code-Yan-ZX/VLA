#!/usr/bin/env python
"""Pre-fixed subset selection for P2 (directives 11,12): stratified by raw
visual-unit count / monotonic proxy, NEVER by answer correctness. seed=20261025.
TextVQA: n_image_full from frozen expB prediction (skipped recs imputed by
median). DocVQA: pixel area of the source image (monotone proxy for post-merge
unit count; expA post runs stored no unit field).
"""
import json, random, hashlib, statistics, os
SEED = 20261025
OUT = "results/cvm2027_supplement_20261026/plan"
SRC_TV = "/media/disk2/YZX/research/vla/results/dcc_presubmit/expB/full_textvqa_rbm_pre_r25.json"
SRC_DOC = "/media/disk2/YZX/research/vla/results/dcc_presubmit/expA/full_docvqa_main.json"
DOC_IMG = "/media/disk2/YZX/research/vla/runs/data/docvqa"
from PIL import Image

def stratified(ids, weights, k, seed=SEED):
    vals = sorted(weights[i] for i in ids if i in weights)
    q = [vals[int(len(vals) * f)] for f in (0.25, 0.5, 0.75)]
    bins = {i: [] for i in range(4)}
    for i in ids:
        b = sum(1 for t in q if weights[i] > t)
        bins[b].append(i)
    rng = random.Random(seed)
    counts = {b: len(v) for b, v in bins.items()}
    total = sum(counts.values())
    alloc = {b: int(round(k * counts[b] / total)) for b in bins}
    diff = k - sum(alloc.values())
    order = sorted(bins, key=lambda b: -(k * counts[b] / total - alloc[b]))
    i = 0
    while diff != 0:
        b = order[i % len(order)]
        adj = 1 if diff > 0 else -1
        if 0 <= alloc[b] + adj <= counts[b]:
            alloc[b] += adj
            diff -= adj
        i += 1
    chosen = []
    for b in bins:
        pool = sorted(bins[b]); rng.shuffle(pool); chosen.extend(pool[:alloc[b]])
    assert len(chosen) == k
    return sorted(chosen)

tv = json.load(open(SRC_TV))
ids = [p["id"] for p in tv["per_sample"]]
vals = [p["n_image_full"] for p in tv["per_sample"] if p.get("n_image_full")]
med = statistics.median(vals)
w = {p["id"]: (p.get("n_image_full") or med) for p in tv["per_sample"]}
sel_tv = stratified(ids, w, 1000)

doc = json.load(open(SRC_DOC))
ids_d, w_d = [], {}
for p in doc["per_sample"]:
    fp = os.path.join(DOC_IMG, f"{p['id']}.jpg")
    if not os.path.exists(fp):
        fp2 = os.path.join(DOC_IMG, p["id"])
        fp = fp2 if os.path.exists(fp2) else None
    try:
        with Image.open(fp) as im:
            a = im.width * im.height
    except Exception:
        a = med
    ids_d.append(p["id"]); w_d[p["id"]] = a
sel_doc = stratified(ids_d, w_d, 500)

open(f"{OUT}/p2_textvqa_1000_ids.txt", "w").write("\n".join(sel_tv) + "\n")
open(f"{OUT}/p2_docvqa_500_ids.txt", "w").write("\n".join(sel_doc) + "\n")
meta = {"seed": SEED, "rule": "pre-fixed before any new results; stratified by visual unit count / pixel area; never by correctness",
        "textvqa_1000": {"n": len(sel_tv),
                         "sha256": hashlib.sha256("\n".join(sel_tv).encode()).hexdigest()[:16],
                         "mean_units_selected": round(sum(w[i] for i in sel_tv) / len(sel_tv), 1),
                         "mean_units_all": round(sum(w[i] for i in ids) / len(ids), 1)},
        "docvqa_500": {"n": len(sel_doc),
                       "sha256": hashlib.sha256("\n".join(sel_doc).encode()).hexdigest()[:16],
                       "strata_proxy": "pixel area (monotone in unit count)",
                       "mean_area_selected": round(sum(w_d[i] for i in sel_doc) / len(sel_doc)),
                       "mean_area_all": round(sum(w_d.values()) / len(w_d))}}
json.dump(meta, open(f"{OUT}/p2_subsets_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
