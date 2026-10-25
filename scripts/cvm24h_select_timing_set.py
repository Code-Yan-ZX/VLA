#!/usr/bin/env python
"""Pre-fix the 200 timing requests (directive 14): 100 TextVQA + 100 OCRBench,
stratified by resolution proxy (pixel area / n_image_full), deterministic.
Selected WITHOUT looking at any score."""
import json, hashlib, statistics
from PIL import Image
OUT = "results/cvm2027_supplement_20261026/plan"
SRC_TV = "/media/disk2/YZX/research/vla/results/dcc_presubmit/expB/full_textvqa_rbm_pre_r25.json"
OCRB_IDS = [l.split('"')[7] if False else json.loads(l)["id"]
            for l in open("eval/full_splits/ocrbench.jsonl")]
OCRB_IMG = "/media/disk2/YZX/research/vla/runs/data/ocrbench"

def strata_pick(items, k):
    """items: list of (id, weight). deterministic: sort by weight then id,
    walk k equal-count bands, take one per band round-robin."""
    items = sorted(items, key=lambda x: (x[1], x[0]))
    n = len(items)
    per = max(1, n // k)
    chosen = []
    for b in range(k):
        start = b * per
        chosen.append(items[start if start < n else n - 1])
    seen = set(); sel = []
    for x in chosen:
        if x[0] not in seen:
            seen.add(x[0]); sel.append(x)
    # fill any dup shortfall from unchosen mid-band items
    b = 0
    while len(sel) < k:
        x = items[(b * 7 + k) % n]
        if x[0] not in seen:
            seen.add(x[0]); sel.append(x)
        b += 1
    return sorted(sel)

tv = json.load(open(SRC_TV))
vals = [p["n_image_full"] for p in tv["per_sample"] if p.get("n_image_full")]
med = statistics.median(vals)
tv_items = [(p["id"], p.get("n_image_full") or med) for p in tv["per_sample"]
            if not p.get("skipped")]
sel_tv = strata_pick(tv_items, 100)

ocr_items = []
for pid in OCRB_IDS:
    fp = f"{OCRB_IMG}/{pid}.jpg" if not str(pid).endswith(".jpg") else f"{OCRB_IMG}/{pid}"
    try:
        with Image.open(fp) as im:
            a = im.width * im.height
    except Exception:
        a = med
    ocr_items.append((str(pid), a))
sel_ocr = strata_pick(ocr_items, 100)

rec = {"rule": "resolution-stratified, pre-fixed before any timing/score result; "
               "200 requests shared by all methods (bs=1, single image)",
       "textvqa_100": [i for i, _ in sel_tv], "textvqa_units": {i: w for i, w in sel_tv},
       "ocrbench_100": [i for i, _ in sel_ocr],
       "ocrbench_area": {i: w for i, w in sel_ocr},
       "sha256_tv": hashlib.sha256(",".join(i for i, _ in sel_tv).encode()).hexdigest()[:16],
       "sha256_ocr": hashlib.sha256(",".join(i for i, _ in sel_ocr).encode()).hexdigest()[:16]}
json.dump(rec, open(f"{OUT}/p3_timing_ids.json", "w"), indent=1)
print("tv n=", len(rec["textvqa_100"]), "units min/med/max=",
      rec["textvqa_units"][rec["textvqa_100"][0]],
      sorted(rec["textvqa_units"].values())[50],
      rec["textvqa_units"][rec["textvqa_100"][-1]], "| sha", rec["sha256_tv"], rec["sha256_ocr"])
