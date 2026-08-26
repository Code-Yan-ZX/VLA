#!/usr/bin/env python3
"""Official-metric rescore for the glm4v stage-law gate (thinking-model aware).

The official scorers (src/v3_premerger/official_scorers.py: TextVQA VQA-acc,
DocVQA ANLS, GQA normalized exact-match) compare the FULL generation against
short gold answers. GLM-4.1V-9B-Thinking wraps answers in a spontaneous
reasoning block (think tags, then an answer/box pair), so RAW text scores 0
on every sample under exact/Levenshtein metrics (verified: 0.0000 on all 9
gate cells despite 0.18-0.785 containment acc). The canonical thinking-model
eval protocol parses out the final answer; we do exactly that here,
IDENTICALLY for all arms:

  extract_final_answer():
    1. content of the LAST box (begin_of_box .. end_of_box),
    2. else text after the LAST closing think tag,
    3. else the full text (truncated reasoning -> scored as generated).

Writes runs/glm4v_gate/glm4v_gate_official_summary.json and prints the
none/pre/post verdict table with d(pre-post) in percentage points.
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.official_scorers import (  # noqa: E402
    score_textvqa_vqaacc, score_docvqa_anls, score_gqa)

THINK_END = "</think>"
BOX_OPEN = "<|begin_of_box|>"
BOX_CLOSE = "<|end_of_box|>"


def extract_final_answer(text: str) -> str:
    t = str(text or "")
    i = t.rfind(BOX_OPEN)
    if i >= 0:
        j = t.find(BOX_CLOSE, i)
        seg = t[i + len(BOX_OPEN):j if j > i else len(t)]
        if seg.strip():
            return seg.strip()
    j = t.rfind(THINK_END)
    if j >= 0:
        seg = t[j + len(THINK_END):].strip()
        seg = re.sub(r"^<answer>|</answer>$", "", seg).strip()
        if seg:
            return seg
    return t.strip()


SCORERS = {"textvqa": score_textvqa_vqaacc,
           "docvqa": score_docvqa_anls,
           "gqa": score_gqa}
METRIC = {"textvqa": "vqa_acc", "docvqa": "anls", "gqa": "exact_match"}

rows = []
for f in sorted(glob.glob(os.path.join(REPO, "runs/glm4v_gate/glm4v_*_n200.json"))):
    d = json.load(open(f))
    b = d.get("benchmark")
    ps = d.get("per_sample") or []
    if b not in SCORERS or not ps:
        continue
    sc = SCORERS[b]
    raw_vals, fin_vals, n_skip, n_box = [], [], 0, 0
    for p in ps:
        if p.get("skipped"):
            n_skip += 1
            continue
        ans = p.get("answer", "")
        gt = str(p.get("gt", ""))
        raw_vals.append(float(sc(ans, gt)))
        fin = extract_final_answer(ans)
        if BOX_OPEN in str(ans):
            n_box += 1
        fin_vals.append(float(sc(fin, gt)))
    row = {"file": os.path.basename(f), "benchmark": b, "mode": d.get("mode"),
           "r": d.get("r"), "n": len(ps), "n_skipped": n_skip,
           "n_boxed": n_box,
           "runner_acc": d.get("acc"),
           "official_metric": METRIC[b],
           "official_score_raw": round(sum(raw_vals) / len(raw_vals), 4),
           "official_score": round(sum(fin_vals) / len(fin_vals), 4),
           "mean_ptid_len": d.get("mean_ptid_len"),
           "wall_s": d.get("wall_s"), "load_s": d.get("load_s"),
           "diag_nk": ((d.get("diag") or {}).get("nk") or [])[:8],
           "proc_placeholder_counts": d.get("proc_placeholder_counts") or []}
    rows.append(row)
    print(f"[rescore] {row['file']}: {METRIC[b]} raw={row['official_score_raw']}"
          f" final={row['official_score']} (boxed {n_box}/{len(ps) - n_skip}, "
          f"runner acc={row['runner_acc']})")

summary = {"gate": "glm4v_stage_law", "model": d.get("model"),
           "keep_frac": 0.25, "selector": "l2",
           "answer_extraction": "last box, else post-think, else full; "
                                "identical for all arms",
           "rows": rows}
with open(os.path.join(REPO, "runs/glm4v_gate/glm4v_gate_official_summary.json"),
          "w") as fh:
    json.dump(summary, fh, indent=2)

tab = {}
for r in rows:
    tab[(r["benchmark"], r["mode"])] = r["official_score"]
print("\n=== OFFICIAL SCORES (none/pre/post), thinking-answer extracted ===")
for b in ("textvqa", "docvqa", "gqa"):
    n, pre, post = (tab.get((b, "none")), tab.get((b, "pre")),
                    tab.get((b, "post")))
    dpp = None if None in (pre, post) else round((pre - post) * 100, 2)
    print(f"{b:8s} none={n}  pre={pre}  post={post}  d(pre-post)pp={dpp}")
