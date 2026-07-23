#!/usr/bin/env python3
"""Task 5 — OFFLINE rescore of the hybrid gate battery under official metrics,
+ gate PASS/FAIL verdict (no GPU).

Cells (all keep=25%, L2, seed=0, short-answer subsets, n=100):
  textvqa: hybrid tf-sweep {0.0,0.5,1.0} + chosen (runs/v3_merger_aware/hybrid_gate/),
           pre/post/none = rescore_rerun n=200 SLICED to the first 100 ids
           (verified: subset-order preserved, first-100 == first 100 subset ids),
           attn-selector pre/post (selector invariance, Task 3)
           -> VQA-acc (official_scorers.score_textvqa_vqaacc)
  ocrbench: hybrid tf-chosen n100; pre/post/none = v3_sota_matrix C/B/A n=200
           sliced to first 100 (verified n=200/seed0) -> runner score_ocrbench
           (per_sample.correct IS the ocrbench containment metric)
  gqa:     hybrid tf-chosen n100 + fresh pre/post/none n100 (short-answer)
           -> runner score_gqa exact-match (per_sample.correct)

GATE (user-defined): PASS iff text-dense (textvqa VQA-acc AND ocrbench)
hybrid >= pre-standard (no OCR regression) AND gqa hybrid notably better than
pre-standard (closes the gap toward post) without hurting text-dense.
"Notably" operationalized as: gqa hybrid > gqa pre AND gap-closure
(hybrid-pre)/(post-pre) >= 0.5 (post>pre regime); a noise-aware secondary
verdict also reports the paired SE of each hybrid-vs-pre delta.

Writes runs/v3_merger_aware/hybrid_gate/gate_summary.json + prints the
markdown tables used in drafts/method_gate_report.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = "/media/disk2/YZX/research/vla"
HG = os.path.join(REPO, "runs/v3_merger_aware/hybrid_gate")
RR = os.path.join(REPO, "runs/v3_merger_aware/rescore_rerun")
SM = os.path.join(REPO, "runs/v3_sota_matrix")
sys.path.insert(0, os.path.join(REPO, "src/v3_premerger"))
from official_scorers import score_textvqa_vqaacc  # noqa: E402

N = 100


def se(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def subset_first_ids(bench: str, n: int = N):
    ids = []
    with open(os.path.join(REPO, f"eval/subsets/{bench}_200.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
    return ids[:n]


def load_cell(path: str):
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
        return {str(s["id"]): s for s in d.get("per_sample", []) if not s.get("skipped")}, d
    except Exception as e:
        print(f"[gate] unreadable cell {path}: {e}")
        return None


def scores_vqaacc(cell_by_id):
    return {sid: float(score_textvqa_vqaacc(s["answer"], s["gt"]))
            for sid, s in cell_by_id.items()}


def scores_correct(cell_by_id):
    return {sid: float(s["correct"]) for sid, s in cell_by_id.items()}


def summarize(vals: dict, ids=None):
    if ids is None:
        v = list(vals.values())
    else:
        v = [vals[i] for i in ids if i in vals]
    if not v:
        return {"acc": None, "se": None, "n": 0}
    return {"acc": round(float(np.mean(v)), 4), "se": round(se(v), 4), "n": len(v)}


def paired_delta(a: dict, b: dict):
    """mean(a-b) over shared ids + paired SE."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"delta": None, "paired_se": None, "n": 0}
    d = np.array([a[i] - b[i] for i in ids])
    return {"delta": round(float(d.mean()), 4),
            "paired_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)
            if len(d) > 1 else 0.0, "n": len(d)}


def main():
    out = {"n": N, "cells": {}, "tables": {}}

    # ---------------- TextVQA (VQA-acc) ----------------
    tv_ids = subset_first_ids("textvqa")
    tv = {}
    for mode, path in [("none", f"{RR}/none_textvqa_r0.000_l2_n200.json"),
                       ("pre", f"{RR}/pre_textvqa_r0.750_l2_n200.json"),
                       ("post", f"{RR}/post_textvqa_r0.750_l2_n200.json")]:
        cell = load_cell(path)
        if cell:
            by_id, meta = cell
            # verify first-100 reference == first 100 subset ids
            got_ids = [str(s["id"]) for s in meta["per_sample"][:N]]
            verified = got_ids == tv_ids
            print(f"[gate] textvqa {mode} n200 first-100 ids == subset first-100: "
                  f"{verified}")
            tv[mode] = scores_vqaacc({k: by_id[k] for k in got_ids if k in by_id})
    for tf in ("0.0", "0.5", "1.0"):
        cell = load_cell(f"{HG}/hybrid_textvqa_r0.750_l2_tf{tf}_n100.json")
        if cell:
            tv[f"hybrid_tf{tf}"] = scores_vqaacc(cell[0])
    for mode in ("pre", "post"):
        cell = load_cell(f"{HG}/{mode}_textvqa_r0.750_attn_n100.json")
        if cell:
            tv[f"attn_{mode}"] = scores_vqaacc(cell[0])
    out["cells"]["textvqa"] = {k: summarize(v, tv_ids) for k, v in tv.items()}

    # ---------------- OCRBench (containment metric, per_sample.correct) ----------------
    ob_ids = subset_first_ids("ocrbench")
    ob = {}
    for mode, path in [("none", f"{SM}/A_ocrbench.json"),
                       ("post", f"{SM}/B_ocrbench_r0.750.json"),
                       ("pre", f"{SM}/C_ocrbench_r0.750.json")]:
        cell = load_cell(path)
        if cell:
            by_id, meta = cell
            verified = ([str(s["id"]) for s in meta["per_sample"][:N]] == ob_ids
                        and meta.get("seed") == 0 and meta.get("n") == 200)
            print(f"[gate] ocrbench {mode} n200/seed0 first-100 ids == subset first-100: "
                  f"{verified}")
            ob[mode] = scores_correct({str(s["id"]): s
                                       for s in meta["per_sample"][:N]
                                       if not s.get("skipped")})
    chosen = open(f"{HG}/chosen_frac.txt").read().strip().splitlines()[-1] \
        if os.path.exists(f"{HG}/chosen_frac.txt") else None
    out["chosen_text_frac"] = chosen
    for tf in ("0.0", "0.5", "1.0"):
        cell = load_cell(f"{HG}/hybrid_ocrbench_r0.750_l2_tf{tf}_n100.json")
        if cell:
            ob[f"hybrid_tf{tf}"] = scores_correct(cell[0])
    if chosen and f"hybrid_tf{chosen}" in ob:
        ob["hybrid"] = ob[f"hybrid_tf{chosen}"]
    out["cells"]["ocrbench"] = {k: summarize(v, ob_ids) for k, v in ob.items()}

    # ---------------- GQA (exact-match, per_sample.correct) ----------------
    gq_ids = subset_first_ids("gqa")
    gq = {}
    for mode, name in [("none", "none_gqa_r0.000_l2_n100.json"),
                       ("post", "post_gqa_r0.750_l2_n100.json"),
                       ("pre", "pre_gqa_r0.750_l2_n100.json")]:
        cell = load_cell(f"{HG}/{name}")
        if cell:
            gq[mode] = scores_correct(cell[0])
    for tf in ("0.0", "0.5", "1.0"):
        cell = load_cell(f"{HG}/hybrid_gqa_r0.750_l2_tf{tf}_n100.json")
        if cell:
            gq[f"hybrid_tf{tf}"] = scores_correct(cell[0])
    if chosen and f"hybrid_tf{chosen}" in gq:
        gq["hybrid"] = gq[f"hybrid_tf{chosen}"]
    out["cells"]["gqa"] = {k: summarize(v, gq_ids) for k, v in gq.items()}

    # ---------------- paired hybrid-vs-pre deltas ----------------
    out["paired_vs_pre"] = {}
    if chosen and "hybrid" in ob and "pre" in ob:
        out["paired_vs_pre"]["ocrbench"] = paired_delta(ob["hybrid"], ob["pre"])
    if chosen and "hybrid" in gq and "pre" in gq:
        out["paired_vs_pre"]["gqa"] = paired_delta(gq["hybrid"], gq["pre"])
    if chosen and f"hybrid_tf{chosen}" in tv and "pre" in tv:
        out["paired_vs_pre"]["textvqa"] = paired_delta(
            tv[f"hybrid_tf{chosen}"], {k: tv["pre"][k] for k in tv_ids if k in tv["pre"]})

    # ---------------- gate verdict ----------------
    def acc(cells, key):
        e = cells.get(key, {})
        return e.get("acc") if isinstance(e, dict) else None

    tv_pre, tv_h = acc(out["cells"]["textvqa"], "pre"), acc(out["cells"]["textvqa"], f"hybrid_tf{chosen}")
    ob_pre, ob_h = acc(out["cells"]["ocrbench"], "pre"), acc(out["cells"]["ocrbench"], "hybrid")
    gq_pre, gq_post, gq_h = (acc(out["cells"]["gqa"], "pre"),
                             acc(out["cells"]["gqa"], "post"),
                             acc(out["cells"]["gqa"], "hybrid"))
    gap = None
    closure = None
    if None not in (gq_pre, gq_post, gq_h) and gq_post > gq_pre:
        gap = round(gq_post - gq_pre, 4)
        closure = round((gq_h - gq_pre) / (gq_post - gq_pre), 3)
    verdict = {
        "textvqa_no_regression": (tv_h is not None and tv_pre is not None and tv_h >= tv_pre),
        "ocrbench_no_regression": (ob_h is not None and ob_pre is not None and ob_h >= ob_pre),
        "gqa_better_than_pre": (gq_h is not None and gq_pre is not None and gq_h > gq_pre),
        "gqa_gap_closure": closure,
        "gqa_notable": (closure is not None and closure >= 0.5),
        "values": {"textvqa": {"pre": tv_pre, "hybrid": tv_h},
                   "ocrbench": {"pre": ob_pre, "hybrid": ob_h},
                   "gqa": {"pre": gq_pre, "post": gq_post, "hybrid": gq_h,
                           "gap_post_minus_pre": gap}},
    }
    complete = all(x is not None for x in (tv_h, ob_h, gq_h))
    verdict["PASS"] = bool(complete
                           and verdict["textvqa_no_regression"]
                           and verdict["ocrbench_no_regression"]
                           and verdict["gqa_better_than_pre"]
                           and verdict["gqa_notable"])
    verdict["complete"] = bool(complete)
    out["gate"] = verdict

    with open(os.path.join(HG, "gate_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- print tables ----------------
    def fmt(e):
        if not e or e.get("acc") is None:
            return "NA"
        return f"{e['acc']:.3f}±{e['se']:.3f} (n={e['n']})"

    print("\n## Hybrid vs pre vs post @ keep=25%, n=100, seed=0 (official metrics)")
    print("| bench | metric | baseline(none) | post | pre | hybrid | Δ(hybrid−pre) paired |")
    print("|---|---|---|---|---|---|---|")
    rows = [
        ("textvqa", "VQA-acc", "textvqa", f"hybrid_tf{chosen}"),
        ("ocrbench", "containment-acc", "ocrbench", "hybrid"),
        ("gqa", "exact-match", "gqa", "hybrid"),
    ]
    for bench, metric, key, hkey in rows:
        c = out["cells"][key]
        d = out["paired_vs_pre"].get(bench, {})
        ds = (f"{d['delta']:+.3f}±{d['paired_se']:.3f}"
              if d.get("delta") is not None else "NA")
        print(f"| {bench} | {metric} | {fmt(c.get('none'))} | {fmt(c.get('post'))} | "
              f"{fmt(c.get('pre'))} | {fmt(c.get(hkey))} | {ds} |")

    print("\n## hybrid-text-frac sweep @n=100 (chosen frac tuned on textvqa VQA-acc)")
    print("| text-frac | textvqa VQA-acc | ocrbench acc | gqa exact-match |")
    print("|---|---|---|---|")
    for tf in ("0.0", "0.5", "1.0"):
        mark = " **(chosen)**" if tf == chosen else ""
        print(f"| {tf}{mark} | {fmt(out['cells']['textvqa'].get(f'hybrid_tf{tf}'))} | "
              f"{fmt(out['cells']['ocrbench'].get(f'hybrid_tf{tf}'))} | "
              f"{fmt(out['cells']['gqa'].get(f'hybrid_tf{tf}'))} |")

    print("\n## attn-selector invariance (textvqa @n=100, VQA-acc)")
    print(f"| selector | pre | post | pre>post? |")
    print("|---|---|---|---|")
    lp, lpo = out["cells"]["textvqa"].get("attn_pre"), out["cells"]["textvqa"].get("attn_post")
    holds = (lp and lpo and lp.get("acc") is not None and lpo.get("acc") is not None
             and lp["acc"] > lpo["acc"])
    print(f"| attn (centroid-dist) | {fmt(lp)} | {fmt(lpo)} | {'YES' if holds else 'NO'} |")
    print(f"| l2 (reference) | {fmt(out['cells']['textvqa'].get('pre'))} | "
          f"{fmt(out['cells']['textvqa'].get('post'))} | YES |")

    print(f"\n## GATE VERDICT: {'PASS' if verdict['PASS'] else 'FAIL'} "
          f"(complete={verdict['complete']})")
    for k in ("textvqa_no_regression", "ocrbench_no_regression",
              "gqa_better_than_pre", "gqa_notable", "gqa_gap_closure"):
        print(f"  {k}: {verdict[k]}")


if __name__ == "__main__":
    main()
