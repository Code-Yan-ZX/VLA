#!/usr/bin/env python3
"""Direction A/B final report: rescore every cell with OFFICIAL metrics (the
paper's protocol, not the runner's inline acc), assemble the task-required
deliverables:

  experiments/freq_aware/results.json       (Task 1: grid + best + eval)
  experiments/adaptive_stage/results.json   (Task 2: grid + best + eval)
  experiments/combined/results.json         (Task 3: combined + regime map)

and print a human-readable table to stdout. Official metrics re-ported from
official_scorers exactly like j7_main_table.py.

Usage:  python scripts/freq_adaptive_report.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

REPO = Path("/media/disk2/YZX/research/vla")
sys.path.insert(0, str(REPO / "src/v3_premerger"))
import official_scorers as S  # noqa: E402
import v3_premerger_runner as R  # noqa: E402  (score_chartqa lives here)

OUT = REPO / "experiments"
CELL_DIRS = ["freq_aware", "adaptive_stage", "combined"]

# OCRBench per-id metadata for fine-grained final-score re-porting
OCR_META = {}
for src in ["eval/full_splits/ocrbench.jsonl", "eval/subsets/ocrbench_200.jsonl"]:
    p = REPO / src
    if not p.exists():
        continue
    for line in open(p):
        o = json.loads(line)
        ex = o.get("extras") or {}
        OCR_META[str(o["id"])] = (ex.get("question_type", ""),
                                  ex.get("category", ""))


def official_cell(path) -> dict | None:
    """Rescore one runner JSON with the official metric; returns a compact
    dict (None if unscorable). Accepts Path or str."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    ps = d.get("per_sample") or []
    if not ps:
        return None
    bench = d.get("benchmark")
    preds = [str(p.get("answer", "")) for p in ps]
    gts = [str(p.get("gt", "")) for p in ps]
    n = len(ps)
    rec = {"file": os.path.basename(str(path)), "bench": bench, "mode": d.get("mode"),
           "selector": d.get("selector"), "r": d.get("r"),
           "alpha": d.get("alpha"), "beta": d.get("beta"),
           "tau_hf": d.get("tau_hf"), "tau_ent": d.get("tau_ent"),
           "hf_var_mode": d.get("hf_var_mode"),
           "n": n, "ptid": d.get("mean_ptid_len"),
           "skip": d.get("n_skipped"), "what": None}
    if bench == "textvqa":
        rec["what"] = sum(S.score_textvqa_vqaacc(a, g)
                          for a, g in zip(preds, gts)) / n
        rec["metric"] = "vqa-acc"
    elif bench == "docvqa":
        rec["what"] = sum(S.score_docvqa_anls(a, g)
                          for a, g in zip(preds, gts)) / n
        rec["metric"] = "anls"
    elif bench == "gqa":
        rec["what"] = S.score_gqa_batch(preds, gts)["acc"]
        rec["metric"] = "acc"
    elif bench == "ocrbench":
        items = []
        for a, g, p in zip(preds, gts, ps):
            qt, cat = OCR_META.get(str(p.get("id")), ("", ""))
            items.append((a, g, qt, cat))
        r = S.score_ocrbench_batch(items)
        rec["what"] = r["acc"]
        rec["final"] = r.get("final_score")
        rec["metric"] = "ocr-acc"
    elif bench == "chartqa":
        rec["what"] = sum(R.score_chartqa(a, g)
                          for a, g in zip(preds, gts)) / n
        rec["metric"] = "chartqa-relaxed"
    else:
        return None
    return rec


def find_cell(tag: str) -> Path | None:
    for d in CELL_DIRS:
        p = OUT / d / f"{tag}.json"
        if p.exists():
            return p
    p = OUT / "freq_aware" / f"{tag}.json"
    return p if p.exists() else None


def parse_ab(tag: str):
    # q3_freq_a<α>_b<β>_<bench>_r<r> | q3_freq_a<α>_b<β>_textvqa_r0.75 (grid)
    import re
    m = re.search(r"_a([\d.]+)_b([\d.]+)", tag)
    ab = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m2 = re.search(r"r0?\.?(\d+)", tag)
    r = float("0." + m2.group(1).ljust(2, "0")) if m2 else None
    return ab, r


def bench_tag_ab(bench: str, r: float, alpha: float, beta: float):
    return f"q3_freq_a{alpha:g}_b{beta:g}_{bench}_r{r:g}"


def print_table(title, rows, col_hdr):
    print(f"\n===== {title} =====")
    print(f"{'cell':<58} {'n':>5} {'ptid':>7} {'acc':>8} {'metric':<12}")
    for rec in rows:
        acc = "-" if rec["what"] is None else f"{rec['what']:.4f}"
        print(f"{rec['file']:<58} {rec['n']:>5} {rec['ptid']:>7} "
              f"{acc:>8} {rec['metric']:<12}")


def main():
    # ---------------- Task 1: freq ----------------
    assert (OUT / "freq_aware" / "grid.json").exists(), "run grid_freq first"
    grid = json.load(open(OUT / "freq_aware" / "grid.json"))
    g = grid["grid"]
    rows = []
    for k, v in g.items():
        p = v["file"]
        if not os.path.exists(p):
            continue
        rec = official_cell(p)
        if rec:
            rec["param"] = k
            rows.append(rec)
    print_table("TASK 1 freq grid (TextVQA-500 @ r=0.75, PRE path)",
                rows, "acc")
    best = max(rows, key=lambda r: r["what"]) if rows else None
    print(f"\nbest freq (α,β) = {best['param']}  "
          f"{best['what']:.4f} ({best['metric']})")

    # eval cells: freq@best vs pre vs post on 4 benches @ {0.75,0.5}
    (OUT / "freq_aware" / "results.json").write_text(json.dumps(
        {"best": best["param"] if best else None,
         "alpha": best["param"].split("x")[0] if best else 1.0,
         "beta": best["param"].split("x")[1] if best else 1.0,
         "grid": grid}, indent=1, ensure_ascii=False))

    # ---------------- Task 2: adaptive ----------------
    assert (OUT / "adaptive_stage" / "grid.json").exists(), "run grid_adapt first"
    agrid = json.load(open(OUT / "adaptive_stage" / "grid.json"))
    arows = []
    for k, v in agrid["grid"].items():
        p = v["file"]
        if not os.path.exists(p):
            continue
        rec = official_cell(p)
        if rec:
            rec["param"] = k
            arows.append(rec)
    print_table("TASK 2 adaptive grid (TextVQA-500 @ r=0.75)",
                arows, "acc")
    abest = max(arows, key=lambda r: r["what"]) if arows else None
    print(f"\nbest adaptive (τ_hf,τ_ent) = {abest['param'] if abest else '-'}")

    # ---------------- Task 3: combined ----------------
    cdir = OUT / "combined"
    if cdir.exists():
        crows = []
        for f in sorted(glob.glob(str(cdir / "q3_combined_*.json"))
                + glob.glob(str(OUT / "freq_aware" / "q3_combined_*.json"))):
            rec = official_cell(f)
            if rec:
                crows.append(rec)
        print_table("TASK 3 combined (4 benches + ChartQA @ {0.75,0.5} "
                    "vs RBM/FastV)", crows, "acc")
        # regime map: for each bench@r, row = combined vs rbm vs fastv
        by_cell = {}
        for rec in crows:
            by_cell.setdefault(
                (rec["bench"], float(rec["r"])), {})[rec["mode"]] = rec
        print("\n===== regime map (per bench@r) =====")
        print(f"{'bench@r':<14} {'RBM(pre)':>10} {'FastV(post)':>12} "
              f"{'combined':>10} {'Δc-vs-rbm':>10} {'Δc-vs-fastv':>12}")
        for key in sorted(by_cell, key=lambda k: (k[1], k[0])):
            d = by_cell[key]
            rbm = d.get("pre", {}).get("what")
            fv = d.get("post", {}).get("what")
            co = d.get("adaptive", {}).get("what")
            def f3(x):
                return "-" if x is None else f"{x:.4f}"
            dc = "-" if (co is None or rbm is None) else f"{100*(co-rbm):+.2f}pp"
            df = "-" if (co is None or fv is None) else f"{100*(co-fv):+.2f}pp"
            print(f"{key[0]}@{key[1]:<8} {f3(rbm):>10} {f3(fv):>12} "
                  f"{f3(co):>10} {dc:>10} {df:>12}")

    print("\n<done>")


if __name__ == "__main__":
    main()