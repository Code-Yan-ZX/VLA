"""Analyzer for the V3 tightening campaign (runs/v3_tighten_cells/*.json).

Reports three things, from whatever cells are present (robust to missing files):
  (1) mean+/-std of the C-B (pre-minus-post) accuracy gap across seeds, on the
      4 headline cells: DocVQA / TextVQA x r=0.75 (seed 0 = the (a) cell, seeds
      1,2 = the (c) repeats).
  (2) iso-token pre-vs-post stage effect: per benchmark x r in {0.75,0.875},
      B (post) vs C (pre) accuracy and their gap (seed 0, L2). Plus baseline A.
  (3) L2-vs-attn selector comparison at r=0.75 (B and C): does the stage effect
      hold under a different selector?

Reads only final JSONs; never touches the GPU. Prints a readable summary and
writes runs/v3_tighten_cells/_summary.json.
"""
from __future__ import annotations
import os, sys, glob, json, math, argparse
from collections import defaultdict


def mean_std(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"mean": m, "std": 0.0, "n": len(xs), "values": xs}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": m, "std": math.sqrt(var), "n": len(xs), "values": xs}


def load(d):
    """Return dict keyed by (mode, bench, r, selector, seed) -> acc (or None)."""
    cells = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p).startswith("_") or os.path.basename(p).startswith("smoke"):
            continue
        try:
            with open(p) as f:
                o = json.load(f)
        except Exception:
            continue
        key = (o.get("mode"), o.get("benchmark"), o.get("r"),
               o.get("selector", "l2"), o.get("seed", 0))
        cells[key] = o.get("acc")
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/v3_tighten_cells")
    a = ap.parse_args()
    C = load(a.dir)
    lines = []
    P = lines.append

    P("=" * 72)
    P("V3 TIGHTENING SUMMARY  (source: %s)" % a.dir)
    P("=" * 72)

    # ---- (1) mean+/-std of C-B across seeds on headline cells ----------------
    P("\n(1) C-B (pre minus post) accuracy gap: mean+/-std over seeds")
    P("    seed 0 = (a) cell; seeds 1,2 = (c) repeats.  L2, r=0.75.")
    summary_cb = {}
    for BENCH in ["docvqa", "textvqa"]:
        gaps = []
        per_seed = []
        for SEED in [0, 1, 2]:
            b = C.get(("post", BENCH, 0.75, "l2", SEED))
            c = C.get(("pre", BENCH, 0.75, "l2", SEED))
            if b is None or c is None:
                per_seed.append((SEED, None, None, None))
                continue
            per_seed.append((SEED, b, c, c - b))
            gaps.append(c - b)
        ms = mean_std(gaps)
        summary_cb[BENCH] = {"per_seed": per_seed, "mean_std": ms}
        P("    %s:" % BENCH)
        for sd, b, c, g in per_seed:
            gb = "%.4f" % g if g is not None else "n/a"
            P("      seed=%d  B(post)=%s C(pre)=%s  C-B=%s"
              % (sd, "%.4f" % b if b is not None else "n/a",
                 "%.4f" % c if c is not None else "n/a", gb))
        if ms:
            P("      >>> C-B mean=%.4f  std=%.4f  (n=%d)"
              % (ms["mean"], ms["std"], ms["n"]))
        else:
            P("      >>> C-B: no data")

    # ---- (2) iso-token stage effect (seed 0, L2) ----------------------------
    P("\n(2) Iso-token pre-vs-post stage effect (seed 0, L2)")
    P("    A=baseline  B=post-merger  C=pre-merger  (acc)")
    summary_stage = {}
    for BENCH in ["docvqa", "textvqa", "gqa"]:
        P("    %s:" % BENCH)
        row = {}
        a0 = C.get(("none", BENCH, 0.0, "l2", 0))
        P("      A  baseline      = %s" % ("%.4f" % a0 if a0 is not None else "n/a"))
        row["A"] = a0
        for R in [0.75, 0.875]:
            b = C.get(("post", BENCH, R, "l2", 0))
            c = C.get(("pre", BENCH, R, "l2", 0))
            gap = (c - b) if (b is not None and c is not None) else None
            row["r%s_B" % R] = b
            row["r%s_C" % R] = c
            row["r%s_CmB" % R] = gap
            P("      r=%.3f  B=%s  C=%s  C-B=%s  %s"
              % (R, "%.4f" % b if b is not None else "n/a",
                 "%.4f" % c if c is not None else "n/a",
                 "%+.4f" % gap if gap is not None else "n/a",
                 "(pre wins)" if (gap is not None and gap > 0)
                 else ("(post wins)" if (gap is not None and gap < 0) else "")))
        summary_stage[BENCH] = row

    # ---- (3) L2-vs-attn selector comparison @ r=0.75 ------------------------
    P("\n(3) Selector comparison @ r=0.75 (does the stage effect survive attn?)")
    P("    B/C under l2 vs attn, and the C-B gap under each selector.")
    summary_sel = {}
    for BENCH in ["docvqa", "textvqa", "gqa"]:
        P("    %s:" % BENCH)
        row = {}
        for SEL in ["l2", "attn"]:
            b = C.get(("post", BENCH, 0.75, SEL, 0))
            c = C.get(("pre", BENCH, 0.75, SEL, 0))
            gap = (c - b) if (b is not None and c is not None) else None
            row[SEL] = {"B": b, "C": c, "CmB": gap}
            P("      %-5s  B=%s  C=%s  C-B=%s"
              % (SEL,
                 "%.4f" % b if b is not None else "n/a",
                 "%.4f" % c if c is not None else "n/a",
                 "%+.4f" % gap if gap is not None else "n/a"))
        l2g = (row.get("l2", {}) or {}).get("CmB")
        atg = (row.get("attn", {}) or {}).get("CmB")
        if l2g is not None and atg is not None:
            agree = (l2g > 0) == (atg > 0)
            P("      >>> stage-effect sign agrees across selectors: %s"
              % ("YES" if agree else "NO  <-- selector-sensitive"))
        summary_sel[BENCH] = row

    P("\n" + "=" * 72)
    out = {"cb_mean_std": summary_cb, "stage_effect": summary_stage,
           "selector_compare": summary_sel}
    with open(os.path.join(a.dir, "_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    P("Wrote %s" % os.path.join(a.dir, "_summary.json"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
