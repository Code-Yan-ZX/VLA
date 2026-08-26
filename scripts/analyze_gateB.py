#!/usr/bin/env python3
"""Gate B analysis: official metrics per cell, position comparison, ratio
comparison (disjoint dev n=64 x 4). Reads runs/merger_repr/gateB/*.json.
Usage: python scripts/analyze_gateB.py [--phase 1|2|3]
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from rescore_official import (  # noqa: E402
    GQA, OCRBENCH, TEXTVQA, DOCVQA, load_ocrbench_meta, rescore_cell,
)

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
OFFICIAL = {TEXTVQA: "vqa_accuracy", DOCVQA: "anls",
            OCRBENCH: "ocrbench_1000", GQA: "exact_match"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "runs/merger_repr/gateB"))
    args = ap.parse_args()
    ocr_meta = load_ocrbench_meta()
    cache: dict = {}
    cells = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        base = os.path.basename(f)
        try:
            cell = json.load(open(f))
        except Exception:
            continue
        rec = rescore_cell(cell, base, cache, ocr_meta=ocr_meta,
                           selected={TEXTVQA, DOCVQA, OCRBENCH, GQA})
        # effective = official where available. textvqa/docvqa: new_metric_mean
        # (vqa_acc / anls). ocrbench: official five-category /1000 extrapolated
        # to 0-1 (the benchmark's official score is the /1000 roll-up).
        if cell.get("benchmark") == OCRBENCH:
            eff = rec.get("ocr_extrap_1000")
            if eff is not None:
                eff = eff / 1000.0
        else:
            eff = rec.get("new_metric_mean")
        if eff is None:
            eff = rec.get("old_acc")
        cells[base] = {"cell": cell, "rec": rec, "eff": eff,
                       "n": rec.get("n_rescored") or rec.get("n_total")}

    def eff_bench(arm, bench):
        for k, v in cells.items():
            if k.startswith(f"{bench}_{arm}"):
                return v["eff"], v["n"]
        return None, None

    def arm_macro(arm):
        vals = []
        for b in BENCHES:
            e, n = eff_bench(arm, b)
            if e is not None:
                vals.append(e)
        return sum(vals) / len(vals) if vals else None

    print("=" * 78)
    print("GATE B — official metrics (disjoint dev n=64 x 4)")
    print("=" * 78)
    hdr = f"{'arm':<14s}" + "".join(f"{b:>12s}" for b in BENCHES) + f"{'macro':>10s}"
    print(hdr)
    for arm in ["full", "rbm_native", "fastv_k3", "c1r2_dup", "c1r2_adj",
                "c1r1_dup", "c1r3_dup", "c2r05_dup", "c2r1_dup", "c2r15_dup"]:
        row = [arm]
        alln = []
        for b in BENCHES:
            e, n = eff_bench(arm, b)
            row.append(f"{e:.4f}" if e is not None else "-")
            if n is not None:
                alln.append(n)
        row.append(f"{arm_macro(arm):.4f}" if arm_macro(arm) is not None else "-")
        print(f"{row[0]:<14s}" + "".join(f"{v:>12s}" for v in row[1:]) +
              f"{row[-1]:>10s}")

    print()
    print("--- Position comparison (C1 ratio=0.2): duplicate vs adjacent ---")
    for b in BENCHES:
        ed, nd = eff_bench("c1r2_dup", b)
        ea, na = eff_bench("c1r2_adj", b)
        if ed is None or ea is None:
            print(f"  {b}: incomplete ({ed}, {ea})")
            continue
        print(f"  {b}: dup={ed:.4f} adj={ea:.4f}  d(dup-adj)={ed-ea:+.4f}")
    md, ma = arm_macro("c1r2_dup"), arm_macro("c1r2_adj")
    print(f"  macro: dup={md:.4f} adj={ma:.4f}  d={ (md-ma) if md is not None and ma is not None else None }")


if __name__ == "__main__":
    main()
