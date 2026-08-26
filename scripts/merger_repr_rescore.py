#!/usr/bin/env python3
"""Rescore merger_repr experiment outputs with the OFFICIAL metrics (CPU).

Same official-scorer machinery as scripts/rescore_official.py (rescore_cell),
pointed at the merger_representation_goal output directory instead of the
repo's standard CELL_DIRS. Writes runs/merger_repr/official/*.json per cell.

Usage:
  python scripts/merger_repr_rescore.py                 # all cells in dir
  python scripts/merger_repr_rescore.py --dir runs/merger_repr/gateB --benchmark textvqa
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from rescore_official import (  # noqa: E402
    ALL_BENCHES, GQA, OCRBENCH, TEXTVQA, DOCVQA,
    load_ocrbench_meta, load_subset_gt, rescore_cell,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "runs/merger_repr"),
                    help="directory of runner output JSON cells (recursive)")
    ap.add_argument("--benchmark", action="append", default=None,
                    choices=ALL_BENCHES)
    ap.add_argument("--pattern", default="*.json",
                    help="glob pattern for cells")
    args = ap.parse_args()
    selected = set(args.benchmark) if args.benchmark else set(ALL_BENCHES)
    files = sorted(glob.glob(os.path.join(args.dir, "**", args.pattern),
                             recursive=True))
    # exclude the official output dir itself
    files = [f for f in files if "/official/" not in f.replace(os.sep, "/")]
    ocr_meta = load_ocrbench_meta() if OCRBENCH in selected else {}
    subset_gt_cache: dict = {}
    out_dir = os.path.join(args.dir, "official")
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for f in files:
        rel = os.path.relpath(f, REPO)
        try:
            cell = json.load(open(f))
        except Exception as e:
            print(f"[skip] {rel}: {e}")
            continue
        rec = rescore_cell(cell, rel, subset_gt_cache,
                           ocr_meta=ocr_meta, selected=selected)
        records.append(rec)
        base = os.path.basename(f).replace(".json", "_official.json")
        with open(os.path.join(out_dir, base), "w") as fh:
            json.dump(rec, fh, indent=1)
        m = rec.get("new_metric_mean")
        eff = m if m is not None else rec.get("old_acc")
        print(f"{os.path.basename(f):42s} n={rec.get('n_rescored') or rec.get('n_total'):>5} "
              f"official={eff if eff is None else round(eff,4)} "
              f"({rec.get('official_metric')}) mode={rec.get('mode')}")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"records": records,
                   "selected_benchmarks": sorted(selected),
                   "n_cells": len(records)}, fh, indent=1)
    print(f"[rescore] {len(records)} cells -> {out_dir}")


if __name__ == "__main__":
    main()
