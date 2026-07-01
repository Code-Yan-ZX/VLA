#!/usr/bin/env python
"""Re-score a saved serve_bench metrics JSON with the current (fixed) scorer.

Usage:
    python -m scripts.rescore [--benchmark gqa|textvqa] runs/p2_probe/<name>_metrics.json [...]

Reads each metrics JSON's "raw" array ({answer, gt, [choices]}), re-scores with
src.serve_bench.score_{gqa,textvqa}, prints old->new accuracy, and optionally
rewrites the file in place (--write) with the new per-row `correct` and agg.

Used to validate the scorer fix on the existing r0 run without re-running the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as `python -m scripts.rescore` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serve_bench import SCORERS, mean_stderr  # noqa: E402


def rescore(path: str, benchmark: str, write: bool) -> tuple[int, int, int]:
    d = json.load(open(path))
    raw = d["raw"]
    scorer = SCORERS[benchmark]
    old_correct = sum(r["correct"] for r in raw)
    # re-score; `choices` is optional and only meaningful for GQA
    for r in raw:
        choices = r.get("choices") or r.get("extra", {}).get("choices")
        r["correct_new"] = scorer(r["answer"], r["gt"], choices)
    new_correct = sum(r["correct_new"] for r in raw)
    n = len(raw)
    old_acc = old_correct / n if n else float("nan")
    new_acc = new_correct / n if n else float("nan")
    print(f"{path}")
    print(f"  benchmark={benchmark} n={n}  old_acc={old_acc:.4f} ({old_correct}/{n})"
          f"  ->  new_acc={new_acc:.4f} ({new_correct}/{n})")
    if write:
        for r in raw:
            r["correct"] = r.pop("correct_new")
        # recompute agg.accuracy + re-derive speedups stay the same (they use latency)
        d["agg"]["accuracy"] = new_acc
        json.dump(d, open(path, "w"), indent=2)
        print(f"  (rewritten in place)")
    return old_correct, new_correct, n


def main():
    ap = argparse.ArgumentParser(description="re-score a serve_bench metrics JSON")
    ap.add_argument("paths", nargs="+", help="metrics JSON path(s)")
    ap.add_argument("--benchmark", required=True, choices=["gqa", "textvqa"])
    ap.add_argument("--write", action="store_true", help="rewrite file with new scores")
    args = ap.parse_args()
    for p in args.paths:
        rescore(p, args.benchmark, args.write)


if __name__ == "__main__":
    main()
