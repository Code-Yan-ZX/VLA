#!/usr/bin/env python3
"""Build an iso-total per-image budget file for D2 (pixel-spectral content
reallocation).

Reads the pixel-frac JSON from scripts/stitch_pixel_calib.py and renormalizes
the fractions so their MEAN equals (1-r) -- making the total budget
approximately iso-token with a uniform-r run (exact to within per-image
rounding). Output is the runner's frac-mode budget file: {"frac": [...]}.

The reallocation story: text-dense images (high spectral energy fraction) get
a HIGHER keep fraction (more tokens survive), simple images get fewer. With
the total held ~constant, content-aware reallocation should improve accuracy
where compression depth costs most (the stage law: text-dense hurts most at
deep compression).

Usage: python scripts/build_d2_budget.py --bench b --r 0.75 --frac X --subset Y --out Z
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--frac", required=True, help="pixel-frac JSON from "
                     "stitch_pixel_calib.py")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frac = json.load(open(args.frac))["frac"]
    samples = [json.loads(l) for l in open(args.subset)][: len(frac)]
    n = min(len(frac), len(samples))
    frac, samples = frac[:n], samples[:n]

    target_mean = 1.0 - args.r            # keep fraction of a uniform run
    f = np.array(frac, dtype=np.float64)
    # guard degenerate (constant) spectra: no reallocation signal -> uniform
    if f.std() < 1e-6:
        frac_n = np.full(n, target_mean)
    else:
        # center on target mean; clip to [0.05, 0.95] so no image collapses
        frac_n = np.clip(f * (target_mean / f.mean()), 0.05, 0.95)
    # scale to hit target mean EXACTLY after clipping
    frac_n = frac_n * (target_mean / frac_n.mean())

    out = {"bench": args.bench, "r": args.r, "n": n, "frac": frac_n.tolist(),
           "target_mean": target_mean, "mean": float(frac_n.mean()),
           "min": float(frac_n.min()), "max": float(frac_n.max())}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: n={n} target_mean={target_mean:.3f} "
          f"mean={frac_n.mean():.3f} range=[{frac_n.min():.3f},"
          f"{frac_n.max():.3f}] std={frac_n.std():.3f}")


if __name__ == "__main__":
    main()
