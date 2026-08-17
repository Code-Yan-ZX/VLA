#!/usr/bin/env python3
"""S6 Phase-2 pixel-domain spectral calib (cache-independent, no GPU).

Computes a per-image complexity proxy from the image's OWN pixels (downscaled
grayscale SVD) -- the same "spectral energy" signal E-AdaPrune reads from the
visual feature matrix, but at the pixel level so it is computable OFFLINE for
ALL n samples (vLLM's vision-encoder cache skips ~37% of images' hooks, so a
feature-side calib only reaches ~63% of samples -- see notes/s6_stitching).

Output: {n, frac: [complexity]} where the per-image fraction is a scale-free
rank/energy signal the runner's budget table maps to keep-k via
round(f * frac) at both the placeholder and the prune site (same pixels ->
same frac -> consistent). kl scheme: two modes:
  spectral  : cumulative energy fraction tau of the downscaled gray matrix's
              singular values (E-AdaPrune).
  erank     : participation-ratio effective rank normalized by the bench mean.
Both are rank-aggregated to [frac_min, frac_max] then the driver normalizes to
iso-total token budget vs uniform r (stitch_budget_alloc-style).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/media/disk2/YZX/research/vla")
# Same max-pixels semantics the runner uses per bench (docvqa/ocrbench: 4M).
MAX_PX = {"textvqa": 0, "docvqa": 4000000, "ocrbench": 4000000, "gqa": 0}


def _spectral_frac_energy(s: np.ndarray, tau: float) -> float:
    w = s * s
    total = float(w.sum())
    if total <= 0:
        return 1.0
    cum = float(np.cumsum(w)[int(np.clip(tau * w.size, 0, w.size - 1))]) if tau < 1 else total
    return float(cum / total) if total > 0 else 1.0


def image_frac(path: str, mode: str, tau: float, erank_ref: float | None = None) -> float:
    """Scale-free complexity in [0, 1]. spectral: normalized cumulative energy
    at index tau*n (larger tau = higher complexity kept for long tails).
    erank: participation ratio capped at a modest absolute scale."""
    im = Image.open(path).convert("L")
    im.thumbnail((128, 128))
    a = np.asarray(im, dtype=np.float64)
    s = np.linalg.svd(a, compute_uv=False)
    if mode == "spectral":
        return _spectral_frac_energy(s, tau)
    w = s * s
    tot = float(w.sum())
    pr = float((tot * tot) / float((w * w).sum())) if tot > 0 else 1.0
    return float(np.clip(pr / 64.0, 0.05, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", choices=["spectral", "erank"], default="spectral")
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.subset)][: args.n]
    fracs, missed = [], 0
    for i, s in enumerate(samples):
        img = s.get("image")
        if isinstance(img, dict):          # some jsonls nest {image: path}
            img = img.get("image", img.get("path"))
        if isinstance(img, list):          # multi-image entry -> keep first
            img = img[0]
        p = Path(str(img))
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            # the runner routes through runs/data/_capped_cache; try there
            p = REPO / "runs/data/_capped_cache" / Path(str(img)).name
        try:
            fracs.append(image_frac(str(p), args.mode, args.tau))
        except Exception:
            missed += 1
            fracs.append(0.5)
    out = {"bench": args.bench, "mode": args.mode, "tau": args.tau,
           "n": len(fracs), "missing": missed, "frac": fracs}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: n={len(fracs)} missing={missed} "
          f"frac mean={np.mean(fracs):.3f} range=[{np.min(fracs):.3f},"
          f"{np.max(fracs):.3f}]")


if __name__ == "__main__":
    main()