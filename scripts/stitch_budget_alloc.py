#!/usr/bin/env python3
"""S6 Phase-2 offline budget allocator (E-AdaPrune/PRUNESID stitch support).

Turns a calib dump (per-image spectral stats from the runner, ordered by the
subset index) into a per-image budget list for a target global budget
B = sum(round(f_i*(1-r))) -- the SAME total a uniform-r run would use, so the
adaptive run is ISO-TOKEN with the uniform incumbent (cleanest comparison).
This is exactly E-AdaPrune's "spectral-energy budget handed to the selector",
with AgilePruner's erank as the complexity measure, mounted on the RBM
(rank-before-merge) selector.

Allocation: k_raw_i = f(spectral energy fraction tau, or erank); clamp per
image to [(1-clamp), (1+clamp)] x base_i (base_i = round(f_i*(1-r))), then
largest-remainder scale the clamps to hit B EXACTLY.

  First run (alpha-mode budget early-delivery; used by the runner via
  --budget-file):
    1) calib by the runner:   --mode pre --budget-calib calib.json ...
    2) alloc here:            python scripts/stitch_budget_alloc.py \
                                 --calib experiments/sota_stitch/calib_all.json \
                                 --r 0.25 --mode spectral --tau 0.9 \
                                 --out experiments/sota_stitch/budgets_sp0.9_r0.25.json
    3) eval by the runner:    --mode pre --r 0.25 --budget-file budgets_*.json ...
"""
from __future__ import annotations

import argparse
import json

import torch


def budget_k_raw(mode: str, tau: float, s: torch.Tensor, f: int) -> int:
    """Raw per-image budget from the singular-value spectrum ``s`` (descending).
    mode='spectral': smallest k with cumulative energy >= tau*total energy
        (E-AdaPrune).
    mode='erank': k proportional to the participation-ratio effective rank
        (AgilePruner), normalized so k_raw == f at erank == f (rare) -- the
        allocator's cross-image normalization makes the absolute scale moot.
    """
    if s.numel() == 0:
        return f
    s = s.double()                                    # avoid overflow in squares
    w = (s * s).clamp_min(0.0)
    total = float(w.sum())
    if not (total > 0.0):                             # degenerate spectrum
        return max(1, f // 2)
    if mode == "spectral":
        cum = w.cumsum(0)
        nel = int((cum < tau * total).sum().item())
        return max(1, nel + 1)
    if mode == "erank":
        sq = float((w * w).sum())
        pr = (total * total / sq) if sq > 0.0 else 1.0
        if not (pr > 0.0) or pr != pr:                # NaN/inf guard
            pr = 1.0
        return max(1, min(f, int(round(f * (pr / f)))))  # erank in [1, f]
    raise ValueError(mode)


def allocate(k_raw: list[int], base: list[int], clamp: float) -> list[int]:
    """Allocate EXACTLY sum(base) tokens (iso-token with the uniform incumbent),
    per image within [(1-clamp), (1+clamp)] x base (floor 1). The deviation
    from the clamped proportional intent is spread greedily: raise uses the
    largest under-capacity, drop uses the largest above-floor headroom."""
    B = sum(base)
    n = len(base)
    lo = [max(1, int(round(b * (1.0 - clamp)))) for b in base]
    hi = [max(lo[i], int(round(base[i] * (1.0 + clamp)))) for i in range(n)]
    k = [min(h, max(l, raw)) for raw, l, h in zip(k_raw, lo, hi)]

    deficit = B - sum(k)
    if deficit > 0:
        for i in sorted(range(n), key=lambda i: -(hi[i] - k[i])):
            if deficit <= 0:
                break
            add = min(hi[i] - k[i], deficit)
            k[i] += add
            deficit -= add
    elif deficit < 0:
        for i in sorted(range(n), key=lambda i: -(k[i] - lo[i])):
            if deficit >= 0:
                break
            sub = min(k[i] - lo[i], -deficit)
            k[i] -= sub
            deficit += sub
    assert sum(k) == B and all(lo[i] <= k[i] <= hi[i] for i in range(n)), \
        (sum(k), B, k, lo, hi)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True,
                    help="runner calib dump (--budget-calib out.json)")
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--mode", choices=["spectral", "erank"], default="spectral")
    ap.add_argument("--tau", type=float, default=0.9,
                    help="spectral energy fraction (mode=spectral)")
    ap.add_argument("--clamp", type=float, default=0.5,
                    help="per-image budget clamp fraction of the uniform base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cal = json.load(open(args.calib))
    entries = cal["per_image"]                    # [{f, svd_desc}...] in index order
    f_list = [e["f"] for e in entries]
    base = [max(1, int(round(f * (1.0 - args.r)))) for f in f_list]
    k_raw = []
    for e in entries:
        s = torch.tensor(e["svd_desc"], dtype=torch.float32)
        k_raw.append(budget_k_raw(args.mode, args.tau, s, e["f"]))
    k = allocate(k_raw, base, args.clamp)
    out = {"r": args.r, "mode": args.mode, "tau": args.tau,
           "n": len(k), "total": sum(k), "total_base": sum(base),
           "mean_k": round(sum(k) / len(k), 3),
           "min_k": min(k), "max_k": max(k),
           "k": k}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: n={len(k)} total={sum(k)} (base {sum(base)}) "
          f"mean_k={out['mean_k']} range=[{min(k)},{max(k)}]")


if __name__ == "__main__":
    main()