#!/usr/bin/env python
"""ElasticVis EV-0 offline go/no-go (ZERO GPU).

Replays measured v2 per-request latency/accuracy via LatPred + AccuracyTerm under
several arrival regimes, and compares allocators on goodput@SLO. This is the
zero-GPU test of the headline claim: "ElasticVis (per-request k) > best fixed-r
on goodput@SLO under open-loop variable load."

  - Sanity:   closed_c64 must reproduce v2 c64 fixed-rate goodput@TTFT<=5s
              (fixed_r0 ~ 1.4-1.8, fixed_r75 ~ 10.8-14.4 req/s). If it doesn't,
              LatPred/sim wiring is wrong -> DO NOT trust the rest.
  - Headline: open-loop regimes -> does Greedy/Lagrangian beat the best Fixed?

Run:  python -m src.elasticvis.run_ev0
      (cd /media/disk2/YZX/research/vla ; python -m src.elasticvis.run_ev0)
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../vla
EV0 = ROOT / "runs" / "elasticvis_ev0"

# ---- ElasticVis constants (LLaVA-1.5-7B, GQA) --------------------------------
N_IMG = 576                       # native image tokens
K_GRID = (144, 288, 576)         # discrete k values with measured accuracy
K_BOUNDS = (144, 576)            # (k_min, k_max) bounds for allocator k_range
SLO_TTFT_5S = 5000.0              # v2 headline SLO: TTFT <= 5s
MAX_SEQS = 64                     # c64 ceiling on 1xA40


def _import():
    """Lazy import so `--help`/errors don't require the modules to exist yet."""
    from .predictors import load_latpred, load_accuracy
    from .allocator import (FixedAllocator, GreedyAllocator,
                            LagrangianAllocator, OracleAllocator)
    from . import sim
    return {
        "load_latpred": load_latpred, "load_accuracy": load_accuracy,
        "Fixed": FixedAllocator, "Greedy": GreedyAllocator,
        "Lagr": LagrangianAllocator, "Oracle": OracleAllocator,
        "sim": sim,
    }


def build_allocators(m):
    return {
        "fixed_r0":   m["Fixed"](0.00, N=N_IMG),
        "fixed_r25":  m["Fixed"](0.25, N=N_IMG),
        "fixed_r50":  m["Fixed"](0.50, N=N_IMG),
        "fixed_r75":  m["Fixed"](0.75, N=N_IMG),
        "greedy":     m["Greedy"](k_grid=K_GRID),
        "lagrangian": m["Lagr"](k_grid=K_GRID),
        "oracle":     m["Oracle"](k_grid=K_GRID),
    }


def regimes(m, n_ids):
    """Arrival-process factories. closed_c64 = v2 bench (sanity); rest = H1/H1b."""
    S = m["sim"]
    N = n_ids * 3
    return {
        "closed_c64":   lambda: S.closed_loop(c=64, n=N),
        # open-loop: admission-time load varies => H1 lives here (headline)
        "openloop_lo":  lambda: S.open_loop_poisson(rate=2.0, n=N, seed=1),
        "openloop_hi":  lambda: S.open_loop_poisson(rate=8.0, n=N, seed=1),
        "bursty":       lambda: S.bursty(rate=5.0, n=N, seed=1, burst_frac=0.5),
        # mixed-SLO (per-request deadlines) over open-loop => H1b
        "mixed_slo":    lambda: S.mixed_slo(S.open_loop_poisson(rate=5.0, n=N, seed=1),
                                             n=N, seed=2),
    }


def run_regime(m, name, factory, lat, acc, ids, allocs):
    """Run all allocators on the SAME arrival trace. Tolerates compare() drift."""
    S = m["sim"]
    kw = dict(slo_ms=SLO_TTFT_5S, slo_type="ttft", k_range=K_BOUNDS,
              max_num_seqs=MAX_SEQS, percentile="p99")
    try:
        return S.compare(factory, allocs, lat, acc, ids, **kw)
    except TypeError as e:
        # compare() signature drifted -> fall back to per-allocator simulate()
        print(f"[{name}] compare() mismatch ({e}); falling back to simulate() per policy",
              file=sys.stderr)
        out = {}
        trace = factory()
        for an, a in allocs.items():
            try:
                out[an] = S.simulate(trace, a, lat, acc, ids, **kw)
            except TypeError as e2:
                print(f"[{name}] {an} simulate() failed: {e2}", file=sys.stderr)
        return out


def row_from(r):
    return {
        "policy":       getattr(r, "policy", "?"),
        "goodput_rate": getattr(r, "goodput_rate", float("nan")),
        "met_rate":     getattr(r, "met_rate", float("nan")),    # unweighted, for v2 sanity
        "frac_met":     getattr(r, "frac_met", float("nan")),
        "mean_acc":     getattr(r, "mean_accuracy", float("nan")),
        "n_met":        getattr(r, "n_met", None),
        "n":            getattr(r, "n", None),
    }


def main():
    m = _import()
    lat = m["load_latpred"](EV0 / "latpred_coeffs.json")
    acc = m["load_accuracy"](EV0 / "accuracy.json")
    # id list straight from accuracy.json (AccuracyTerm may not expose per_image as an attr)
    try:
        _aj = json.loads((EV0 / "accuracy.json").read_text())
        ids = sorted(_aj.get("per_image", {}).keys()) or [f"img{i}" for i in range(200)]
    except Exception:
        ids = [f"img{i}" for i in range(200)]
    allocs = build_allocators(m)

    out = {}
    for name, factory in regimes(m, len(ids)).items():
        res = run_regime(m, name, factory, lat, acc, ids, allocs)
        out[name] = {an: row_from(r) for an, r in res.items()}
    (EV0 / "go_nogo.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- report ----
    for name, rows in out.items():
        print(f"\n=== {name}   goodput@TTFT<={SLO_TTFT_5S:.0f}ms (p99 gate) ===")
        print(f"{'policy':12s} {'goodput/s':>10s} {'met/s':>8s} {'frac_met':>9s} {'mean_acc':>9s}")
        for r in sorted(rows.values(), key=lambda x: -(x["goodput_rate"] or 0)):
            print(f"{r['policy']:12s} {r['goodput_rate']:10.3f} {r['met_rate']:8.3f} "
                  f"{r['frac_met']:9.3f} {r['mean_acc']:9.3f}")

    # ---- sanity vs v2 c64 closed-loop ----
    cl = out.get("closed_c64", {})
    g = lambda p: cl.get(p, {}).get("met_rate")
    print("\n[sanity] closed_c64 met_rate: fixed_r0=%.2f (v2~1.4-1.8)  fixed_r75=%.2f (v2~10.8-14.4)"
          % (g("fixed_r0") or -1, g("fixed_r75") or -1))

    # ---- headline verdict ----
    for ol in ("openloop_lo", "openloop_hi", "bursty", "mixed_slo"):
        if ol not in out:
            continue
        rows = out[ol]
        best_fixed = max((rows[p]["goodput_rate"] for p in rows if p.startswith("fixed_")),
                         default=float("nan"))
        for adaptive in ("greedy", "lagrangian", "oracle"):
            if adaptive in rows:
                v = rows[adaptive]["goodput_rate"]
                win = (v - best_fixed) / best_fixed * 100 if best_fixed else float("nan")
                tag = "WIN" if (v or 0) > (best_fixed or 0) else "no-win"
                print(f"[headline {ol}] {adaptive:11s} goodput={v:.3f} vs best_fixed={best_fixed:.3f} "
                      f"({win:+.1f}%) -> {tag}")
    print(f"\nwrote {EV0/'go_nogo.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
