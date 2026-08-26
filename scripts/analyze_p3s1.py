#!/usr/bin/env python3
"""Analyze the P3-step-1 max-tokens=32 Pareto re-validation (refined
num_running controller). Prints:
  1. The Pareto table: {adaptive, fixed-r25, fixed-r50} x {GQA, TextVQA}
     x {bursty} -> req/s, acc, realized-r (min/mean/max), kept{min,max}.
  2. The Pareto-dominance verdict per benchmark.
  3. The step-profile realized-r time-series summary (the controller figure):
     r should rise to ~r_max in the high phase, fall to ~r_min in the low.

Reads runs/p2_d/p3s1_<bench>_<config>_<profile>_mt32.json.
"""
import json, os
from collections import Counter

D = "/media/disk2/YZX/research/vla/runs/p2_d"


def load(name):
    p = os.path.join(D, name + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def rdist(j):
    """Realized-r min/mean/max string from a controller-bearing run."""
    if not j or not j.get("controller"):
        return "—", "—", "—", "—"
    c = j["controller"]
    rs = c.get("realized_summary", {})
    if not rs or not rs.get("n"):
        return "—", "—", "—", "—"
    rmin, rmean, rmax = rs.get("r_min", 0), rs.get("r_mean", 0), rs.get("r_max", 0)
    cf = f"{rs.get('conc_frac_min', 0):.2f}-{rs.get('conc_frac_max', 0):.2f}"
    return f"{rmin:.3f}", f"{rmean:.3f}", f"{rmax:.3f}", cf


def kept_str(j):
    ku = j.get("hook", {}).get("kept_counts_unique", []) if j else []
    return f"{{{ku[0]},{ku[-1]}}}" if ku else "—"


BENCHES = ["gqa", "textvqa"]
CONFIGS = [("adaptive", "adaptive"), ("fixed_r25", "fixed r25"), ("fixed_r50", "fixed r50")]

print("\n=== P3-step-1 Pareto table (max-tokens=32, num_running signal) ===")
print(f"{'bench':<8} {'config':<11} {'req/s':>7} {'acc':>7} {'kept{min,max}':>14} "
      f"{'r_min':>6} {'r_mean':>7} {'r_max':>6} {'conc_frac_range':>16}")

results = {}
for bench in BENCHES:
    for ck, label in CONFIGS:
        fn = f"p3s1_{bench}_{ck}_bursty_mt32"
        j = load(fn)
        if j is None:
            print(f"{bench:<8} {label:<11} {'MISSING':>7}  ({fn})")
            continue
        req_s = j["agg"]["served_req_s"]["mean"]
        acc = j["agg"]["accuracy"]
        rmin, rmean, rmax, cfr = rdist(j)
        print(f"{bench:<8} {label:<11} {req_s:>7.2f} {acc:>7.3f} {kept_str(j):>14} "
              f"{rmin:>6} {rmean:>7} {rmax:>6} {cfr:>16}")
        results[(bench, ck)] = (req_s, acc)

# ---- Pareto-dominance verdict per benchmark ----
print("\n=== Pareto-dominance verdict (adaptive vs both fixed points) ===")
verdicts = {}
for bench in BENCHES:
    a = results.get((bench, "adaptive"))
    r25 = results.get((bench, "fixed_r25"))
    r50 = results.get((bench, "fixed_r50"))
    if not (a and r25 and r50):
        print(f"  {bench}: incomplete (missing a row)")
        verdicts[bench] = "incomplete"
        continue
    d_req = a[0] - r25[0]
    d_acc = a[1] - r50[1]
    win_req = d_req > 0
    win_acc = d_acc > 0
    if win_req and win_acc:
        tag = "PARETO-DOMINATES (HEADLINE)"
    elif win_req:
        tag = "beats r25 on req/s only"
    elif win_acc:
        tag = "beats r50 on acc only"
    else:
        tag = "no dominance"
    verdicts[bench] = tag
    # also report adaptive vs r25 on acc and vs r50 on req/s (full picture)
    d_acc_r25 = a[1] - r25[1]
    d_req_r50 = a[0] - r50[0]
    print(f"  {bench}: adaptive req/s={a[0]:.2f} (vs r25 {r25[0]:.2f}, Δ{d_req:+.2f} "
          f"{'WIN' if win_req else 'loss'} | vs r50 {r50[0]:.2f}, Δ{d_req_r50:+.2f}) || "
          f"acc={a[1]:.3f} (vs r50 {r50[1]:.3f}, Δ{d_acc:+.3f} "
          f"{'WIN' if win_acc else 'loss'} | vs r25 {r25[1]:.3f}, Δ{d_acc_r25:+.3f}) "
          f"-> {tag}")

# ---- step-profile realized-r time-series (the controller figure) ----
print("\n=== Step-profile realized-r time-series (controller figure) ===")
j = load("p3s1_gqa_adaptive_step_mt32")
if j is None:
    print("  MISSING p3s1_gqa_adaptive_step_mt32")
else:
    c = j.get("controller", {})
    rs = c.get("realized_summary", {})
    rea = c.get("realized", [])
    rseq = [round(x["r"], 3) for x in rea]
    print(f"  n decisions: {len(rseq)}")
    print(f"  r range: min={rs.get('r_min', 0):.3f} mean={rs.get('r_mean', 0):.3f} "
          f"max={rs.get('r_max', 0):.3f}")
    print(f"  conc_frac range: {rs.get('conc_frac_min', 0):.2f} - "
          f"{rs.get('conc_frac_max', 0):.2f}")
    print(f"  r value counts: {dict(sorted(Counter(rseq).items()))}")
    # show the swing: group consecutive runs of equal r
    if rseq:
        groups = []
        cur_r, cur_n = rseq[0], 1
        for r in rseq[1:]:
            if abs(r - cur_r) < 1e-6:
                cur_n += 1
            else:
                groups.append((cur_r, cur_n))
                cur_r, cur_n = r, 1
        groups.append((cur_r, cur_n))
        gstr = " -> ".join(f"{r:.2f}x{n}" for r, n in groups[:12])
        print(f"  r run-length pattern (first 12): {gstr}")
    swung = (rs.get("r_max", 0) - rs.get("r_min", 0)) > 0.15
    print(f"  -> controller {'SWUNG across the full range' if swung else 'DID NOT swing'} "
          f"(r_max - r_min = {rs.get('r_max', 0) - rs.get('r_min', 0):.3f})")

print("\n=== Summary ===")
for b in BENCHES:
    print(f"  {b}: {verdicts.get(b, '?')}")
