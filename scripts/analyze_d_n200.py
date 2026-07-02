#!/usr/bin/env python3
"""Analyze the P2 method-D n=200 validation: adaptive vs fixed under varying
load, on GQA + TextVQA. Null-safe (handles missing controller/realized fields).
Prints the Pareto table + the dominance verdict (the method's headline claim).

Reads runs/p2_d/{gqa,textvqa}_{adaptive,fixed_r25,fixed_r50}_{bursty,constant,step}_n200.json
(plus the n=100 dval_* files as fallback if n=200 missing).
"""
import json, os
D = "/media/disk2/YZX/research/vla/runs/p2_d"

def load(bench, config, profile):
    """config in {adaptive, fixed_r25, fixed_r50}; profile in {bursty, constant, step}."""
    # n=200 paths only (the matched fixed points are n=200). The dval_* n=100
    # files are a DIFFERENT run (different code path) -- do NOT mix them in.
    p = os.path.join(D, f"{bench}_{config}_{profile}_n200.json")
    if os.path.exists(p):
        return json.load(open(p)), f"{bench}_{config}_{profile}_n200.json"
    return None, None

def rdist(j):
    """Null-safe realized-r (min/mean/max) extraction. Returns ('—' if no controller)."""
    if not j:
        return "—"
    c = j.get("controller")
    if not c or not isinstance(c, dict):
        return "—"
    rs = c.get("realized_summary")
    if not rs or not isinstance(rs, dict):
        return "—"
    n = rs.get("n", 0)
    if not n:
        return "—"
    return f"{rs.get('r_min',0):.3f}/{rs.get('r_mean',0):.3f}/{rs.get('r_max',0):.3f}"

def row(j):
    """Returns (req_s, acc, kept_unique_str) or None."""
    if not j:
        return None
    agg = j.get("agg", {})
    req_s = agg.get("served_req_s", {}).get("mean", float("nan"))
    acc = agg.get("accuracy", float("nan"))
    ku = j.get("hook", {}).get("kept_counts_unique", [])
    ku_str = f"{{{ku[0]},{ku[-1]}}}" if ku else "—"
    return (req_s, acc, ku_str)

BENCHES = ["gqa", "textvqa"]
PROFILES = ["bursty", "constant"]
CONFIGS = [("adaptive", "adaptive"), ("fixed_r25", "fixed r25"), ("fixed_r50", "fixed r50")]

print("\n=== Method-D n=200 validation: adaptive vs fixed (Pareto table) ===")
print(f"{'bench':<8} {'profile':<9} {'config':<12} {'req/s':>7} {'acc':>7} "
      f"{'kept{min,max}':>14} {'realized r (min/mean/max)':>26}  {'file':<34}")

results = {}  # (bench, profile, config_key) -> (req_s, acc)
for bench in BENCHES:
    for profile in PROFILES:
        for ck, label in CONFIGS:
            j, fn = load(bench, ck, profile)
            r = row(j)
            if r is None:
                print(f"{bench:<8} {profile:<9} {label:<12} {'MISSING':>7}")
                continue
            req_s, acc, ku_str = r
            print(f"{bench:<8} {profile:<9} {label:<12} {req_s:>7.2f} {acc:>7.3f} "
                  f"{ku_str:>14} {rdist(j):>26}  {fn:<34}")
            results[(bench, profile, ck)] = (req_s, acc)

# ---- Pareto-dominance verdict per (bench, profile) ----
print("\n=== Pareto-dominance verdict (adaptive vs fixed points) ===")
for bench in BENCHES:
    for profile in PROFILES:
        a = results.get((bench, profile, "adaptive"))
        r25 = results.get((bench, profile, "fixed_r25"))
        r50 = results.get((bench, profile, "fixed_r50"))
        if not (a and r25 and r50):
            print(f"  {bench}/{profile}: incomplete (missing a row)")
            continue
        d_req = a[0] - r25[0]
        d_acc = a[1] - r50[1]
        win_req = d_req > 0
        win_acc = d_acc > 0
        tag = "PARETO-DOMINATES (HEADLINE)" if (win_req and win_acc) else \
              ("beats r25 on req/s only" if win_req else
               "beats r50 on acc only" if win_acc else "no dominance")
        print(f"  {bench}/{profile}: adaptive req/s={a[0]:.2f} (vs r25 {r25[0]:.2f}, "
              f"Δ{d_req:+.2f} {'WIN' if win_req else 'loss'}) | "
              f"acc={a[1]:.3f} (vs r50 {r50[1]:.3f}, Δ{d_acc:+.3f} "
              f"{'WIN' if win_acc else 'loss'}) -> {tag}")

# ---- constant-vs-bursty contrast (load-tracking check) ----
print("\n=== Constant-vs-bursty contrast (load-tracking sanity) ===")
for bench in BENCHES:
    a_burst = results.get((bench, "bursty", "adaptive"))
    a_const = results.get((bench, "constant", "adaptive"))
    if a_burst and a_const:
        print(f"  {bench} adaptive: bursty req/s={a_burst[0]:.2f} acc={a_burst[1]:.3f} | "
              f"constant req/s={a_const[0]:.2f} acc={a_const[1]:.3f} | "
              f"constant faster (max load all the time)? {a_const[0] > a_burst[0]}")
