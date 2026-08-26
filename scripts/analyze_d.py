#!/usr/bin/env python3
"""Analyze the P2 method-D validation matrix (adaptive vs fixed under varying
load). Prints the req/s + accuracy + realized-r distribution table and the
Pareto-dominance verdict (the method's headline claim)."""
import json, os, sys
D = "/media/disk2/YZX/research/vla/runs/p2_d"

def load(name):
    p = os.path.join(D, name + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))

NAMES = {
    "dval_adaptive_bursty":   "adaptive (bursty)",
    "dval_fixed_r25_bursty":  "fixed r25 (bursty)",
    "dval_fixed_r50_bursty":  "fixed r50 (bursty)",
    "dval_adaptive_constant": "adaptive (constant)",
}

rows = {}
for name, label in NAMES.items():
    j = load(name)
    if j is None:
        print(f"MISSING: {name}")
        continue
    rows[name] = {
        "label": label,
        "req_s": j["agg"]["served_req_s"]["mean"],
        "tok_s": j["agg"]["served_tok_s"]["mean"],
        "ttft": j["agg"]["ttft_ms"]["mean"],
        "wall": j["wall_s"],
        "acc": j["agg"]["accuracy"],
        "n": j["n"],
        "controller": j.get("controller"),
        "kept_unique": j["hook"].get("kept_counts_unique"),
    }

print("\n=== Method-D validation: adaptive vs fixed under varying load (GQA) ===")
print(f"{'config':<24} {'req/s':>8} {'wall_s':>8} {'acc':>7} {'kept{min,max}}':>16} {'realized r (mean,min,max)':>28}")
for name, d in rows.items():
    cu = d["kept_unique"]
    kept_str = f"{{{cu[0]},{cu[-1]}}}" if cu else "—"
    if d["controller"]:
        rs = d["controller"].get("realized_summary", {})
        rstr = f"{rs.get('r_mean',0):.3f}/{rs.get('r_min',0):.3f}/{rs.get('r_max',0):.3f}"
    else:
        rstr = "—"
    print(f"{d['label']:<24} {d['req_s']:>8.2f} {d['wall']:>8.1f} {d['acc']:>7.3f} "
          f"{kept_str:>16} {rstr:>28}")

# ---- Pareto-dominance verdict (the method claim) ----
# adaptive should: (a) higher req/s than fixed-r25, AND (b) higher acc than fixed-r50
print("\n=== Pareto-dominance verdict (adaptive vs the two fixed points) ===")
a = rows.get("dval_adaptive_bursty")
r25 = rows.get("dval_fixed_r25_bursty")
r50 = rows.get("dval_fixed_r50_bursty")
if a and r25 and r50:
    req_win_25 = a["req_s"] - r25["req_s"]
    acc_win_50 = a["acc"] - r50["acc"]
    print(f"  req/s: adaptive={a['req_s']:.2f}  fixed-r25={r25['req_s']:.2f}  "
          f"delta={req_win_25:+.2f}  ({'WIN' if req_win_25 > 0 else 'LOSS'} vs r25)")
    print(f"  acc  : adaptive={a['acc']:.3f}  fixed-r50={r50['acc']:.3f}  "
          f"delta={acc_win_50:+.3f}  ({'WIN' if acc_win_50 > 0 else 'LOSS'} vs r50)")
    pareto = (req_win_25 > 0) and (acc_win_50 > 0)
    # softer: also report if adaptive beats EITHER fixed on BOTH axes (the
    # fixed points are themselves on the frontier; adaptive dominating one of
    # them on both axes is already meaningful)
    beats_r25_both = (req_win_25 > 0) and (a["acc"] - r25["acc"] >= -0.001)
    beats_r50_both = (acc_win_50 > 0) and (a["req_s"] - r50["req_s"] >= -0.05)
    verdict = ("PARETO-DOMINATES the fixed points (higher req/s than r25 AND "
               "higher acc than r50) -> HEADLINE RESULT")
    if not pareto:
        if beats_r25_both:
            verdict = ("beats fixed-r25 on req/s at iso-or-better acc (adaptive "
                       "strictly dominates the accuracy-favoring fixed point)")
        elif beats_r50_both:
            verdict = ("beats fixed-r50 on acc at iso-or-better req/s (adaptive "
                       "strictly dominates the throughput-favoring fixed point)")
        else:
            verdict = ("does NOT Pareto-dominate (claim not supported at this "
                       "setting — tune r_min/r_max/thresholds or check the load profile)")
    print(f"  -> {verdict}")

# ---- realized-r adaptation check (did the controller actually adapt?) ----
print("\n=== Realized-r distribution (adaptation proof) ===")
for name in ["dval_adaptive_bursty", "dval_adaptive_constant"]:
    d = rows.get(name)
    if not d or not d["controller"]:
        continue
    rs = d["controller"].get("realized_summary", {})
    rea = d["controller"].get("realized", [])
    occs = [x["kv_occupancy"] for x in rea if x["kv_occupancy"] is not None]
    nrs = [x["num_running"] for x in rea if x["num_running"] is not None]
    occ_str = (f"occ[min={min(occs):.2f},mean={sum(occs)/len(occs):.2f},"
               f"max={max(occs):.2f}]") if occs else "occ=N/A"
    nr_str = (f"num_running[min={min(nrs)},mean={sum(nrs)/len(nrs):.1f},"
              f"max={max(nrs)}]") if nrs else "num_running=N/A"
    adapting = (rs.get("r_max", 0) - rs.get("r_min", 0)) > 0.01
    adapt_str = "ADAPTING (r varies)" if adapting else "FLAT (r stuck — load never crossed thresholds)"
    print(f"  {d['label']:<24} r[min={rs.get('r_min',0):.3f},mean={rs.get('r_mean',0):.3f},"
          f"max={rs.get('r_max',0):.3f}] {occ_str} {nr_str} -> {adapt_str}")
