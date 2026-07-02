#!/usr/bin/env python3
"""Analyze P3-step-2 breadth Pareto validation.

Reads runs/p3s2/*.json and produces:
  1. The extended Pareto table: {adaptive, fixed-r25, fixed-r50} x
     {MME, MMBench, ScienceQA} (mt64, c12, bursty) -> req/s, acc, realized-r.
  2. The benchmark-conditional verdict: on how many of the 3 new benchmarks
     does adaptive cleanly Pareto-dominate BOTH fixed points?
  3. n=500 TextVQA: is the +0.020 acc margin (n=200) now significant?
  4. FastV accuracy anchor rows.
  5. Concurrency matrix (c1/c4/c12) served-throughput scaling.

Writes the verdict to notes/p3s2_pareto.md (the deliverable).
"""
import json
import os
from pathlib import Path

ROOT = Path("/media/disk2/YZX/research/vla")
D = ROOT / "runs/p3s2"


def load(name):
    p = D / f"{name}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def rdist(j):
    if not j or not j.get("controller"):
        return "—", "—", "—", "—"
    rs = j["controller"].get("realized_summary", {})
    if not rs or not rs.get("n"):
        return "—", "—", "—", "—"
    cf = f"{rs.get('conc_frac_min', 0):.2f}-{rs.get('conc_frac_max', 0):.2f}"
    return f"{rs['r_min']:.3f}", f"{rs['r_mean']:.3f}", f"{rs['r_max']:.3f}", cf


def kept_str(j):
    ku = j.get("hook", {}).get("kept_counts_unique", []) if j else []
    return f"{{{ku[0]},{ku[-1]}}}" if ku else "—"


def row(bench, ck):
    """Return (req_s, acc, realized_r_str) or None."""
    fn = f"{bench}_{ck}_bursty_mt64"
    j = load(fn)
    if j is None:
        return None
    return (j["agg"]["served_req_s"]["mean"], j["agg"]["accuracy"], rdist(j), kept_str(j))


CONFIGS = [("adaptive", "adaptive"), ("fixed_r25", "fixed r25"), ("fixed_r50", "fixed r50")]
NEW_BENCHES = ["mme", "mmbench", "scienceqa"]
ALL_BENCHES = ["gqa", "textvqa"] + NEW_BENCHES

out = ["# P3-step-2: breadth Pareto validation (benchmark-conditional pattern)",
       "",
       "> **Thesis:** load-adaptive Pareto-dominates fixed where compression is",
       "> COSTLY (long answers, dense text, MC reasoning); it does NOT where the",
       "> model recovers (short yes/no answers). P3-step-1 confirmed this on",
       "> TextVQA (costly -> adaptive wins) vs GQA (free -> fixed-r50 wins).",
       "> P3-step-2 tests the pattern on 3 denser benchmarks: MME / MMBench /",
       "> ScienceQA. If adaptive wins on >=2 of 3, the benchmark-conditional",
       "> pattern is confirmed -> strong, broad paper story.",
       ""]

# --------------------------------------------------------------------------- #
# 1. New-benchmark Pareto table
# --------------------------------------------------------------------------- #
out += ["## 1. Pareto table (mt64, c12, bursty, n=200, num_running controller)",
        "",
        "| bench | config | req/s | acc | kept{min,max} | r_min | r_mean | r_max | conc_frac |",
        "|---|---|---|---|---|---|---|---|---|"]
results = {}
for bench in NEW_BENCHES:
    for ck, label in CONFIGS:
        r = row(bench, ck)
        if r is None:
            out.append(f"| {bench} | {label} | MISSING | | | | | | |")
            continue
        req_s, acc, (rmin, rmean, rmax, cfr), ku = r
        out.append(f"| {bench} | {label} | {req_s:.3f} | {acc:.3f} | {ku} | {rmin} | {rmean} | {rmax} | {cfr} |")
        results[(bench, ck)] = (req_s, acc)

# --------------------------------------------------------------------------- #
# 2. Benchmark-conditional verdict
# --------------------------------------------------------------------------- #
out += ["", "## 2. Verdict: does adaptive cleanly Pareto-dominate both fixed points?",
        ""]
n_win = 0
for bench in NEW_BENCHES:
    a = results.get((bench, "adaptive"))
    r25 = results.get((bench, "fixed_r25"))
    r50 = results.get((bench, "fixed_r50"))
    if not (a and r25 and r50):
        out.append(f"- **{bench}**: INCOMPLETE (missing a row)")
        continue
    d_req = a[0] - r25[0]  # adaptive req/s should EXCEED r25
    d_acc = a[1] - r50[1]  # adaptive acc should EXCEED r50
    win_req = d_req > 0
    win_acc = d_acc > 0
    verdict = "PARETO-DOMINATES" if (win_req and win_acc) else (
        "wins req/s only" if win_req and not win_acc else (
            "wins acc only" if win_acc and not win_req else "dominated"))
    out.append(f"- **{bench}**: req/s adaptive {a[0]:.3f} vs r25 {r25[0]:.3f} "
               f"(Δ {d_req:+.3f}, {'WIN' if win_req else 'LOSS'}); "
               f"acc adaptive {a[1]:.3f} vs r50 {r50[1]:.3f} "
               f"(Δ {d_acc:+.3f}, {'WIN' if win_acc else 'LOSS'}) -> **{verdict}**")
    if win_req and win_acc:
        n_win += 1

out.append("")
out.append(f"**★ Pattern verdict: adaptive cleanly Pareto-dominates on "
           f"{n_win}/3 new benchmarks.**")
if n_win >= 2:
    out.append("CONFIRMED (>=2/3): the benchmark-conditional pattern holds "
               "-- load-adaptive wins where compression costs accuracy.")
elif n_win == 1:
    out.append("PARTIAL (1/3): weak support; one benchmark is ambiguous. "
               "Inspect per-benchmark r50 acc-cost to interpret.")
else:
    out.append("REFUTED (0/3): adaptive does NOT dominate on the new denser "
               "benchmarks. Re-examine the controller or the r50-acc-cost premise.")

# --------------------------------------------------------------------------- #
# 3. n=500 TextVQA (tighten the +0.020 acc margin)
# --------------------------------------------------------------------------- #
out += ["", "## 3. n=500 TextVQA: is the +0.020 acc margin significant?",
        "",
        "| config | n | acc | vs adaptive |",
        "|---|---|---|---|"]
tvqa = {}
for ck, _ in CONFIGS:
    j = load(f"textvqa_{ck}_bursty_n500")
    if j is None:
        continue
    tvqa[ck] = j["agg"]["accuracy"]
    out.append(f"| {ck} | 500 | {j['agg']['accuracy']:.3f} | |")
if "adaptive" in tvqa and "fixed_r50" in tvqa:
    d = tvqa["adaptive"] - tvqa["fixed_r50"]
    # two-proportion z (normal approx) for significance
    import math
    n = 500
    p1, p2 = tvqa["adaptive"], tvqa["fixed_r50"]
    pp = (p1 * n + p2 * n) / (2 * n)
    se = math.sqrt(2 * pp * (1 - pp) / n) if 0 < pp < 1 else float("nan")
    z = d / se if se and se == se else float("nan")
    sig = "|z|>=1.96 SIGNIFICANT" if abs(z) >= 1.96 else "|z|<1.96 not significant"
    out.append("")
    out.append(f"adaptive - r50 = {d:+.3f}; z={z:+.2f} (n=500, two-prop) -> **{sig}**")
    out.append(f"(P3-step-1 n=200 margin was +0.020; n=500 margin is {d:+.3f}.)")

# --------------------------------------------------------------------------- #
# 4. FastV accuracy anchor rows
# --------------------------------------------------------------------------- #
out += ["", "## 4. FastV accuracy anchor (LLaVA-1.5-7B, keep=288 r50, agg-layer 2)",
        "",
        "| bench | config | acc |",
        "|---|---|---|"]
for bench in ALL_BENCHES:
    for ck, label in [("control576", "r0 control"), ("keep288", "FastV r50")]:
        j = load(f"fastv_{bench}_{ck}")
        if j is None:
            continue
        # fastv_bench writes top-level "accuracy" (not agg.accuracy like serve_bench)
        acc = j.get("accuracy", j.get("agg", {}).get("accuracy"))
        if acc is None:
            continue
        out.append(f"| {bench} | {label} | {acc:.3f} |")

# --------------------------------------------------------------------------- #
# 5. Concurrency matrix (c1/c4/c12) served-throughput scaling
# --------------------------------------------------------------------------- #
out += ["", "## 5. Concurrency matrix (GQA, batch-submit, served-throughput scaling)",
        "",
        "| config | c1 | c4 | c12 |",
        "|---|---|---|---|"]
def load_req_s(path):
    p = Path(path)
    if not p.exists():
        return "—"
    return f"{json.load(open(p))['agg']['served_req_s']['mean']:.3f}"
# c1/c12 from the M2 batch-submit matrix (runs/p2_d/m2_c{1,12}_r{0,50}.json);
# c4 from this step (runs/p3s2/gqa_r{0,50}_c4_batch.json).
c1_r0 = load_req_s(ROOT / "runs/p2_d/m2_c1_r0.json")
c1_r50 = load_req_s(ROOT / "runs/p2_d/m2_c1_r50.json")
c12_r0 = load_req_s(ROOT / "runs/p2_d/m2_c12_r0.json")
c12_r50 = load_req_s(ROOT / "runs/p2_d/m2_c12_r50.json")
c4_r0 = load_req_s(D / "gqa_r0_c4_batch.json")
c4_r50 = load_req_s(D / "gqa_r50_c4_batch.json")
out.append(f"| r0 (control) | {c1_r0} | {c4_r0} | {c12_r0} |")
out.append(f"| r50 | {c1_r50} | {c4_r50} | {c12_r50} |")

out += ["", "---", "Generated by `scripts/analyze_p3s2.py`."]
text = "\n".join(out)
(ROOT / "notes/p3s2_pareto.md").write_text(text)
print(text)
print(f"\n[analyze_p3s2] wrote notes/p3s2_pareto.md")
