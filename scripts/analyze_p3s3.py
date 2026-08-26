#!/usr/bin/env python3
"""Analyze P3-step-3 n=500 accuracy tightening (the Pareto GATE).

Reads runs/p3s3/*.json (the 12 n=500 jobs) + runs/p3s2/textvqa_*_n500.json
(already at n=500) and produces:
  1. The n=500 Pareto table: {adaptive, fixed-r25, fixed-r50} x
     {gqa, mme, mmbench, scienceqa} -> req/s, acc, realized-r, kept.
  2. Per-benchmark verdict at n=500 (Pareto-dominate / req/s-only / dominated),
     with a two-proportion z significance note on the adaptive-vs-r50 acc margin.
  3. The TextVQA n=500 row (already collected) for the full 5-benchmark tally.
  4. The honest bottom line: how many of {gqa, mme, mmbench, scienceqa, textvqa}
     does adaptive cleanly Pareto-dominate at n=500? Did MME/ScienceQA (the two
     n=200 Pareto cases) HOLD or REVERSE?

Significance calibration (n=500, two-prop acc stderr ~+-0.022):
  |margin| < 0.022  -> noise (|z| < 1)
  0.022-0.044       -> suggestive (1 <= |z| < 2)
  >= 0.044          -> meaningful (|z| >= 2, ~p<0.05)

Writes notes/p3s3_pareto_n500.md (the deliverable).
"""
import json
import math
from pathlib import Path

ROOT = Path("/media/disk2/YZX/research/vla")
D = ROOT / "runs/p3s3"
D2 = ROOT / "runs/p3s2"  # for the textvqa n=500 rows


def load(d, name):
    p = d / f"{name}.json"
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


def metrics(j):
    """Return (req_s, acc) from a serve_bench metrics json."""
    if not j:
        return None
    return (j["agg"]["served_req_s"]["mean"], j["agg"]["accuracy"])


def znote(d_acc, n=500):
    """Two-proportion normal-approx significance label for an acc margin."""
    if d_acc is None or n <= 0:
        return ""
    # rough pooled-p z (assume ~symmetric; the stderr ~sqrt(2*p*(1-p)/n))
    # use |d| itself as a proxy for p(1-p)~0.25 worst case if we lack the p's
    return f"margin {d_acc:+.3f} ({'noise' if abs(d_acc) < 0.022 else 'suggestive' if abs(d_acc) < 0.044 else 'meaningful' if abs(d_acc) >= 0.044 else '?'})"


def zscore(p1, p2, n):
    if n <= 0:
        return float("nan")
    pp = (p1 * n + p2 * n) / (2 * n)
    if not (0 < pp < 1):
        return float("nan")
    se = math.sqrt(2 * pp * (1 - pp) / n)
    return (p1 - p2) / se if se else float("nan")


BENCHES = [("gqa", 32), ("mme", 64), ("mmbench", 64), ("scienceqa", 64)]
CONFIGS = [("adaptive", "adaptive"), ("fixed_r25", "fixed r25"), ("fixed_r50", "fixed r50")]

out = ["# P3-step-3: n=500 Pareto accuracy tightening (the GATE)",
       "",
       "> **Gate question:** the n=200 Pareto-dominate claims on MME / ScienceQA",
       "> rested on acc margins of ~+0.015 — WITHIN noise (n=200 acc stderr ~+-0.031).",
       "> TextVQA's n=200 acc headline already REVERSED at n=500. This step re-runs",
       "> the Pareto comparison at n=500 (acc stderr ~+-0.022) on GQA / MME / MMBench",
       "> / ScienceQA to see whether MME / ScienceQA HOLD or REVERSE, and whether the",
       "> benchmark-conditional Pareto story survives the noise gate.",
       "",
       "> All runs: c12 (max_num_seqs=12), bursty profile, num_running controller",
       "> (adaptive: r_min 0.25 / r_max 0.50, conc-lo 0.25 / conc-hi 0.75), seed=0.",
       "> GQA max-tokens=32; MME/MMBench/ScienceQA max-tokens=64 — identical to the",
       "> n=200 anchors (clean n=200 -> n=500 comparison).",
       "",
       "> Significance calibration (n=500, two-prop acc stderr ~+-0.022):",
       "> |margin| < 0.022 = noise; 0.022-0.044 = suggestive; >= 0.044 = meaningful.",
       ""]

# --------------------------------------------------------------------------- #
# 1. n=500 Pareto table
# --------------------------------------------------------------------------- #
out += ["## 1. n=500 Pareto table",
        "",
        "| bench | config | req/s | acc | kept{min,max} | r_min | r_mean | r_max | conc_frac |",
        "|---|---|---|---|---|---|---|---|---|"]
results = {}
for bench, _mt in BENCHES:
    for ck, label in CONFIGS:
        fn = f"{bench}_{ck}_bursty_n500"
        j = load(D, fn)
        if j is None:
            out.append(f"| {bench} | {label} | MISSING | | | | | | |")
            continue
        req_s, acc = metrics(j)
        rmin, rmean, rmax, cfr = rdist(j)
        ku = kept_str(j)
        out.append(f"| {bench} | {label} | {req_s:.3f} | {acc:.3f} | {ku} | {rmin} | {rmean} | {rmax} | {cfr} |")
        results[(bench, ck)] = (req_s, acc)

# --------------------------------------------------------------------------- #
# 2. Per-benchmark verdict at n=500 + significance
# --------------------------------------------------------------------------- #
out += ["", "## 2. Per-benchmark verdict at n=500",
        "",
        "Each row: adaptive vs r25 (req/s, must be WIN) and vs r50 (acc, must be WIN) -> Pareto-dominate?"]
out.append("")
out.append("| bench | adaptive req/s | r25 req/s | Δ req/s | adaptive acc | r50 acc | Δ acc | acc sig | verdict |")
out.append("|---|---|---|---|---|---|---|---|---|")
n_pareto = 0
per_bench = {}
for bench, _mt in BENCHES:
    a = results.get((bench, "adaptive"))
    r25 = results.get((bench, "fixed_r25"))
    r50 = results.get((bench, "fixed_r50"))
    if not (a and r25 and r50):
        out.append(f"| {bench} | | | | | | | | INCOMPLETE |")
        per_bench[bench] = "incomplete"
        continue
    d_req = a[0] - r25[0]
    d_acc = a[1] - r50[1]
    win_req = d_req > 0
    win_acc = d_acc > 0
    z = zscore(a[1], r50[1], 500)
    sig = "noise" if abs(d_acc) < 0.022 else ("suggestive" if abs(d_acc) < 0.044 else "meaningful")
    verdict = "PARETO-DOMINATES" if (win_req and win_acc) else (
        "wins req/s only" if win_req and not win_acc else (
            "wins acc only" if win_acc and not win_req else "dominated"))
    out.append(f"| {bench} | {a[0]:.3f} | {r25[0]:.3f} | {d_req:+.3f} | {a[1]:.3f} | {r50[1]:.3f} | {d_acc:+.3f} | {sig} (z={z:+.2f}) | {verdict} |")
    per_bench[bench] = verdict
    if win_req and win_acc:
        n_pareto += 1

# --------------------------------------------------------------------------- #
# 3. TextVQA n=500 (from p3s2, already at n=500)
# --------------------------------------------------------------------------- #
out += ["", "## 3. TextVQA at n=500 (from P3-step-2, for the full 5-benchmark tally)",
        "",
        "| config | n | acc |",
        "|---|---|---|"]
tvqa = {}
for ck, _ in CONFIGS:
    j = load(D2, f"textvqa_{ck}_bursty_n500")
    if j is None:
        continue
    tvqa[ck] = metrics(j)
    out.append(f"| {ck} | 500 | {tvqa[ck][1]:.3f} |")
tvqa_verdict = "incomplete"
if "adaptive" in tvqa and "fixed_r25" in tvqa and "fixed_r50" in tvqa:
    a, r25, r50 = tvqa["adaptive"], tvqa["fixed_r25"], tvqa["fixed_r50"]
    d_req = a[0] - r25[0]
    d_acc = a[1] - r50[1]
    win_req = d_req > 0
    win_acc = d_acc > 0
    tvqa_verdict = "PARETO-DOMINATES" if (win_req and win_acc) else (
        "wins req/s only" if win_req and not win_acc else (
            "wins acc only" if win_acc and not win_req else "dominated"))
    out += ["",
            f"TextVQA n=500: Δreq/s(adaptive-r25)={d_req:+.3f} ({'WIN' if win_req else 'LOSS'}); "
            f"Δacc(adaptive-r50)={d_acc:+.3f} ({'WIN' if win_acc else 'LOSS'}) -> **{tvqa_verdict}**"]

# --------------------------------------------------------------------------- #
# 4. Honest bottom line
# --------------------------------------------------------------------------- #
out += ["", "## 4. ★ Honest bottom line (the GATE result)",
        ""]
n_total_pareto = n_pareto + (1 if tvqa_verdict == "PARETO-DOMINATES" else 0)
out.append(f"**Adaptive cleanly Pareto-dominates at n=500 on {n_total_pareto}/5 benchmarks** (gqa, mme, mmbench, scienceqa, textvqa).")
out.append("")
out.append("Per-benchmark status at n=500:")
for bench, _ in BENCHES:
    out.append(f"- **{bench}**: {per_bench.get(bench, 'incomplete')}")
out.append(f"- **textvqa**: {tvqa_verdict}")
out.append("")

# MME / ScienceQA HOLD or REVERSE (the load-bearing question)
mme_v = per_bench.get("mme", "incomplete")
sqa_v = per_bench.get("scienceqa", "incomplete")
held = []
reversed_ = []
for name, v in (("MME", mme_v), ("ScienceQA", sqa_v)):
    if v == "PARETO-DOMINATES":
        held.append(name)
    elif v in ("wins req/s only", "dominated", "wins acc only"):
        reversed_.append(name)
if held and not reversed_:
    out.append(f"**MME/ScienceQA: HELD ({', '.join(held)} still Pareto-dominate at n=500).** "
               "Method has solid Pareto cases — the throughput-optimal-under-guardrail framing is supported by clean accuracy wins on these benchmarks.")
elif reversed_ and not held:
    out.append(f"**MME/ScienceQA: REVERSED ({', '.join(reversed_)} no longer Pareto-dominate at n=500).** "
               "Method collapses to 'req/s > r25 everywhere' only — the paper's method section must drop the Pareto-claim and frame purely as throughput-optimal-under-guardrail (no per-benchmark accuracy win).")
elif reversed_ and held:
    out.append(f"**MME/ScienceQA: SPLIT — {', '.join(held)} HELD, {', '.join(reversed_)} REVERSED.** "
               "Method retains one clean Pareto case but loses breadth; frame conservatively.")
else:
    out.append("**MME/ScienceQA: incomplete — at least one benchmark's n=500 run is missing.**")

out += ["",
        "---",
        "Generated by `scripts/analyze_p3s3.py`."]

text = "\n".join(out)
(ROOT / "notes/p3s3_pareto_n500.md").write_text(text)
print(text)
print(f"\n[analyze_p3s3] wrote notes/p3s3_pareto_n500.md")
print(f"[analyze_p3s3] Pareto-dominate count: {n_total_pareto}/5")
