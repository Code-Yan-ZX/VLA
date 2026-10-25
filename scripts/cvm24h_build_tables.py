#!/usr/bin/env python
"""Generate supplement tables (markdown) + figures (svg) from score/timing JSONs.
Deterministic, reads only plan/scores/*.json + plan/timing_*.json.
Quality target rule: >=95% of same-protocol None ATTEMPTED score; cheapest
MEASURED budget point meeting it, else NOT-MET; OCRBench comparisons flag
skip-asymmetry with the common-answered number.
"""
import json, os, glob, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = "results/cvm2027_supplement_20261026/plan"
S = f"{Q}/scores"
FIG = "reports/figures_cvm2027_supplement"
os.makedirs(FIG, exist_ok=True)

def load(name):
    p = f"{S}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None

M = {"textvqa": 100.0, "ocrbench": 1000.0, "docvqa": 100.0}
def fmt(bench, v):  # score scale
    return v * M[bench]

md = ["# CVM2027 supplement — auto tables (2026-10-26)", ""]

# ---------- P1 main table ----------
md.append("## P1 main table — keep=25%, full splits, official attempted denominator")
md.append("")
md.append("| Benchmark | None@100% | RBM@25 | FastV-k3@25 | HiLo@25 |")
md.append("|---|---:|---:|---:|---:|")
row = {}
for bench, pref in (("textvqa", "p1_tv"), ("ocrbench", "p1_oc")):
    rb_none = load(f"{pref}_rb_none"); hl = load(f"{pref}_hl_none")
    none_v = rb_none["B"]["attempted_score"] if rb_none else None
    cells = {
        "None": none_v,
        "RBM": rb_none["A"]["attempted_score"] if rb_none else None,
        "FastV": load(f"{pref}_fv_none")["A"]["attempted_score"] if load(f"{pref}_fv_none") else None,
        "HiLo": hl["A"]["attempted_score"] if hl else None,
    }
    unit = "acc" if bench == "textvqa" else "pts/1000"
    md.append(f"| {bench} ({unit}) | " + " | ".join(
        (f"{fmt(bench, cells[k]):.2f}" if cells[k] is not None else "—")
        for k in ["None", "RBM", "FastV", "HiLo"]) + " |")
md.append("")
md.append("Pairs (A−B, paired bootstrap 20k, attempted denom; * = common-answered sensitivity where skip sets differ):")
for bench, pref in (("textvqa", "p1_tv"), ("ocrbench", "p1_oc")):
    for pair in ["rb_fv", "rb_none", "fv_none", "hl_none", "hl_rb", "hl_fv"]:
        d = load(f"{pref}_{pair}")
        if not d: continue
        pa, pb = d["paired_attempted_all"], d["paired_common_answered_sensitivity"]
        scale = 1.0 if bench == "textvqa" else 10.0
        flag = "*" if abs(pa["delta_pp"] * scale - pb["delta_pp"] * scale) > 5 else ""
        md.append(f"- {bench} {pair}: Δ={pa['delta_pp'] * scale:+.2f}{flag} "
                  f"CI[{pa['ci95_pp'][0] * scale:+.2f},{pa['ci95_pp'][1] * scale:+.2f}] "
                  f"p={pa['sign_flip_p_two_sided']:.4f} (common-answered Δ={pb['delta_pp'] * scale:+.2f}, n={pb['n_paired']})")
md.append("")

# ---------- budget curves ----------
for bench, pref, unit in (("textvqa", "curve_tv", "acc"), ("ocrbench", "curve_oc", "pts/1000")):
    md.append(f"## Budget curve — {bench} ({unit}; {'TextVQA 1000 stratified subset' if bench=='textvqa' else 'OCRBench full 1000'})")
    md.append("| keep | RBM | FastV | HiLo | Δvs None (RBM/FastV/HiLo, attempted*) |")
    md.append("|---|---:|---:|---:|---|")
    for K in ["0.25", "0.50", "0.75"]:
        vals, ds = [], []
        for Mm in ["rb", "fv", "hl"]:
            d = load(f"{pref}_{Mm}_{K}")
            if not d: vals.append("—"); ds.append("—"); continue
            vals.append(f"{fmt(bench, d['A']['attempted_score']):.2f}")
            ds.append(f"{d['paired_attempted_all']['delta_pp'] * (1 if bench=='textvqa' else 10):+.2f}")
        md.append(f"| {K} | {vals[0]} | {vals[1]} | {vals[2]} | {ds[0]} / {ds[1]} / {ds[2]} |")
    md.append("")

# ---------- DocVQA ----------
md.append("## DocVQA-500 stress (subset, unified 2M px cap, 0 skips all arms)")
md.append("| arm | ANLS attempted | Δ vs None | p |")
md.append("|---|---:|---:|---:|")
for m, lbl in (("pre", "RBM@25"), ("fastv", "FastV@25"), ("hilo", "HiLo@25")):
    d = load(f"doc_{m}")
    if d:
        pa = d["paired_attempted_all"]
        md.append(f"| {lbl} | {d['A']['attempted_score']*100:.2f} | {pa['delta_pp']:+.2f}pp CI[{pa['ci95_pp'][0]:+.1f},{pa['ci95_pp'][1]:+.1f}] | {pa['sign_flip_p_two_sided']:.4f} |")
        b = d["B"]["attempted_score"] * 100
    if m == "pre":
        md.append(f"| None@100% | {b:.2f} | — | — |")
for pair in ["rb_hl", "rb_fv"]:
    d = load(f"doc_{pair}")
    if d:
        pa = d["paired_attempted_all"]
        md.append(f"- direct pair {pair}: Δ={pa['delta_pp']:+.2f}pp CI[{pa['ci95_pp'][0]:+.2f},{pa['ci95_pp'][1]:+.2f}] p={pa['sign_flip_p_two_sided']:.4f}")
md.append("")

# ---------- 95% quality target table ----------
md.append("## Quality target: >=95% of same-protocol None (attempted). Cheapest MEASURED point meeting it")
for bench, pref in (("textvqa", "curve_tv"), ("ocrbench", "curve_oc")):
    dref = load(f"{pref}_rb_0.25")
    if not dref: continue
    none_ref = dref["B"]["attempted_score"]
    for lbl, key in (("RBM", "rb"), ("FastV", "fv"), ("HiLo", "hl")):
        got = None
        for K in ["0.25", "0.50", "0.75"]:
            d = load(f"{pref}_{key}_{K}")
            if d and d["A"]["attempted_score"] >= 0.95 * none_ref:
                got = K; break
        md.append(f"- {bench} {key}: " + (f"MEETS at keep={got}" if got else "NOT-MET at measured points"))
md.append("")

# ---------- timing ----------
for eng in ("timing_harness", "timing_hilo"):
    p = f"{Q}/{eng}.json"
    if not os.path.exists(p):
        md.append(f"## {eng}: MISSING (partial)"); continue
    d = json.load(open(p))
    md.append(f"## {eng} — per-request cost (200 requests x 3 repeats, bs=1)")
    md.append("| config | p50 total_s | p95 total_s | p50 TTFT_s | out_len med | peak_alloc GB | work_wall_s |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for tag, blk in d["results"].items():
        reqs = [r for r in blk["requests"] if not r.get("oom") and "total_s" in r]
        if not reqs:
            md.append(f"| {tag} | ALL OOM | — | — | — | — | — |"); continue
        tot = sorted(r["total_s"] for r in reqs)
        if eng == "timing_hilo":
            tts = sorted(r["ttft_s"] for r in reqs if r.get("ttft_s") is not None)
        else:
            tts = sorted((r.get("vision_merger_s", 0) or 0) + (r.get("selection_s") or 0)
                         + (r.get("prefill_s") or 0) for r in reqs)
        ttft = f"{tts[len(tts)//2]:.3f}" if tts else "—"
        ol = statistics.median(r["out_len"] for r in reqs)
        md.append(f"| {tag} | {tot[len(tot)//2]:.3f} | {tot[int(len(tot)*.95)]:.3f} | "
                  f"{ttft} | {ol:.1f} | {blk.get('peak_alloc_GB','—')} | {blk.get('workload_wall_s','—')} |")
    md.append("")

open("reports/cvm2027_supplement_tables.md", "w").write("\n".join(md))

# ---------- figures ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
COL = {"rb": "#1f77b4", "fv": "#d62728", "hl": "#2ca02c"}
for ax, (bench, pref) in zip(axes, [("textvqa", "curve_tv"), ("ocrbench", "curve_oc")]):
    dref = load(f"{pref}_rb_0.25")
    none_ref = fmt(bench, dref["B"]["attempted_score"])
    ax.axhline(none_ref, ls="--", c="k", lw=1, label="None@100%")
    ax.axhline(0.95 * none_ref, ls=":", c="gray", lw=1, label="95% target")
    for key, lbl in (("rb", "RBM"), ("fv", "FastV-k3"), ("hl", "Hi-Lo")):
        xs, ys = [], []
        for K in [0.25, 0.5, 0.75]:
            d = load(f"{pref}_{key}_{str(K)}" if K != 0.25 else f"{pref}_{key}_0.25")
            d = load(f"{pref}_{key}_{'0.25' if K==0.25 else ('0.50' if K==0.5 else '0.75')}")
            if d:
                xs.append(K * 100); ys.append(fmt(bench, d["A"]["attempted_score"]))
        ax.plot(xs, ys, "o-", color=COL[key], label=lbl)
    ax.set_xlabel("keep (%)"); ax.set_ylabel("official score" if bench == "textvqa" else "points /1000")
    ax.set_title(f"{bench} budget curve ({'TV-1000 subset' if bench=='textvqa' else 'OCRBench full'})")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/budget_curves.svg")

if os.path.exists(f"{Q}/timing_harness.json") and os.path.exists(f"{Q}/timing_hilo.json"):
    th = json.load(open(f"{Q}/timing_harness.json")); tl = json.load(open(f"{Q}/timing_hilo.json"))
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    name_map = {"pre_0.25": "RBM25", "pre_0.5": "RBM50", "pre_0.75": "RBM75",
                "fastv_0.25": "FastV25", "fastv_0.5": "FastV50", "fastv_0.75": "FastV75",
                "none": "None"}
    for tag, blk in th["results"].items():
        ok = [r for r in blk["requests"] if "total_s" in r and not r.get("oom")]
        if not ok: continue
        tot = sorted(r["total_s"] for r in ok)
        lat = tot[len(tot)//2]
        sc = {"none": None}
        key = tag.replace("pre", "rb").replace("fastv", "fv")
        d = load(f"curve_tv_{key.split('_')[0]}_{key.split('_')[1] if '_' in key else '0.25'}")
        # map tags: none / pre_0.25 ... fastv_0.75
        parts = tag.split("_")
        if parts[0] == "none":
            score = fmt("textvqa", load("curve_tv_rb_0.25")["B"]["attempted_score"])
        else:
            ab = {"pre": "rb", "fastv": "fv"}[parts[0]]
            kk = {"0.25": "0.25", "0.5": "0.50", "0.75": "0.75"}[parts[1]]
            score = fmt("textvqa", load(f"curve_tv_{ab}_{kk}")["A"]["attempted_score"])
        ax2.scatter(lat, score, label=name_map.get(tag, tag))
    for tag, blk in tl["results"].items():
        okl = [r for r in blk["requests"] if "total_s" in r and not r.get("oom")]
        if not okl: continue
        tot = sorted(r["total_s"] for r in okl)
        kk = {"0.25": "0.25", "0.5": "0.50", "0.75": "0.75"}[tag.split("_")[1]]
        score = fmt("textvqa", load(f"curve_tv_hl_{kk}")["A"]["attempted_score"])
        ax2.scatter(tot[len(tot)//2], score, marker="^", label=f"HiLo{kk}")
    ax2.set_xlabel("p50 total request latency (s, TextVQA set)")
    ax2.set_ylabel("TextVQA official score (subset-1k)")
    ax2.set_title("quality vs measured cost (full workload mix incl. OCRBench)")
    ax2.legend(fontsize=7)
    fig2.tight_layout(); fig2.savefig(f"{FIG}/score_vs_latency.svg")

print("tables+figures written")
