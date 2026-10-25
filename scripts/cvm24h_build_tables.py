#!/usr/bin/env python
"""Build supplement tables/figures from score JSONs + timing JSON.
Outputs: reports/cvm2027_supplement_tables.md + figures (svg/pdf via matplotlib).
Quality target rule (directive 17/38): target = 95% of same-protocol None score;
report cheapest MEASURED budget point meeting it, else NOT-MET; no interpolation.
"""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = "results/cvm2027_supplement_20261026/plan"
S = f"{Q}/scores"
FIG = "reports/figures_cvm2027_supplement"
os.makedirs(FIG, exist_ok=True)

def load(p):
    if os.path.exists(p):
        return json.load(open(p))
    return None

rows = []
for f in sorted(glob.glob(f"{S}/*.json")):
    d = load(f)
    if d:
        rows.append((os.path.basename(f)[:-5], d))

md = ["# CVM2027 supplement tables (auto-generated)", ""]
md.append("## P1 main table (keep=25%, official attempted-denominator scores)")
md.append("| bench | method | score | vs None delta_pp | CI95 | p |")
md.append("|---|---|---:|---:|---:|---:|")
for name, d in rows:
    if name.startswith("p1_"):
        bench = "textvqa" if "_tv_" in name else "ocrbench"
        who = name.split("_")[2]
        sc = d["A"]["attempted_score"]
        sc_fmt = sc * 100 if bench == "textvqa" else sc * 1000
        unit = "%" if bench == "textvqa" else "/1000"
        md.append(f"| {bench} | {who} | {sc_fmt:.3f}{unit} (A-side vs B=None) | see pair | | |")
md.append("")
md.append("## Pair detail")
for name, d in rows:
    pa = d.get("paired_attempted_all") or {}
    pb = d.get("paired_common_answered_sensitivity") or {}
    md.append(f"### {name}")
    md.append(f"- A: {d['label_a']} attempted={d['A']['attempted_score']:.6f} "
              f"(skipped {d['A']['skipped']})")
    md.append(f"- B: {d['label_b']} attempted={d['B']['attempted_score']:.6f} "
              f"(skipped {d['B']['skipped']})")
    md.append(f"- Δpp(A−B)={pa.get('delta_pp', 0):.4f} CI[{pa.get('ci95_pp',[0,0])[0]:.4f},"
              f"{pa.get('ci95_pp',[0,0])[1]:.4f}] p={pa.get('sign_flip_p_two_sided', 1):.4f} "
              f"(n={pa.get('n_paired')})")
    md.append(f"- common-answered sensitivity: Δ={pb.get('delta_pp', 0):.4f} "
              f"CI{pb.get('ci95_pp')} n={pb.get('n_paired')}")
    md.append(f"- failures: A-only {len(d['failures']['a_only_skip'])}, "
              f"B-only {len(d['failures']['b_only_skip'])}, both {d['failures']['both_skip']}")
    md.append("")

# --- quality-vs-latency figure ---
tim_h = load(f"{Q}/timing_harness.json")
tim_l = load(f"{Q}/timing_hilo.json")
def score_of(name):
    for n, d in rows:
        if n == name:
            b = "tv" if "textvqa" in name else "oc"
            v = d["A"]["attempted_score"]
            return v * (100 if b == "tv" else 1000)
    return None

def e2e_p50(block, tag):
    reqs = block["results"][tag]["requests"]
    med = sorted(r["total_s"] for r in reqs)
    return med[len(med)//2], med[int(len(med)*0.95)]

fig_done = False
if tim_h:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    per_map = {"none": ("c_none", 100),
               "pre_0.25": ("p1_tv_rb_none", None)}
    # score vs latency per (method,budget) on TextVQA cells
    import statistics
    for ax, (bench, skey) in zip(axes, [("textvqa", "tv"), ("ocrbench", "oc")]):
        pass
    fig_done = True
md.append("## Timing (see timing_harness.json / timing_hilo.json; p50/p95 per config)")
for src, d in (("harness", tim_h), ("hilo", tim_l)):
    if not d:
        md.append(f"- {src}: MISSING")
        continue
    for tag, blk in d["results"].items():
        if "requests" not in blk or not blk["requests"]:
            continue
        tot = [r.get("total_s") for r in blk["requests"] if r.get("total_s")]
        ttft = [r.get("ttft_s") or (r.get("vision_merger_s", 0) + r.get("selection_s", 0)
                if r.get("ttft_s") is None and src != "harness" else None)
                for r in blk["requests"]] if src == "hilo" else None
        md.append(f"- {src}/{tag}: p50 total {statistics.median(tot):.3f}s, "
                  f"p95 {sorted(tot)[int(len(tot)*.95)]:.3f}s, "
                  f"peak_alloc {blk.get('peak_alloc_GB')}GB, "
                  f"workload_wall {blk.get('workload_wall_s')}s")
open("reports/cvm2027_supplement_tables.md", "w").write("\n".join(md))
print("wrote reports/cvm2027_supplement_tables.md; figures pending fill-in")
