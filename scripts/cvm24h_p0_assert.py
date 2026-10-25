#!/usr/bin/env python
"""P0 gate assertions over smoke runs + timing estimates (directives 4,5).
Reads predictions/smoke/*.json, writes plan/p0_assert_report.json +
plan/timing_estimates.json.  Fails loud on any no-op/K violation.
"""
import json, os, sys

D = "results/cvm2027_supplement_20261026"
S = f"{D}/predictions/smoke"
NEED = ["none", "noop_pre", "noop_post", "noop_fastv", "pre_keep25", "pre_keep50",
        "pre_keep75", "fastv_keep25", "fastv_keep50", "fastv_keep75",
        "pre_keep25_mimic"]
load = {}
missing = [t for t in NEED if not os.path.exists(f"{S}/{t}.json")]
for t in NEED:
    if os.path.exists(f"{S}/{t}.json"):
        load[t] = json.load(open(f"{S}/{t}.json"))

def ans_map(d):
    return {p["id"]: (p.get("answer"), p.get("n_image_full"),
                      p.get("n_image_kept")) for p in d["per_sample"]}

rep = {"missing_runs": missing, "checks": {}, "failures": []}
def check(name, ok, detail=""):
    rep["checks"][name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        rep["failures"].append(f"{name}: {detail}")

base = load.get("none")
if base:
    bm = ans_map(base)
    # (4a) no-op: answers + kept counts identical to native none
    for t in ["noop_pre", "noop_post", "noop_fastv"]:
        if t in load:
            m = ans_map(load[t])
            adiff = [(k, bm[k][0], m[k][0]) for k in bm if k in m
                     and (bm[k][0] or "").strip() != (m[k][0] or "").strip()]
            kdiff = [(k, m[k][1], m[k][2]) for k in m if m[k][1] != m[k][2]]
            check(f"noop_answers_{t}", not adiff and not kdiff,
                  f"ans_mismatch={len(adiff)} {adiff[:3]} kept_ne_full={len(kdiff)}")
    # (4c) per-image K consistency.  NOTE field-semantics trap (directive 18):
    # in mode=pre the per-sample n_image_full/n_image_kept are recorded AFTER
    # the pre-merge selection (LLM-visible rows), so the true FULL baseline is
    # the `none` run's n_image_full (== pre-merge unit count 1:1 with post-merge
    # rows, L442-483); in fastv/none modes the field is the raw full count.
    for keep, t in [(0.25, "pre_keep25"), (0.5, "pre_keep50"), (0.75, "pre_keep75")]:
        if t in load:
            bad = []
            for p in load[t]["per_sample"]:
                full = bm[p["id"]][1]                      # from none run
                exp = max(1, round(full * keep))
                if p["n_image_kept"] != exp:
                    bad.append((p["id"], p["n_image_kept"], exp))
            check(f"K_formula_{t}", not bad, f"bad={len(bad)} {bad[:5]}")
    # monotone budget ladder: kept25 <= kept50 <= kept75 per id (all methods)
    for fam in ("pre", "fastv"):
        ks = {c: {p["id"]: p["n_image_kept"] for p in load[f"{fam}_keep{c}"]["per_sample"]}
              for c in (25, 50, 75) if f"{fam}_keep{c}" in load}
        if len(ks) == 3:
            viol = [i for i in ks[25] if not ks[25][i] <= ks[50][i] <= ks[75][i]]
            check(f"ladder_{fam}", not viol, str(viol[:5]))
    for drop, t in [(0.75, "fastv_keep25"), (0.5, "fastv_keep50"), (0.25, "fastv_keep75")]:
        if t in load:
            bad = [(p["id"], p["n_image_kept"], max(1, round(p["n_image_full"] * (1 - drop))))
                   for p in load[t]["per_sample"]
                   if p["n_image_kept"] != max(1, round(p["n_image_full"] * (1 - drop)))]
            check(f"K_formula_{t}", not bad, f"bad={len(bad)} {bad[:5]}")
    # (4d) native vs mimic answer divergence report (NOT a gate; diagnosis)
    if "pre_keep25_mimic" in load:
        m = ans_map(load["pre_keep25_mimic"])
        n = ans_map(load["pre_keep25"])
        diff = sum(1 for k in n if (n[k][0] or "") != (m[k][0] or ""))
        rep["checks"]["mrope_native_vs_mimic"] = {
            "pass": True, "detail": f"{diff}/{len(n)} answers differ on smoke32"}
# probe gate
if os.path.exists(f"{D}/plan/p0_probe.json"):
    pr = json.load(open(f"{D}/plan/p0_probe.json"))
    check("probe_gate", pr.get("GATE_P0_pass"), f"see p0_probe.json; "
          f"max logit diffs: {[s['noop_pre']['logits']['max_abs'] for s in pr['samples']]}")

# (5) timing estimates -> machine-readable plan
tim = {}
for t, d in load.items():
    n = len(d["per_sample"])
    wall = d.get("wall_s", 0)          # loop wall (model load excluded, see L2161)
    load_s = d.get("load_s", 0)
    gen_s = max(wall, 0.1)
    tim[t] = {"n": n, "wall_s": wall, "load_s": load_s, "req_per_s_gen": round(n / gen_s, 3),
              "sec_per_req_gen": round(gen_s / n, 2),
              "n_skipped": d.get("n_skipped"),
              "proj_textvqa5000_h": round(5000 * (gen_s / n) / 3600, 2),
              "proj_ocrbench1000_h": round(1000 * (gen_s / n) / 3600, 2)}
json.dump(tim, open(f"{D}/plan/timing_estimates.json", "w"), indent=1)
rep["timing"] = tim
rep["P0_GATE"] = "PASS" if not rep["failures"] and not missing else "FAIL/PARTIAL"
json.dump(rep, open(f"{D}/plan/p0_assert_report.json", "w"), indent=1)
print(json.dumps({"P0_GATE": rep["P0_GATE"], "failures": rep["failures"],
                  "missing": missing}, indent=1))
print(json.dumps(tim, indent=1))
