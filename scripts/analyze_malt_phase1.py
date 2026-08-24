#!/usr/bin/env python3
"""MALT goal-mode Phase-1 analysis: causal path isolation (task 2026-08-25).

Loads runs/malt_goal_mode/explore_{arm}_{bench}_n64.json (arms h0..h4,nb) and
answers the five causal questions using the OFFICIAL per-sample scorer flags
(the runner scores with the repo's official binary/ANLS/OCRBench scorers):

  Q1 text<-transient : H2 vs H1 (and vs H0) -- blocking text reads
  Q2 anchor<-transient: H3 vs H1 (and vs H0) -- blocking anchor reads
  Q3 transient self-update: H4 vs H1 (kv_only strips the update)
  Q4 K/V-only reproduce?   : H4 vs H1 and H4 vs H0
  Q5 main receiver         : whichever ablation collapses the gain

Summary rule (documented, directional at n=64): an ablation arm "kills" the
gain if its macro ~= H0 macro (within noise); "no effect" if ~= H1.
Outputs a human-readable table + a machine-readable JSON digest.
"""
import json
import os
import math
import sys

HERE = "/media/disk2/YZX/research/vla"
OUT = f"{HERE}/runs/malt_goal_mode"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
ARMS = ["h0", "h1", "h2", "h3", "h4", "nb"]
ARM_LABEL = {"h0": "H0 immediate-RBM", "h1": "H1 MALT-1",
             "h2": "H2 no_text_read", "h3": "H3 no_anchor_read",
             "h4": "H4 kv_only", "nb": "H5 no_both"}


def load_arm(arm, bench):
    with open(f"{OUT}/explore_{arm}_{bench}_n64.json") as f:
        d = json.load(f)
    return d


def per_sample(arm, bench):
    d = load_arm(arm, bench)
    return {str(r["id"]): r for r in d["per_sample"] if not r.get("skipped")}


def acc(arm, bench):
    d = load_arm(arm, bench)
    ps = [r for r in d["per_sample"] if not r.get("skipped")]
    if not ps:
        return float("nan")
    return sum(int(r["correct"]) for r in ps) / len(ps)


def macro(arm):
    return sum(acc(arm, b) for b in BENCHES) / len(BENCHES)


def mcnemar_z(a_ps, b_ps):
    """z = (b - a) / sqrt(b + a) for discordant pairs (same id both present).
    b wins where a=0,b=1; a wins where a=1,b=0."""
    n_ab = n_ba = 0
    for i, r in a_ps.items():
        if i not in b_ps:
            continue
        if r["correct"] and not b_ps[i]["correct"]:
            n_ab += 1
        elif not r["correct"] and b_ps[i]["correct"]:
            n_ba += 1
    den = math.sqrt(n_ab + n_ba) if (n_ab + n_ba) else 0.0
    z = (n_ba - n_ab) / den if den else 0.0
    return z, n_ab, n_ba


def agreement(a_ps, b_ps):
    n = same = 0
    for i, r in a_ps.items():
        if i in b_ps:
            n += 1
            if r["answer"] == b_ps[i]["answer"]:
                same += 1
    return same, n


def main():
    print("=== MALT Phase-1 causal ablation: accuracy table (n=64 x 4) ===")
    hdr = f"{'bench':9s}" + "".join(f"{ARM_LABEL[a]:>16s}" for a in ARMS)
    print(hdr)
    tab = {}
    for b in BENCHES:
        row = [f"{b:9s}"]
        for a in ARMS:
            v = acc(a, b)
            row.append(f"{v:>16.3f}")
            tab[(a, b)] = v
        print("".join(row))
    mrow = ["macro    "] + [f"{macro(a):>16.3f}" for a in ARMS]
    print("".join(mrow))
    for a in ARMS:
        tab[(a, "macro")] = macro(a)

    # ---- per-arm gain vs H0 and gap vs H1 ----
    print("\n=== gains vs H0 / gaps vs H1 (macro) ===")
    for a in ARMS[1:]:
        print(f"  {ARM_LABEL[a]:>16s}: vsH0 {macro(a)-macro('h0'):+.3f}"
              f"  vsH1 {macro(a)-macro('h1'):+.3f}")

    # ---- per-sample comparisons (same-id, official scoring) ----
    print("\n=== paired comparisons (z=McNemar, W/L = discordant pairs) ===")
    summary = {}
    for name, (x, y) in {"Q1 text<-trans": ("h2", "h1"),
                         "Q1b text<-trans vsH0": ("h2", "h0"),
                         "Q2 anchor<-trans": ("h3", "h1"),
                         "Q2b anchor<-trans vsH0": ("h3", "h0"),
                         "Q3 self-update": ("h4", "h1"),
                         "Q4 kv_only vsH0": ("h4", "h0"),
                         "no_both vsH1": ("nb", "h1"),
                         "no_both vsH0": ("nb", "h0")}.items():
        ztot, w_tot, l_tot, n = 0.0, 0, 0, 0
        for b in BENCHES:
            a_ps, b_ps = per_sample(x, b), per_sample(y, b)
            z, w, l = mcnemar_z(a_ps, b_ps)
            ztot += z * z
            w_tot += w
            l_tot += l
            n += sum(1 for i in a_ps if i in b_ps)
        zcomb = math.sqrt(ztot / len(BENCHES)) if ztot else 0.0
        # combined z across benches (approx: sum z / sqrt(n_benches))
        zsum = sum(mcnemar_z(per_sample(x, b), per_sample(y, b))[0]
                   for b in BENCHES)
        z_comb = zsum / math.sqrt(len(BENCHES))
        print(f"  {name:>18s}: {ARM_LABEL[x]:>16s} vs {ARM_LABEL[y]:<16s}"
              f" z_comb={z_comb:+.2f}  W={w_tot} L={l_tot} n={n}")
        summary[name] = {"arm": x, "ref": y, "z_comb": round(z_comb, 3),
                         "W": w_tot, "L": l_tot, "n": n}
    for name, (x, y) in {"Q1 text<-trans": ("h2", "h1"),
                         "Q2 anchor<-trans": ("h3", "h1"),
                         "Q3 self-update": ("h4", "h1")}.items():
        for b in BENCHES:
            a_ps, b_ps = per_sample(x, b), per_sample(y, b)
            z, w, l = mcnemar_z(a_ps, b_ps)
            same, n = agreement(a_ps, b_ps)
            print(f"    {name} {b}: z={z:+.2f} W={w} L={l} "
                  f"ans_same={same}/{n}")

    # ---- interpretation (directional; n=64 is underpowered) ----
    g1 = macro("h1") - macro("h0")          # MALT-1 total gain
    print(f"\n=== directional interpretation (gain vs H0 = {g1:+.3f}) ===")
    for arm, q in [("h2", "Q1 text<-transient"), ("h3", "Q2 anchor<-transient"),
                   ("h4", "Q4 kv_only")]:
        g = macro(arm) - macro("h0")
        rel = g / g1 if g1 else float("nan")
        print(f"  {q}: {ARM_LABEL[arm]} gain-vs-H0 = {g:+.3f} "
              f"({rel*100:.0f}% of MALT-1 gain)")
    with open(f"{OUT}/phase1_digest.json", "w") as f:
        json.dump({"tab": {f"{a}/{k}": round(v, 4) for (a, k), v in
                           tab.items()},
                   "summary": summary,
                   "gain_vs_h0": {a: round(macro(a) - macro("h0"), 4)
                                  for a in ARMS}}, f, indent=2)
    print(f"\n[digest] {OUT}/phase1_digest.json")


if __name__ == "__main__":
    main()
