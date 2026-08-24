#!/usr/bin/env python3
"""MALT goal-mode compute model: per-arm ΣN_l (token passes) and ΣN_l²
(attention cost) + a per-layer FLOPs model separating the MALT-1 full-update
cost from the H4-style K/V-only (transient) cost.

Uses each run's per-sample layer_visual_counts (active VISUAL tokens per layer)
and n_text.  N_l = n_text + visual_active(layer l).

For MALT-1 (arms h1..nb): every token is a FULL update in layers 0..K-1.
For an H4-REAL ragged implementation (modeled, NOT what we ran): each transient
token in layers 0..K-1 contributes a fraction f_kv of a full token (K/V
projection + serving as keys/values to others), skipping query/attention-row/
o_proj/MLP/residual.  We report f_kv at 0.0 and 0.5 as bounds.
"""
import json
import os
import sys

HERE = "/media/disk2/YZX/research/vla"
OUT = f"{HERE}/runs/malt_goal_mode"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
ARMS = ["h0", "h1", "h2", "h3", "h4", "nb"]


def load(arm, bench):
    with open(f"{OUT}/explore_{arm}_{bench}_n64.json") as f:
        return json.load(f)


def sums(arm, bench):
    """Per-sample (sumN, sumN2) over layers; returns list of tuples."""
    d = load(arm, bench)
    out = []
    for r in d["per_sample"]:
        if r.get("skipped"):
            continue
        lvc = r.get("layer_visual_counts") or []
        nt = r["n_text"]
        Ns = [nt + int(v) for v in lvc]
        out.append((sum(Ns), sum(n * n for n in Ns)))
    return out


def main():
    print("=== per-arm compute (ΣN_l = token-layer passes; ΣN_l² = attn cost) ===")
    hdr = f"{'bench':9s}" + "".join(f"{a:>14s}" for a in ARMS)
    print(hdr)
    agg = {}
    for b in BENCHES:
        print(f"{b:9s}", end="")
        for a in ARMS:
            ss = sums(a, b)
            sumN = sum(s[0] for s in ss) / len(ss)
            sumN2 = sum(s[1] for s in ss) / len(ss)
            print(f"{sumN:>8.0f}/{sumN2:>12.0f}", end="")
            agg[(a, b)] = (sumN, sumN2)
        print()
    print("\n=== ratio vs H1 (macro of means) ===")
    for a in ARMS:
        m1 = sum(agg[(a, b)][0] for b in BENCHES) / 4
        m2 = sum(agg[(a, b)][1] for b in BENCHES) / 4
        h1_1 = sum(agg[("h1", b)][0] for b in BENCHES) / 4
        h1_2 = sum(agg[("h1", b)][1] for b in BENCHES) / 4
        print(f"  {a}: ΣN_l={m1:,.0f} ({m1/h1_1*100:.0f}% of H1)  "
              f"ΣN_l²={m2:,.0f} ({m2/h1_2*100:.0f}% of H1)")
    # H4-real ragged model: transient tokens in layers 0..K-1 at fraction f_kv
    print("\n=== H4-REAL ragged compute model (f_kv of a full token per "
          "transient in the transient lifetime) ===")
    for f_kv in (0.0, 0.3, 0.5):
        tot_h1 = tot_h4 = 0
        for b in BENCHES:
            for r in load("h1", b)["per_sample"]:
                if r.get("skipped"):
                    continue
                lvc = r.get("layer_visual_counts") or []
                nt = r["n_text"]
                n_trans = r["n_image_full"] - r["n_image_kept"]
                # H1: all full; H4-real: transients count f_kv in layers 0..1
                for li, v in enumerate(lvc):
                    N1 = nt + int(v)
                    if li <= 1:  # transient lifetime (fastv-k 1 -> layers 0,1)
                        N4 = nt + int(v) - int(n_trans) * (1.0 - f_kv)
                    else:
                        N4 = N1
                    tot_h1 += N1
                    tot_h4 += N4
        print(f"  f_kv={f_kv}: ΣN_l H4-real/H1 = {tot_h4/tot_h1*100:.1f}%")
    print("\n[note] attention cost is dominated by ΣN_l² for the pruned "
          "prefill; the ragged kernel also removes the transient QUERY rows in "
          "layers 0..1 -> (n_text+n_anchor) x L instead of L x L.")


if __name__ == "__main__":
    main()
