#!/usr/bin/env python3
"""Gate C verdict: locked n=200 x 4, candidate vs Full / native RBM / FastV-k3.
Computes the 8 formal success criteria. Paired per-sample via the official
scorer (paired_stats score_sample)."""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "v3_premerger"))
from paired_stats import load_cell_scores, run_pair, score_sample  # noqa: E402

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "runs/merger_repr/gateC"))
    args = ap.parse_args()

    # official per-sample scoring via paired_stats.score_sample (official
    # scorers; gqa normalized EM, textvqa vqa-acc, docvqa anls, ocrbench
    # containment) -- NOT the runner's inline containment.
    cells = {}
    for f in os.listdir(args.dir):
        if not f.endswith(".json"):
            continue
        cells[f.replace(".json", "")] = os.path.join(args.dir, f)

    def load(tag, bench):
        f = cells.get(f"{bench}_{tag}")
        if not f:
            return None, None, None, False
        return load_cell_scores(os.path.relpath(f, REPO), bench, False)

    scores = {}
    for b in BENCHES:
        scores[b] = {}
        for tag in ["full", "rbm_native", "fastv_k3", "candidate"]:
            sc, n, skip, ok = load(tag, b)
            scores[b][tag] = sc if ok else None

    print("=" * 72)
    print("GATE C — locked n=200 x 4 (official per-sample)")
    print("=" * 72)
    print(f"{'bench':<10s}{'full':>10s}{'RBM':>10s}{'FastV':>10s}{'cand':>10s}"
          f"{'cand-RBM':>10s}{'cand-parent':>14s}{'n':>6s}")
    macro = {t: [] for t in ["full", "rbm_native", "fastv_k3", "candidate"]}
    for b in BENCHES:
        row = [b]
        for t in ["full", "rbm_native", "fastv_k3", "candidate"]:
            sc = scores[b][t]
            m = (sum(sc.values()) / len(sc)) if sc else None
            if m is not None:
                macro[t].append(m)
            row.append(f"{m:.4f}" if m is not None else "-")
        rb = scores[b]["rbm_native"]
        ca = scores[b]["candidate"]
        parent = None
        for t in ["full", "rbm_native", "fastv_k3"]:
            sc = scores[b][t]
            if sc:
                m = sum(sc.values()) / len(sc)
                parent = m if parent is None else max(parent, m)
        if rb and ca:
            mrb = sum(rb.values()) / len(rb)
            mca = sum(ca.values()) / len(ca)
            row.append(f"{mca - mrb:+.4f}")
            row.append(f"{mca - parent:+.4f}" if parent is not None else "-")
        else:
            row += ["-", "-"]
        n = len(scores[b]["candidate"]) if scores[b]["candidate"] else 0
        row.append(str(n))
        print(f"{row[0]:<10s}" + "".join(f"{v:>10s}" for v in row[1:5]) +
              "".join(f"{v:>10s}" for v in row[5:7]) + f"{row[7]:>6s}")
    print(f"{'macro':<10s}" +
          "".join(f"{(sum(macro[t])/len(macro[t])):.4f}" if macro[t] else "-"
                  for t in ["full", "rbm_native", "fastv_k3", "candidate"]))

    print()
    print("--- paired significance (candidate vs RBM / vs stronger parent) ---")
    for b in BENCHES:
        ca = scores[b]["candidate"]
        rb = scores[b]["rbm_native"]
        if not ca or not rb:
            print(f"  {b}: incomplete")
            continue
        ids = set(ca) & set(rb)
        for label, other in [("RBM", rb)]:
            o = {k: other[k] for k in ids}
            c = {k: ca[k] for k in ids}
            res = run_pair(c, o, b, "candidate", label, 20000, 0,
                           n_nominal=len(ids))
            print(f"  {b} cand-{label}: d={res['mean_delta_pp']:+.3f}pp "
                  f"ci95={res['ci95_pp']} p={res['perm_p_two_sided']:.4f} "
                  f"n={res['n_paired']}")


if __name__ == "__main__":
    main()
