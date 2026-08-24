#!/usr/bin/env python
"""Render audit.json into compact markdown tables for the feasibility report.
Usage: python scripts/audit_malt_summary.py  (reads audit.json, prints md)"""
import json
import os

HERE = "/media/disk2/YZX/research/vla"
OUT = f"{HERE}/experiments/malt_adaptive_gate_feasibility"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]


def main():
    a = json.load(open(f"{OUT}/audit.json"))
    print("## G0 (K0-route) per-bench  [P=precision, R=recall, FSR=1-P, "
          "AUROC, AUPRC, route%]")
    for m in a["results"]:
        print(f"\n### model: {m}")
        print("| bench | P | R | FSR(routed) | FSR(neg-cond) | AUROC | AUPRC | "
              "route% | routedK0acc |")
        print("|---|---|---|---|---|---|---|---|---|")
        for b in a["results"][m]["g0"]:
            g = a["results"][m]["g0"][b]
            def f(x):
                return "nan" if isinstance(x, float) and x != x else f"{x:.3f}"
            print(f"| {b} | {f(g['precision'])} | {f(g['recall'])} | "
                  f"{f(g['false_safe_rate_routed'])} | "
                  f"{f(g['false_safe_rate_neg_cond'])} | {f(g['auroc'])} | "
                  f"{f(g['auprc'])} | {f(g['routed_frac']*100)}% | "
                  f"{f(g['routed_acc_pK0correct'])} |")
    print("\n## G1 (K8-extension) per-bench  [rescueP, rescueR, AUROC, AUPRC, "
          "route%, unnec%, harm%]")
    for m in a["results"]:
        print(f"\n### model: {m}")
        print("| bench | rescueP | rescueR | AUROC | AUPRC | route% | unnec% | harm% |")
        print("|---|---|---|---|---|---|---|---|")
        for b in a["results"][m]["g0"]:
            g = a["results"][m]["g1"][b]
            def f(x):
                return "nan" if isinstance(x, float) and x != x else f"{x:.3f}"
            print(f"| {b} | {f(g['rescue_precision'])} | {f(g['recall'])} | "
                  f"{f(g['auroc'])} | {f(g['auprc'])} | "
                  f"{f(g['routed_frac']*100)}% | {f(g['unnecessary_ext_rate'])} | "
                  f"{f(g['harmful_ext_rate'])} |")
    print("\n## Policy (5-fold OOF)")
    print("| model | macro | dK1 | sumNl2 | vsK1 | vsK8 | K0/K1/K8 routing |")
    print("|---|---|---|---|---|---|---|")
    bl = a["baselines"]
    for m, p in a["policy"].items():
        rc = {b: p["routing_counts"][b] for b in p["routing_counts"]}
        tot = {k: sum(rc[b][k] for b in rc) for k in ("K0", "K1", "K8")}
        acc = a["acceptance"][m]
        def f(x):
            return "nan" if isinstance(x, float) and x != x else f"{x:.4f}"
        print(f"| {m} | {f(p['macro_acc'])} | {f(acc['delta_vs_K1'])} | "
              f"{p['total_sum_Nl2']:.2e} | {f(acc['compute_vs_K1'])} | "
              f"{f(acc['compute_vs_K8'])} | "
              f"{tot['K0']}/{tot['K1']}/{tot['K8']} |")
    print("\n## Baselines")
    print("| bench | K0 | K1 | K8 | oracle |")
    print("|---|---|---|---|---|")
    for b in bl["per_bench"]["0"]:
        print(f"| {b} | {bl['per_bench']['0'][b]['acc']:.4f} | "
              f"{bl['per_bench']['1'][b]['acc']:.4f} | "
              f"{bl['per_bench']['8'][b]['acc']:.4f} | "
              f"{bl['oracle_acc'][b]:.4f} |")
    print(f"macro: K0={bl['macro']['0']:.4f} K1={bl['macro']['1']:.4f} "
          f"K8={bl['macro']['8']:.4f} oracle={bl['oracle_macro']:.4f}")
    print("compute sumNl2: K0=%.2e K1=%.2e K8=%.2e oracle=%.2e" % (
        bl["compute"]["sum_Nl2"]["0"], bl["compute"]["sum_Nl2"]["1"],
        bl["compute"]["sum_Nl2"]["8"], bl["compute"]["sum_Nl2"]["oracle"]))
    print("\n## Acceptance")
    for m, acc in a["acceptance"].items():
        print(f"{m}: acc_pass={acc['accuracy_oriented_pass']} "
              f"eff_pass={acc['efficiency_oriented_pass']} "
              f"worst_drop={acc['worst_single_dataset_drop_vs_K1']:.4f} "
              f"lodo_macro={acc.get('lodo_policy_macro')}")
        print(f"   lodo_per_bench={ {k: round(v,4) if v else v for k,v in acc['lodo_policy_per_bench'].items()} }")


if __name__ == "__main__":
    main()
