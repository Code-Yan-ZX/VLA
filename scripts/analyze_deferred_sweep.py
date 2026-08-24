#!/usr/bin/env python
"""Deferred-RBM fixed-lifetime K sweep analysis (phase 2, task 2026-08-24).

Arms per benchmark (n=200, official rescore):
  K=0 : immediate-RBM (reused runs/cascade/gate_pre25_*)
  K=1 : new sweep_deferred_K1_*
  K=3 : reused round-1 runs/deferred_rbm/locked_deferred_*_n200.json
  K=5 : new sweep_deferred_K5_*
  K=8 : new sweep_deferred_K8_*
  FastV : runs/rankbridge/locked_fst3_*_n200.json
  Full  : runs/rankbridge/locked_none_*_n200.json

Efficiency comes from the fresh instrumented runs:
  K=0 : runs/deferred_rbm/sweep_pre_*_n200.json
  K in {1,3,5,8} : runs/deferred_rbm/sweep_deferred_K{K}_*_n200.json
  (K=3 re-run must reproduce round-1 accuracy; K=0 re-run must reproduce parents.)

Outputs:
  experiments/deferred_rbm_n200_data/sweep_analysis.json
  experiments/deferred_rbm_n200_data/sweep_per_sample.json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/media/disk2/YZX/research/vla/src/v3_premerger")
from official_scorers import (score_textvqa_vqaacc, score_docvqa_anls,
                              score_gqa, score_ocrbench_batch)

HERE = "/media/disk2/YZX/research/vla"
RB = f"{HERE}/runs/deferred_rbm"
CASCADE = f"{HERE}/runs/cascade"
RANKB = f"{HERE}/runs/rankbridge"
OUT = f"{HERE}/experiments/deferred_rbm_n200_data"
os.makedirs(OUT, exist_ok=True)

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
KS = [0, 1, 3, 5, 8]
rng = np.random.default_rng(0)
NBOOT = 5000


def load(p):
    return json.load(open(p))


def official_per_sample(bench, cell):
    out = {}
    for s in cell.get("per_sample", []):
        if s.get("skipped"):
            continue
        a, g = s.get("answer", ""), s.get("gt", "")
        if bench == "ocrbench":
            v = float(score_ocrbench_batch(
                [(a, g, s.get("question_type", ""), s.get("category"))])
                ["per_item"][0])
        elif bench == "textvqa":
            v = float(score_textvqa_vqaacc(a, g))
        elif bench == "docvqa":
            v = float(score_docvqa_anls(a, g))
        else:
            v = float(score_gqa(a, g))
        out[str(s["id"])] = (v, 1 if v >= 0.5 else 0)
    return out


def agg(per, ids):
    vals = [per[i][0] for i in ids]
    n = len(vals)
    m = sum(vals) / n if n else 0.0
    if n > 1:
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return m, se, n


def mcnemar(per_a, per_b, ids):
    b10 = b01 = 0
    for i in ids:
        da, db = per_a[i][1], per_b[i][1]
        b10 += int(da == 1 and db == 0)
        b01 += int(da == 0 and db == 1)
    z = ((b10 - b01) / math.sqrt(b10 + b01)) if (b10 + b01) else 0.0
    return z, b10, b01


def boot_ci(diffs):
    ids = list(diffs)
    if not ids:
        return (float("nan"), float("nan"))
    arr = np.array([diffs[i] for i in ids])
    means = np.empty(NBOOT)
    for k in range(NBOOT):
        idx = rng.integers(0, len(arr), len(arr))
        means[k] = arr[idx].mean()
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def cell_recs(cell):
    return {str(r["id"]): r for r in cell["per_sample"] if not r.get("skipped")}


def main():
    # ---- accuracy cells (reused for K=0/K=3, fresh for K=1/5/8) ----
    acc_cells = {}
    for b in BENCHES:
        acc_cells[(b, 0)] = load(f"{CASCADE}/gate_pre25_{b}.json")
        acc_cells[(b, 3)] = load(f"{RB}/locked_deferred_{b}_n200.json")
        for K in (1, 5, 8):
            acc_cells[(b, K)] = load(f"{RB}/sweep_deferred_K{K}_{b}_n200.json")
        acc_cells[(b, "fastv")] = load(f"{RANKB}/locked_fst3_{b}_n200.json")
        acc_cells[(b, "full")] = load(f"{RANKB}/locked_none_{b}_n200.json")
    # ---- efficiency cells (fresh instrumented runs) ----
    eff_cells = {}
    for b in BENCHES:
        eff_cells[(b, 0)] = load(f"{RB}/sweep_pre_{b}_n200.json")
        for K in (1, 3, 5, 8):
            eff_cells[(b, K)] = load(f"{RB}/sweep_deferred_K{K}_{b}_n200.json")

    # invariance: K=3 sweep re-run == round-1; K=0 sweep re-run == parents
    print("=== invariance of re-runs (must be identical) ===")
    inv_ok = True
    for b in BENCHES:
        k3_new = cell_recs(eff_cells[(b, 3)])
        k3_old = cell_recs(acc_cells[(b, 3)])
        p0_new = cell_recs(eff_cells[(b, 0)])
        p0_old = cell_recs(acc_cells[(b, 0)])
        s3 = sum(1 for i in k3_new if i in k3_old
                 and k3_new[i]["answer"] == k3_old[i]["answer"])
        s0 = sum(1 for i in p0_new if i in p0_old
                 and p0_new[i]["answer"] == p0_old[i]["answer"])
        ok = s3 == len(k3_old) and s0 == len(p0_old)
        inv_ok = inv_ok and ok
        print(f"  {b}: K3 rerun match={s3}/{len(k3_old)}  pre rerun match="
              f"{s0}/{len(p0_old)}  {'OK' if ok else 'MISMATCH'}")
    if not inv_ok:
        print("  [!!] re-runs do not reproduce reused accuracy -- investigate")

    per = {(b, K): official_per_sample(b, acc_cells[(b, K)])
           for b in BENCHES for K in KS}
    per.update({(b, "fastv"): official_per_sample(b, acc_cells[(b, "fastv")])
                for b in BENCHES})
    per.update({(b, "full"): official_per_sample(b, acc_cells[(b, "full")])
                for b in BENCHES})

    # ================= 5.1 ACCURACY =================
    print("\n=== ACCURACY (official rescore, common IDs) ===")
    common = {}
    for b in BENCHES:
        sets = [set(per[(b, K)]) for K in KS] + [set(per[(b, "fastv")]),
                                                 set(per[(b, "full")])]
        common[b] = sorted(set.intersection(*sets))
    hdr = f"{'bench':9s}" + "".join(f"{'K='+str(K):>9s}" for K in KS) + \
          f"{'fastv':>9s}{'full':>9s}  n"
    print(hdr)
    mean_tbl = {}
    se_tbl = {}
    for b in BENCHES:
        ids = common[b]
        row = []
        for K in KS + ["fastv", "full"]:
            m, se, n = agg(per[(b, K)], ids)
            mean_tbl[(b, K)] = m
            se_tbl[(b, K)] = se
            row.append(f"{m:.4f}")
        print(f"{b:9s}" + "".join(f"{v:>9s}" for v in row) + f"  {len(ids)}")
    macro = {str(K): float(np.mean([mean_tbl[(b, K)] for b in BENCHES]))
             for K in KS + ["fastv", "full"]}
    print("macro    " + "".join(f"{macro[str(K)]:>9.4f}" for K in KS) +
          f"{macro['fastv']:>9.4f}{macro['full']:>9.4f}")

    # differences vs RBM(K=0) / FastV / stronger parent
    print("\n[paired] per-benchmark (def_K minus ...)")
    stats = {}
    for b in BENCHES:
        ids = common[b]
        d = per[(b, 0)]
        spm = {i: max(per[(b, "fastv")][i][0], per[(b, 0)][i][0]) for i in ids}
        entry = {"n_common": len(ids)}
        for K in KS:
            defp = per[(b, K)]
            diff0 = {i: defp[i][0] - d[i][0] for i in ids}
            difff = {i: defp[i][0] - per[(b, "fastv")][i][0] for i in ids}
            diffs = {i: defp[i][0] - spm[i] for i in ids}
            z, b10, b01 = mcnemar(defp, d, ids)
            zf, f10, f01 = mcnemar(defp, per[(b, "fastv")], ids)
            # win/tie/loss vs K=0 and vs fastv and vs stronger(per-sample)
            wtl = {}
            for tag, par in (("vs_k0", d), ("vs_fastv", per[(b, "fastv")])):
                w = l = t = 0
                for i in ids:
                    dd, pp = defp[i][1], par[i][1]
                    w += int(dd == 1 and pp == 0)
                    l += int(dd == 0 and pp == 1)
                    t += int(dd == pp)
                wtl[tag] = (w, t, l)
            entry[str(K)] = {
                "diff_vs_k0": sum(diff0.values()) / len(ids),
                "diff_vs_fastv": sum(difff.values()) / len(ids),
                "diff_vs_stronger_per_sample": sum(diffs.values()) / len(ids),
                "ci95_vs_k0": boot_ci(diff0),
                "ci95_vs_fastv": boot_ci(difff),
                "ci95_vs_stronger_ps": boot_ci(diffs),
                "mcnemar_z_vs_k0": z, "b10": b10, "b01": b01,
                "mcnemar_z_vs_fastv": zf,
                "win_tie_loss": wtl,
            }
        stats[b] = entry
        for K in KS:
            e = entry[str(K)]
            print(f"  {b} K={K}: vsK0={e['diff_vs_k0']:+.4f} "
                  f"vsFastV={e['diff_vs_fastv']:+.4f} "
                  f"vsStrongerPS={e['diff_vs_stronger_per_sample']:+.4f} "
                  f"zK0={e['mcnemar_z_vs_k0']:+.2f} "
                  f"wtl_vsK0={e['win_tie_loss']['vs_k0']}")

    # ================= 5.2 EFFICIENCY =================
    print("\n=== EFFICIENCY (median over common non-skipped samples) ===")
    eff = {}
    for b in BENCHES:
        # common efficiency sample set: non-skipped in the K=0 pre rerun and
        # all deferred K runs (matched comparison)
        sets = [set(cell_recs(eff_cells[(b, 0)]))]
        for K in (1, 3, 5, 8):
            sets.append(set(cell_recs(eff_cells[(b, K)])))
        eff_ids = sorted(set.intersection(*sets))
        e0 = {str(K): {} for K in [0] + [1, 3, 5, 8]}
        for K in [0] + [1, 3, 5, 8]:
            recs = cell_recs(eff_cells[(b, K)])
            lc = [recs[i]["layer_visual_counts"] for i in eff_ids
                  if recs[i].get("layer_visual_counts")]
            area = [sum(l) for l in lc]
            comp = [sum(x * x for x in l) for l in lc]
            e0[str(K)] = {
                "n_measured": len(eff_ids),
                "area_sum_Nl": med(area),
                "compute_sum_Nl2": med(comp),
                "prefill_s_med": med([recs[i].get("prefill_s") for i in eff_ids]),
                "ttft_s_med": med([recs[i].get("ttft_s") for i in eff_ids]),
                "peak_mem_mb_med": med([recs[i].get("peak_mem_mb") for i in eff_ids]),
            }
        eff[b] = e0
        print(f"  {b} (n={len(eff_ids)}):")
        for K in [0, 1, 3, 5, 8]:
            e = e0[str(K)]
            print(f"    K={K}: area={e['area_sum_Nl']:.0f} compute={e['compute_sum_Nl2']:.1e} "
                  f"prefill={e['prefill_s_med']:.4f}s ttft={e['ttft_s_med']:.4f}s "
                  f"peak={e['peak_mem_mb_med']:.0f}MB")

    # ================= 5.3 HETEROGENEITY =================
    print("\n=== LIFETIME HETEROGENEITY ===")
    het = {}
    for b in BENCHES:
        ids = common[b]
        # best fixed K
        best = max(KS, key=lambda K: mean_tbl[(b, K)])
        # earliest-correct K per sample (binary correctness over K=0,1,3,5,8)
        eck = {}
        for i in ids:
            ck = [K for K in KS if per[(b, K)][i][1] == 1]
            eck[i] = min(ck) if ck else None
        oracle_acc = sum(1 for i in ids if eck[i] is not None) / len(ids)
        oracle_mean_k = float(np.mean([eck[i] for i in ids
                                       if eck[i] is not None])) if oracle_acc else None
        dist = {}
        for K in KS:
            dist[K] = sum(1 for i in ids if eck[i] == K)
        dist["never"] = sum(1 for i in ids if eck[i] is None)
        # transitions
        trans = {}
        for Ka, Kb in ((3, 5), (5, 8)):
            wc = cc = uc = 0   # wrong->correct, correct->wrong, answer unchanged
            for i in ids:
                ca, cb = per[(b, Ka)][i][1], per[(b, Kb)][i][1]
                aa, ab = per[(b, Ka)][i][0], per[(b, Kb)][i][0]
                if aa == ab:
                    uc += 1
                elif ca == 0 and cb == 1:
                    wc += 1
                elif ca == 1 and cb == 0:
                    cc += 1
                else:
                    uc += 1
            trans[f"{Ka}->{Kb}"] = {"wrong_to_correct": wc,
                                    "correct_to_wrong": cc,
                                    "answer_unchanged": uc}
        het[b] = {"best_fixed_K": best,
                  "best_fixed_acc": mean_tbl[(b, best)],
                  "oracle_accuracy": oracle_acc,
                  "oracle_mean_K": oracle_mean_k,
                  "earliest_correct_dist": dist,
                  "transitions": trans}
        print(f"  {b}: best_K={best}({mean_tbl[(b,best)]:.4f}) "
              f"oracle_acc={oracle_acc:.4f} oracle_mean_K={oracle_mean_k} "
              f"dist={dist} trans3to5={trans['3->5']} trans5to8={trans['5->8']}")

    # ================= 6 A/B/C/D =================
    print("\n=== SCENARIO JUDGMENT (A/B/C/D) ===")
    # A: clear preference for different K across datasets/samples
    best_per_bench = {b: het[b]["best_fixed_K"] for b in BENCHES}
    oracle_gap = {b: het[b]["oracle_accuracy"] - max(mean_tbl[(b, K)] for K in KS)
                  for b in BENCHES}
    # C: K=3 isolated peak
    k3_is_peak = all(het[b]["best_fixed_K"] == 3 for b in BENCHES)
    # B: same K best everywhere
    same_k = len(set(best_per_bench.values())) == 1
    # D: accuracy increases with K monotonically
    mono_up = all(mean_tbl[(b, 8)] >= mean_tbl[(b, 5)] >= mean_tbl[(b, 3)] >=
                  mean_tbl[(b, 1)] for b in BENCHES)

    print(f"  best fixed K per bench: {best_per_bench}")
    print(f"  oracle accuracy - best-fixed: { {b: round(v,4) for b,v in oracle_gap.items()} }")
    print(f"  K=3 isolated peak everywhere: {k3_is_peak}")
    print(f"  same K best on all benches: {same_k}")
    print(f"  monotone accuracy up in K: {mono_up}")

    if not same_k and any(v != max(best_per_bench.values()) for v in best_per_bench.values()):
        scenario = "A"
    elif same_k:
        scenario = "B"
    elif k3_is_peak:
        scenario = "C"
    elif mono_up:
        scenario = "D"
    else:
        scenario = "MIXED/other"
    print(f"\n  SCENARIO: {scenario}")

    out = {
        "mean_tbl": {f"{b}/K{K}": mean_tbl[(b, K)] for b in BENCHES for K in KS}
        | {f"{b}/fastv": mean_tbl[(b, "fastv")] for b in BENCHES}
        | {f"{b}/full": mean_tbl[(b, "full")] for b in BENCHES},
        "se_tbl": {f"{b}/K{K}": se_tbl[(b, K)] for b in BENCHES for K in KS},
        "macro": macro,
        "paired": stats,
        "efficiency": eff,
        "heterogeneity": het,
        "scenario": scenario,
        "best_per_bench": best_per_bench,
        "oracle_gap": oracle_gap,
        "invariance_ok": inv_ok,
        "git": {"commit": os.popen("git rev-parse --short HEAD").read().strip(),
                "branch": os.popen("git branch --show-current").read().strip()},
    }
    with open(f"{OUT}/sweep_analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- per-sample combined output ----
    combined = {}
    for b in BENCHES:
        recs0 = cell_recs(acc_cells[(b, 0)])
        for K in KS:
            for i, r in cell_recs(acc_cells[(b, K)]).items():
                key = f"{b}/K{K}/{i}"
                combined[key] = {
                    "bench": b, "K": K, "sample_id": i,
                    "answer": r["answer"], "gt": r.get("gt"),
                    "metric": per[(b, K)][i][0],
                    "correct": per[(b, K)][i][1],
                    "anchor_indices": r.get("rb", {}).get("kept_per_image"),
                    "n_image_full": r.get("n_image_full"),
                    "n_image_kept": r.get("n_image_kept"),
                    "n_text": r.get("n_text"),
                    "L_after": r.get("prompt_token_ids"),
                    "fired": r.get("rb", {}).get("fired"),
                    "prefill_s": r.get("prefill_s"),
                    "ttft_s": r.get("ttft_s"),
                    "peak_mem_mb": r.get("peak_mem_mb"),
                    "layer_visual_counts": r.get("layer_visual_counts"),
                }
    with open(f"{OUT}/sweep_per_sample.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[written] {OUT}/sweep_analysis.json + sweep_per_sample.json")


if __name__ == "__main__":
    main()
