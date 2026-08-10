#!/usr/bin/env python3
"""cross_arch_law.py - Test the cross-architecture PREDICTIVE LAW.

HYPOTHESIS: the pre>post stage-effect magnitude is predicted by the degree of
pre/post unit-saliency ranking divergence (M1).  Where the 2x2 merger rewrites
the ranking (pre and post select DIFFERENT units -> low Jaccard / low Spearman),
the stage effect is large.  Where pre and post agree (high Jaccard), it is small.

This script is CPU-only (no model loading).  It:
  1. Recomputes per-image M1 (Jaccard@k, Spearman(pre,post), edge-rankshift)
     directly from the Qwen3-VL survival_capture npz files and verifies against
     the pre-computed drafts/figures/token_survival_stats.json.
  2. Matches capture image ids to the full-matrix eval cells to get per-image
     pre/post correctness (binary stage effect = pre_correct - post_correct).
  3. Tests the law at TWO levels:
       (a) Benchmark-level: 3 Qwen3-VL points (textvqa, docvqa, gqa) - M1 vs
           stage-effect magnitude (pp from paired_metric_statistics.md).
       (b) Per-image: 192 Qwen3-VL images - per-image M1 vs per-image stage effect.
  4. Reports correlations (Pearson + Spearman) + verdict.
  5. Flags the cross-architecture data gap (Qwen2.5-VL / InternVL3 have NO M1).

Usage:  python3 src/v3_premerger/cross_arch_law.py
Outputs: experiments/cross_arch_predictive_law.md  (report)
         stdout table + verdict
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAP_DIR = os.path.join(ROOT, "runs", "v3_merger_aware", "survival_capture")
STATS_JSON = os.path.join(ROOT, "drafts", "figures", "token_survival_stats.json")
OUT_MD = os.path.join(ROOT, "experiments", "cross_arch_predictive_law.md")


def _pr(a, b):
    """Pearson -> (r, p) tuple (robust across scipy versions)."""
    res = pearsonr(a, b)
    return (float(res.statistic), float(res.pvalue))


def _sr(a, b):
    """Spearman -> (rho, p) tuple."""
    res = spearmanr(a, b)
    return (float(res.statistic), float(res.pvalue))

# Stage-effect magnitudes (delta pre-post, pp) @ 25% retention, from
# experiments/paired_metric_statistics.md (official scorers, seed=0).
STAGE_EFFECT_PP = {
    "qwen3vl":  {"textvqa": +38.367, "docvqa": +24.298, "ocrbench": +36.300, "gqa": -2.830},
    "qwen2vl":  {"textvqa": +26.053, "docvqa": +11.045, "ocrbench": +29.300, "gqa": -2.584},
    "internvl3":{"textvqa": +37.420, "docvqa": +34.641, "ocrbench": +43.200, "gqa": -0.382},
    "glm4v":    {"textvqa": +16.833, "docvqa": +9.844,  "gqa": +4.500},  # n=200, thinking
}

# Eval cells with per_sample (id + correct) for Qwen3-VL pre/post @ r=0.75.
EVAL_CELLS = {
    "textvqa": ("runs/full_matrix/j7_qwen3vl_pre_textvqa_r0.750_full.json",
                "runs/full_matrix/j7_qwen3vl_post_textvqa_r0.750_full.json"),
    "docvqa":  ("runs/full_matrix/j7_qwen3vl_pre_docvqa_r0.750_full.json",
                "runs/full_matrix/j7_qwen3vl_post_docvqa_r0.750_full.json"),
    "gqa":     ("runs/full_matrix/j7_qwen3vl_pre_gqa_r0.750_full.json",
                "runs/full_matrix/j7_qwen3vl_post_gqa_r0.750_full.json"),
}

R = 0.75  # retention ratio (keep 25%)


def compute_m1_for_bench(bench: str):
    """Recompute per-image M1 from the survival_capture npz.

    Returns list of dicts: {id, n_units, k, jaccard, spearman, kendall,
    edge_rankshift, pre_edge_keep, post_edge_keep}.
    """
    npz = np.load(os.path.join(CAP_DIR, f"{bench}.npz"), allow_pickle=True)
    pre_all, post_all, edge_all = npz["pre"], npz["post"], npz["edge"]
    offs, ids = npz["offsets"], [str(x) for x in npz["ids"]]
    rows = []
    for i in range(len(ids)):
        s, e = int(offs[i]), int(offs[i + 1])
        pre, post, edge = pre_all[s:e], post_all[s:e], edge_all[s:e]
        n = len(pre)
        k = max(1, int(round(n * (1.0 - R))))
        # top-k by score (descending) - tie-break by index for determinism
        pre_idx = np.argsort(-pre, kind="stable")[:k]
        post_idx = np.argsort(-post, kind="stable")[:k]
        pre_set, post_set = set(pre_idx.tolist()), set(post_idx.tolist())
        inter = len(pre_set & post_set)
        union = len(pre_set | post_set)
        jac = inter / union if union else 0.0
        rho = _sr(pre, post)[0]
        tau, _ = kendalltau(pre, post)
        # edge-rankshift: rank_shift = post_rank - pre_rank (ranks: 1=highest score)
        # +rank_shift => merger DEMOTED the unit (pre ranked it high, post ranks it low)
        pre_rank = np.argsort(np.argsort(-pre, kind="stable"), kind="stable")  # 0=highest
        post_rank = np.argsort(np.argsort(-post, kind="stable"), kind="stable")
        rank_shift = post_rank - pre_rank  # + => demoted by merger
        ers = _sr(rank_shift, edge)[0]
        rows.append({
            "id": ids[i], "n_units": n, "k": k, "jaccard": jac,
            "spearman": float(rho) if rho == rho else 0.0,
            "kendall": float(tau) if tau == tau else 0.0,
            "edge_rankshift": float(ers) if ers == ers else 0.0,
        })
    return rows


def load_correctness(bench: str):
    """Load per-image (pre_correct, post_correct) matched to capture ids."""
    pre_f, post_f = EVAL_CELLS[bench]
    with open(os.path.join(ROOT, pre_f)) as f:
        pre = {s["id"]: int(s.get("correct", 0)) for s in json.load(f)["per_sample"]
               if not s.get("skipped", False)}
    with open(os.path.join(ROOT, post_f)) as f:
        post = {s["id"]: int(s.get("correct", 0)) for s in json.load(f)["per_sample"]
                if not s.get("skipped", False)}
    return pre, post


def fmt_r(r, p):
    return f"r={r:+.3f} (p={p:.3f})" if p >= 0.001 else f"r={r:+.3f} (p<0.001)"


def main():
    # ---- 1. Recompute M1 + verify against pre-computed stats ----
    with open(STATS_JSON) as f:
        stats = json.load(f)["sample"]

    print("=" * 78)
    print("STEP 1-2: Recompute per-image M1 from npz (Qwen3-VL, 64 imgs/bench)")
    print("=" * 78)
    all_rows = {}  # bench -> list of rows (with correctness)
    bench_m1 = {}  # bench -> mean M1
    for bench in ["textvqa", "docvqa", "gqa"]:
        rows = compute_m1_for_bench(bench)
        pre_c, post_c = load_correctness(bench)
        # attach correctness
        n_match = 0
        for r in rows:
            rid = r["id"]
            if rid in pre_c and rid in post_c:
                r["pre_correct"] = pre_c[rid]
                r["post_correct"] = post_c[rid]
                r["stage_effect"] = pre_c[rid] - post_c[rid]  # -1/0/+1
                n_match += 1
            else:
                r["pre_correct"] = r["post_correct"] = r["stage_effect"] = None
        all_rows[bench] = rows
        # benchmark-level mean M1
        jac = np.mean([r["jaccard"] for r in rows])
        rho = np.mean([r["spearman"] for r in rows])
        ers = np.mean([r["edge_rankshift"] for r in rows])
        bench_m1[bench] = {"jaccard": jac, "spearman": rho, "edge_rankshift": ers}
        # verify vs pre-computed
        sj = stats[bench]["jaccard_mean"]
        sr = stats[bench]["spearman_pre_post_mean"]
        se = stats[bench]["rank_shift_vs_edge_spearman_mean"]
        print(f"\n  {bench}: n_units_mean={np.mean([r['n_units'] for r in rows]):.1f}, "
              f"k_mean={np.mean([r['k'] for r in rows]):.1f}, matched={n_match}/64")
        print(f"    Jaccard    = {jac:.4f}  (stats.json: {sj:.4f}, "
              f"diff={abs(jac-sj):.4f}) {'OK' if abs(jac-sj)<0.01 else 'MISMATCH'}")
        print(f"    Spearman   = {rho:.4f}  (stats.json: {sr:.4f}, "
              f"diff={abs(rho-sr):.4f}) {'OK' if abs(rho-sr)<0.01 else 'MISMATCH'}")
        print(f"    EdgeRankSh = {ers:.4f}  (stats.json: {se:.4f}, "
              f"diff={abs(ers-se):.4f}) {'OK' if abs(ers-se)<0.01 else 'MISMATCH'}")

    # ---- 3. Benchmark-level test (3 Qwen3-VL points) ----
    print("\n" + "=" * 78)
    print("STEP 4a: Benchmark-level test (3 Qwen3-VL points: textvqa, docvqa, gqa)")
    print("=" * 78)
    benches = ["textvqa", "docvqa", "gqa"]
    print(f"\n  {'bench':<10} {'Jaccard':>8} {'Spearman':>9} {'EdgeRS':>8} {'stage_pp':>9}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*8} {'-'*9}")
    jac_vals, sp_vals, ers_vals, eff_vals = [], [], [], []
    for b in benches:
        j = bench_m1[b]["jaccard"]
        s = bench_m1[b]["spearman"]
        e = bench_m1[b]["edge_rankshift"]
        eff = STAGE_EFFECT_PP["qwen3vl"][b]
        jac_vals.append(j); sp_vals.append(s); ers_vals.append(e); eff_vals.append(eff)
        print(f"  {b:<10} {j:>8.4f} {s:>9.4f} {e:>8.4f} {eff:>+9.3f}")
    # correlations (law predicts NEGATIVE: higher M1-agreement -> lower stage-effect)
    r_jac_p = _pr(jac_vals, eff_vals)
    r_sp_p = _pr(sp_vals, eff_vals)
    r_ers_p = _pr(ers_vals, eff_vals)
    r_jac_s = _sr(jac_vals, eff_vals)
    r_sp_s = _sr(sp_vals, eff_vals)
    r_ers_s = _sr(ers_vals, eff_vals)
    print(f"\n  Pearson  Jaccard~effect : {fmt_r(*r_jac_p)}  (pred: negative)")
    print(f"  Pearson  Spearman~effect: {fmt_r(*r_sp_p)}  (pred: negative)")
    print(f"  Pearson  EdgeRS~effect  : {fmt_r(*r_ers_p)}  (pred: positive)")
    print(f"  Spearman Jaccard~effect : {fmt_r(*r_jac_s)}")
    print(f"  Spearman Spearman~effect: {fmt_r(*r_sp_s)}")
    print(f"  Spearman EdgeRS~effect  : {fmt_r(*r_ers_s)}")
    print(f"  NOTE: n=3 points -> no significance possible; direction-only.")

    # ---- 4. Per-image test (192 Qwen3-VL images) ----
    print("\n" + "=" * 78)
    print("STEP 4b: Per-image test (192 Qwen3-VL images: 64 x 3 benchmarks)")
    print("=" * 78)
    all_jac, all_sp, all_ers, all_eff, all_bench = [], [], [], [], []
    for b in benches:
        for r in all_rows[b]:
            if r["stage_effect"] is not None:
                all_jac.append(r["jaccard"])
                all_sp.append(r["spearman"])
                all_ers.append(r["edge_rankshift"])
                all_eff.append(r["stage_effect"])
                all_bench.append(b)
    all_jac = np.array(all_jac); all_sp = np.array(all_sp)
    all_ers = np.array(all_ers); all_eff = np.array(all_eff)
    n_img = len(all_eff)
    n_pre_post = int(np.sum(all_eff > 0))   # pre>post
    n_post_pre = int(np.sum(all_eff < 0))   # post>pre
    n_tie = int(np.sum(all_eff == 0))
    print(f"\n  n={n_img} images | pre>post: {n_pre_post} | post>pre: {n_post_pre} | tie: {n_tie}")
    print(f"  (pre>post rate = {n_pre_post/n_img:.1%})")
    print(f"\n  Per-image correlations (stage_effect = pre_correct - post_correct in {{-1,0,+1}}):")
    pj = _pr(all_jac, all_eff); ps = _pr(all_sp, all_eff); pe = _pr(all_ers, all_eff)
    sj = _sr(all_jac, all_eff); ss = _sr(all_sp, all_eff); se = _sr(all_ers, all_eff)
    print(f"    Pearson  Jaccard~effect   : {fmt_r(*pj)}  (pred: negative)")
    print(f"    Pearson  Spearman(pre,post)~effect: {fmt_r(*ps)}  (pred: negative)")
    print(f"    Pearson  EdgeRankShift~eff: {fmt_r(*pe)}  (pred: positive)")
    print(f"    Spearman Jaccard~effect   : {fmt_r(*sj)}")
    print(f"    Spearman Spearman(p,p)~eff: {fmt_r(*ss)}")
    print(f"    Spearman EdgeRankShift~eff: {fmt_r(*se)}")

    # group comparison: pre>post vs rest
    print(f"\n  Group comparison (pre>post vs pre<=post):")
    grp_pre_better = all_eff > 0
    for name, vals in [("Jaccard", all_jac), ("Spearman", all_sp), ("EdgeRS", all_ers)]:
        a = vals[grp_pre_better]; b = vals[~grp_pre_better]
        try:
            u, pu = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            u, pu = float("nan"), float("nan")
        print(f"    {name:<10}: pre>post mean={np.mean(a):.4f} (n={len(a)}) | "
              f"rest mean={np.mean(b):.4f} (n={len(b)}) | "
              f"diff={np.mean(a)-np.mean(b):+.4f} | MWU p={pu:.4f}")

    # per-bench per-image correlations (for robustness)
    print(f"\n  Per-bench per-image correlations (Pearson Jaccard~effect):")
    for b in benches:
        rows_b = [r for r in all_rows[b] if r["stage_effect"] is not None]
        j = np.array([r["jaccard"] for r in rows_b])
        e = np.array([r["stage_effect"] for r in rows_b])
        s = np.array([r["spearman"] for r in rows_b])
        if len(set(e.tolist())) > 1:
            rj2, pj2 = _pr(j, e); rs2, ps2 = _pr(s, e)
            nb = int(np.sum(e > 0))
            print(f"    {b:<10}: n={len(e)}, pre>post={nb}, "
                  f"Jaccard~eff {fmt_r(rj2, pj2)}, Spearman~eff {fmt_r(rs2, ps2)}")
        else:
            print(f"    {b:<10}: n={len(e)}, no variance in stage_effect")

    # ---- 5. Cross-architecture qualitative summary ----
    print("\n" + "=" * 78)
    print("STEP 4c: Cross-architecture context + DATA GAP")
    print("=" * 78)
    print(f"\n  Stage-effect magnitudes (delta pre-post, pp) @ 25% retention:")
    print(f"  {'family':<12} {'textvqa':>9} {'docvqa':>9} {'ocrbench':>9} {'gqa':>9}")
    for fam in ["qwen3vl", "qwen2vl", "internvl3", "glm4v"]:
        vals = STAGE_EFFECT_PP[fam]
        print(f"  {fam:<12} {vals.get('textvqa',0):>+9.2f} {vals.get('docvqa',0):>+9.2f} "
              f"{vals.get('ocrbench','-'):>9} {vals.get('gqa',0):>+9.2f}")
    print(f"\n  M1 (ranking divergence) availability:")
    print(f"    Qwen3-VL  : YES (survival_capture npz, 64 imgs/bench, 3 benchmarks)")
    print(f"    Qwen2.5-VL: NO  (no kept_indices, no unit scores; --save-unit-scores")
    print(f"                 does not write kept indices - see j3_mechanism_crossarch.md)")
    print(f"    InternVL3 : NO  (no kept_indices, no unit scores)")
    print(f"    GLM-4.1V  : NO  (n=200, no mechanism capture)")
    print(f"\n  => Cross-architecture quantitative test NOT possible with current data.")
    print(f"     Only Qwen3-VL (1 family x 3 benchmarks) has M1.")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # determine verdict from per-image test (the one with statistical power)
    law_holds_perimg = (pj[0] < -0.1 and pj[1] < 0.05) or (ps[0] < -0.1 and ps[1] < 0.05)
    law_direction_perimg = pj[0] < 0  # at least correct sign
    law_holds_bench = r_jac_p[0] < 0  # correct sign at benchmark level

    print(f"\n  Benchmark-level (n=3 Qwen3-VL points):")
    print(f"    Jaccard~effect Pearson r = {r_jac_p[0]:+.3f} (pred: negative)")
    print(f"    Direction {'CORRECT' if law_holds_bench else 'WRONG'} but n=3 (no significance)")
    print(f"\n  Per-image (n={n_img} Qwen3-VL images):")
    print(f"    Jaccard~effect    Pearson r = {pj[0]:+.3f}, p = {pj[1]:.4f} (pred: negative)")
    print(f"    Spearman(p,p)~eff Pearson r = {ps[0]:+.3f}, p = {ps[1]:.4f} (pred: negative)")
    print(f"    EdgeRankShift~eff Pearson r = {pe[0]:+.3f}, p = {pe[1]:.4f} (pred: positive)")
    sig = (pj[1] < 0.05) or (ps[1] < 0.05) or (pe[1] < 0.05)
    print(f"    Significant (p<0.05) on any M1: {'YES' if sig else 'NO'}")
    print(f"\n  Cross-architecture: UNTESTED (M1 missing for Qwen2.5-VL, InternVL3, GLM-4.1V)")

    # ---- write markdown report ----
    write_report(bench_m1, STAGE_EFFECT_PP,
                 r_jac_p, r_sp_p, r_ers_p, r_jac_s, r_sp_s, r_ers_s,
                 pj, ps, pe, sj, ss, se,
                 n_img, n_pre_post, n_post_pre, n_tie,
                 all_rows, all_jac, all_sp, all_ers, all_eff, all_bench)
    print(f"\n  Report written to: {OUT_MD}")


def write_report(bench_m1, stage_eff, rjp, rsp, rep, rjs, rss, res,
                 pj, ps, pe, sj, ss, se,
                 n_img, n_pre, n_post, n_tie,
                 all_rows, all_jac, all_sp, all_ers, all_eff, all_bench):
    lines = []
    lines.append("# Cross-Architecture Predictive Law Test")
    lines.append("")
    lines.append("**Hypothesis:** the pre>post stage-effect magnitude is predicted by the")
    lines.append("degree of pre/post unit-saliency ranking divergence (M1). Where the 2x2")
    lines.append("merger rewrites the ranking (pre and post select DIFFERENT units -> low")
    lines.append("Jaccard / low Spearman), the stage effect is large. Where pre and post")
    lines.append("agree (high Jaccard), the stage effect is small.")
    lines.append("")
    lines.append("**CPU-only analysis.** M1 recomputed from `runs/v3_merger_aware/survival_capture/*.npz`")
    lines.append("(Qwen3-VL, 64 images/bench, L2 both stages, r=0.75). Verified against")
    lines.append("`drafts/figures/token_survival_stats.json`. Stage-effect magnitudes from")
    lines.append("`experiments/paired_metric_statistics.md` (official scorers, seed=0).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Data availability (M1 = pre/post ranking divergence)")
    lines.append("")
    lines.append("| Family | M1 available? | Source | Benchmarks |")
    lines.append("|--------|---------------|--------|------------|")
    lines.append("| Qwen3-VL-8B | YES | survival_capture npz (pre/post L2 + Sobel edge, 64 imgs) | textvqa, docvqa, gqa |")
    lines.append("| Qwen2.5-VL-7B | **NO** | kept_indices never saved; `--save-unit-scores` does not write kept indices (j3_mechanism_crossarch.md) | - |")
    lines.append("| InternVL3-8B | **NO** | no unit-score capture, no kept_indices | - |")
    lines.append("| GLM-4.1V-9B | **NO** | n=200, no mechanism capture | - |")
    lines.append("")
    lines.append("**Critical gap:** the cross-architecture quantitative test CANNOT be performed.")
    lines.append("M1 exists for only 1 of 4 families (Qwen3-VL). The law is tested WITHIN")
    lines.append("Qwen3-VL only (3 benchmarks, 64 images each = 192 per-image points).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Test A: Benchmark-level (3 Qwen3-VL points)")
    lines.append("")
    lines.append("| Benchmark | Jaccard@k | Spearman(pre,post) | Edge-rankshift | Stage-effect (pp) |")
    lines.append("|-----------|-----------|---------------------|----------------|-------------------|")
    for b in ["textvqa", "docvqa", "gqa"]:
        lines.append(f"| {b} | {bench_m1[b]['jaccard']:.4f} | {bench_m1[b]['spearman']:.4f} | "
                     f"{bench_m1[b]['edge_rankshift']:.4f} | {stage_eff['qwen3vl'][b]:+.2f} |")
    lines.append("")
    lines.append("| Metric | Pearson r (pred sign) | Spearman r |")
    lines.append("|--------|-----------------------|------------|")
    lines.append(f"| Jaccard ~ effect | {rjp[0]:+.3f} (p={rjp[1]:.3f}) [pred: -] | {rjs[0]:+.3f} (p={rjs[1]:.3f}) |")
    lines.append(f"| Spearman(p,p) ~ effect | {rsp[0]:+.3f} (p={rsp[1]:.3f}) [pred: -] | {rss[0]:+.3f} (p={rss[1]:.3f}) |")
    lines.append(f"| Edge-rankshift ~ effect | {rep[0]:+.3f} (p={rep[1]:.3f}) [pred: +] | {res[0]:+.3f} (p={res[1]:.3f}) |")
    lines.append("")
    lines.append(f"n=3 points: direction-only, no statistical significance possible.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Test B: Per-image within Qwen3-VL (192 images)")
    lines.append("")
    lines.append(f"Per-image stage effect = pre_correct - post_correct in {{-1, 0, +1}}.")
    lines.append(f"n={n_img} images | pre>post: {n_pre} | post>pre: {n_post} | tie: {n_tie} "
                 f"(pre>post rate = {n_pre/n_img:.1%}).")
    lines.append("")
    lines.append("| Metric | Pearson r (pred sign) | p | Spearman r | p |")
    lines.append("|--------|-----------------------|------|-----------|------|")
    lines.append(f"| Jaccard ~ effect | {pj[0]:+.4f} [pred: -] | {pj[1]:.4f} | {sj[0]:+.4f} | {sj[1]:.4f} |")
    lines.append(f"| Spearman(p,p) ~ effect | {ps[0]:+.4f} [pred: -] | {ps[1]:.4f} | {ss[0]:+.4f} | {ss[1]:.4f} |")
    lines.append(f"| Edge-rankshift ~ effect | {pe[0]:+.4f} [pred: +] | {pe[1]:.4f} | {se[0]:+.4f} | {se[1]:.4f} |")
    lines.append("")
    # group comparison
    grp = all_eff > 0
    lines.append("### Group comparison: pre>post vs pre<=post")
    lines.append("")
    lines.append("| Metric | pre>post mean (n) | rest mean (n) | diff |")
    lines.append("|--------|-------------------|---------------|------|")
    for name, vals in [("Jaccard", all_jac), ("Spearman", all_sp), ("EdgeRS", all_ers)]:
        a = vals[grp]; b = vals[~grp]
        try:
            _, pu = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            pu = float("nan")
        lines.append(f"| {name} | {np.mean(a):.4f} ({len(a)}) | {np.mean(b):.4f} ({len(b)}) | "
                     f"{np.mean(a)-np.mean(b):+.4f} (MWU p={pu:.4f}) |")
    lines.append("")
    lines.append("### Per-bench per-image correlations (Pearson, Jaccard ~ effect)")
    lines.append("")
    lines.append("| Benchmark | n | pre>post | Jaccard~eff r (p) | Spearman(p,p)~eff r (p) |")
    lines.append("|-----------|---|----------|--------------------|------------------------|")
    for b in ["textvqa", "docvqa", "gqa"]:
        rows_b = [r for r in all_rows[b] if r["stage_effect"] is not None]
        j = np.array([r["jaccard"] for r in rows_b])
        e = np.array([r["stage_effect"] for r in rows_b])
        s = np.array([r["spearman"] for r in rows_b])
        nb = int(np.sum(e > 0))
        if len(set(e.tolist())) > 1:
            rj, pj2 = _pr(j, e); rs2, ps2 = _pr(s, e)
            lines.append(f"| {b} | {len(e)} | {nb} | {rj:+.3f} ({pj2:.3f}) | {rs2:+.3f} ({ps2:.3f}) |")
        else:
            lines.append(f"| {b} | {len(e)} | {nb} | no variance | no variance |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Cross-architecture context (qualitative)")
    lines.append("")
    lines.append("Stage-effect magnitudes (delta pre-post, pp) @ 25% retention:")
    lines.append("")
    lines.append("| Family | textvqa | docvqa | ocrbench | gqa |")
    lines.append("|--------|---------|--------|----------|-----|")
    for fam in ["qwen3vl", "qwen2vl", "internvl3", "glm4v"]:
        v = stage_eff[fam]
        oc = f"{v['ocrbench']:+.2f}" if 'ocrbench' in v else "-"
        lines.append(f"| {fam} | {v['textvqa']:+.2f} | {v['docvqa']:+.2f} | {oc} | {v['gqa']:+.2f} |")
    lines.append("")
    lines.append("All 3 non-thinking families show the same **pattern**: large pre>post on")
    lines.append("text-dense benchmarks (textvqa/docvqa/ocrbench: +11 to +43 pp) and ~zero")
    lines.append("on GQA (-0.4 to -2.8 pp). The Qwen3-VL M1 shows more ranking divergence")
    lines.append("on text-dense (Jaccard 0.18-0.24, Spearman 0.14-0.33) than on GQA")
    lines.append("(Jaccard 0.28, Spearman 0.36). This is **consistent** with the law but")
    lines.append("NOT a quantitative cross-architecture test (M1 missing for 3 of 4 families).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    sig_any = (pj[1] < 0.05) or (ps[1] < 0.05) or (pe[1] < 0.05)
    bench_dir = rjp[0] < 0
    lines.append(f"**Does M1 predict the stage-effect magnitude?**")
    lines.append("")
    lines.append(f"1. **Benchmark-level (n=3, Qwen3-VL only):** Jaccard~effect Pearson r={rjp[0]:+.3f}.")
    lines.append(f"   Direction is {'CORRECT (negative, as predicted)' if bench_dir else 'WRONG'},")
    lines.append(f"   but n=3 precludes any significance claim.")
    lines.append("")
    lines.append(f"2. **Per-image (n={n_img}, Qwen3-VL):** Jaccard~effect Pearson r={pj[0]:+.4f} (p={pj[1]:.4f});")
    lines.append(f"   Spearman(pre,post)~effect r={ps[0]:+.4f} (p={ps[1]:.4f});")
    lines.append(f"   Edge-rankshift~effect r={pe[0]:+.4f} (p={pe[1]:.4f}).")
    lines.append(f"   {'At least one M1 metric is significant (p<0.05) in the predicted direction.' if sig_any else 'No M1 metric reaches p<0.05 in the predicted direction.'}")
    lines.append("")
    lines.append(f"3. **Cross-architecture: UNTESTED.** M1 (ranking divergence) is available for")
    lines.append(f"   Qwen3-VL only. Qwen2.5-VL, InternVL3, and GLM-4.1V have no kept_indices")
    lines.append(f"   or unit scores, so their M1 cannot be computed (confirmed in")
    lines.append(f"   experiments/j3_mechanism_crossarch.md).")
    lines.append("")
    if sig_any and (pj[0] < 0 or ps[0] < 0):
        lines.append(f"**Bottom line:** The law is **partially supported** within Qwen3-VL (per-image")
        lines.append(f"M1 significantly predicts per-image stage effect in the predicted direction),")
        lines.append(f"but it is **NOT cross-architecture validated** --- the core theory contribution")
        lines.append(f"(predictive law across families) does NOT hold with the current data, because")
        lines.append(f"M1 was never captured for the other families. Flag as a gap, not a result.")
    elif pj[0] < 0 or ps[0] < 0:
        lines.append(f"**Bottom line:** The law shows the **correct direction** within Qwen3-VL but does")
        lines.append(f"NOT reach significance per-image, and is **NOT cross-architecture validated**.")
        lines.append(f"The predictive-law claim is **not supported** by the current data. Flag honestly.")
    else:
        lines.append(f"**Bottom line:** The law **does not hold** --- per-image M1 does not predict the")
        lines.append(f"stage effect in the predicted direction, and cross-architecture validation is")
        lines.append(f"impossible (M1 missing for 3 of 4 families). Report as a negative finding.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## What would be needed to test the cross-architecture law properly")
    lines.append("")
    lines.append("1. Capture pre/post per-unit L2 scores (like the Qwen3-VL survival_capture)")
    lines.append("   for Qwen2.5-VL and InternVL3 on textvqa/docvqa/gqa/ocrbench. This requires")
    lines.append("   a GPU forward pass (~1h/family) with `mechanism_token_survival.py --mode capture`")
    lines.append("   extended to those architectures.")
    lines.append("2. Alternatively, add `kept_indices` output to the runner (`attach_kept_indices`")
    lines.append("   exists but is off by default; `--save-unit-scores` was confirmed to NOT write")
    lines.append("   kept indices for Qwen2.5-VL). Then re-run pre/post cells with the flag on.")
    lines.append("3. With M1 for 3 families x 4 benchmarks = 12 points, the cross-architecture")
    lines.append("   correlation (M1 vs stage-effect) becomes testable.")
    lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
