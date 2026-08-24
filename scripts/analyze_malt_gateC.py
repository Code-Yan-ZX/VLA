#!/usr/bin/env python3
"""MALT goal-mode Gate C: locked n=200 paired confirmation of the best
candidate vs MALT-1 (task 2026-08-25).

Usage: python analyze_malt_gateC.py <cand_tag> [--ref h1]
Loads runs/malt_goal_mode/{tag}_{bench}_n200.json for tag in {cand, ref} and
reports per the task spec:
  * paired comparison: accuracy per bench + macro
  * paired bootstrap 95% CI of the macro difference (macro = mean per-bench acc)
  * sign-flip / McNemar z per bench (and combined)
  * compute: mean (sum N_l), (sum N_l^2) from layer_visual_counts
  * runtime: prefill_s, ttft_s, throughput proxy, max_memory_allocated
  * keep-set equality: rb.kept_per_image identical to the ref on every sample
"""
import json
import os
import sys
import math

HERE = "/media/disk2/YZX/research/vla"
OUT = f"{HERE}/runs/malt_goal_mode"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]


def load(tag, bench):
    with open(f"{OUT}/{tag}_{bench}_n200.json") as f:
        return json.load(f)


def ps(tag, bench):
    d = load(tag, bench)
    return {str(r["id"]): r for r in d["per_sample"] if not r.get("skipped")}


def acc_map(tag, bench):
    a = ps(tag, bench)
    return {i: int(r["correct"]) for i, r in a.items()}


def macro_acc(tag):
    return sum(sum(acc_map(tag, b).values()) / max(1, len(acc_map(tag, b)))
               for b in BENCHES) / len(BENCHES)


def paired_bootstrap_ci(tag_c, tag_r, n_boot=5000, seed=0):
    """Bootstrap 95% CI of (macro_cand - macro_ref) on the same-id pairs."""
    import random
    rng = random.Random(seed)
    diffs = []
    per = {b: [] for b in BENCHES}
    for b in BENCHES:
        c, r = acc_map(tag_c, b), acc_map(tag_r, b)
        ids = [i for i in c if i in r]
        per[b] = [(c[i], r[i]) for i in ids]
    for _ in range(n_boot):
        d = 0.0
        for b in BENCHES:
            data = per[b]
            if not data:
                continue
            cc = rr = 0
            for _ in range(len(data)):
                c, r = rng.choice(data)
                cc += c
                rr += r
            d += (cc / len(data) - rr / len(data))
        diffs.append(d / len(BENCHES))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, hi, diffs[len(diffs) // 2]


def mcnemar(tag_c, tag_r, bench):
    c, r = acc_map(tag_c, bench), acc_map(tag_r, bench)
    n_ba = n_ab = 0
    for i in c:
        if i in r:
            if c[i] and not r[i]:
                n_ab += 1
            elif not c[i] and r[i]:
                n_ba += 1
    z = (n_ba - n_ab) / math.sqrt(n_ba + n_ab) if (n_ba + n_ab) else 0.0
    return z, n_ab, n_ba


def eff_stats(tag, bench):
    d = load(tag, bench)
    recs = [r for r in d["per_sample"] if not r.get("skipped")]
    sums = [(sum((r.get("n_text", 0) + v)
                 for v in (r.get("layer_visual_counts") or [])),
             sum((r.get("n_text", 0) + v) ** 2
                 for v in (r.get("layer_visual_counts") or [])))
            for r in recs]
    return (sum(s[0] for s in sums) / len(sums),
            sum(s[1] for s in sums) / len(sums),
            sum(r.get("prefill_s") or 0 for r in recs) / len(recs),
            sum(r.get("ttft_s") or 0 for r in recs) / len(recs),
            max((r.get("peak_mem_mb") or 0) for r in recs))


def _kept(rec):
    """kept-per-image from either the rb (deferred) or pre (immediate) diag."""
    for key in ("rb", "pre"):
        if rec.get(key, {}).get("kept_per_image"):
            return frozenset(tuple(x) for x in rec[key]["kept_per_image"])
    return None


def keepset_eq(tag_c, tag_r, bench):
    c, r = ps(tag_c, bench), ps(tag_r, bench)
    n_ok = n = 0
    for i in c:
        if i not in r:
            continue
        n += 1
        kc, kr = _kept(c[i]), _kept(r[i])
        if kc is None or kr is None:
            continue          # not comparable (no diag); not a mismatch
        n_ok += int(kc == kr)
    return n_ok, n


def main():
    cand = sys.argv[1] if len(sys.argv) > 1 else "h1"
    ref = "h1"
    print(f"=== GATE C: candidate={cand} vs ref={ref} (locked n=200) ===")
    print(f"{'bench':9s}{'cand':>8s}{'ref':>8s}{'diff':>8s}"
          f"{'z(McN)':>8s}{'W':>4s}{'L':>4s}{'keep=ref':>10s}")
    for b in BENCHES:
        c, r = acc_map(cand, b), acc_map(ref, b)
        ids = [i for i in c if i in r]
        ca = sum(c[i] for i in ids) / len(ids) if ids else float("nan")
        ra = sum(r[i] for i in ids) / len(ids) if ids else float("nan")
        z, w, l = mcnemar(cand, ref, b)
        k_eq, k_n = keepset_eq(cand, ref, b)
        print(f"{b:9s}{ca:>8.3f}{ra:>8.3f}{ca-ra:>+8.3f}{z:>+8.2f}"
              f"{w:>4d}{l:>4d}  keep={k_eq}/{k_n}")
    lo, hi, med = paired_bootstrap_ci(cand, ref)
    print(f"\nmacro cand={macro_acc(cand):.3f} ref={macro_acc(ref):.3f} "
          f"diff={macro_acc(cand)-macro_acc(ref):+.3f}")
    print(f"paired bootstrap 95% CI of diff: [{lo:+.3f}, {hi:+.3f}] "
          f"(median {med:+.3f})")
    print("\n=== efficiency (means over samples; n=200 locked) ===")
    print(f"{'bench':9s}{'sumN_c':>12s}{'sumN_r':>12s}{'sumN2_c':>14s}"
          f"{'sumN2_r':>14s}{'prefill_c':>10s}{'prefill_r':>10s}"
          f"{'mem_c':>8s}")
    for b in BENCHES:
        sc, s2c, pf_c, tt_c, mc = eff_stats(cand, b)
        sr, s2r, pf_r, tt_r, mr = eff_stats(ref, b)
        print(f"{b:9s}{sc:>12,.0f}{sr:>12,.0f}{s2c:>14,.0f}{s2r:>14,.0f}"
              f"{pf_c:>10.3f}{pf_r:>10.3f}{mc:>8.0f}")
    # aggregate sumN/sumN2 ratios
    tot_c = [eff_stats(cand, b) for b in BENCHES]
    tot_r = [eff_stats(ref, b) for b in BENCHES]
    print(f"\nΣN_l ratio cand/ref = "
          f"{sum(x[0] for x in tot_c)/sum(x[0] for x in tot_r)*100:.1f}%")
    print(f"ΣN_l² ratio cand/ref = "
          f"{sum(x[1] for x in tot_c)/sum(x[1] for x in tot_r)*100:.1f}%")


if __name__ == "__main__":
    main()
