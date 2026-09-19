#!/usr/bin/env python
"""DCC pre-submission Experiment A analysis: Qwen3 main-only Post-L2 control.

Arm "post-main" (new, results/dcc_presubmit/expA/) ranks post-merger units by
the main-merger column block ONLY (vLLM qwen3_vl.py:654 cats main first), but
keeps FULL main+deepstack rows. Compared against the EXISTING pre-final arm
(results/acmmm_final_controls/p0_1/ full; runs/qwen3_prefinal_control/ n=200)
-- pre-final ranks at the main-merger INPUT on the same pre-merge feature
space. Both arms keep k_i = round(full_i * 0.25) units/image (r=0.75), so the
iso-token contract requires per-sample prompt_token_ids to match exactly.

Stages:
  validate  n=200 technical check: branch executed (diag), ID sets match,
            per-sample prompt_token_ids match, no unexpected skips.
  full      official scores (VQA-acc / ANLS via paired_stats.score_sample) +
            paired bootstrap 95% CI + permutation p + McNemar on the per-sample
            difference (post-main - pre-final). Reported regardless of sign.

Usage: python scripts/analyze_dcc_expA.py --stage {validate,full} --out-dir
       results/dcc_presubmit/expA [--extra]
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.paired_stats import (score_sample, run_pair,  # noqa: E402
                                       METRIC_NAME)

PREFINAL_N200 = os.path.join(REPO, "runs", "qwen3_prefinal_control")
PREFINAL_FULL = os.path.join(REPO, "results", "acmmm_final_controls", "p0_1")
BENCHES = ["textvqa", "docvqa"]
BENCH_EXTRA = ["ocrbench", "gqa"]
N_FULL = {"textvqa": 5000, "docvqa": 5349, "ocrbench": 1000, "gqa": 12578}


def exp_a_path(out_dir, fn):
    d = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    return os.path.join(d, fn)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_scores(d, bench):
    """{id: official score} over non-skipped samples + (n_total, n_skip)."""
    out, n_skip = {}, 0
    for s in d.get("per_sample") or []:
        if s.get("skipped"):
            n_skip += 1
            continue
        out[s["id"]] = score_sample(bench, s.get("answer", ""),
                                    str(s.get("gt", "")), s["id"], False)
    return out, (len(d.get("per_sample") or []), n_skip)


def check_arm_diag(d, tag):
    diag = d.get("diag") or {}
    ok = (diag.get("score_scope") == "main"
          and diag.get("n_deepstack") == 3
          and int(diag.get("main_hidden") or 0) == 4096
          and int(diag.get("fires") or 0) > 0)
    print(f"  [{tag}] mode={d.get('mode')} scope={diag.get('score_scope')} "
          f"n_ds={diag.get('n_deepstack')} main_hidden={diag.get('main_hidden')} "
          f"fires={diag.get('fires')} nk={diag.get('nk', [])[:2]} -> "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def iso_token_check(new_d, old_d, tag_new, tag_old):
    """Per-sample prompt_token_ids + ID-set equality between two arms."""
    new_ps = {s["id"]: s for s in new_d.get("per_sample") or []}
    old_ps = {s["id"]: s for s in old_d.get("per_sample") or []}
    ids_only_new = sorted(set(new_ps) - set(old_ps))
    ids_only_old = sorted(set(old_ps) - set(new_ps))
    common = sorted(set(new_ps) & set(old_ps))
    mism = [i for i in common
            if new_ps[i].get("prompt_token_ids")
            != old_ps[i].get("prompt_token_ids")]
    skip_new = sum(1 for s in new_ps.values() if s.get("skipped"))
    skip_old = sum(1 for s in old_ps.values() if s.get("skipped"))
    print(f"  [{tag_new} vs {tag_old}] n_ids new={len(new_ps)} old={len(old_ps)} "
          f"common={len(common)} only_new={len(ids_only_new)} "
          f"only_old={len(ids_only_old)} ptid_mismatch={len(mism)} "
          f"skipped new={skip_new} old={skip_old}")
    if mism[:5]:
        print(f"    first mismatches: {mism[:5]}")
    return (not ids_only_new and not ids_only_old and not mism)


def stage_validate(out_dir):
    ok = True
    for bench in BENCHES:
        print(f"== {bench} n=200 ==")
        new = load_json(exp_a_path(out_dir, f"n200_{bench}_main.json"))
        old = load_json(os.path.join(
            PREFINAL_N200, f"qwen3_pre-final_{bench}_r0.750_n200.json"))
        if new is None or old is None:
            print("  MISSING run file; abort")
            return 1
        ok &= check_arm_diag(new, f"n200 {bench} post-main")
        ok &= iso_token_check(new, old, "post-main", "pre-final")
    print(f"VALIDATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def stage_full(out_dir, benches, extra=False):
    results = {}
    for bench in benches:
        print(f"== {bench} full (n={N_FULL[bench]}) ==")
        new = load_json(exp_a_path(out_dir, f"full_{bench}_main.json"))
        old = load_json(os.path.join(
            PREFINAL_FULL,
            f"p0_1_qwen3_pre-final_{bench}_r0.750_full.json"))
        if new is None:
            print("  post-main full run MISSING; skipping")
            continue
        if old is None:
            print("  pre-final anchor MISSING; skipping")
            continue
        check_arm_diag(new, f"full {bench} post-main")
        sa, (na, ska) = load_scores(new, bench)
        sb, (nb, skb) = load_scores(old, bench)
        print(f"  complete/total: post-main {na - ska}/{na} (skip {ska}), "
              f"pre-final {nb - skb}/{nb} (skip {skb})")
        common = [i for i in sa if i in sb]
        print(f"  paired n={len(common)}")
        # A = post-main (new), B = pre-final (anchor). Report regardless of sign.
        pair = run_pair(sa, sb, bench,
                        "post-main(L2@main-block)", "pre-final(stage, L2@merger-input)",
                        n_resamples=20000, seed=20260919,
                        n_nominal=N_FULL[bench])
        if pair["mean_A"] is None:
            print("  paired n=0 -> no statistics (arm-level skip mismatch); "
                  "see skip counts above")
            results[bench] = pair
            continue
        print(f"  {METRIC_NAME[bench]} post-main(mean_A)={pair['mean_A']:.4f} "
              f"pre-final(mean_B)={pair['mean_B']:.4f}")
        print(f"  delta(A-B) = {pair['mean_delta_pp']:+.2f} pp, "
              f"95% CI {pair['ci95_pp']}, perm p={pair['perm_p_two_sided']}")
        if "mcnemar" in pair:
            print(f"  McNemar: A-only={pair['mcnemar']['A_only_correct']} "
                  f"B-only={pair['mcnemar']['B_only_correct']} "
                  f"p={pair['mcnemar']['p_exact_two_sided']}")
        results[bench] = pair
    out = exp_a_path(out_dir, "analysis_full.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["validate", "full"])
    ap.add_argument("--out-dir", default="results/dcc_presubmit/expA")
    ap.add_argument("--extra", action="store_true")
    a = ap.parse_args()
    if a.stage == "validate":
        sys.exit(stage_validate(a.out_dir))
    benches = BENCHES + (BENCH_EXTRA if a.extra else [])
    sys.exit(stage_full(a.out_dir, benches, a.extra))


if __name__ == "__main__":
    main()
