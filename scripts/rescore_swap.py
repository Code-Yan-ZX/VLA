#!/usr/bin/env python3
"""Offline rescore of the M3 ranking-swap control cells with the OFFICIAL
metrics (CPU only, no GPU, no network).

Reads:
  runs/v3_merger_aware/swap/swap_{textvqa,docvqa}_r0.750_l2_n200.json  (M3)
  runs/v3_merger_aware/rescore_rerun/{none,post,pre}_{bench}_r*_l2_n200.json

For each bench computes, over per_sample[]:
    TextVQA -> official VQA accuracy (src.v3_premerger.official_scorers)
    DocVQA  -> official ANLS
mean +/- binomial stderr, and the decisive M3 comparison:

    swap (post forward path + PRE ranking)  vs  pre-standard  vs  post-standard

PREDICTION (unit equivalence, drafts/v3_merger_aware_design.md §2): swap must
reproduce pre-standard accuracy almost exactly -- same selected units => same
merged tokens => same greedy-decoded output -- proving the pre>post gap is
100% a RANKING effect with the forward path held constant.

Also reports the per-sample answer-string agreement swap vs pre (ids joined)
and the paired score difference (swap - pre) with its paired SE, to quantify
"how close" (run noise from GPU-kernel non-determinism only).

Writes: runs/v3_merger_aware/swap/rescore_swap_summary.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

REPO = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.official_scorers import (  # noqa: E402
    score_textvqa_vqaacc,
    score_docvqa_anls,
)

SCORERS = {
    "textvqa": ("VQA-acc", score_textvqa_vqaacc),
    "docvqa": ("anls", score_docvqa_anls),
}

CELLS = {
    "textvqa": {
        "baseline": "runs/v3_merger_aware/rescore_rerun/none_textvqa_r0.000_l2_n200.json",
        "post":     "runs/v3_merger_aware/rescore_rerun/post_textvqa_r0.750_l2_n200.json",
        "pre":      "runs/v3_merger_aware/rescore_rerun/pre_textvqa_r0.750_l2_n200.json",
        "swap":     "runs/v3_merger_aware/swap/swap_textvqa_r0.750_l2_n200.json",
    },
    "docvqa": {
        "baseline": "runs/v3_merger_aware/rescore_rerun/none_docvqa_r0.000_l2_n200.json",
        "post":     "runs/v3_merger_aware/rescore_rerun/post_docvqa_r0.750_l2_n200.json",
        "pre":      "runs/v3_merger_aware/rescore_rerun/pre_docvqa_r0.750_l2_n200.json",
        "swap":     "runs/v3_merger_aware/swap/swap_docvqa_r0.750_l2_n200.json",
    },
}


def binom_stderr(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(0.0, p * (1 - p)) / n)


def score_cell(path: str, bench: str) -> dict:
    cell = json.load(open(os.path.join(REPO, path)))
    metric, fn = SCORERS[bench]
    per = cell["per_sample"]
    scores, by_id = [], {}
    n_skipped = 0
    for s in per:
        if s.get("skipped"):
            n_skipped += 1
            continue
        v = fn(s["answer"], s["gt"])
        scores.append(v)
        by_id[str(s["id"])] = dict(score=v, answer=s["answer"])
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    return dict(file=path, metric=metric, n=n, n_skipped=n_skipped,
                mean=round(mean, 4),
                stderr=round(binom_stderr(mean, n), 4),
                stored_acc=cell.get("acc"),
                mask_ranking=cell.get("mask_ranking", "stage"),
                _by_id=by_id)


def main():
    out = {"generated": __import__("datetime").date.today().isoformat(),
           "metric_note": {"textvqa": "official VQA accuracy",
                           "docvqa": "official ANLS"},
           "benches": {}}
    for bench, paths in CELLS.items():
        res = {mode: score_cell(p, bench) for mode, p in paths.items()}
        swap, pre, post = res["swap"], res["pre"], res["post"]

        # per-sample agreement swap vs pre (the causal-control diagnostic)
        common = sorted(set(swap["_by_id"]) & set(pre["_by_id"]))
        ans_same = sum(1 for i in common
                       if swap["_by_id"][i]["answer"] == pre["_by_id"][i]["answer"])
        diffs = [swap["_by_id"][i]["score"] - pre["_by_id"][i]["score"]
                 for i in common]
        d_mean = sum(diffs) / len(diffs) if diffs else 0.0
        d_se = (statistics.stdev(diffs) / math.sqrt(len(diffs))
                if len(diffs) > 1 else 0.0)
        # swap vs post as well (swap should be FAR from post on text-dense)
        common_p = sorted(set(swap["_by_id"]) & set(post["_by_id"]))
        diffs_p = [swap["_by_id"][i]["score"] - post["_by_id"][i]["score"]
                   for i in common_p]
        dp_mean = sum(diffs_p) / len(diffs_p) if diffs_p else 0.0

        diag = {"swap_vs_pre": {
                    "n_common": len(common),
                    "identical_answers": ans_same,
                    "identical_answers_frac": round(ans_same / len(common), 4)
                    if common else 0.0,
                    "score_diff_swap_minus_pre_mean": round(d_mean, 4),
                    "score_diff_paired_se": round(d_se, 4)},
                "swap_vs_post": {
                    "score_diff_swap_minus_post_mean": round(dp_mean, 4)}}
        for r in res.values():
            r.pop("_by_id", None)
        out["benches"][bench] = {"cells": res, "comparison": diag}

        print(f"\n=== {bench.upper()} (metric={swap['metric']}, n={swap['n']}) ===")
        for mode in ["baseline", "post", "pre", "swap"]:
            r = res[mode]
            print(f"  {mode:8s}: {r['mean']:.4f} +/- {r['stderr']:.4f} "
                  f"(stored {r['stored_acc']}, skipped {r['n_skipped']})")
        print(f"  pre-post gap   : {res['pre']['mean'] - res['post']['mean']:+.4f}")
        print(f"  swap-pre delta : {d_mean:+.4f} (paired SE {d_se:.4f}); "
              f"identical answers {ans_same}/{len(common)} "
              f"({diag['swap_vs_pre']['identical_answers_frac']:.1%})")
        print(f"  swap-post delta: {dp_mean:+.4f}  (swap should sit with PRE)")

    outpath = os.path.join(REPO, "runs/v3_merger_aware/swap",
                           "rescore_swap_summary.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
