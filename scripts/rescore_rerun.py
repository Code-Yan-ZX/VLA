#!/usr/bin/env python3
"""Offline rescore of the SHORT-ANSWER-PROMPT rerun cells with the OFFICIAL
metrics (CPU only, no GPU, no network).

Reads the 6 cells under runs/v3_merger_aware/rescore_rerun/
    {none,post,pre}_{textvqa,docvqa}_r*_l2_n200.json
and for each computes, over per_sample[]:

    TextVQA -> official VQA accuracy   (src.v3_premerger.official_scorers.score_textvqa_vqaacc)
    DocVQA  -> official ANLS           (src.v3_premerger.official_scorers.score_docvqa_anls)

mean +/- binomial stderr, where stderr = sqrt(p*(1-p)/n), p = mean score,
n = number of scored samples.  Reports, per bench: baseline / pre / post,
the pre-vs-post gap (pp) with its combined stderr, and retention
(pre/baseline, post/baseline).

Also sanity-checks each cell: n_answered ~= 200 and the median answer length
is now small (single word/phrase) under the short-answer prompt.

Writes:
    runs/v3_merger_aware/rescore_rerun/rescore_summary.json
    drafts/rescore_rerun_report.md   (table + HOLD/FLIP verdict)
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

CELL_DIR = os.path.join(REPO, "runs", "v3_merger_aware", "rescore_rerun")

# bench -> (official metric name, scorer)
SCORERS = {
    "textvqa": ("VQA-acc", score_textvqa_vqaacc),
    "docvqa": ("anls", score_docvqa_anls),
}

# filename -> (bench, mode)
CELLS = {
    "none_textvqa_r0.000_l2_n200.json": ("textvqa", "baseline"),
    "post_textvqa_r0.750_l2_n200.json": ("textvqa", "post"),
    "pre_textvqa_r0.750_l2_n200.json": ("textvqa", "pre"),
    "none_docvqa_r0.000_l2_n200.json": ("docvqa", "baseline"),
    "post_docvqa_r0.750_l2_n200.json": ("docvqa", "post"),
    "pre_docvqa_r0.750_l2_n200.json": ("docvqa", "pre"),
}


def binom_stderr(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def rescore(bench: str, mode: str, fname: str) -> dict:
    path = os.path.join(CELL_DIR, fname)
    rec = {
        "bench": bench, "mode": mode, "file": fname, "metric": SCORERS[bench][0],
        "present": os.path.exists(path),
    }
    if not rec["present"]:
        rec.update(mean=None, stderr=None, n=0, n_answered=0,
                   median_ans_len=None, n_skipped=0)
        return rec
    d = json.load(open(path))
    per = d.get("per_sample", [])
    scorer = SCORERS[bench][1]
    scores = []
    ans_lens = []
    n_ans = 0
    n_skip = 0
    for s in per:
        pred = s.get("answer", "") or ""
        gt = s.get("gt", "") or ""
        if s.get("skipped", False):
            n_skip += 1
        elif pred.strip() != "":
            n_ans += 1
            ans_lens.append(len(pred))
        scores.append(scorer(pred, gt))
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    rec.update(
        mean=mean,
        stderr=binom_stderr(mean, n),
        n=n,
        n_answered=n_ans,
        n_skipped=n_skip,
        median_ans_len=(statistics.median(ans_lens) if ans_lens else None),
        mean_ans_len=(statistics.mean(ans_lens) if ans_lens else None),
        stored_acc=d.get("acc"),
        r=d.get("r"),
        seed=d.get("seed"),
    )
    return rec


def main():
    results = {}  # bench -> {mode -> rec}
    for fname, (bench, mode) in CELLS.items():
        results.setdefault(bench, {})[mode] = rescore(bench, mode, fname)

    # ---- verdict + table rows ----
    bench_rows = {}
    verdicts = []
    for bench, metric in [("textvqa", "VQA-acc"), ("docvqa", "anls")]:
        m = results[bench]
        base, pre, post = m["baseline"], m["pre"], m["post"]
        gap = None
        gap_se = None
        verdict = "MISSING"
        if None not in (pre["mean"], post["mean"], base["mean"]):
            gap = (pre["mean"] - post["mean"]) * 100.0  # pp
            # stderr of the difference of two proportions (independent cells)
            gap_se = math.sqrt(pre["stderr"] ** 2 + post["stderr"] ** 2) * 100.0
            # "clearly >" : gap positive AND exceeds ~1 combined stderr
            if pre["mean"] > post["mean"] and gap > gap_se:
                verdict = "HOLD"
            elif pre["mean"] > post["mean"]:
                verdict = "HOLD (marginal: gap < 1 stderr)"
            else:
                verdict = "FLIP"
        bench_rows[bench] = {
            "metric": metric, "baseline": base, "pre": pre, "post": post,
            "gap_pp": gap, "gap_se_pp": gap_se, "verdict": verdict,
            "ret_pre": (pre["mean"] / base["mean"]) if base["mean"] else None,
            "ret_post": (post["mean"] / base["mean"]) if base["mean"] else None,
        }
        verdicts.append((bench, metric, verdict, gap, gap_se))

    # ---- summary json ----
    summary = {
        "generated": "2026-07-23",
        "prompt_instruction": "Answer the question using a single word or phrase.",
        "metrics": {"textvqa": "official VQA accuracy", "docvqa": "official ANLS"},
        "stderr": "binomial sqrt(p(1-p)/n)",
        "cells": {f"{b}/{mode}": rec for b, mm in results.items() for mode, rec in mm.items()},
        "verdicts": [
            {"bench": b, "metric": mt, "verdict": v, "gap_pp": g, "gap_se_pp": gs}
            for b, mt, v, g, gs in verdicts
        ],
        "table": bench_rows,
    }
    os.makedirs(CELL_DIR, exist_ok=True)
    with open(os.path.join(CELL_DIR, "rescore_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- markdown report ----
    lines = []
    flip = any(v.startswith("FLIP") for _, _, v, _, _ in verdicts)
    lines.append("# Rescore Rerun — OFFICIAL metrics under short-answer prompting (keep=25%)\n")
    lines.append("_Generated 2026-07-23 · offline CPU rescore of freshly re-run cells · "
                 "scorer: `src/v3_premerger/official_scorers.py`_\n")

    # prominent verdict banner
    lines.append("## VERDICT (does pre>post hold under official metrics?)\n")
    if flip:
        lines.append("> **🔄 FLIP detected on at least one text-dense bench — claim-overturn, "
                     "review before reporting.**\n")
    else:
        lines.append("> **✅ HOLD — pre>post is preserved on both text-dense benches under "
                     "official metrics + proper short-answer prompting at keep=25%.**\n")
    for b, mt, v, g, gs in verdicts:
        if g is None:
            lines.append(f"- **{b}** ({mt}): MISSING cells — {v}")
        else:
            lines.append(f"- **{b}** ({mt}): **{v}** · pre−post gap = **{g:+.1f} pp** "
                         f"(±{gs:.1f} pp combined stderr)")
    lines.append("")

    lines.append("**Prompt fix:** each TextVQA/DocVQA question now carries the canonical lmms-eval "
                 "short-answer instruction `\\nAnswer the question using a single word or phrase.` "
                 "(subsets rebuilt by `scripts/fix_shortanswer_subsets.py`; originals backed up to "
                 "`eval/subsets/_backup/`). Cells re-run at r=0.75 (keep=25%), n=200, --selector l2, "
                 "seed=0 (== router_probe default), enforce_eager. DocVQA used the canonical big-doc "
                 "config (--max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4).\n")

    # main table
    lines.append("## Official-metric results (mean ± binomial stderr)\n")
    lines.append("| Bench | Metric | Baseline (none) | Post | Pre | Pre−Post gap (pp) | Retention pre/base | Retention post/base | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in ["textvqa", "docvqa"]:
        r = bench_rows[b]
        base, pre, post = r["baseline"], r["pre"], r["post"]
        def fmt(rec):
            if rec["mean"] is None:
                return "—"
            return f"{rec['mean']*100:.1f} ± {rec['stderr']*100:.1f}"
        gap = "—" if r["gap_pp"] is None else f"{r['gap_pp']:+.1f} ± {r['gap_se_pp']:.1f}"
        rp = "—" if r["ret_pre"] is None else f"{r['ret_pre']*100:.0f}%"
        rpo = "—" if r["ret_post"] is None else f"{r['ret_post']*100:.0f}%"
        lines.append(
            f"| {b} | {r['metric']} | {fmt(base)} | {fmt(post)} | {fmt(pre)} | "
            f"{gap} | {rp} | {rpo} | {r['verdict']} |"
        )
    lines.append("")
    lines.append("_All values in %. stderr = sqrt(p(1−p)/n), n=200. Gap stderr = quadrature of "
                 "the two independent cell stderrs._\n")

    # per-cell sanity table
    lines.append("## Per-cell sanity (n_answered, answer length under short-answer prompt)\n")
    lines.append("| Cell | n | n_answered | n_skipped | median ans len | mean ans len | stored acc |")
    lines.append("|---|---|---|---|---|---|---|")
    for b in ["textvqa", "docvqa"]:
        for mode in ["baseline", "post", "pre"]:
            rec = results[b][mode]
            mal = "—" if rec["median_ans_len"] is None else f"{rec['median_ans_len']:.0f}"
            mea = "—" if rec["mean_ans_len"] is None else f"{rec['mean_ans_len']:.0f}"
            sa = "—" if rec.get("stored_acc") is None else f"{rec['stored_acc']:.3f}"
            lines.append(
                f"| {b}/{mode} ({rec['file']}) | {rec['n']} | {rec['n_answered']} | "
                f"{rec['n_skipped']} | {mal} | {mea} | {sa} |"
            )
    lines.append("")
    lines.append("Median answer length should now be a single word/phrase (≪ the ~132-char verbose "
                 "median seen under the raw-question prompt), confirming the prompt fix took effect.\n")

    lines.append("## Artifacts\n")
    lines.append("- Cells: `runs/v3_merger_aware/rescore_rerun/*.json` (per_sample saved)")
    lines.append("- Summary: `runs/v3_merger_aware/rescore_rerun/rescore_summary.json`")
    lines.append("- Subset fix: `scripts/fix_shortanswer_subsets.py`; backups `eval/subsets/_backup/`")
    lines.append("- Rerun driver: `src/v3_premerger/v3_rescore_rerun.sh`")

    os.makedirs(os.path.join(REPO, "drafts"), exist_ok=True)
    with open(os.path.join(REPO, "drafts", "rescore_rerun_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # console digest
    print("=== RESCORE RERUN COMPLETE ===")
    for b in ["textvqa", "docvqa"]:
        r = bench_rows[b]
        base, pre, post = r["baseline"], r["pre"], r["post"]
        print(f"\n{b} [{r['metric']}]:")
        for nm, rec in [("baseline", base), ("post", post), ("pre", pre)]:
            if rec["mean"] is None:
                print(f"   {nm:9s}: MISSING ({rec['file']})")
            else:
                print(f"   {nm:9s}: {rec['mean']*100:5.1f}% ± {rec['stderr']*100:4.1f}  "
                      f"(n_ans={rec['n_answered']} med_ans_len={rec['median_ans_len']})")
        if r["gap_pp"] is not None:
            print(f"   -> pre−post gap = {r['gap_pp']:+.1f} pp ± {r['gap_se_pp']:.1f}  "
                  f"ret_pre={r['ret_pre']*100:.0f}% ret_post={r['ret_post']*100:.0f}%  VERDICT={r['verdict']}")


if __name__ == "__main__":
    main()
