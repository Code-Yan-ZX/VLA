#!/usr/bin/env python
"""Deferred-RBM n=200 life-or-death gate analysis (task 2026-08-24).

Official rescore + paired statistics + GO/NO-GO per the pre-registered task
criteria (six).  Arms: Full (locked_none), RBM (gate_pre25), FastV-K3
(locked_fst3), Deferred (locked_deferred, rankbridge rho=1.0 @ K=3 keep 25%).
Reuses the official_scorers contract from runs/deferred_rbm/gate_analyze.py.

Outputs:
  results/deferred_rbm_n200/analysis.json   -- machine-readable
  results/deferred_rbm_n200/per_sample.json -- per-sample combined (anchors,
                                               pre/post visual counts, 4 arms)
  stdout                                    -- human-readable summary
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
OUTDIR = f"{HERE}/results/deferred_rbm_n200"
os.makedirs(OUTDIR, exist_ok=True)

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
ARMS = ["full", "rbm", "fastv", "deferred"]

rng = np.random.default_rng(0)
NBOOT = 5000


def load(p):
    return json.load(open(p))


def official_per_sample(bench, cell):
    """id -> (metric_value, binary_correct)."""
    out = {}
    ps = cell.get("per_sample", [])
    for s in ps:
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
    bins = [per[i][1] for i in ids]
    n = len(vals)
    m = sum(vals) / n if n else 0.0
    if n > 1:
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return m, se, n, sum(bins)


def mcnemar(per_a, per_b, ids):
    b10 = b01 = 0
    for i in ids:
        da, db = per_a[i][1], per_b[i][1]
        b10 += int(da == 1 and db == 0)
        b01 += int(da == 0 and db == 1)
    z = ((b10 - b01) / math.sqrt(b10 + b01)) if (b10 + b01) else 0.0
    return z, b10, b01


def boot_ci(diffs):
    """diffs: dict id -> float (per-sample deferred - stronger-parent diff)."""
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


def per_sample_cells():
    cells = {}
    for b in BENCHES:
        cells[(b, "deferred")] = load(f"{RB}/locked_deferred_{b}_n200.json")
        cells[(b, "rbm")] = load(f"{CASCADE}/gate_pre25_{b}.json")
        cells[(b, "fastv")] = load(f"{RANKB}/locked_fst3_{b}_n200.json")
        cells[(b, "full")] = load(f"{RANKB}/locked_none_{b}_n200.json")
    return cells


def main():
    cells = per_sample_cells()
    per = {(b, a): official_per_sample(b, cells[(b, a)])
           for b in BENCHES for a in ARMS}

    # ---------------- per-sample combined output ----------------
    combined = {}
    full_recs = {}
    for b in BENCHES:
        full_recs[b] = {str(r["id"]): r
                        for r in cells[(b, "full")]["per_sample"]}
    for b in BENCHES:
        recs = {str(r["id"]): r for r in cells[(b, "deferred")]["per_sample"]}
        rbm_recs = {str(r["id"]): r for r in cells[(b, "rbm")]["per_sample"]}
        fst_recs = {str(r["id"]): r for r in cells[(b, "fastv")]["per_sample"]}
        for i in per[(b, "deferred")]:
            dr = recs[i]
            combined[f"{b}/{i}"] = {
                "bench": b, "sample_id": i,
                "gt": dr.get("gt"),
                "question": dr.get("question"),
                # answers
                "answer_full": full_recs[b].get(i, {}).get("answer"),
                "answer_rbm": rbm_recs[i]["answer"] if i in rbm_recs else None,
                "answer_fastv": fst_recs[i]["answer"] if i in fst_recs else None,
                "answer_deferred": dr["answer"],
                # correctness
                "correct_full": per[(b, "full")].get(i, (None, None))[1],
                "correct_rbm": per[(b, "rbm")].get(i, (None, None))[1],
                "correct_fastv": per[(b, "fastv")].get(i, (None, None))[1],
                "correct_deferred": per[(b, "deferred")][i][1],
                "metric_deferred": per[(b, "deferred")][i][0],
                # anchors + visual token counts
                "anchor_indices_deferred": dr.get("rb", {}).get("kept_per_image"),
                "anchor_indices_rbm": rbm_recs.get(i, {}).get(
                    "pre", {}).get("kept_per_image") if i in rbm_recs else None,
                "n_image_full": dr.get("n_image_full"),
                "n_image_kept": dr.get("n_image_kept"),
                "n_text": dr.get("n_text"),
                "L_after": dr.get("prompt_token_ids"),
                "fired": dr.get("rb", {}).get("fired"),
                "k_per_image": dr.get("rb", {}).get("k_per_image"),
            }
    with open(f"{OUTDIR}/per_sample.json", "w") as f:
        json.dump(combined, f, indent=2)

    # ---------------- common IDs + aggregates ----------------
    res = {}
    print("=== DEFERRED-RBM n=200 GATE (official rescore, K=3, keep 25%, "
          "rho=1.0) ===")
    print(f"{'bench':9s} " + " ".join(f"{a:>10s}" for a in ARMS))
    aggrow = {b: {} for b in BENCHES}
    common = {}
    for b in BENCHES:
        ids = sorted(set.intersection(*[
            {k for k in per[(b, a)]} for a in ARMS]))
        common[b] = ids
        row = []
        for a in ARMS:
            m, se, n, ncorr = agg(per[(b, a)], ids)
            aggrow[b][a] = (m, se, n)
            row.append(f"{m:.4f}")
        print(f"{b:9s} " + " ".join(f"{v:>10s}" for v in row)
              + f"   (n_common={len(ids)})")

    # ---------------- paired differences vs parents ----------------
    print("\n[paired] deferred minus parent (per-benchmark, common IDs)")
    table = {}
    for b in BENCHES:
        ids = common[b]
        d = per[(b, "deferred")]
        diff_pre = {i: d[i][0] - per[(b, "rbm")][i][0] for i in ids}
        diff_fst = {i: d[i][0] - per[(b, "fastv")][i][0] for i in ids}
        # stronger parent per sample (binary-correctness based for win/loss)
        stronger_per_sample = {}
        for i in ids:
            pv = max(per[(b, "rbm")][i][1], per[(b, "fastv")][i][1])
            stronger_per_sample[i] = max(
                per[(b, "rbm")][i][0], per[(b, "fastv")][i][0])
        diff_max = {i: d[i][0] - stronger_per_sample[i] for i in ids}

        m_pre = sum(diff_pre.values()) / len(ids)
        m_fst = sum(diff_fst.values()) / len(ids)
        m_max = sum(diff_max.values()) / len(ids)
        ci_pre = boot_ci(diff_pre)
        ci_fst = boot_ci(diff_fst)
        ci_max = boot_ci(diff_max)
        # stronger parent identity per benchmark (by aggregate mean)
        pkey = "rbm" if aggrow[b]["rbm"][0] >= aggrow[b]["fastv"][0] else "fastv"
        # benchmark-level diff: deferred_mean - stronger_parent_mean (criteria)
        m_max_level = (aggrow[b]["deferred"][0]
                       - max(aggrow[b]["rbm"][0], aggrow[b]["fastv"][0]))
        z, b10, b01 = mcnemar(per[(b, "deferred")], per[(b, pkey)], ids)
        # win/tie/loss vs each parent and vs stronger parent
        wtl = {}
        for tag, par in (("vs_rbm", "rbm"), ("vs_fastv", "fastv")):
            w = l = t = 0
            for i in ids:
                dd, pp = per[(b, "deferred")][i][1], per[(b, par)][i][1]
                w += int(dd == 1 and pp == 0)
                l += int(dd == 0 and pp == 1)
                t += int(dd == pp)
            wtl[tag] = (w, t, l)
        w = l = t = 0
        for i in ids:
            dd, pp = per[(b, "deferred")][i][1], per[(b, pkey)][i][1]
            w += int(dd == 1 and pp == 0)
            l += int(dd == 0 and pp == 1)
            t += int(dd == pp)
        wtl["vs_stronger"] = (w, t, l)

        table[b] = {
            "mean": {a: aggrow[b][a][0] for a in ARMS},
            "se": {a: aggrow[b][a][1] for a in ARMS},
            "n_common": len(ids),
            "stronger_parent": pkey,
            "diff_vs_rbm": m_pre, "diff_vs_fastv": m_fst,
            "diff_vs_stronger_per_sample": m_max,
            "diff_vs_stronger_level": m_max_level,
            "ci95_vs_rbm": ci_pre, "ci95_vs_fastv": ci_fst,
            "ci95_vs_stronger_per_sample": ci_max,
            "mcnemar_z_vs_stronger": z, "mcnemar_b10": b10, "mcnemar_b01": b01,
            "win_tie_loss": wtl,
        }
        print(f"  {b:9s} def-rbm={m_pre:+.4f} (CI{ci_pre[0]:+.3f},"
              f"{ci_pre[1]:+.3f})  def-fst={m_fst:+.4f}  "
              f"def-max(level)={m_max_level:+.4f}  def-max(per-samp)="
              f"{m_max:+.4f} (CI{ci_max[0]:+.3f},{ci_max[1]:+.3f})  "
              f"stronger={pkey}  McNemar z={z:+.2f} (b10={b10},b01={b01})")
        print(f"        win/tie/loss vs rbm={wtl['vs_rbm']}  vs fastv="
              f"{wtl['vs_fastv']}  vs stronger={wtl['vs_stronger']}")

    # ---------------- macro average ----------------
    macro_def = np.mean([table[b]["mean"]["deferred"] for b in BENCHES])
    macro_sp = np.mean([max(table[b]["mean"]["rbm"], table[b]["mean"]["fastv"])
                        for b in BENCHES])
    macro_rbm = np.mean([table[b]["mean"]["rbm"] for b in BENCHES])
    macro_fst = np.mean([table[b]["mean"]["fastv"] for b in BENCHES])
    macro_full = np.mean([table[b]["mean"]["full"] for b in BENCHES])
    print(f"\n[macro] full={macro_full:.4f} rbm={macro_rbm:.4f} "
          f"fastv={macro_fst:.4f} deferred={macro_def:.4f} "
          f"stronger-parent={macro_sp:.4f}  (def - stronger = "
          f"{macro_def - macro_sp:+.4f})")

    # ---------------- GO/NO-GO (user criteria) ----------------
    print("\n[GO/NO-GO criteria (task 2026-08-24)]")
    c1 = macro_def >= macro_sp - 1e-12
    hits3 = []
    for b in BENCHES:
        if table[b]["diff_vs_stronger_level"] >= 0.03:
            hits3.append(b)
    c2 = any(b in ("ocrbench", "gqa") for b in hits3)
    worst = {b: table[b]["diff_vs_stronger_level"] for b in BENCHES}
    c3 = all(v >= -0.05 for v in worst.values())
    # c4: no eval-failure / answer-format / missing-sample artifact
    skip = {}
    for b in BENCHES:
        skip[b] = {a: cells[(b, a)]["n_skipped"] for a in ARMS}
    c4 = all(len(set(skip[b].values())) == 1 for b in BENCHES)
    go = c1 and c2 and c3 and c4
    print(f"  c1 macro_def({macro_def:.4f}) >= macro_stronger({macro_sp:.4f}): "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  c2 OCRBench/GQA +>=3pp: hits={hits3} "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"  c3 no bench below stronger-5pp: worst={ {k: round(v,4) for k,v in worst.items()} } "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"  c4 no skip/format artifact (skips equal across arms): "
          f"{skip} {'PASS' if c4 else 'FAIL'}")
    verdict = "GO (mechanism worth continuing)" if go else "NO-GO"
    print(f"\nVERDICT: {verdict}")

    # ---------------- n=64 vs n=200 ----------------
    dev = {}
    for b in BENCHES:
        d64 = load(f"{RB}/dev_deferred_{b}_n64.json")
        ids64 = sorted(set.intersection(
            {k for k in official_per_sample(b, d64)},
            {k for k in per[(b, "rbm")]},
            {k for k in per[(b, "fastv")]}))
        def_m, _, _, _ = agg(official_per_sample(b, d64), ids64)
        sp_m = max(agg(per[(b, "rbm")], ids64)[0], agg(per[(b, "fastv")], ids64)[0])
        dev[b] = (def_m, sp_m, len(ids64))

    # ---------------- failure-case classification ----------------
    print("\n[failure classification] deferred vs stronger parent")
    fail = {}
    for b in BENCHES:
        ids = common[b]
        pkey = table[b]["stronger_parent"]
        dloss = [i for i in ids
                 if per[(b, "deferred")][i][1] == 0 and per[(b, pkey)][i][1] == 1]
        dwin = [i for i in ids
                if per[(b, "deferred")][i][1] == 1 and per[(b, pkey)][i][1] == 0]
        fail[b] = {"n_win": len(dwin), "n_loss": len(dloss)}
        # OCRBench question-type breakdown of losses
        if b == "ocrbench":
            cat = {}
            for i in dloss:
                s = next(x for x in cells[(b, "deferred")]["per_sample"]
                         if str(x["id"]) == i)
                c = s.get("question_type", "?") or "?"
                cat[c] = cat.get(c, 0) + 1
            fail[b]["loss_categories"] = cat
        # empty-answer or same-answer-but-diff-correctness artifact check
        fmt_artifact = 0
        for i in ids:
            dd = cells[(b, "deferred")]["per_sample"]
            rr = cells[(b, pkey)]["per_sample"]
            di = next(x for x in dd if str(x["id"]) == i)
            ri = next(x for x in rr if str(x["id"]) == i)
            if (per[(b, "deferred")][i][1] == 1 and per[(b, pkey)][i][1] == 0
                    and di["answer"] == ri["answer"]):
                fmt_artifact += 1
        fail[b]["win_with_same_answer_as_parent"] = fmt_artifact
        print(f"  {b:9s} wins={len(dwin)} losses={len(dloss)} "
              f"(same-answer wins={fmt_artifact})")

    out = {
        "verdict": verdict, "go": go,
        "criteria": {"c1_macro": bool(c1), "c2_hits": hits3,
                     "c3_worst": worst, "c4_skips_equal": bool(c4)},
        "macro": {"full": macro_full, "rbm": macro_rbm, "fastv": macro_fst,
                  "deferred": macro_def, "stronger_parent": macro_sp,
                  "diff_deferred_minus_stronger": macro_def - macro_sp},
        "per_benchmark": table,
        "dev_n64": {b: {"deferred": dev[b][0], "stronger": dev[b][1],
                        "n": dev[b][2]} for b in BENCHES},
        "failure_classification": fail,
        "git": {"commit": os.popen("git rev-parse --short HEAD").read().strip(),
                "branch": os.popen("git branch --show-current").read().strip()},
    }
    with open(f"{OUTDIR}/analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[written] {OUTDIR}/analysis.json + per_sample.json")


if __name__ == "__main__":
    main()
