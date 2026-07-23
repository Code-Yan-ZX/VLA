#!/usr/bin/env python3
"""Rescore EXISTING saved predictions with the OFFICIAL metrics, offline (CPU).

TextVQA -> standard VQA accuracy.  DocVQA -> ANLS.
gqa / chartqa / ocrbench / mme / mmbench / scienceqa keep their existing stored
metric (we just report the stored aggregate `acc`).

Writes:
  runs/rescore_official/summary.json   (per-cell old vs new)
  drafts/rescore_official_report.md    (table + verdict + direction-flip flags)

New files only; does not touch the runner or serve_bench.py. No GPU.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

REPO = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.official_scorers import (  # noqa: E402
    score_textvqa_vqaacc,
    score_docvqa_anls,
)

# dirs to walk (cell JSONs that may carry per_sample)
CELL_DIRS = [
    "runs/v3_router_probe",
    "runs/v3_sota_matrix",
    "runs/v3_attn_robust",
]
EXTRA_CELLS = [
    "runs/v3_premerger_cells/C_vzstyle_docvqa_r0.750_l2_n200.json",
]
SUBSET_DIR = "eval/subsets"

# benchmarks whose OFFICIAL metric we recompute
TEXTVQA = "textvqa"
DOCVQA = "docvqa"


def load_subset_gt(bench: str) -> dict[str, str]:
    """Return id -> gt from eval/subsets/<bench>_200.jsonl (if present)."""
    path = os.path.join(REPO, SUBSET_DIR, f"{bench}_200.jsonl")
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in o:
                out[str(o["id"])] = str(o.get("gt", ""))
    return out


def parse_meta(cell: dict, fname: str) -> dict:
    """Benchmark / mode / keep-ratio / selector from JSON fields (authoritative),
    falling back to filename conventions."""
    bench = cell.get("benchmark")
    mode = cell.get("mode")
    r = cell.get("r")
    selector = cell.get("selector")
    vz = cell.get("visionzip_style")

    # filename fallbacks: <mode>_<bench>_r<R>_[<sel>_]n<N>.json  or  <P>_<bench>[_r<R>].json
    base = os.path.splitext(os.path.basename(fname))[0]
    if not bench:
        for b in ["textvqa", "docvqa", "chartqa", "ocrbench", "mmbench",
                  "scienceqa", "mme", "gqa"]:
            if b in base:
                bench = b
                break
    if mode is None or mode == "":
        if base.startswith("pre_") or base.startswith("C_"):
            mode = "pre"
        elif base.startswith("post_") or base.startswith("B_") or base.startswith("vz"):
            mode = "post"
        elif base.startswith("A_"):
            mode = "baseline"
    if r is None:
        if "_r0." in base:
            try:
                r = float(base.split("_r")[1].split("_")[0])
            except Exception:
                r = None
        elif base.startswith("A_"):
            r = 0.0

    # keep ratio = fraction of visual tokens KEPT = 1 - r (r = drop ratio)
    keep = None
    if isinstance(r, (int, float)):
        keep = round(1.0 - float(r), 4)
    # normalize mode label
    if mode == "none":
        mode = "baseline"
    return {
        "bench": bench,
        "mode": mode,
        "r": r,
        "keep": keep,
        "selector": selector,
        "visionzip_style": bool(vz) if vz is not None else None,
    }


def rescore_cell(cell: dict, fname: str, subset_gt_cache: dict) -> dict:
    meta = parse_meta(cell, fname)
    bench = meta["bench"]
    old_acc = cell.get("acc")
    n_total = cell.get("n")
    per = cell.get("per_sample")
    has_ps = isinstance(per, list) and len(per) > 0

    rec = {
        "cell": fname,
        "bench": bench,
        "mode": meta["mode"],
        "r": meta["r"],
        "keep": meta["keep"],
        "selector": meta["selector"],
        "visionzip_style": meta["visionzip_style"],
        "old_acc": old_acc,
        "n_total": n_total,
        "has_per_sample": has_ps,
        "official_metric": None,
        "new_metric_mean": None,
        "n_rescored": 0,
        "n_answered_rescored": 0,
        "n_nonzero_official": 0,
        "gt_rejoined": 0,
        "note": None,
    }

    # Only textvqa/docvqa get rescored; everything else keeps stored metric.
    if bench not in (TEXTVQA, DOCVQA):
        rec["official_metric"] = None  # keep existing metric
        if not has_ps:
            rec["note"] = "no per_sample; reported stored acc only"
        return rec

    if not has_ps:
        rec["note"] = "MISSING per_sample -> cannot recompute official metric; reported stored acc only"
        return rec

    # subset gt (authoritative) for re-joining truncated per_sample gt
    if bench not in subset_gt_cache:
        subset_gt_cache[bench] = load_subset_gt(bench)
    subgt = subset_gt_cache[bench]

    metric = "vqa_accuracy" if bench == TEXTVQA else "anls"
    scorer = score_textvqa_vqaacc if bench == TEXTVQA else score_docvqa_anls

    scores = []
    n_ans = 0
    rejoined = 0
    for s in per:
        sid = str(s.get("id"))
        gt = s.get("gt") or ""
        # re-join from subset jsonl if per_sample gt looks truncated/empty
        if sid in subgt:
            canon = subgt[sid]
            if (gt is None or len(str(gt)) < len(canon)):
                gt = canon
                rejoined += 1
        pred = s.get("answer", "") or ""
        if not s.get("skipped", False) and pred.strip() != "":
            n_ans += 1
        scores.append(scorer(pred, gt))

    rec["official_metric"] = metric
    rec["new_metric_mean"] = sum(scores) / len(scores) if scores else 0.0
    rec["n_rescored"] = len(scores)
    rec["n_answered_rescored"] = n_ans
    rec["n_nonzero_official"] = sum(1 for x in scores if x > 0)
    rec["gt_rejoined"] = rejoined
    return rec


def main():
    files = []
    for d in CELL_DIRS:
        dp = os.path.join(REPO, d)
        if not os.path.isdir(dp):
            continue
        for fn in sorted(os.listdir(dp)):
            if not fn.endswith(".json"):
                continue
            low = fn.lower()
            # skip aggregate/summary artifacts that are NOT prediction cells
            if any(tok in low for tok in ("summary", "sweeps", "curves")):
                continue
            files.append(os.path.join(d, fn))
    for ex in EXTRA_CELLS:
        if os.path.exists(os.path.join(REPO, ex)):
            files.append(ex)

    subset_gt_cache: dict = {}
    records = []
    skipped_no_preds = []
    for rel in files:
        full = os.path.join(REPO, rel)
        try:
            cell = json.load(open(full))
        except Exception as e:
            print(f"[skip] cannot read {rel}: {e}")
            continue
        rec = rescore_cell(cell, rel, subset_gt_cache)
        records.append(rec)
        if rec["bench"] in (TEXTVQA, DOCVQA) and not rec["has_per_sample"]:
            skipped_no_preds.append(rel)

    # ---- aggregate verdict table: (bench, mode, keep) ----
    # For textvqa/docvqa use new_metric_mean; else old_acc.
    def effective(rec):
        if rec["bench"] in (TEXTVQA, DOCVQA) and rec["new_metric_mean"] is not None:
            return rec["new_metric_mean"]
        return rec["old_acc"]

    # Build pre/post lookup keyed by (bench, keep, selector-tag) for verdict
    def sel_tag(rec):
        sel = rec["selector"] or "l2"
        vz = "vz" if rec["visionzip_style"] else "std"
        return f"{sel}/{vz}"

    rows = defaultdict(dict)  # (bench,keep,tag) -> {mode: {old,new,eff}}
    for rec in records:
        if rec["bench"] is None:
            continue
        keep = rec["keep"]
        if keep is None:
            continue
        key = (rec["bench"], keep, sel_tag(rec))
        eff = effective(rec)
        rows[key][rec["mode"]] = {
            "old_acc": rec["old_acc"],
            "new_metric": rec["new_metric_mean"],
            "official_metric": rec["official_metric"],
            "effective": eff,
            "n": rec["n_rescored"] or rec["n_total"],
            "cell": rec["cell"],
        }

    summary = {
        "generated": "2026-07-23",
        "metrics": {
            "textvqa": "vqa_accuracy (official)",
            "docvqa": "anls (official)",
            "gqa/chartqa/ocrbench/mme/mmbench/scienceqa": "existing stored metric (unchanged)",
        },
        "cells_skipped_missing_preds": skipped_no_preds,
        "records": records,
    }

    out_dir = os.path.join(REPO, "runs/rescore_official")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------------- Verdict computations ----------------
    # router_probe pre/post at keep=25% (r=0.75), selector l2/std
    def pp(bench, keep=0.25, tag="l2/std"):
        m = rows.get((bench, keep, tag), {})
        return m.get("pre"), m.get("post")

    verdicts = []
    # textvqa (NEW vqa-acc)
    for bench, metric in [(TEXTVQA, "VQA-acc"), (DOCVQA, "ANLS"), ("gqa", "stored")]:
        pre, post = pp(bench)
        verdicts.append((bench, metric, pre, post))

    # ocrbench / chartqa pre(C_)=mode pre vs post(B_)=mode post from sota_matrix, keep25
    for bench in ["ocrbench", "chartqa"]:
        pre, post = pp(bench)
        verdicts.append((bench, "stored", pre, post))

    # ---------------- Markdown report ----------------
    lines = []
    lines.append("# Rescore with OFFICIAL metrics — claim-overturn gate\n")
    lines.append("_Generated 2026-07-23 · offline · CPU-only · scorer: `src/v3_premerger/official_scorers.py`_\n")
    lines.append("**Metrics applied:** TextVQA → official VQA accuracy; DocVQA → official ANLS. "
                 "GQA/ChartQA/OCRBench/MME/MMBench/ScienceQA keep their existing stored metric (unchanged).\n")
    lines.append("**Important caveat:** VQA-acc and ANLS are computed on the RAW stored generations "
                 "(`per_sample[].answer`, often verbose multi-sentence text). No answer extraction is "
                 "applied, so these are an *honest lower bound* of the official metric for these runs.\n")

    lines.append("### Root cause of the near-zero official scores\n")
    lines.append("The router-probe runs were generated with a free-form prompt (no short-answer "
                 "constraint; `max_tokens=32` but outputs are full sentences, median ~133 chars). "
                 "Official VQA-acc requires the normalized prediction to EXACTLY equal a short gt "
                 "answer, and official ANLS requires high string overlap with a short gt — both fail "
                 "against verbose sentences. Consequently textvqa VQA-acc is **exactly 0.0** for every "
                 "pre AND post sample, and docvqa ANLS collapses to ~0.02 (pre) / 0.0 (post), driven by "
                 "a handful of coincidentally short 'yes'/'no' answers. The OLD containment rule passed "
                 "whenever the short gt appeared *anywhere* inside the verbose answer, which is why it "
                 "credited 0.695 / 0.725 where the official metric sees ~0. **These runs are therefore "
                 "not comparable on the official scale; the containment numbers over-state absolute "
                 "performance by ~100×.** Re-running with a short-answer prompt would be needed to get "
                 "meaningful official-metric values; that is a data-generation change, out of scope here.\n")

    # ---- prominent direction-flip banner ----
    lines.append("## ⚑ DIRECTION-FLIP CHECK (top priority)\n")
    flip_any = False
    flip_lines = []
    for bench, metric, pre, post in verdicts:
        if not pre or not post:
            continue
        old_pre, old_post = pre.get("old_acc"), post.get("old_acc")
        new_pre = pre.get("effective")
        new_post = post.get("effective")
        if None in (old_pre, old_post, new_pre, new_post):
            continue
        old_dir = "pre>post" if old_pre > old_post else ("post>pre" if old_post > old_pre else "tie")
        new_dir = "pre>post" if new_pre > new_post else ("post>pre" if new_post > new_pre else "tie")
        flipped = old_dir != new_dir
        if flipped:
            flip_any = True
        flag = "🔄 **FLIP**" if flipped else "✅ hold"
        flip_lines.append(
            f"- {flag} **{bench}** ({metric}): OLD containment {old_dir} "
            f"({old_pre:.3f} vs {old_post:.3f}) → NEW {new_dir} "
            f"({new_pre:.3f} vs {new_post:.3f})"
        )
    if flip_any:
        lines.append("> **⚠️ AT LEAST ONE DIRECTION FLIPPED under the official metrics — review the "
                     "affected conclusion(s) below before reporting.**\n")
    else:
        lines.append("> **No direction flips** at keep=25% — the pre/post ordering is preserved under "
                     "the official metrics (though absolute magnitudes may change).\n")
    lines.extend(flip_lines)
    lines.append("")

    # ---- verdict detail ----
    lines.append("## VERDICT at keep=25% (r=0.75)\n")
    lines.append("| Benchmark | Metric | OLD pre | OLD post | NEW pre | NEW post | OLD dir | NEW dir | NEW gap (pre−post, pp) | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for bench, metric, pre, post in verdicts:
        if not pre or not post:
            lines.append(f"| {bench} | {metric} | — | — | — | — | — | — | — | MISSING pre/post cell |")
            continue
        old_pre, old_post = pre.get("old_acc"), post.get("old_acc")
        new_pre, new_post = pre.get("effective"), post.get("effective")
        def d(a, b):
            if a is None or b is None:
                return "—"
            return "pre>post" if a > b else ("post>pre" if b > a else "tie")
        gap_pp = "" if new_pre is None or new_post is None else f"{(new_pre-new_post)*100:+.1f}"
        old_dir = d(old_pre, old_post)
        new_dir = d(new_pre, new_post)
        verdict_word = "HOLD" if old_dir == new_dir else "**FLIP**"
        # near-floor warning for the rescored benches
        if bench in (TEXTVQA, DOCVQA) and new_pre is not None and new_post is not None \
                and max(new_pre, new_post) < 0.05:
            verdict_word += " ⚠near-floor"
        def f3(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(
            f"| {bench} | {metric} | {f3(old_pre)} | {f3(old_post)} | "
            f"{f3(new_pre)} | {f3(new_post)} | {old_dir} | {new_dir} | {gap_pp} | {verdict_word} |"
        )
    lines.append("")
    lines.append("Interpretation: **textvqa/docvqa** NEW pre/post use the official metric; "
                 "**gqa/chartqa/ocrbench** NEW == OLD (existing metric, so direction cannot change).\n")

    # ---- full OLD vs NEW table ----
    lines.append("## Full per-cell table (OLD containment vs NEW official)\n")
    lines.append("| Cell | Bench | Mode | keep% | sel | OLD acc | NEW metric | NEW value | nonzero/n | n | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rec in sorted(records, key=lambda r: (r["bench"] or "", str(r["keep"]), str(r["mode"]), r["cell"])):
        keep = "—" if rec["keep"] is None else f"{rec['keep']*100:.0f}"
        nm = rec["official_metric"] or "kept"
        nv = "—" if rec["new_metric_mean"] is None else f"{rec['new_metric_mean']:.3f}"
        oa = "—" if rec["old_acc"] is None else f"{rec['old_acc']:.3f}"
        n = rec["n_rescored"] or rec["n_total"] or "—"
        nz = f"{rec['n_nonzero_official']}/{rec['n_rescored']}" if rec["n_rescored"] else "—"
        lines.append(
            f"| {os.path.basename(rec['cell'])} | {rec['bench']} | {rec['mode']} | {keep} | "
            f"{rec['selector']} | {oa} | {nm} | {nv} | {nz} | {n} | {rec['note'] or ''} |"
        )
    lines.append("")

    # ---- skipped cells ----
    lines.append("## Cells skipped (missing per_sample preds)\n")
    if skipped_no_preds:
        for c in skipped_no_preds:
            lines.append(f"- `{c}` (textvqa/docvqa cell without `per_sample`; kept stored acc only)")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Disk artifacts\n")
    lines.append("- Summary JSON: `runs/rescore_official/summary.json`")
    lines.append("- Scorer module: `src/v3_premerger/official_scorers.py`")
    lines.append("- This report: `drafts/rescore_official_report.md`")

    os.makedirs(os.path.join(REPO, "drafts"), exist_ok=True)
    with open(os.path.join(REPO, "drafts/rescore_official_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # console digest of key numbers
    print("=== RESCORE COMPLETE ===")
    print(f"cells processed: {len(records)}")
    print(f"textvqa/docvqa cells skipped (no per_sample): {len(skipped_no_preds)}")
    for c in skipped_no_preds:
        print("   -", c)
    for bench, metric, pre, post in verdicts:
        if pre and post and pre.get("effective") is not None and post.get("effective") is not None:
            print(f"{bench:9s} [{metric}] NEW pre={pre['effective']:.3f} post={post['effective']:.3f} "
                  f"gap={(pre['effective']-post['effective'])*100:+.1f}pp "
                  f"OLD(pre={pre.get('old_acc')} post={post.get('old_acc')})")
        else:
            print(f"{bench:9s} [{metric}] MISSING pre/post cell")


if __name__ == "__main__":
    main()
