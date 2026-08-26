#!/usr/bin/env python3
"""Rescore EXISTING saved predictions with the OFFICIAL metrics, offline (CPU).

TextVQA  -> standard VQA accuracy.   DocVQA -> ANLS.
GQA      -> normalized EXACT match (official GQA eval).
OCRBench -> per-sample containment keyed on question_type, rolled up into the
            5 official skills (each /200, total /1000).
chartqa / mme / mmbench / scienceqa keep their existing stored metric (we just
report the stored aggregate `acc`).

Writes:
  runs/rescore_official/summary.json   (per-cell old vs new)
  drafts/rescore_official_report.md    (table + verdict + direction-flip flags)

Usage:
  python scripts/rescore_official.py                     # all 4 official benches
  python scripts/rescore_official.py --benchmark gqa     # only GQA
  python scripts/rescore_official.py --benchmark ocrbench
  python scripts/rescore_official.py --benchmark textvqa --benchmark docvqa

Does not touch the runner or serve_bench.py. No GPU (the runner's containment
rules are re-implemented inline below so we never import torch/vllm).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

REPO = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.official_scorers import (  # noqa: E402
    score_textvqa_vqaacc,
    score_docvqa_anls,
    score_gqa,
    score_ocrbench,
    ocrbench_category,
    OCRBENCH_CATEGORIES,
    OCRBENCH_HME,
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
# extra dirs that carry the GQA / OCRBench cells we also rescore
GQA_OCR_DIRS = [
    "runs/v3_merger_aware/hybrid_gate",
    "runs/v3_merger_aware/rescore_rerun",
    "runs/v3_merger_aware/router",
]
SUBSET_DIR = "eval/subsets"

# benchmarks whose OFFICIAL metric we recompute
TEXTVQA = "textvqa"
DOCVQA = "docvqa"
GQA = "gqa"
OCRBENCH = "ocrbench"
ALL_BENCHES = [TEXTVQA, DOCVQA, GQA, OCRBENCH]


# ---------------------------------------------------------------------------
# Faithful CPU re-implementations of the RUNNER's ad-hoc containment scorers
# (src/v3_premerger/v3_premerger_runner.py, which copies src/serve_bench.py).
# Used ONLY to compute the "official vs runner-containment" difference on the
# SAME denominator -- we do NOT import the runner (it pulls in torch/vllm).
# ---------------------------------------------------------------------------

def _norm_words(s: str) -> list:
    """Runner _norm_words (L107): lowercase, alnum/space only, split."""
    return "".join(c if (c.isalnum() or c.isspace()) else " "
                   for c in str(s).strip().lower()).split()


def _singular(tok: str) -> str:
    """Runner _singular (L116)."""
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 2 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 1 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def runner_containment_gqa(pred: str, gt: str) -> int:
    """Runner score_gqa (L125-160): yes/no lead-word, else word-level
    containment with singular tolerance (multi-word gt -> substring)."""
    if not gt:
        return 0
    p_words = _norm_words(pred)
    g_norm = "".join(c for c in gt.strip().lower()
                     if c.isalnum() or c.isspace()).strip()
    g_words = g_norm.split()
    if not g_words:
        return 0
    if g_norm in {"yes", "no"}:
        lead = next((w for w in p_words if w not in {"a", "an", "the"}), None)
        return 1 if (lead in {"yes", "no"} and lead == g_norm) else 0
    syns = {g_norm, _singular(g_norm) if len(g_words) == 1 else g_norm}
    p_text = " ".join(p_words)
    for s in syns:
        s_words = s.split()
        if len(s_words) == 1:
            sg = _singular(s)
            if any(w == s or _singular(w) == sg for w in p_words):
                return 1
        elif s in p_text:
            return 1
    return 0


def runner_containment_ocrbench(pred: str, gt: str, nospace: bool) -> int:
    """Runner score_ocrbench (L262-280): lowercase containment; HME rows
    (nospace) strip all spaces. (Runner lowercases HME too -> the one
    difference vs the official case-sensitive HME branch.)"""
    if not gt:
        return 0
    p = str(pred).lower().strip().replace("\n", " ")
    for g in str(gt).split(";"):
        g = g.lower().strip().replace("\n", " ")
        if not g:
            continue
        if nospace:
            if g.replace(" ", "") in p.replace(" ", ""):
                return 1
        elif g in p:
            return 1
    return 0


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


def load_ocrbench_meta() -> dict:
    """Return id -> {'question_type', 'category', 'nospace'} for OCRBench.

    The runner per_sample carries NO category/question_type (extras are not
    passed through), so we recover them by ID. PREFERRED source is the
    authoritative full split eval/full_splits/ocrbench.jsonl (1000 rows; carries
    the data-side `category` field = TR/HTR/ST-VQA/DT-VQA/KIE, 200 each). We
    fall back to eval/subsets/ocrbench_200.jsonl (has question_type but no
    category -> category is then derived from question_type by the scorer).
    `nospace` = HME100k rows (choices has '__nospace__'), equivalently
    question_type == 'Handwritten Mathematical Expression Recognition'."""
    candidates = [
        os.path.join(REPO, "eval/full_splits/ocrbench.jsonl"),
        os.path.join(REPO, SUBSET_DIR, "ocrbench_200.jsonl"),
    ]
    out: dict = {}
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" not in o:
                    continue
                sid = str(o["id"])
                if sid in out:
                    continue  # first (preferred) source wins
                qt = str(o.get("question_type") or "")
                cat = o.get("category") or ""
                nospace = (bool(o.get("choices")) and "__nospace__" in o.get("choices")) \
                    or (qt == OCRBENCH_HME)
                out[sid] = {"question_type": qt, "category": cat, "nospace": nospace}
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


def rescore_cell(cell: dict, fname: str, subset_gt_cache: dict,
                 ocr_meta: dict = None, selected=None) -> dict:
    meta = parse_meta(cell, fname)
    bench = meta["bench"]
    old_acc = cell.get("acc")
    n_total = cell.get("n")
    per = cell.get("per_sample")
    has_ps = isinstance(per, list) and len(per) > 0
    if selected is None:
        selected = set(ALL_BENCHES)

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
        "containment_recomputed": None,   # runner ad-hoc rule, same /n_total
        "official_vs_containment_pp": None,
        "note": None,
    }

    # Benchmarks not selected for official rescoring keep their stored metric.
    if bench not in selected:
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

    # ---- TextVQA / DocVQA (unchanged official path) ----------------------
    if bench in (TEXTVQA, DOCVQA):
        metric = "vqa_accuracy" if bench == TEXTVQA else "anls"
        scorer = score_textvqa_vqaacc if bench == TEXTVQA else score_docvqa_anls
        scores, n_ans, rejoined = [], 0, 0
        for s in per:
            sid = str(s.get("id"))
            gt = s.get("gt") or ""
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

    # ---- GQA (official normalized exact match) ---------------------------
    if bench == GQA:
        off, cont, n_ans = [], [], 0
        for s in per:
            sid = str(s.get("id"))
            gt = s.get("gt") or ""
            if sid in subgt:
                canon = subgt[sid]
                if (gt is None or len(str(gt)) < len(canon)):
                    gt = canon
                    rec["gt_rejoined"] += 1
            pred = s.get("answer", "") or ""
            if not s.get("skipped", False) and pred.strip() != "":
                n_ans += 1
            off.append(score_gqa(pred, gt))
            cont.append(runner_containment_gqa(pred, gt))
        rec["official_metric"] = "exact_match(normalized)"
        rec["new_metric_mean"] = sum(off) / len(off) if off else 0.0
        rec["containment_recomputed"] = sum(cont) / len(cont) if cont else 0.0
        rec["official_vs_containment_pp"] = (
            (rec["new_metric_mean"] - rec["containment_recomputed"]) * 100.0)
        rec["n_rescored"] = len(off)
        rec["n_answered_rescored"] = n_ans
        rec["n_nonzero_official"] = sum(1 for x in off if x > 0)
        return rec

    # ---- OCRBench (official per-sample containment + 5-skill roll-up) ----
    if bench == OCRBENCH:
        ocr_meta = ocr_meta or {}
        off, cont, n_ans = [], [], 0
        cat_correct = defaultdict(int)
        cat_total = defaultdict(int)
        for s in per:
            sid = str(s.get("id"))
            gt = s.get("gt") or ""
            if sid in subgt:
                canon = subgt[sid]
                if (gt is None or len(str(gt)) < len(canon)):
                    gt = canon
                    rec["gt_rejoined"] += 1
            m = ocr_meta.get(sid, {})
            qt = m.get("question_type", "")
            nospace = m.get("nospace", False)
            # data-side 5-category (prefer the authoritative `category` field,
            # recovered by id from eval/full_splits/ocrbench.jsonl; else
            # derived from question_type by the scorer)
            cat = ocrbench_category(qt, m.get("category"))
            pred = s.get("answer", "") or ""
            if not s.get("skipped", False) and pred.strip() != "":
                n_ans += 1
            c = score_ocrbench(pred, gt, qt, nospace=nospace)
            off.append(c)
            cont.append(runner_containment_ocrbench(pred, gt, nospace))
            cat_total[cat] += 1
            if c:
                cat_correct[cat] += 1
        rec["official_metric"] = "containment(per-category)"
        rec["new_metric_mean"] = sum(off) / len(off) if off else 0.0
        rec["containment_recomputed"] = sum(cont) / len(cont) if cont else 0.0
        rec["official_vs_containment_pp"] = (
            (rec["new_metric_mean"] - rec["containment_recomputed"]) * 100.0)
        rec["n_rescored"] = len(off)
        rec["n_answered_rescored"] = n_ans
        rec["n_nonzero_official"] = sum(off)
        # official 5-category breakdown + /1000 extrapolation (each category's
        # acc scaled to the official 200-per-category basis)
        categories = {}
        extrap = 0.0
        for cat in OCRBENCH_CATEGORIES:
            tot = cat_total.get(cat, 0)
            cor = cat_correct.get(cat, 0)
            acc = (cor / tot) if tot else 0.0
            categories[cat] = {"correct": cor, "total": tot, "acc": round(acc, 4)}
            if tot:
                extrap += acc * 200.0
        rec["ocr_categories"] = categories
        rec["ocr_extrap_1000"] = round(extrap, 2)
        rec["ocr_final_score_raw"] = int(sum(off))
        return rec

    # any other benchmark keeps stored metric
    return rec


def _collect(dirs, source, files):
    for d in dirs:
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
            files.append((os.path.join(d, fn), source))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="append", default=None,
                    choices=ALL_BENCHES,
                    help="benchmark(s) to rescore with the OFFICIAL metric "
                         "(repeatable; default: all four).")
    args = ap.parse_args()
    selected = set(args.benchmark) if args.benchmark else set(ALL_BENCHES)

    # (relpath, source) -- 'core' dirs keep the validated textvqa/docvqa
    # verdict; 'ext' dirs add the GQA / OCRBench cells (merger_aware runs).
    files = []
    _collect(CELL_DIRS, "core", files)
    for ex in EXTRA_CELLS:
        if os.path.exists(os.path.join(REPO, ex)):
            files.append((ex, "core"))
    if selected & {GQA, OCRBENCH}:
        _collect(GQA_OCR_DIRS, "ext", files)
    # de-dup (a dir may appear in both lists), keep first (core) source
    seen, uniq = set(), []
    for rel, src in files:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append((rel, src))
    files = uniq

    ocr_meta = load_ocrbench_meta() if OCRBENCH in selected else {}
    subset_gt_cache: dict = {}
    records = []
    skipped_no_preds = []
    for rel, src in files:
        full = os.path.join(REPO, rel)
        try:
            cell = json.load(open(full))
        except Exception as e:
            print(f"[skip] cannot read {rel}: {e}")
            continue
        rec = rescore_cell(cell, rel, subset_gt_cache,
                           ocr_meta=ocr_meta, selected=selected)
        rec["source"] = src
        records.append(rec)
        if rec["bench"] in (TEXTVQA, DOCVQA) and rec["bench"] in selected \
                and not rec["has_per_sample"]:
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

    # rows feed the textvqa/docvqa/chartqa direction-flip verdict; built from
    # CORE dirs only so the validated verdict is unchanged by the added
    # merger_aware cells. GQA / OCRBench get their own dedicated sections.
    rows = defaultdict(dict)  # (bench,keep,tag) -> {mode: {old,new,eff}}
    for rec in records:
        if rec.get("source") != "core":
            continue
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
        "selected_benchmarks": sorted(selected),
        "metrics": {
            "textvqa": "vqa_accuracy (official)",
            "docvqa": "anls (official)",
            "gqa": "normalized exact match (official GQA eval)",
            "ocrbench": "per-category containment + 5-skill /1000 (official OCRBench eval)",
            "chartqa/mme/mmbench/scienceqa": "existing stored metric (unchanged)",
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
    # textvqa (NEW vqa-acc), docvqa (NEW anls), chartqa (stored, not rescored)
    for bench, metric in [(TEXTVQA, "VQA-acc"), (DOCVQA, "ANLS"), ("chartqa", "stored")]:
        pre, post = pp(bench)
        verdicts.append((bench, metric, pre, post))
    # GQA and OCRBench are now officially rescored too; their pre/post verdicts
    # live in dedicated sections below (they pull the merger_aware / sota_matrix
    # cells rather than the core-only rows dict used above).

    # ---------------- Markdown report ----------------
    lines = []
    lines.append("# Rescore with OFFICIAL metrics — claim-overturn gate\n")
    lines.append("_Generated 2026-07-23 · offline · CPU-only · scorer: `src/v3_premerger/official_scorers.py`_\n")
    lines.append("**Metrics applied:** TextVQA → official VQA accuracy; DocVQA → official ANLS; "
                 "GQA → official normalized exact match; OCRBench → official per-category containment "
                 "rolled into the 5 skills (/1000). ChartQA/MME/MMBench/ScienceQA keep their stored metric.\n")
    lines.append("**Important caveat:** VQA-acc and ANLS are computed on the RAW stored generations "
                 "(`per_sample[].answer`, often verbose multi-sentence text). No answer extraction is "
                 "applied, so these are an *honest lower bound* of the official metric for these runs. "
                 "GQA/OCRBench generations are short (single-word prompts), so the official metric is "
                 "meaningful there (see dedicated sections below).\n")

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
                 "**chartqa** NEW == OLD (existing metric, direction cannot change). "
                 "**GQA / OCRBench** are rescored officially in their dedicated sections below.\n")

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

    def _dir(a, b):
        if a is None or b is None:
            return "—"
        return "pre>post" if a > b else ("post>pre" if b > a else "tie")

    # ---- GQA official exact-match section ----
    if GQA in selected:
        gqa_recs = [r for r in records
                    if r["bench"] == GQA and r["new_metric_mean"] is not None]
        lines.append("## GQA — official normalized EXACT match vs runner containment\n")
        lines.append("Official GQA eval (`nlp.stanford.edu/data/gqa/eval.zip` → `eval.py` L350: "
                     "`correct = (predicted == gold)`) is a string exact match. We canonicalize both "
                     "sides with VQA normalization (lowercase / de-punctuate / drop articles a-an-the / "
                     "number-word→digit) then exact-match — a NO-OP on these clean single-word outputs, "
                     "so it equals the strict official raw-exact number. `containment_recomp` is the "
                     "runner's word-containment rule recomputed on the same /n_total.\n")
        lines.append("| Cell | src | Mode | keep% | n | stored(cont) | OFFICIAL exact | containment_recomp | Δ(off−cont) pp |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(gqa_recs, key=lambda r: (r["source"], str(r["keep"]), str(r["mode"]), r["cell"])):
            keep = "—" if r["keep"] is None else f"{r['keep']*100:.0f}"
            lines.append(
                f"| {os.path.basename(r['cell'])} | {r['source']} | {r['mode']} | {keep} | "
                f"{r['n_rescored']} | {r['old_acc']:.3f} | {r['new_metric_mean']:.3f} | "
                f"{r['containment_recomputed']:.3f} | {r['official_vs_containment_pp']:+.1f} |"
            )
        lines.append("")
        lines.append("### GQA pre/post TIE check (writing red-line: merger pre≈post must hold)\n")
        lines.append("| Source | n | cont pre | cont post | OFFICIAL pre | OFFICIAL post | OFFICIAL dir | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")

        def _pick_gqa(src, keep, mode):
            cands = [r for r in gqa_recs
                     if r["source"] == src and r["mode"] == mode
                     and abs((r["keep"] if r["keep"] is not None else -1) - keep) < 1e-6]
            # canonical = largest-n cell (n=100 over cap64 n=64)
            return max(cands, key=lambda r: r["n_rescored"]) if cands else None

        for src, label in [("ext", "merger_aware/hybrid_gate"), ("core", "router_probe")]:
            pre = _pick_gqa(src, 0.25, "pre")
            post = _pick_gqa(src, 0.25, "post")
            if pre and post:
                od = _dir(pre["old_acc"], post["old_acc"])
                nd = _dir(pre["new_metric_mean"], post["new_metric_mean"])
                verdict = "HOLD" if od == nd else "**FLIP**"
                if nd == "tie":
                    verdict += " (exact tie)"
                if max(pre["new_metric_mean"], post["new_metric_mean"]) < 0.05:
                    verdict += " ⚠near-floor (verbose-prompt run → official ~0; not meaningful)"
                lines.append(
                    f"| {label} | {pre['n_rescored']} | {pre['old_acc']:.3f} | {post['old_acc']:.3f} | "
                    f"{pre['new_metric_mean']:.3f} | {post['new_metric_mean']:.3f} | {nd} | {verdict} |"
                )
            else:
                lines.append(f"| {label} | — | — | — | — | — | — | MISSING pre/post cell |")
        lines.append("")

    # ---- OCRBench official section ----
    if OCRBENCH in selected:
        ocr_recs = [r for r in records
                    if r["bench"] == OCRBENCH and r["new_metric_mean"] is not None]
        lines.append("## OCRBench — official per-category containment + 5-category /1000\n")
        lines.append("Official OCRBench eval (Yuliang-Liu/OCRBench; canonical mirror VLMEvalKit "
                     "`vlmeval/dataset/utils/ocrbench.py:OCRBench_eval` L8-60): per-sample containment "
                     "keyed on question_type — HME = space-insensitive AND case-SENSITIVE (no lowercase), "
                     "all other types lowercase + `\\n`→space. This is the SAME rule as the runner's "
                     "containment, so official ≈ containment (only HME case-sensitivity differs; 0 "
                     "disagreeing samples on our cells). The 10 fine types roll into the data-side 5 "
                     "categories × 200 = /1000 (TR/HTR/ST-VQA/DT-VQA/KIE).\n")
        lines.append("> **Category source:** runner `per_sample` carries no category (extras not passed "
                     "through), so each sample's category is recovered **by id** from the authoritative "
                     "`eval/full_splits/ocrbench.jsonl` (1000 rows, data-side `category` field). The "
                     "HME per-sample rule still keys on the fine `question_type` (nospace+case-sensitive).\n")
        lines.append("| Cell | Mode | keep% | n | stored | OFFICIAL acc | containment_recomp | Δ pp | Final(correct/n) | extrap /1000 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(ocr_recs, key=lambda r: (str(r["keep"]), str(r["mode"]), r["cell"])):
            keep = "—" if r["keep"] is None else f"{r['keep']*100:.0f}"
            lines.append(
                f"| {os.path.basename(r['cell'])} | {r['mode']} | {keep} | {r['n_rescored']} | "
                f"{r['old_acc']:.3f} | {r['new_metric_mean']:.3f} | {r['containment_recomputed']:.3f} | "
                f"{r['official_vs_containment_pp']:+.1f} | {r['ocr_final_score_raw']}/{r['n_rescored']} | "
                f"{r['ocr_extrap_1000']:.1f} |"
            )
        lines.append("")
        lines.append("### OCRBench 5-category breakdown (correct/total, acc) — data-side TR/HTR/ST-VQA/DT-VQA/KIE\n")
        lines.append("| Cell | Text Recognition (TR) | Handwriting TR (HTR) | Scene VQA (ST) | Doc VQA (DT) | KIE |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(ocr_recs, key=lambda r: (str(r["keep"]), str(r["mode"]), r["cell"])):
            cats = r["ocr_categories"]

            def fmt(s):
                d = cats.get(s, {"correct": 0, "total": 0, "acc": 0.0})
                return f"{d['correct']}/{d['total']} ({d['acc']:.2f})"
            lines.append(
                f"| {os.path.basename(r['cell'])} | {fmt('Text Recognition')} | "
                f"{fmt('Handwriting Text Recognition')} | {fmt('Scene Text-centric VQA')} | "
                f"{fmt('Document Text-centric VQA')} | {fmt('Key Information Extraction')} |"
            )
        lines.append("")
        lines.append("### OCRBench pre/post verdict (keep=25%, l2/std: C=pre vs B=post)\n")

        def _pick_ocr(keep, mode):
            cands = [r for r in ocr_recs
                     if r["mode"] == mode and not r["visionzip_style"]
                     and abs((r["keep"] if r["keep"] is not None else -1) - keep) < 1e-6]
            # prefer canonical l2 selector (over attn), then largest n
            l2 = [r for r in cands if (r["selector"] or "l2") == "l2"]
            pool = l2 or cands
            return max(pool, key=lambda r: r["n_rescored"]) if pool else None

        pre = _pick_ocr(0.25, "pre")
        post = _pick_ocr(0.25, "post")
        if pre and post:
            od = _dir(pre["old_acc"], post["old_acc"])
            nd = _dir(pre["new_metric_mean"], post["new_metric_mean"])
            verdict = "HOLD" if od == nd else "**FLIP**"
            lines.append(f"- containment: pre={pre['old_acc']:.3f} vs post={post['old_acc']:.3f} → {od}")
            lines.append(f"- **official**: pre={pre['new_metric_mean']:.3f} vs post="
                         f"{post['new_metric_mean']:.3f} → {nd}  ⇒ {verdict} "
                         f"(Δ off−cont: pre {pre['official_vs_containment_pp']:+.1f}pp, "
                         f"post {post['official_vs_containment_pp']:+.1f}pp)")
        else:
            lines.append("- MISSING pre/post ocrbench cell at keep=25%.")
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

    if GQA in selected:
        gqa_recs = [r for r in records if r["bench"] == GQA and r["new_metric_mean"] is not None]
        print(f"--- GQA official exact-match ({len(gqa_recs)} cells) ---")
        for r in sorted(gqa_recs, key=lambda r: (r["source"], str(r["mode"]), r["cell"])):
            print(f"  {os.path.basename(r['cell'])[:42]:42s} [{r['source']}/{r['mode']}] "
                  f"official={r['new_metric_mean']:.3f} cont={r['containment_recomputed']:.3f} "
                  f"stored={r['old_acc']:.3f} Δ={r['official_vs_containment_pp']:+.1f}pp n={r['n_rescored']}")
    if OCRBENCH in selected:
        ocr_recs = [r for r in records if r["bench"] == OCRBENCH and r["new_metric_mean"] is not None]
        print(f"--- OCRBench official ({len(ocr_recs)} cells) ---")
        for r in sorted(ocr_recs, key=lambda r: (str(r["mode"]), r["cell"])):
            print(f"  {os.path.basename(r['cell'])[:42]:42s} [{r['mode']}] "
                  f"official={r['new_metric_mean']:.3f} cont={r['containment_recomputed']:.3f} "
                  f"stored={r['old_acc']:.3f} Δ={r['official_vs_containment_pp']:+.1f}pp "
                  f"extrap/1000={r['ocr_extrap_1000']:.0f} n={r['n_rescored']}")


if __name__ == "__main__":
    main()
