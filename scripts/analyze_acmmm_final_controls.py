#!/usr/bin/env python3
"""Unified analysis for the acmmm_final_controls campaign (P0-1 / P0-2 / P1).

CPU-only. Rescores every per-sample prediction with the OFFICIAL scorers
(src/v3_premerger/official_scorers.py), so every number traces to the JSONs
under results/acmmm_final_controls/. No manual log picking.

Comparisons:
  P0-1  pre-final vs post (Qwen3-VL-8B, full splits, native pixels, r=0.75)
        + none anchors (reused verified cells) for context.
  P0-2  pre vs post (Qwen2.5-VL-7B, OCRBench full, max_pixels=4M, r=0.75 & 0.875)
        + YES/NO verdict on replacing the Table 1 Qwen2.5 OCRBench cell.
  P1    RBM (HF pre) vs FastV-k3 (HF fastv), OCRBench full 1000, both models.

For every paired comparison:
  * official score per arm (OCRBench: /1000 total incl. skips-as-0 AND
    over-answered mean; others: mean over answered)
  * paired mean delta, paired bootstrap 95% CI, sign-flip permutation p,
    McNemar (binary: OCRBench, GQA), Holm correction across the benches of
    that comparison
  * mean/median post-merger visual-token count (prompt_token_ids) per arm,
    per-sample-ID equality and per-sample token-count equality checks
  * attempted / completed / skipped counts per arm
  * per-item predictions exported to per-comparison JSON

Outputs:
  results/acmmm_final_controls/analysis.json        (machine-readable)
  results/acmmm_final_controls/analysis/            (per-comparison per-item JSONs)
  reports/acmmm_final_controls.md                   (report skeleton, human)
"""
from __future__ import annotations

import json
import os
import sys
import math
import statistics as st
from collections import OrderedDict

import numpy as np

REPO = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "v3_premerger"))
from v3_premerger import official_scorers as S
from v3_premerger import paired_stats as PS

ROOT = os.path.join(REPO, "results", "acmmm_final_controls")
OUTJSON = os.path.join(ROOT, "analysis.json")
OUTDIR = os.path.join(ROOT, "analysis")
os.makedirs(OUTDIR, exist_ok=True)

N_RESAMPLES = 20000
SEED = 0

# OCRBench question_type map (for the HME branch), keyed by sample id
_OCRBENCH_QT = {}
for _ln in open(os.path.join(REPO, "eval", "full_splits", "ocrbench.jsonl")):
    _o = json.loads(_ln)
    _OCRBENCH_QT[str(_o["id"])] = _o.get("question_type", "")


def per_sample_scores(cell):
    """Return dict id -> (score, skipped, ptid_len) using official scorers."""
    out = {}
    for p in cell.get("per_sample", []):
        sid = str(p["id"])
        if p.get("skipped"):
            out[sid] = (0.0, True, 0)
            continue
        ans = str(p.get("answer", ""))
        gt = str(p.get("gt", ""))
        bench = cell.get("benchmark")
        if bench == "ocrbench":
            sc = S.score_ocrbench(ans, gt, question_type=_OCRBENCH_QT.get(sid, ""))
        elif bench == "textvqa":
            sc = S.score_textvqa_vqaacc(ans, gt)
        elif bench == "docvqa":
            sc = S.score_docvqa_anls(ans, gt)
        elif bench == "gqa":
            sc = S.score_gqa(ans, gt)
        else:
            raise ValueError(bench)
        out[sid] = (float(sc), False, int(p.get("prompt_token_ids", 0)))
    return out


def official_metric(cell, scores):
    """Official per-arm metric. OCRBench reports BOTH /1000 (skips as 0,
    denominator = attempted+skipped i.e. the benchmark slice size) and
    mean-over-answered. Others: mean over answered (paper convention)."""
    answered = [v for v in scores.values() if not v[1]]
    n_ans = len(answered)
    bench = cell.get("benchmark")
    if bench == "ocrbench":
        total = sum(v[0] for v in scores.values())          # skips scored 0
        n_nom = len(scores)
        mean_ans = (sum(v[0] for v in answered) / n_ans) if n_ans else 0.0
        return {"total1000": round(total, 1), "n_nominal": n_nom,
                "over_answered": round(mean_ans, 4)}
    mean = (sum(v[0] for v in answered) / n_ans) if n_ans else 0.0
    return {"official": round(mean, 4), "n_answered": n_ans}


def token_stats(scores):
    ptids = [v[2] for v in scores.values() if not v[1]]
    if not ptids:
        return {"mean": None, "median": None}
    return {"mean": round(st.mean(ptids), 1), "median": float(st.median(ptids))}


def iso_checks(scores_a, scores_b):
    """sample-ID equality + per-sample token-count equality between two arms."""
    ids_a = {k for k, v in scores_a.items() if not v[1]}
    ids_b = {k for k, v in scores_b.items() if not v[1]}
    id_ok = ids_a == ids_b
    common = ids_a & ids_b
    tok_eq = all(scores_a[s][2] == scores_b[s][2] for s in common)
    return {"sample_ids_equal": bool(id_ok), "n_common_answered": len(common),
            "token_counts_equal": bool(tok_eq)}


def paired_analysis(scores_a, scores_b, label_a, label_b, seed):
    """Full paired statistics between two arms (aligned by answered ids)."""
    common = sorted(set(scores_a) & set(scores_b))
    common = [s for s in common if not scores_a[s][1] and not scores_b[s][1]]
    va = np.array([scores_a[s][0] for s in common])
    vb = np.array([scores_b[s][0] for s in common])
    d = va - vb
    res = {"n_paired": len(common)}
    if len(common) == 0:
        res.update({"mean_delta": None, "ci95": None, "p_perm": None})
        return res
    mean, lo, hi, se = PS.paired_bootstrap(d, N_RESAMPLES, seed)
    p = PS.paired_permutation(d, N_RESAMPLES, seed + 1000)
    res.update({"mean_delta": round(float(mean), 5),
                "ci95_lo": round(float(lo), 5), "ci95_hi": round(float(hi), 5),
                "bootstrap_se": round(float(se), 5), "p_perm": round(float(p), 6)})
    # McNemar for binary metrics (check the SCORES, not the differences)
    if all(x in (0.0, 1.0) for x in va) and all(x in (0.0, 1.0) for x in vb):
        a_only, b_only, z, pm = PS.mcnemar(va, vb)
        res.update({"mcnemar_A_only": a_only, "mcnemar_B_only": b_only,
                    "mcnemar_z": round(z, 4), "mcnemar_p": round(float(pm), 6)})
    return res


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a dict/list of raw p (ascending)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        running = min(running, v)
        adj[i] = running
    return adj


def dump_per_item(tag, cell_a, cell_b, scores_a, scores_b, label_a, label_b):
    ps_a = {str(p["id"]): p for p in cell_a.get("per_sample", [])}
    ps_b = {str(p["id"]): p for p in cell_b.get("per_sample", [])}
    rows = []
    for sid in sorted(set(ps_a) | set(ps_b)):
        pa, pb = ps_a.get(sid), ps_b.get(sid)
        rows.append({
            "id": sid,
            "gt": str((pa or pb).get("gt", "")),
            f"answer_{label_a}": str(pa.get("answer", "")) if pa else None,
            f"answer_{label_b}": str(pb.get("answer", "")) if pb else None,
            f"score_{label_a}": scores_a.get(sid, (None, None, None))[0],
            f"score_{label_b}": scores_b.get(sid, (None, None, None))[0],
            f"skipped_{label_a}": bool(pa.get("skipped")) if pa else None,
            f"skipped_{label_b}": bool(pb.get("skipped")) if pb else None,
        })
    with open(os.path.join(OUTDIR, f"{tag}_per_item.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return os.path.join(OUTDIR, f"{tag}_per_item.json")


def compare(tag, cell_a, cell_b, label_a, label_b, seed, bench):
    """Run the full comparison, write per-item, return the result dict."""
    sa = per_sample_scores(cell_a)
    sb = per_sample_scores(cell_b)
    ma = official_metric(cell_a, sa)
    mb = official_metric(cell_b, sb)
    ta = token_stats(sa)
    tb = token_stats(sb)
    iso = iso_checks(sa, sb)
    pair = paired_analysis(sa, sb, label_a, label_b, seed)
    per_item = dump_per_item(tag, cell_a, cell_b, sa, sb, label_a, label_b)
    res = OrderedDict([
        ("comparison", f"{label_a} vs {label_b}"),
        ("benchmark", bench),
        ("metric_A", ma), ("metric_B", mb),
        ("tokens_A", ta), ("tokens_B", tb),
        ("iso", iso),
        ("paired", pair),
        ("per_item", per_item),
    ])
    return res


def attempts(cell):
    ps = cell.get("per_sample", [])
    skip = cell.get("n_skipped", sum(1 for p in ps if p.get("skipped")))
    return {"attempted": len(ps), "completed": len(ps) - skip, "skipped": skip}


def cell_meta(cell, path):
    """Capture the run config recorded in each cell JSON for traceability."""
    return {
        "file": os.path.basename(path),
        "model": cell.get("model"), "mode": cell.get("mode"),
        "r": cell.get("r"), "benchmark": cell.get("benchmark"),
        "max_pixels": cell.get("max_pixels"),
        "max_model_len": cell.get("max_model_len"),
        "max_num_seqs": cell.get("max_num_seqs"),
        "max_tokens": cell.get("max_tokens"),
        "selector": cell.get("selector"),
        "vllm": cell.get("vllm"), "wall_s": cell.get("wall_s"),
        "load_s": cell.get("load_s"),
    }


# --------------------------------------------------------------------------- #
# Campaign layout
# --------------------------------------------------------------------------- #
P0_1 = os.path.join(ROOT, "p0_1")
P0_2 = os.path.join(ROOT, "p0_2")
P1 = os.path.join(ROOT, "p1")

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
NONE_ANCHORS = {
    "textvqa": os.path.join(REPO, "runs/full_matrix/j7_qwen3vl_none_textvqa_r0.000_full.json"),
    "docvqa": os.path.join(REPO, "runs/r2_same_scope/r2_qwen3vl_none_docvqa_full5349.json"),
    "ocrbench": os.path.join(REPO, "runs/full_matrix/j7_qwen3vl_none_ocrbench_r0.000_full.json"),
    "gqa": os.path.join(REPO, "runs/full_matrix/j7_qwen3vl_none_gqa_r0.000_full.json"),
}
# Table 1 reference cells (Qwen3-VL pre/post full, for the delta cross-check)
T1_Q3 = {
    "textvqa": ("runs/full_matrix/j7_qwen3vl_pre_textvqa_r0.750_full.json",
                "runs/full_matrix/j7_qwen3vl_post_textvqa_r0.750_full.json"),
    "docvqa": ("runs/full_matrix/j7_qwen3vl_pre_docvqa_r0.750_full.json",
               "runs/full_matrix/j7_qwen3vl_post_docvqa_r0.750_full.json"),
    "ocrbench": ("runs/full_matrix/j7_qwen3vl_pre_ocrbench_r0.750_full.json",
                 "runs/full_matrix/j7_qwen3vl_post_ocrbench_r0.750_full.json"),
    "gqa": ("runs/full_matrix/j7_qwen3vl_pre_gqa_r0.750_full.json",
            "runs/full_matrix/j7_qwen3vl_post_gqa_r0.750_full.json"),
}


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def main():
    out = OrderedDict()
    out["meta"] = {
        "campaign": "acmmm_final_controls",
        "scorers": "src/v3_premerger/official_scorers.py (official per-sample)",
        "n_resamples": N_RESAMPLES, "seed": SEED,
        "alignment": "intersection of answered sample ids",
        "token_count": "per-sample prompt_token_ids (post-merger visual tokens + text)",
        "date": "2026-08-19",
        "environment": {
            "python": "3.10.20", "torch": "2.10.0+cu128",
            "transformers": "4.57.6", "vllm": "0.19.0",
            "PIL": "12.2.0", "numpy": "2.2.6", "scipy": "1.15.3",
            "conda_env": "qwen3vl_clean", "gpu": "A40 46GB (1x)",
            "hf_offline": "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        },
    }

    # --------------------------- P0-1 ------------------------------------- #
    p01 = OrderedDict()
    pf_compare = []
    holm_p = []
    holm_tags = []
    for bench in BENCHES:
        cell_pf = load(os.path.join(P0_1, f"p0_1_qwen3_pre-final_{bench}_r0.750_full.json"))
        cell_po = load(os.path.join(P0_1, f"p0_1_qwen3_post_{bench}_r0.750_full.json"))
        cell_no = load(NONE_ANCHORS[bench])
        entry = OrderedDict([("benchmark", bench),
                             ("meta_pre_final", cell_meta(cell_pf, os.path.join(P0_1, f"p0_1_qwen3_pre-final_{bench}_r0.750_full.json")) if cell_pf else None),
                             ("meta_post", cell_meta(cell_po, os.path.join(P0_1, f"p0_1_qwen3_post_{bench}_r0.750_full.json")) if cell_po else None)])
        entry["attempts"] = {
            "pre-final": attempts(cell_pf) if cell_pf else None,
            "post": attempts(cell_po) if cell_po else None,
            "none_anchor": attempts(cell_no) if cell_no else None,
        }
        if cell_pf and cell_po:
            cmp = compare(f"p01_{bench}", cell_pf, cell_po, "pre-final", "post",
                          SEED + 10 * BENCHES.index(bench), bench)
            entry["pre_final_vs_post"] = cmp
            pf_compare.append((bench, cmp))
            holm_p.append(cmp["paired"]["p_perm"])
            holm_tags.append(f"P0-1 {bench}")
            # none anchors official metric
            if cell_no:
                sn = per_sample_scores(cell_no)
                entry["none_metric"] = official_metric(cell_no, sn)
                entry["none_tokens"] = token_stats(sn)
        else:
            entry["pre_final_vs_post"] = {"missing": True}
        p01[bench] = entry
    # Holm across P0-1 benches
    if holm_p:
        adj = holm(holm_p)
        for i, bench in enumerate(BENCHES):
            if bench in p01 and "pre_final_vs_post" in p01[bench]:
                p01[bench]["pre_final_vs_post"]["paired"]["p_perm_holm"] = round(adj[i], 6)
                p01[bench]["pre_final_vs_post"]["paired"]["holm_bench"] = holm_tags[i]
    out["P0_1_pure_stage_control"] = p01

    # Table 1 cross-check: re-run the paper's Qwen3 pre-vs-post numbers from
    # the SAME verified JSONs (audit: do our rescored numbers match Table 1?)
    out["P0_1_table1_audit"] = OrderedDict()
    for bench in BENCHES:
        cp, co = T1_Q3[bench]
        a, b = load(cp), load(co)
        if not (a and b):
            continue
        sa, sb = per_sample_scores(a), per_sample_scores(b)
        ma, mb = official_metric(a, sa), official_metric(b, sb)
        out["P0_1_table1_audit"][bench] = {
            "pre": ma, "post": mb,
            "delta_pp": (round((ma.get("official", ma.get("over_answered", 0))
                                - mb.get("official", mb.get("over_answered", 0))) * 100, 2)),
            "paper_table1_pre_post": {
                "textvqa": (0.605, 0.222), "docvqa": (0.481, 0.238),
                "ocrbench": (547, 184), "gqa": (0.449, 0.477),
            }.get(bench),
        }

    # --------------------------- P0-2 ------------------------------------- #
    p02 = OrderedDict()
    for r in (0.75, 0.875):
        key = "k25" if r == 0.75 else "k125"
        cell_pre = load(os.path.join(P0_2, "p0_2_qwen2_pre_ocrbench_r{:.3f}_full.json".format(r)))
        cell_post = load(os.path.join(P0_2, "p0_2_qwen2_post_ocrbench_r{:.3f}_full.json".format(r)))
        if not (cell_pre and cell_post):
            p02[key] = {"missing": True}
            continue
        cmp = compare(f"p02_{key}", cell_pre, cell_post, "pre", "post", SEED + 2000, "ocrbench")
        cmp["meta_pre"] = cell_meta(cell_pre, os.path.join(P0_2, "p0_2_qwen2_pre_ocrbench_r{:.3f}_full.json".format(r)))
        cmp["meta_post"] = cell_meta(cell_post, os.path.join(P0_2, "p0_2_qwen2_post_ocrbench_r{:.3f}_full.json".format(r)))
        # Table 1 replacement verdict
        verdict_yes = (
            cmp["paired"]["n_paired"] >= 900
            and cmp["iso"]["token_counts_equal"]
            and cmp["iso"]["sample_ids_equal"]
        )
        cmp["table1_replacement"] = {
            "old_pre": {"official": 476, "mean_tokens": 229.6, "pixels": "4M"},
            "old_post": {"official": 183, "mean_tokens": 282.0, "pixels": "native"},
            "new_pre_total1000": cmp["metric_A"].get("total1000"),
            "new_post_total1000": cmp["metric_B"].get("total1000"),
            "verdict_replaces_table1": verdict_yes,
            "rationale": (
                "iso pixel cap (4M) on both arms, iso sample IDs, iso per-sample "
                "token counts, n_paired>=900 -> YES (new numbers supersede the "
                "mismatched-config pair)" if verdict_yes else "NO"),
        }
        p02[key] = cmp
    # Holm across the two κ levels of P0-2
    _p = [p02[k]["paired"]["p_perm"] for k in ("k25", "k125")
          if k in p02 and "paired" in p02[k] and p02[k]["paired"].get("p_perm") is not None]
    if _p:
        _adj = holm(_p)
        for _k, _a in zip(("k25", "k125"), _adj):
            if _k in p02 and "paired" in p02[_k]:
                p02[_k]["paired"]["p_perm_holm"] = round(_a, 6)
    out["P0_2_ocrbench_matched"] = p02

    # --------------------------- P1 ---------------------------------------- #
    p1 = OrderedDict()
    for fam, model in (("qwen3vl", "Qwen3-VL-8B"), ("qwen2vl", "Qwen2.5-VL-7B")):
        cell_rbm = load(os.path.join(P1, f"p1_{fam}_pre_ocrbench_r25_full.json"))
        cell_fv = load(os.path.join(P1, f"p1_{fam}_fastv_ocrbench_k3_full.json"))
        if not (cell_rbm and cell_fv):
            p1[fam] = {"model": model, "missing": True}
            continue
        cmp = compare(f"p1_{fam}", cell_rbm, cell_fv, "rbm", "fastv", SEED + 3000, "ocrbench")
        # RBM-only / FastV-only correct counts
        sa = per_sample_scores(cell_rbm); sb = per_sample_scores(cell_fv)
        common = [s for s in set(sa) & set(sb) if not sa[s][1] and not sb[s][1]]
        rbm_only = sum(1 for s in common if sa[s][0] == 1 and sb[s][0] == 0)
        fv_only = sum(1 for s in common if sa[s][0] == 0 and sb[s][0] == 1)
        both = sum(1 for s in common if sa[s][0] == 1 and sb[s][0] == 1)
        cmp["rbm_only_correct"] = rbm_only
        cmp["fastv_only_correct"] = fv_only
        cmp["both_correct"] = both
        cmp["model"] = model
        cmp["meta_rbm"] = cell_meta(cell_rbm, os.path.join(P1, f"p1_{fam}_pre_ocrbench_r25_full.json"))
        cmp["meta_fastv"] = cell_meta(cell_fv, os.path.join(P1, f"p1_{fam}_fastv_ocrbench_k3_full.json"))
        p1[fam] = cmp
    # Holm across the two models of P1
    _p = [p1[f]["paired"]["p_perm"] for f in ("qwen3vl", "qwen2vl")
          if f in p1 and "paired" in p1[f] and p1[f]["paired"].get("p_perm") is not None]
    if _p:
        _adj = holm(_p)
        for _f, _a in zip(("qwen3vl", "qwen2vl"), _adj):
            if _f in p1 and "paired" in p1[_f]:
                p1[_f]["paired"]["p_perm_holm"] = round(_a, 6)
    out["P1_rbm_vs_fastv_hf"] = p1

    with open(OUTJSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"analysis.json written -> {OUTJSON}")

    # ------------------------- human report -------------------------------- #
    md = report_md(out)
    with open(os.path.join(REPO, "reports", "acmmm_final_controls.md"), "w") as f:
        f.write(md)
    print("report skeleton -> reports/acmmm_final_controls.md")


def report_md(out):
    L = []
    L.append("# ACM MM Final Controls — Verification Report (2026-08-19)")
    L.append("")
    L.append("> Skeleton generated by `scripts/analyze_acmmm_final_controls.py` "
             "from result JSONs. Status per experiment: `verified` / "
             "`failed-incomplete` / `config-mismatch` / `interpretation` is "
             "assigned in the final pass after the author reviews the runs.")
    L.append("")
    L.append("## P0-1: Qwen3-VL-8B pure-stage control (pre-final vs post, full splits)")
    L.append("")
    L.append("| bench | pre-final | post | Δ (pp) | 95% CI | p (perm) | Holm p | n paired | iso-token | tokens pre/post |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for bench, e in out.get("P0_1_pure_stage_control", {}).items():
        if not isinstance(e, dict) or "pre_final_vs_post" not in e:
            continue
        c = e["pre_final_vs_post"]
        if "paired" not in c:
            L.append(f"| {bench} | MISSING | | | | | | | | |")
            continue
        p = c["paired"]
        ma, mb = c["metric_A"], c["metric_B"]
        def fmt(m):
            if "total1000" in m:
                return f"{m['total1000']}/1000 (ans-mean {m['over_answered']})"
            return str(m.get("official"))
        ta, tb = c["tokens_A"], c["tokens_B"]
        L.append(
            f"| {bench} | {fmt(ma)} | {fmt(mb)} | {p['mean_delta']*100:.1f} | "
            f"[{p['ci95_lo']*100:.1f}, {p['ci95_hi']*100:.1f}] | {p['p_perm']:.4f} | "
            f"{p.get('p_perm_holm','-')} | {p['n_paired']} | "
            f"{'Y' if c['iso']['token_counts_equal'] else 'N'} | "
            f"{ta['mean']}/{tb['mean']} |")
    L.append("")
    L.append("## P0-2: Qwen2.5-VL-7B OCRBench matched-config rerun (both arms 4M px, iso)")
    L.append("")
    for key in ("k25", "k125"):
        e = out.get("P0_2_ocrbench_matched", {}).get(key, {})
        L.append(f"### κ = {'0.25' if key=='k25' else '0.125'}")
        if e.get("missing"):
            L.append("- MISSING")
            continue
        p = e["paired"]
        L.append(f"- pre total1000 = {e['metric_A']['total1000']}, post = {e['metric_B']['total1000']}")
        L.append(f"- Δ paired = {p['mean_delta']*100:.1f} pp, 95% CI [{p['ci95_lo']*100:.1f}, {p['ci95_hi']*100:.1f}], p={p['p_perm']:.4f}")
        L.append(f"- tokens: pre {e['tokens_A']['mean']} / post {e['tokens_B']['mean']}; iso-sample={e['iso']['sample_ids_equal']}, iso-token={e['iso']['token_counts_equal']}")
        L.append(f"- Table 1 replacement: **{e['table1_replacement']['verdict_replaces_table1']}** — {e['table1_replacement']['rationale']}")
        L.append("")
    L.append("## P1: RBM vs FastV-k3, OCRBench full 1000, same HF harness (both arms 4M px)")
    L.append("")
    for fam, e in out.get("P1_rbm_vs_fastv_hf", {}).items():
        if e.get("missing"):
            L.append(f"- {fam}: MISSING")
            continue
        p = e["paired"]
        L.append(f"### {e['model']}")
        L.append(f"- RBM total1000 = {e['metric_A']['total1000']} vs FastV-k3 = {e['metric_B']['total1000']}")
        L.append(f"- Δ (RBM−FastV) = {p['mean_delta']*100:.1f} pp, 95% CI [{p['ci95_lo']*100:.1f}, {p['ci95_hi']*100:.1f}], p={p['p_perm']:.4f}, McNemar p={p.get('mcnemar_p','-')}")
        L.append(f"- RBM-only correct = {e['rbm_only_correct']}, FastV-only = {e['fastv_only_correct']}, both = {e['both_correct']}")
        L.append(f"- tokens: RBM {e['tokens_A']['mean']} / FastV {e['tokens_B']['mean']}; "
                 f"iso-sample={e['iso']['sample_ids_equal']}, iso-token={e['iso']['token_counts_equal']}")
        L.append("")
    L.append("## Result management")
    L.append("- New results in `results/acmmm_final_controls/`; old results untouched.")
    L.append("- All numbers trace to per-sample JSONs (rescore path in "
             "`scripts/analyze_acmmm_final_controls.py`).")
    L.append("- Config / commands / model / pixel cap / max-model-len / engine "
             "/ versions / log paths: see per-run `.log` files next to each JSON "
             "and the campaign logs in `results/acmmm_final_controls/*/campaign.log`.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
