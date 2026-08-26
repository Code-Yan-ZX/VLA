#!/usr/bin/env python3
"""Task 4 — Adaptive stage router: OFFLINE analysis, no new GPU.

Joins Agent B's per-image capture (runs/v3_merger_aware/survival_capture/*.npz:
per-unit PRE/POST L2 scores + Sobel edge, deterministic seed-0 sample of n=64
images/bench from the *_200 subsets) with per-image CORRECTNESS of pre- vs
post-merger pruning under OFFICIAL metrics:

  textvqa -> VQA-acc (official_scorers.score_textvqa_vqaacc), correctness from
             runs/v3_merger_aware/rescore_rerun/{pre,post}_textvqa_r0.750_l2_n200
  docvqa  -> ANLS (official_scorers.score_docvqa_anls), rescore_rerun docvqa
  gqa     -> exact-match (runner score_gqa, per_sample.correct), from the
             cap64 cells runs/v3_merger_aware/router/{pre,post}_gqa_cap64_*.json
             (the 64 captured ids, re-run under the short-answer prompt)

Per-image router signal: disagreement = 1 - Spearman(pre, post) over ALL units
(+ 1 - Jaccard@k and mean Sobel edge as variants). Routers compared per-bench
and pooled (n=64/bench):
  * always-pre, always-post
  * ORACLE (per-image best of pre/post — upper bound)
  * ptid-threshold router (OLD signal: prompt-token count; route pre if
    ptid_pre >= threshold, sweep threshold)
  * disagreement-router (route pre if disagreement > tau, else post; sweep tau)
  * text-gated disagreement-router (route pre if disagreement > tau AND
    mean_edge > edge quantile — the mechanism's high-dis x high-text rule)

Thresholds are tuned on the SAME pooled sample (in-sample sweep; reported as
sensitivity, not as an out-of-sample claim). Mechanism prediction: high-
disagreement images are text-dense images where pre wins, so the disagreement
router should beat the ptid router and approach the oracle.

Output: runs/v3_merger_aware/router/router_comparison.json + markdown tables.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO = "/media/disk2/YZX/research/vla"
CAP = os.path.join(REPO, "runs/v3_merger_aware/survival_capture")
RR = os.path.join(REPO, "runs/v3_merger_aware/rescore_rerun")
ROUTER = os.path.join(REPO, "runs/v3_merger_aware/router")
sys.path.insert(0, os.path.join(REPO, "src/v3_premerger"))
from official_scorers import score_textvqa_vqaacc, score_docvqa_anls  # noqa: E402

R_KEEP = 0.75
OFFICIAL = {"textvqa": score_textvqa_vqaacc, "docvqa": score_docvqa_anls}


def se(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def load_capture(bench: str):
    """Per-image dict: id -> {pre, post, edge, n_units, dis, jaccard, mean_edge, k}."""
    d = np.load(os.path.join(CAP, f"{bench}.npz"), allow_pickle=True)
    pre, post, edge = d["pre"], d["post"], d["edge"]
    offs, ids = d["offsets"], [str(x) for x in d["ids"]]
    out = {}
    for i, sid in enumerate(ids):
        lo, hi = int(offs[i]), int(offs[i + 1])
        p, q, e = pre[lo:hi].astype(np.float64), post[lo:hi].astype(np.float64), edge[lo:hi].astype(np.float64)
        f = p.shape[0]
        k = max(1, int(round(f * (1.0 - R_KEEP))))
        top_p = set(np.argsort(-p)[:k].tolist())
        top_q = set(np.argsort(-q)[:k].tolist())
        rho = spearmanr(p, q).statistic if f > 1 else float("nan")
        out[sid] = {"pre": p, "post": q, "edge": e, "n_units": f, "k": k,
                    "dis": 1.0 - float(rho),
                    "jaccard": len(top_p & top_q) / k,
                    "mean_edge": float(e.mean())}
    return out


def load_correctness(bench: str):
    """id -> {pre: score, post: score, ptid_pre, ptid_post} under official metric."""
    out = {}
    if bench in OFFICIAL:
        scorer = OFFICIAL[bench]
        cells = {"pre": f"pre_{bench}_r0.750_l2_n200.json",
                 "post": f"post_{bench}_r0.750_l2_n200.json"}
        data = {m: json.load(open(os.path.join(RR, p)))["per_sample"]
                for m, p in cells.items()}
        for m, ps in data.items():
            for s in ps:
                if s.get("skipped"):
                    continue
                row = out.setdefault(str(s["id"]), {})
                row[m] = float(scorer(s["answer"], s["gt"]))
                row[f"ptid_{m}"] = int(s.get("prompt_token_ids", 0))
    else:  # gqa: cap64 cells, runner score_gqa correctness already per-sample
        cells = {"pre": "pre_gqa_cap64_r0.750_l2_n64.json",
                 "post": "post_gqa_cap64_r0.750_l2_n64.json"}
        for m, p in cells.items():
            path = os.path.join(ROUTER, p)
            if not os.path.exists(path):
                print(f"[router] WARNING: missing {p} (gqa correctness); "
                      f"gqa joins will be empty until the cap64 cells exist")
                continue
            for s in json.load(open(path))["per_sample"]:
                if s.get("skipped"):
                    continue
                row = out.setdefault(str(s["id"]), {})
                row[m] = float(s["correct"])
                row[f"ptid_{m}"] = int(s.get("prompt_token_ids", 0))
    return out


def build_rows():
    rows = []
    for bench in ("textvqa", "docvqa", "gqa"):
        cap = load_capture(bench)
        corr = load_correctness(bench)
        joined = 0
        for sid, c in cap.items():
            if sid in corr and "pre" in corr[sid] and "post" in corr[sid]:
                rows.append({"bench": bench, "id": sid, **c, **corr[sid]})
                joined += 1
        print(f"[router] {bench}: captured={len(cap)} correct={len(corr)} joined={joined}")
    return rows


def eval_router(rows, route_pre_fn):
    """Router accuracy + fraction routed to pre."""
    scores = [r["pre"] if route_pre_fn(r) else r["post"] for r in rows]
    n_pre = sum(1 for r in rows if route_pre_fn(r))
    return float(np.mean(scores)), n_pre / len(rows) if rows else 0.0


def sweep_threshold(rows, key, label, direction="gt"):
    """Sweep threshold over pooled quantiles of `key`; route pre if key>thresh
    (direction='gt') or key<thresh ('lt'). Returns list of result dicts + best."""
    vals = sorted({r[key] for r in rows})
    qs = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 19)))
    res = []
    for t in qs:
        fn = (lambda r, t=t: r[key] > t) if direction == "gt" else (lambda r, t=t: r[key] < t)
        acc, fpre = eval_router(rows, fn)
        res.append({"signal": label, "threshold": round(float(t), 4),
                    "acc": acc, "frac_pre": fpre})
    best = max(res, key=lambda x: x["acc"])
    return res, best


def main():
    rows = build_rows()
    pools = {"textvqa": [r for r in rows if r["bench"] == "textvqa"],
             "docvqa": [r for r in rows if r["bench"] == "docvqa"],
             "gqa": [r for r in rows if r["bench"] == "gqa"],
             "pooled": rows}

    report = {"n_per_bench": {k: len(v) for k, v in pools.items()}, "routers": {}}

    def acc_se(rows_, key):
        v = [r[key] for r in rows_]
        return round(float(np.mean(v)), 4), round(se(v), 4)

    for pool_name, pool in pools.items():
        if not pool:
            continue
        pre_acc, pre_se = acc_se(pool, "pre")
        post_acc, post_se = acc_se(pool, "post")
        oracle_v = [max(r["pre"], r["post"]) for r in pool]
        entry = {
            "n": len(pool),
            "always_pre": {"acc": pre_acc, "se": pre_se},
            "always_post": {"acc": post_acc, "se": post_se},
            "oracle": {"acc": round(float(np.mean(oracle_v)), 4),
                       "se": round(se(oracle_v), 4)},
        }
        # mechanism diagnostics: is high-disagreement where pre wins?
        med = float(np.median([r["dis"] for r in pool]))
        hi = [r for r in pool if r["dis"] > med]
        lo = [r for r in pool if r["dis"] <= med]
        entry["diag_by_disagreement"] = {
            "median_dis": round(med, 4),
            "high_dis_pre_minus_post": round(
                float(np.mean([r["pre"] - r["post"] for r in hi])), 4) if hi else None,
            "low_dis_pre_minus_post": round(
                float(np.mean([r["pre"] - r["post"] for r in lo])), 4) if lo else None,
            "high_dis_pre_ge_post_frac": round(
                float(np.mean([r["pre"] >= r["post"] for r in hi])), 3) if hi else None,
            "low_dis_pre_ge_post_frac": round(
                float(np.mean([r["pre"] >= r["post"] for r in lo])), 3) if lo else None,
        }
        report["routers"][pool_name] = entry

    # ---- threshold routers, tuned on the POOLED sample ----
    pooled = pools["pooled"]
    if pooled:
        ptid_sweep, ptid_best = sweep_threshold(pooled, "ptid_pre", "ptid>=t")
        dis_sweep, dis_best = sweep_threshold(pooled, "dis", "dis>t")
        jac_sweep, jac_best = sweep_threshold(pooled, "jaccard", "jaccard<t", "lt")
        # text-gated: route pre if dis>td AND mean_edge>te (coarse 5x5 grid of
        # pooled tercile/quintile cuts)
        dqs = np.unique(np.quantile([r["dis"] for r in pooled], [0.3, 0.5, 0.7]))
        eqs = np.unique(np.quantile([r["mean_edge"] for r in pooled], [0.3, 0.5, 0.7]))
        comb_best, comb_sweep = None, []
        comb_or_best, comb_or_sweep = None, []
        for td in dqs:
            for te in eqs:
                fn = lambda r, td=td, te=te: r["dis"] > td and r["mean_edge"] > te
                acc, fpre = eval_router(pooled, fn)
                rec = {"signal": "dis>td AND edge>te", "td": round(float(td), 4),
                       "te": round(float(te), 4), "acc": acc, "frac_pre": fpre}
                comb_sweep.append(rec)
                if comb_best is None or acc > comb_best["acc"]:
                    comb_best = rec
                fn2 = lambda r, td=td, te=te: r["dis"] > td or r["mean_edge"] > te
                acc2, fpre2 = eval_router(pooled, fn2)
                rec2 = {"signal": "dis>td OR edge>te", "td": round(float(td), 4),
                        "te": round(float(te), 4), "acc": acc2, "frac_pre": fpre2}
                comb_or_sweep.append(rec2)
                if comb_or_best is None or acc2 > comb_or_best["acc"]:
                    comb_or_best = rec2
        report["routers"]["pooled"].update({
            "ptid_router": ptid_best,
            "disagreement_router": dis_best,
            "jaccard_router": jac_best,
            "text_gated_disagreement_router_AND": comb_best,
            "text_gated_disagreement_router_OR": comb_or_best,
        })
        report["sweeps"] = {"ptid": ptid_sweep, "dis": dis_sweep,
                            "jaccard": jac_sweep, "text_gated_AND": comb_sweep,
                            "text_gated_OR": comb_or_sweep}

    os.makedirs(ROUTER, exist_ok=True)
    with open(os.path.join(ROUTER, "router_comparison.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- markdown table ----
    def fmt(e):
        return f"{e['acc']:.3f}±{e['se']:.3f}" if e else "NA"

    lines = ["# Router comparison (per-image stage routing; official metrics; n=64/bench)", ""]
    hdr = "| pool | n | always-pre | always-post | oracle | ptid-router | dis-router | text-gated-dis |"
    sep = "|---|---|---|---|---|---|---|---|"
    print(hdr); print(sep)
    lines += [hdr, sep]
    for pn in ("textvqa", "docvqa", "gqa", "pooled"):
        e = report["routers"].get(pn)
        if not e:
            continue
        pt = e.get("ptid_router"); dr = e.get("disagreement_router")
        tg = e.get("text_gated_disagreement_router_AND")
        tgo = e.get("text_gated_disagreement_router_OR")
        row = (f"| {pn} | {e['n']} | {fmt(e['always_pre'])} | {fmt(e['always_post'])} | "
               f"{fmt(e['oracle'])} | "
               f"{pt['acc']:.3f} (t={pt['threshold']}, pre%={pt['frac_pre']:.2f}) | "
               f"{dr['acc']:.3f} (τ={dr['threshold']}, pre%={dr['frac_pre']:.2f}) | "
               f"AND {tg['acc']:.3f} / OR {tgo['acc']:.3f} |" if pt else
               f"| {pn} | {e['n']} | {fmt(e['always_pre'])} | {fmt(e['always_post'])} | "
               f"{fmt(e['oracle'])} | - | - | - |")
        print(row)
        lines.append(row)
    with open(os.path.join(ROUTER, "router_comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # headline deltas (pooled)
    p = report["routers"].get("pooled", {})
    if p and p.get("disagreement_router"):
        print("\n[headline] pooled: "
              f"pre={p['always_pre']['acc']:.3f} post={p['always_post']['acc']:.3f} "
              f"oracle={p['oracle']['acc']:.3f} "
              f"ptid={p['ptid_router']['acc']:.3f} "
              f"dis={p['disagreement_router']['acc']:.3f} "
              f"text-gated-AND={p['text_gated_disagreement_router_AND']['acc']:.3f} "
              f"text-gated-OR={p['text_gated_disagreement_router_OR']['acc']:.3f}")
        for pn in ("textvqa", "docvqa", "gqa"):
            e = report["routers"].get(pn)
            if not e:
                continue
            d = e["diag_by_disagreement"]
            print(f"  [{pn}] high-dis pre-post={d['high_dis_pre_minus_post']} "
                  f"low-dis pre-post={d['low_dis_pre_minus_post']} "
                  f"high-dis pre>=post={d['high_dis_pre_ge_post_frac']} "
                  f"low-dis pre>=post={d['low_dis_pre_ge_post_frac']}")


if __name__ == "__main__":
    main()
