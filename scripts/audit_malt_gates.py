#!/usr/bin/env python
"""MALT adaptive-gate feasibility audit -- PART 3/4: G0/G1 predictability audit +
offline 3-stage policy simulation (CPU only).

Models allowed by the audit charter: single-feature threshold, logistic
regression, max_depth<=2 decision tree.  NO MLP / embeddings / VLM finetune.
All standardization, feature selection and threshold selection happen ONLY on
the training fold; predictions are strict out-of-fold (stratified 5-fold CV
within each dataset AND leave-one-dataset-out).

Labels (official rescore, K0<->K1<->K8):
  G0  safe-to-K0 : pos=K0 correct, neg=K0 wrong & K1 correct, neutral=K0W&K1W
                   (neutral excluded from classifier training/metrics; routed
                   neutrals keep their K0-wrong label in the policy sim).
  G1  extend-to-K8: pos=K1 wrong & K8 correct (rescue), neg=everything else;
                   harmful extension = K1 correct & K8 wrong (reported).

Policy (strict OOF):  G0 safe -> K0 ; else run to K1 ; G1 extend -> K8.
Compute proxies: sum_l N_l and sum_l N_l^2 from the recorded layer_visual_counts
at the routed K (K=0 from the fresh pre rerun, K=1/8 from the sweep runs).

Output: experiments/malt_adaptive_gate_feasibility/audit.json  (all metrics,
OOF decisions, policy sim, PASS/NO-GO).
"""
import argparse
import json
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_curve)
from sklearn.tree import DecisionTreeClassifier

HERE = "/media/disk2/YZX/research/vla"
FEATDIR = f"{HERE}/experiments/malt_adaptive_gate_feasibility"
SWEEP = f"{HERE}/experiments/deferred_rbm_n200_data/sweep_per_sample.json"
OUT = f"{HERE}/experiments/malt_adaptive_gate_feasibility"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]

# feature columns (G0 pre-decoder + G1 layer-1), excluding labels/ids
FEATURES = [
    # G0 pre-decoder
    "n_img_log", "n_text_log", "q_len_log", "log_aspect",
    "l2_mean", "l2_std", "l2_cv", "l2_gini", "l2_entropy",
    "top25_mass", "topk_margin_rel", "kth_gap_rel",
    "pre_post_jaccard", "group_var_rel",
    "anchor_bbox_cov", "n_conn_comp", "anchor_ctx_cos_pre",
    # G1 layer-1 dynamic
    "t2a_mass", "t2a_head_std", "t2a_head_max",
    "t2c_mass", "t2c_head_std", "t2c_head_max",
    "a2c_mass", "a2c_head_std", "a2c_head_max",
    "ctx_entropy_all", "ctx_entropy_head_std", "ctx_entropy_img",
    "text_top5_conc", "text_top5_head_std",
    "cos_anchor_01", "cos_text_01",
    "sep_anchor_ctx_1", "sep_anchor_ctx_0",
    "attnout_norm_anchor", "attnout_norm_ctx",
    "attnout_norm_text", "attnout_norm_all",
]
N_FEAT = len(FEATURES)


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_data():
    sweep = json.load(open(SWEEP))
    sw = {(v["bench"], v["sample_id"]): v for v in sweep.values()}
    rows = []
    for b in BENCHES:
        path = f"{FEATDIR}/features_{b}.json"
        if not os.path.exists(path):
            print(f"[warn] missing {path} -- skipping bench", flush=True)
            continue
        feats = json.load(open(path))
        if not feats:
            print(f"[warn] empty {path} -- skipping bench", flush=True)
            continue
        for r in feats:
            sid = r["sample_id"]
            c = {K: r.get(f"correct_K{K}") for K in (0, 1, 8)}
            if any(x is None for x in c.values()):
                continue
            row = {f: r.get(f) for f in FEATURES}
            row.update({"bench": b, "sample_id": sid, **{f"c{K}": c[K]
                                                         for K in (0, 1, 8)}})
            rows.append(row)
    return rows


def compute_lookup():
    """bench -> sample_id -> {K: (sum_Nl, sum_Nl2)} from layer_visual_counts."""
    sweep = json.load(open(SWEEP))
    out = {b: {} for b in BENCHES}
    for v in sweep.values():
        b, K, sid = v["bench"], v["K"], v["sample_id"]
        lc = v.get("layer_visual_counts")
        if K in (1, 8) and lc:
            out[b].setdefault(sid, {})[K] = (sum(lc), sum(x * x for x in lc))
    for b in BENCHES:
        pre = json.load(open(f"{HERE}/runs/deferred_rbm/sweep_pre_{b}_n200.json"))
        for s in pre["per_sample"]:
            if s.get("skipped"):
                continue
            lc = s.get("layer_visual_counts")
            if lc:
                out[b].setdefault(str(s["id"]), {})[0] = (sum(lc), sum(x * x for x in lc))
    return out


# --------------------------------------------------------------------------- #
# strict CV helpers
# --------------------------------------------------------------------------- #
def impute_train(X_tr, X_te):
    """NaN -> train-fold median per feature (never touches test)."""
    med = np.nanmedian(X_tr, axis=0)
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    return X_tr, X_te, med


def std_train(X_tr, X_te):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    return (X_tr - mu) / sd, (X_te - mu) / sd


def fit_predict(model_name, X_tr, y_tr, X_te):
    """Return P(y=1) on test fold.  model_name: 'threshold' uses each feature's
    best single-feature logistic direction (rank by train AUC); 'logistic' and
    'tree' are sklearn."""
    if model_name == "logistic":
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    if model_name == "tree":
        m = DecisionTreeClassifier(max_depth=2, random_state=0,
                                   class_weight="balanced")
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    if model_name == "threshold":
        # single-feature: pick the feature with best train-fold AUC, use its
        # signed score (a single scalar per sample = best single feature value)
        best = None
        for j in range(X_tr.shape[1]):
            x = X_tr[:, j]
            if np.isnan(x).any():
                continue
            # sign: both directions allowed (use -x if AUC<0.5)
            try:
                fpr, tpr, _ = roc_curve(y_tr, x)
                a = auc(fpr, tpr)
            except Exception:
                continue
            a = max(a, 1 - a)
            if best is None or a > best[0]:
                best = (a, j, 1 if a == a else 0)
        j = best[1]
        # signed score: +x if AUC>0.5 else -x
        fpr, tpr, _ = roc_curve(y_tr, X_tr[:, j])
        sign = 1 if auc(fpr, tpr) >= 0.5 else -1
        s_tr = sign * X_tr[:, j]
        s_te = sign * X_te[:, j]
        # threshold selected on train: argmax precision@>=0.9 for G0 handled by
        # caller via the returned score; here we return the raw signed score
        return s_te, (best[0], FEATURES[j])


def select_threshold(score_tr, y_tr, target_precision):
    """Threshold on train score maximizing recall subject to precision >=
    target_precision (precision-prioritized).  Returns (thr, prec, rec)."""
    if np.isnan(score_tr).all():
        return 0.0, 0.0, 0.0
    s = score_tr[~np.isnan(score_tr)]
    if s.size == 0:
        return 0.0, 0.0, 0.0
    cands = sorted(set(np.quantile(s, np.linspace(0, 1, 101))))
    best = (None, 0.0, 0.0, 0.0)          # (rec, thr, prec, rec)
    fallback = None                       # best-effort: highest precision
    for thr in cands:
        pred = (score_tr >= thr).astype(int)
        tp = int(((pred == 1) & (y_tr == 1)).sum())
        fp = int(((pred == 1) & (y_tr == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / max(1, int(y_tr.sum()))
        if fallback is None or (prec > fallback[2]
                                or (prec == fallback[2] and rec > fallback[0])):
            fallback = (rec, thr, prec, rec)
        if prec >= target_precision - 1e-9:
            if best[0] is None or rec > best[0]:
                best = (rec, thr, prec, rec)
    if best[0] is None:            # nothing meets target -> best-effort precision
        best = fallback
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def gate_metrics(y, score, thr, neutral_mask=None):
    """y: 1=positive, 0=negative (neutral already dropped for classifier);
    neutral_mask: over the FULL sample list, marks samples excluded from the
    2-class metrics.  Returns dict."""
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr_neg = fp / (fp + tn) if (fp + tn) else float("nan")   # P(route|neg)
    fpr, tpr, _ = roc_curve(y, score)
    pr, rc, _ = precision_recall_curve(y, score)
    return {
        "precision": prec, "recall": rec,
        "false_safe_rate_routed": 1 - prec if not math.isnan(prec) else float("nan"),
        "false_safe_rate_neg_cond": fpr_neg,
        "n_tp": tp, "n_fp": fp, "n_fn": fn, "n_tn": tn,
        "auroc": float(auc(fpr, tpr)),
        "auprc": float(average_precision_score(y, score)),
    }


def eval_oof(model_name, X, y, target_precision):
    """Stratified 5-fold CV -> OOF scores + per-sample thresholds (strict:
    each held-out sample's decision threshold is selected on the other 4 folds
    only)."""
    from sklearn.model_selection import StratifiedKFold
    n = len(y)
    oof = np.full(n, np.nan)
    thr_per_sample = np.full(n, np.nan)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                 .split(np.zeros(n), y))
    for k in range(5):
        tr_idx, te_idx = folds[k]
        X_tr, X_te = X[tr_idx], X[te_idx]
        X_tr, X_te, _ = impute_train(X_tr, X_te)
        if model_name != "tree":
            X_tr, X_te = std_train(X_tr, X_te)
        if model_name == "threshold":
            s_te, _ = fit_predict("threshold", X_tr, y[tr_idx], X_te)
            s_tr = fit_predict("threshold", X_tr, y[tr_idx], X_tr)[0]
        else:
            s_te = fit_predict(model_name, X_tr, y[tr_idx], X_te)
            s_tr = fit_predict(model_name, X_tr, y[tr_idx], X_tr)
        thr = select_threshold(s_tr, y[tr_idx], target_precision)[0]
        oof[te_idx] = s_te
        thr_per_sample[te_idx] = thr
    return oof, thr_per_sample


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logistic,tree,threshold")
    ap.add_argument("--g0-precision", type=float, default=0.90)
    ap.add_argument("--g1-precision", type=float, default=0.50)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = load_data()
    X_all = np.array([[r[f] if r[f] is not None else np.nan for f in FEATURES]
                      for r in rows])
    bench_arr = np.array([r["bench"] for r in rows])
    sid_arr = np.array([r["sample_id"] for r in rows])
    c0 = np.array([r["c0"] for r in rows])
    c1 = np.array([r["c1"] for r in rows])
    c8 = np.array([r["c8"] for r in rows])
    compute = compute_lookup()

    # --- G0/G1 label masks ---
    g0_keep = (c0 == 1) | ((c0 == 0) & (c1 == 1))       # pos ∪ high-risk
    g0_y = (c0 == 1).astype(int)
    g0_neutral = ((c0 == 0) & (c1 == 0)).astype(bool)
    g1_y = ((c1 == 0) & (c8 == 1)).astype(int)          # rescue positives
    g1_harm = ((c1 == 1) & (c8 == 0)).astype(bool)

    report = {"n_total": int(len(rows)),
              "n_per_bench": {b: int((bench_arr == b).sum()) for b in BENCHES},
              "g0_pos_neg_neutral": {
                  b: {"pos": int((g0_y[bench_arr == b]).sum()),
                      "neg": int(((g0_keep & (g0_y == 0))[bench_arr == b]).sum()),
                      "neutral": int((g0_neutral[bench_arr == b]).sum())}
                  for b in BENCHES},
              "g1_pos_neg_harm": {
                  b: {"pos": int((g1_y[bench_arr == b]).sum()),
                      "neg": int(((g1_y == 0)[bench_arr == b]).sum()),
                      "harm": int((g1_harm[bench_arr == b]).sum())}
                  for b in BENCHES},
              "results": {}, "policy": {}}

    active_benches = [b for b in BENCHES if (bench_arr == b).sum() > 0]
    for mname in models:
        # ---------- within-dataset 5-fold CV (per dataset) ----------
        g0_oof = np.full(len(rows), np.nan)
        g1_oof = np.full(len(rows), np.nan)
        g0_thr = np.full(len(rows), np.nan)
        g1_thr = np.full(len(rows), np.nan)
        for b in active_benches:
            m = (bench_arr == b) & g0_keep
            if m.sum() < 5:
                continue
            oof, thr = eval_oof(mname, X_all[m], g0_y[m], args.g0_precision)
            g0_oof[m] = oof
            g0_thr[m] = thr
            m1 = (bench_arr == b)
            oof1, thr1 = eval_oof(mname, X_all[m1], g1_y[m1], args.g1_precision)
            g1_oof[m1] = oof1
            g1_thr[m1] = thr1
        # G0 metrics per bench (2-class) + routed stats (per-sample threshold)
        g0_res = {}
        for b in active_benches:
            m = (bench_arr == b) & g0_keep
            if m.sum() == 0:
                continue
            met = gate_metrics(g0_y[m], g0_oof[m], g0_thr[m])
            all_m = (bench_arr == b)
            pred0 = (g0_oof >= g0_thr)          # full-length; NaN -> False
            routed = all_m & pred0
            n_routed = int(routed.sum())
            n_route_correct = int((routed & (c0 == 1)).sum())
            n_route_highrisk = int((routed & (c0 == 0) & (c1 == 1)).sum())
            g0_res[b] = {**met,
                         "routed_frac": n_routed / int(all_m.sum()),
                         "routed_acc_pK0correct": (n_route_correct / n_routed
                                                   if n_routed else float("nan")),
                         "routed_n_highrisk": n_route_highrisk,
                         "routed_n_neutral": int((routed & g0_neutral).sum())}
        # G1 metrics per bench (per-sample threshold)
        g1_res = {}
        for b in active_benches:
            m = (bench_arr == b)
            met = gate_metrics(g1_y[m], g1_oof[m], g1_thr[m])
            pred1 = (g1_oof[m] >= g1_thr[m]).astype(int)
            routed = pred1 == 1
            n_r = int(routed.sum())
            n_rescue = int((routed & (g1_y[m] == 1)).sum())
            n_unnecc = int((routed & (g1_y[m] == 0)).sum())
            n_harm = int((routed & g1_harm[m]).sum())
            g1_res[b] = {**met,
                         "routed_frac": n_r / int(m.sum()),
                         "rescue_precision": (n_rescue / n_r if n_r else float("nan")),
                         "unnecessary_ext_rate": (n_unnecc / n_r if n_r else float("nan")),
                         "harmful_ext_rate": (n_harm / n_r if n_r else float("nan")),
                         "n_routed": n_r, "n_rescue": n_rescue,
                         "n_unnecessary": n_unnecc, "n_harmful": n_harm}

        # ---------- policy simulation (5-fold OOF decisions) ----------
        policy = simulate_policy(rows, bench_arr, c0, c1, c8,
                                 g0_oof, g1_oof, g0_thr, g1_thr, compute)
        report["results"][mname] = {"g0": g0_res, "g1": g1_res}
        report["policy"][mname] = policy
        # per-sample strict OOF predictions (deliverable)
        oof_recs = []
        for i in range(len(rows)):
            d0 = (not np.isnan(g0_oof[i])) and g0_oof[i] >= g0_thr[i]
            d1 = (not np.isnan(g1_oof[i])) and g1_oof[i] >= g1_thr[i]
            k = 0 if d0 else (8 if d1 else 1)
            oof_recs.append({
                "bench": bench_arr[i], "sample_id": rows[i]["sample_id"],
                "c0": int(c0[i]), "c1": int(c1[i]), "c8": int(c8[i]),
                "g0_score": None if np.isnan(g0_oof[i]) else float(g0_oof[i]),
                "g0_thr": None if np.isnan(g0_thr[i]) else float(g0_thr[i]),
                "g1_score": None if np.isnan(g1_oof[i]) else float(g1_oof[i]),
                "g1_thr": None if np.isnan(g1_thr[i]) else float(g1_thr[i]),
                "routed_K": int(k),
                "routed_correct": int(([c0, c1, c8][{0: 0, 1: 1, 8: 2}[k]])[i]),
            })
        report["results"][mname]["oof_predictions"] = oof_recs

        # ---------- leave-one-dataset-out ----------
        if len(active_benches) >= 2:
            report["results"][mname]["lodo"] = lodo(
                model=mname, rows=rows, X=X_all, bench_arr=bench_arr,
                c0=c0, c1=c1, c8=c8, g0_y=g0_y, g0_keep=g0_keep,
                g1_y=g1_y, g1_harm=g1_harm, args=args)
        else:
            report["results"][mname]["lodo"] = {}

    # ---------- fixed baselines + oracle ----------
    report["baselines"] = fixed_baselines(rows, bench_arr, c0, c1, c8, compute)

    # ---------- acceptance check ----------
    report["acceptance"] = acceptance(report)

    with open(f"{OUT}/audit.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1)[:5000])
    print(f"\n[written] {OUT}/audit.json")


def simulate_policy(rows, bench_arr, c0, c1, c8,
                    g0_oof, g1_oof, g0_thr, g1_thr, compute):
    """3-stage policy with strict OOF decisions (per-sample thresholds).
    Returns per-bench + macro."""
    ks = np.zeros(len(rows), dtype=int)
    for i in range(len(rows)):
        if not np.isnan(g0_oof[i]) and g0_oof[i] >= g0_thr[i]:
            ks[i] = 0
        elif not np.isnan(g1_oof[i]) and g1_oof[i] >= g1_thr[i]:
            ks[i] = 8
        else:
            ks[i] = 1
    # per-sample correctness at routed K
    corr = np.where(ks == 0, c0, np.where(ks == 1, c1, c8))
    # compute proxies at routed K
    sum_nl = np.zeros(len(rows)); sum_nl2 = np.zeros(len(rows))
    for i in range(len(rows)):
        b, sid, k = bench_arr[i], rows[i]["sample_id"], ks[i]
        c = compute[b].get(sid, {}).get(k)
        if c:
            sum_nl[i], sum_nl2[i] = c
        else:
            sum_nl[i] = sum_nl2[i] = np.nan
    active = [b for b in BENCHES if (bench_arr == b).sum() > 0]
    out = {"per_bench": {}, "routing_counts": {}}
    for b in active:
        m = bench_arr == b
        out["per_bench"][b] = {
            "acc": float(corr[m].mean()),
            "mean_K": float(ks[m].mean()),
            "sum_Nl": float(np.nansum(sum_nl[m])),
            "sum_Nl2": float(np.nansum(sum_nl2[m])),
            "mean_Nl": float(np.nanmean(sum_nl[m])),
            "mean_Nl2": float(np.nanmean(sum_nl2[m])),
            "n": int(m.sum()),
        }
        out["routing_counts"][b] = {
            "K0": int((ks[m] == 0).sum()), "K1": int((ks[m] == 1).sum()),
            "K8": int((ks[m] == 8).sum())}
    out["macro_acc"] = float(np.mean(
        [out["per_bench"][b]["acc"] for b in active]))
    out["macro_mean_K"] = float(np.mean(
        [out["per_bench"][b]["mean_K"] for b in active]))
    out["total_sum_Nl"] = float(np.nansum(sum_nl))
    out["total_sum_Nl2"] = float(np.nansum(sum_nl2))
    return out


def fixed_baselines(rows, bench_arr, c0, c1, c8, compute):
    active = [b for b in BENCHES if (bench_arr == b).sum() > 0]
    out = {"per_bench": {}, "macro": {}}
    for K, c in ((0, c0), (1, c1), (8, c8)):
        per = {}
        for b in active:
            m = bench_arr == b
            per[b] = {"acc": float(c[m].mean())}
        out["per_bench"][K] = per
        out["macro"][str(K)] = float(np.mean([per[b]["acc"] for b in active]))
    # compute per fixed K (sum over samples at that K)
    comp = {"sum_Nl": {}, "sum_Nl2": {}}
    for K in (0, 1, 8):
        nl = nl2 = 0.0
        for i in range(len(rows)):
            c = compute[bench_arr[i]][rows[i]["sample_id"]].get(K)
            if c:
                nl += c[0]; nl2 += c[1]
        comp["sum_Nl"][str(K)] = nl
        comp["sum_Nl2"][str(K)] = nl2
    out["compute"] = comp
    # oracle upper bound: accuracy = fraction correct at SOME K; compute routes
    # to the earliest-correct K (never-correct -> K0).
    oracle_acc = {}
    k_oracle = np.zeros(len(rows), dtype=int)
    corr_by_K = {0: c0, 1: c1, 8: c8}
    for i in range(len(rows)):
        for K in (0, 1, 8):
            if corr_by_K[K][i]:
                k_oracle[i] = K
                break
        else:
            k_oracle[i] = 0
    for b in active:
        m = bench_arr == b
        ever = np.any(np.stack([c0[m], c1[m], c8[m]], -1), axis=-1)
        oracle_acc[b] = float(ever.mean())
    out["oracle_acc"] = oracle_acc
    out["oracle_macro"] = float(np.mean(list(oracle_acc.values())))
    # oracle compute (sum_Nl2 at earliest-correct K)
    nl = nl2 = 0.0
    for i in range(len(rows)):
        c = compute[bench_arr[i]][rows[i]["sample_id"]].get(k_oracle[i])
        if c:
            nl += c[0]; nl2 += c[1]
    out["compute"]["sum_Nl"]["oracle"] = nl
    out["compute"]["sum_Nl2"]["oracle"] = nl2
    return out


def acceptance(report):
    """PASS/NO-GO per the audit charter (any single dataset drop >1pp vs fixed
    K=1 -> fail; accuracy-oriented OR efficiency-oriented pass; OOF + LODO not
    fully collapsed + model direction consistency)."""
    bl = report["baselines"]
    acc_k1 = bl["macro"]["1"]
    acc_k8 = bl["macro"]["8"]
    comp_k1 = bl["compute"]["sum_Nl2"]["1"]
    comp_k8 = bl["compute"]["sum_Nl2"]["8"]
    out = {}
    for mname, pol in report["policy"].items():
        mac = pol["macro_acc"]
        comp = pol["total_sum_Nl2"]
        # single-dataset drop check vs fixed K=1 (over benches present)
        worst_drop = max(bl["per_bench"][1][b]["acc"]
                         - pol["per_bench"][b]["acc"]
                         for b in bl["per_bench"][1])
        acc_pass = (mac >= acc_k1 + 0.010
                    and comp <= 0.75 * comp_k8
                    and worst_drop <= 0.011)
        eff_pass = (mac >= acc_k1 - 0.005
                    and comp <= 0.85 * comp_k1
                    and worst_drop <= 0.011)
        out[mname] = {
            "macro_acc": mac, "macro_acc_K1": acc_k1, "macro_acc_K8": acc_k8,
            "delta_vs_K1": mac - acc_k1,
            "total_Nl2": comp, "K1_Nl2": comp_k1, "K8_Nl2": comp_k8,
            "compute_vs_K8": comp / comp_k8 if comp_k8 else None,
            "compute_vs_K1": comp / comp_k1 if comp_k1 else None,
            "worst_single_dataset_drop_vs_K1": worst_drop,
            "accuracy_oriented_pass": bool(acc_pass),
            "efficiency_oriented_pass": bool(eff_pass),
        }
    # leave-one-dataset-out macro (policy accuracy on held-out) per model
    for mname in report["results"]:
        lodo_pol = report["results"][mname].get("lodo", {})
        vals = [v["policy_acc"] for v in lodo_pol.values()] if lodo_pol else []
        out[mname]["lodo_policy_macro"] = (float(np.mean(vals)) if vals else None)
        out[mname]["lodo_policy_per_bench"] = {
            k: v["policy_acc"] for k, v in lodo_pol.items()}
    return out


def lodo(model, rows, X, bench_arr, c0, c1, c8, g0_y, g0_keep, g1_y, g1_harm, args):
    """Leave-one-dataset-out: train on 3 datasets, predict on the held-out one.
    Reports held-out G0/G1 metrics and policy accuracy (strict)."""
    out = {}
    for held in [b for b in BENCHES if (bench_arr == b).sum() > 0]:
        tr = bench_arr != held
        te = bench_arr == held
        # G0
        trm = tr & g0_keep
        X_tr, X_te = X[trm], X[te & g0_keep]
        X_tr, X_te, _ = impute_train(X_tr, X_te)
        std_tr, std_te = std_train(X_tr, X_te) if model != "tree" else (X_tr, X_te)
        if model == "threshold":
            s_tr = fit_predict("threshold", std_tr, g0_y[trm], std_tr)[0]
            s_te = fit_predict("threshold", std_tr, g0_y[trm], std_te)[0]
        else:
            s_tr = fit_predict(model, std_tr, g0_y[trm], std_tr)
            s_te = fit_predict(model, std_tr, g0_y[trm], std_te)
        thr0 = select_threshold(s_tr, g0_y[trm], args.g0_precision)[0]
        g0_met = gate_metrics(g0_y[te & g0_keep], s_te, thr0)
        # G1
        X_tr1, X_te1 = X[tr], X[te]
        X_tr1, X_te1, _ = impute_train(X_tr1, X_te1)
        std_tr1, std_te1 = std_train(X_tr1, X_te1) if model != "tree" else (X_tr1, X_te1)
        if model == "threshold":
            s_tr1 = fit_predict("threshold", std_tr1, g1_y[tr], std_tr1)[0]
            s_te1 = fit_predict("threshold", std_tr1, g1_y[tr], std_te1)[0]
        else:
            s_tr1 = fit_predict(model, std_tr1, g1_y[tr], std_tr1)
            s_te1 = fit_predict(model, std_tr1, g1_y[tr], std_te1)
        thr1 = select_threshold(s_tr1, g1_y[tr], args.g1_precision)[0]
        g1_met = gate_metrics(g1_y[te], s_te1, thr1)
        # policy on held-out: G0 score exists only for g0_keep samples
        # (neutral samples are never routed to K0, matching the 5-fold policy)
        pred0 = np.zeros(int(te.sum()), dtype=bool)
        g0_te = g0_keep[te]
        pred0[g0_te] = s_te >= thr0
        pred1 = s_te1 >= thr1
        k = np.where(pred0, 0, np.where(pred1, 8, 1))
        corr = np.where(k == 0, c0[te], np.where(k == 1, c1[te], c8[te]))
        out[held] = {"g0": g0_met, "g1": g1_met,
                     "policy_acc": float(corr.mean()),
                     "n": int(te.sum())}
    return out


if __name__ == "__main__":
    main()
