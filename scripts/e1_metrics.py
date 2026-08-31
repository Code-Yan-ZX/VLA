#!/usr/bin/env python
"""Q-RBM E1 — metrics & pre-registered GO/NO-GO evaluation (CPU only).

Reads runs/e1/{dev,heldout}_{bench}.jsonl produced by e1_causal_import.py
(schema in notes/qrbm_e1_plan.md §3 / §6) and evaluates the four
pre-registered acceptance criteria (G1-G4) on the held-out set only.

Predictors are fit on the dev split ONLY; all gate numbers come from the
held-out split. Pre-registered in notes/qrbm_e1_plan.md §4.

Usage:
  /home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/e1_metrics.py
      [--out runs/e1] [--resamples 2000] [--seed 0] [--g 8]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src/v3_premerger"))
from paired_stats import paired_bootstrap, paired_permutation  # noqa: E402

BENCHES = ["textvqa", "docvqa", "gqa", "ocrbench"]
G2_BENCHES = ["textvqa", "docvqa", "ocrbench"]
G3_BENCH = "gqa"
K_FRAC = 0.25          # nDCG@25% / Recall@25%
RIDGE_LAMBDA = 1.0     # pre-registered; no tuning
LAMBDA_Q = 0.5         # diagnostic J5-style blend weight (diagnostic only)


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def blocks_per_sample(sample, g):
    """Return (signals array (g,4), causal array (g,)) for a sample's blocks."""
    sig = np.zeros((g, 4))
    caus = np.zeros(g)
    for blk in sample["blocks"]:
        i = int(blk["g"])
        sig[i] = [blk["pre_l2"], blk["post_l2"], blk["fastv"], blk["qsim"]]
        caus[i] = blk["causal_util"]
    return sig, caus


def zscore_rows(x):
    """Per-sample z-score across rows (guard: all-equal -> 0)."""
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return (x - m) / s


def ndcg_at_k(rank_order, rel, k):
    """nDCG@k: rank_order = indices of predicted-top-k (desc rel order)."""
    idcg_idx = np.argsort(rel)[::-1][:k]
    idcg = sum(rel[i] / np.log2(r + 2) for r, i in enumerate(idcg_idx))
    if idcg <= 0:
        return 0.5  # neutral: no detectable causal signal
    dcg = sum(rel[i] / np.log2(r + 2) for r, i in enumerate(rank_order[:k]))
    return dcg / idcg


def rec_at_k(pred_top, causal_top, k):
    if len(causal_top) == 0:
        return float("nan")
    return len(set(pred_top) & set(causal_top)) / k


def ridge_fit(Xtr, ytr, lam=RIDGE_LAMBDA):
    """Centered ridge regression. Returns (w, mean)."""
    mean = Xtr.mean(axis=0)
    Xc = Xtr - mean
    yc = ytr - ytr.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    return w, mean


def ridge_pred(X, w, mean):
    return (X - mean) @ w


def run_bench(b, args, split_sets):
    dev, ho = split_sets[b]
    g = args.g
    k = max(1, int(round(K_FRAC * g)))

    # ---- pooled block features from dev ----
    Fdev, ydev = [], []
    dev_meta = []  # per (sample idx) arrays
    for s in dev:
        sig, caus = blocks_per_sample(s, g)
        z = zscore_rows(caus.reshape(1, -1))[0]
        Fdev.append(sig)
        ydev.append(z)
        dev_meta.append((sig, caus, z))
    Xdev = np.vstack(Fdev)
    ydev = np.concatenate(ydev)

    # predictor A: [pre_l2, post_l2]; predictor B: [pre_l2, post_l2, qsim]
    def feat(sig, colsa, colsb):
        return sig[:, colsa], sig[:, colsb]
    cola = [0, 1]
    colb = [0, 1, 3]  # pre_l2, post_l2, qsim

    XA = Xdev[:, cola]
    XB = Xdev[:, colb]
    wA, mA = ridge_fit(XA, ydev)
    wB, mB = ridge_fit(XB, ydev)

    # ---- held-out per-sample metrics ----
    rows = []
    for s in ho:
        sig, caus = blocks_per_sample(s, g)
        z = zscore_rows(caus.reshape(1, -1))[0]
        idx = np.arange(g)
        # rankings
        rA = np.argsort(ridge_pred(sig[:, cola], wA, mA))[::-1]
        rB = np.argsort(ridge_pred(sig[:, colb], wB, mB))[::-1]
        rPre = np.argsort(sig[:, 0])[::-1]           # pre-L2 (RBM)
        rPost = np.argsort(sig[:, 1])[::-1]          # post-L2
        rFastV = np.argsort(sig[:, 2])[::-1]         # FastV
        rQ = np.argsort(_qblend(sig))[::-1]          # diagnostic λ-blend
        causal_top = set(idx[np.argsort(z)[::-1][:k]])
        row = {
            "id": s["id"],
            "ndcg_A": ndcg_at_k(rA, z, k), "ndcg_B": ndcg_at_k(rB, z, k),
            "ndcg_pre": ndcg_at_k(rPre, z, k), "ndcg_post": ndcg_at_k(rPost, z, k),
            "ndcg_fastv": ndcg_at_k(rFastV, z, k), "ndcg_qdiag": ndcg_at_k(rQ, z, k),
            "rec_B": rec_at_k(rB[:k], causal_top, k),
            "rec_RBM": rec_at_k(rPre[:k], causal_top, k),
            "rec_FastV": rec_at_k(rFastV[:k], causal_top, k),
        }
        # per-sample signal-vs-causal Spearman (over g blocks)
        row["sp"] = {}
        for name, col in [("pre_l2", 0), ("post_l2", 1), ("fastv", 2), ("qsim", 3)]:
            sp, _ = stats.spearmanr(sig[:, col], z)
            row["sp"][name] = float(sp) if not np.isnan(sp) else 0.0
        # G4 feature: alignment = Spearman(qsim, pre_l2)
        sp_align, _ = stats.spearmanr(sig[:, 3], sig[:, 0])
        row["align"] = float(sp_align) if not np.isnan(sp_align) else 0.0
        rows.append(row)

    return {"rows": rows, "dev_n": len(dev), "ho_n": len(ho), "g": g, "k": k}


def _qblend(sig):
    """Diagnostic J5-style blend: minmax(pre_l2) + λ·minmax(qsim)."""
    def mm(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return mm(sig[:, 0]) + LAMBDA_Q * mm(sig[:, 3])


def _paired_stats(d, args, label):
    mean, lo, hi, se = paired_bootstrap(np.asarray(d, float), args.resamples, args.seed)
    p = paired_permutation(np.asarray(d, float), args.resamples, args.seed)
    return {"mean": float(mean), "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(se), "p_perm": float(p), "n": int(len(d)),
            "n_positive": int(np.sum(np.asarray(d) > 0)), "label": label}


def evaluate(results, args):
    out = {"g": args.g, "k_frac": K_FRAC, "ridge_lambda": RIDGE_LAMBDA,
           "lambda_q_diag": LAMBDA_Q, "benches": {}}

    # ---- G1: KEY - held-out ΔnDCG@25% (B − A), per bench ----
    g1_d = {}
    for b in BENCHES:
        rs = results[b]["rows"]
        d = [r["ndcg_B"] - r["ndcg_A"] for r in rs]
        g1_d[b] = _paired_stats(d, args, "G1 B-A")
    # pooled (all benches)
    g1_pool = [v for b in BENCHES for v in
               [r["ndcg_B"] - r["ndcg_A"] for r in results[b]["rows"]]]
    out["G1"] = {"per_bench": g1_d,
                 "pooled": _paired_stats(g1_pool, args, "G1 pooled B-A"),
                 "n_benches_positive": int(sum(g1_d[b]["mean"] > 0 for b in BENCHES))}

    # ---- G2: Recall@25% predictor-B vs RBM (textvqa/docvqa/ocrbench) ----
    g2 = {}
    for b in G2_BENCHES:
        rs = results[b]["rows"]
        rB = np.array([r["rec_B"] for r in rs if not np.isnan(r["rec_B"])])
        rR = np.array([r["rec_RBM"] for r in rs if not np.isnan(r["rec_RBM"])])
        # paired on common non-nan samples
        mask = [not (np.isnan(r["rec_B"]) or np.isnan(r["rec_RBM"])) for r in rs]
        dB = np.array([r["rec_B"] for r, m in zip(rs, mask) if m])
        dR = np.array([r["rec_RBM"] for r, m in zip(rs, mask) if m])
        g2[b] = {"rec_B": float(dB.mean()) if len(dB) else float("nan"),
                 "rec_RBM": float(dR.mean()) if len(dR) else float("nan"),
                 "delta": float((dB - dR).mean()) if len(dB) else float("nan"),
                 "paired_n": int(len(dB))}
    out["G2"] = g2

    # ---- G3: GQA nDCG@25% B vs RBM ----
    rs = results[G3_BENCH]["rows"]
    d = [r["ndcg_B"] - r["ndcg_pre"] for r in rs]
    out["G3"] = _paired_stats(d, args, "G3 B-RBM(gqa ndcg)")

    # ---- G4: per-sample win predictor (logistic on align) ----
    def logistic_ll(theta, X, y):
        z = theta[0] + X * theta[1]
        return -np.mean(y * z - np.logaddexp(0, z))

    def fit_logistic(Xtr, ytr):
        from scipy.optimize import minimize
        r = minimize(logistic_ll, np.zeros(2), args=(Xtr, ytr),
                     method="BFGS")
        return r.x

    g4 = {}
    for b in BENCHES:
        dev = results[b]["rows"]
        # labels: winner(FastV vs RBM) by higher causal Recall@25%
        def lab(r):
            a, b2 = r["rec_FastV"], r["rec_RBM"]
            if np.isnan(a) or np.isnan(b2):
                return None
            return 1 if a > b2 else (0 if b2 > a else None)
        Xtr = np.array([r["align"] for r in dev if lab(r) is not None])
        ytr = np.array([lab(r) for r in dev if lab(r) is not None])
        ho = results[b]["rows"]
        Xte = np.array([r["align"] for r in ho if lab(r) is not None])
        yte = np.array([lab(r) for r in ho if lab(r) is not None])
        if len(Xtr) >= 16 and len(Xte) >= 16:
            th = fit_logistic(Xtr, ytr)
            pred = (th[0] + Xte * th[1] > 0).astype(int)
            acc = (pred == yte).mean()
            acc_ci = _paired_stats((pred == yte).astype(float), args, "G4 acc")
            g4[b] = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                     "acc": float(acc), "acc_ci_lo": acc_ci["ci_lo"],
                     "acc_ci_hi": acc_ci["ci_hi"]}
        else:
            g4[b] = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                     "acc": float("nan"), "note": "too few labelled samples"}
    out["G4"] = g4

    # ---- signal-vs-causal per-sample Spearman (means) ----
    out["signal_causal_spearman"] = {}
    for b in BENCHES:
        rs = results[b]["rows"]
        out["signal_causal_spearman"][b] = {
            name: float(np.mean([r["sp"][name] for r in rs]))
            for name in ["pre_l2", "post_l2", "fastv", "qsim"]}

    # ---- verdicts ----
    G1o = out["G1"]
    g1_mean = G1o["pooled"]["mean"]
    g1_ok = (g1_mean >= 0.05 and G1o["pooled"]["ci_lo"] > 0
             and G1o["n_benches_positive"] >= 3)
    g1_no = (g1_mean <= 0.02 or G1o["pooled"]["ci_lo"] <= 0)
    G2o = out["G2"]
    g2_ok = sum(1 for b in G2_BENCHES
                if G2o[b]["rec_B"] >= G2o[b]["rec_RBM"]) >= 2 and all(
        not np.isnan(G2o[b]["delta"]) and G2o[b]["delta"] >= -0.05
        for b in G2_BENCHES)
    G3o = out["G3"]
    g3_ok = G3o["mean"] >= 0.05
    G4o = out["G4"]
    ok4 = [G4o[b] for b in BENCHES if not np.isnan(G4o[b]["acc"])]
    g4_ok = len(ok4) > 0 and max(o["acc"] for o in ok4) > 0.60 and all(
        o["acc_ci_lo"] > 0.50 for o in ok4 if not np.isnan(o["acc"]))

    out["verdicts"] = {
        "G1": "GO" if g1_ok else ("NO-GO" if g1_no else "INCONCLUSIVE"),
        "G2": "GO" if g2_ok else "NO-GO",
        "G3": "GO" if g3_ok else "NO-GO",
        "G4": "GO" if g4_ok else "NO-GO",
    }
    out["global"] = ("NO-GO (G1 failed — do NOT enter scorer training)"
                     if out["verdicts"]["G1"] == "NO-GO"
                     else ("GO-PROVISIONAL (G1 passed; G2-G4 confirmatory)"
                           if out["verdicts"]["G1"] == "GO"
                           else "INCONCLUSIVE"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs/e1"))
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g", type=int, default=8)
    args = ap.parse_args()

    split_sets = {}
    missing = []
    for b in BENCHES:
        dpath = os.path.join(args.out, f"dev_{b}.jsonl")
        hpath = os.path.join(args.out, f"heldout_{b}.jsonl")
        if not (os.path.exists(dpath) and os.path.exists(hpath)):
            missing.append(b)
            continue
        split_sets[b] = (load(dpath), load(hpath))
    if missing:
        print(f"[e1-metrics] MISSING splits for: {missing} "
              f"(run e1_causal_import.py first) — evaluating only present benches",
              flush=True)
    if not split_sets:
        print("[e1-metrics] nothing to evaluate", flush=True)
        return 1

    results = {b: run_bench(b, args, split_sets) for b in split_sets}
    out = evaluate(results, args)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=== E1 gate metrics (held-out) ===", flush=True)
    print(f"G1 KEY ΔnDCG@25% (B−A) pooled: {out['G1']['pooled']['mean']:+.4f} "
          f"CI[{out['G1']['pooled']['ci_lo']:+.4f}, "
          f"{out['G1']['pooled']['ci_hi']:+.4f}] p={out['G1']['pooled']['p_perm']:.4f} "
          f"pos_benches={out['G1']['n_benches_positive']}/4 → {out['verdicts']['G1']}",
          flush=True)
    for b in BENCHES:
        g = out["G1"]["per_bench"][b]
        print(f"  {b:10s} Δ={g['mean']:+.4f} CI[{g['ci_lo']:+.4f},{g['ci_hi']:+.4f}]",
              flush=True)
    for b in G2_BENCHES:
        g2 = out["G2"][b]
        print(f"G2 {b:10s} Recall@25% B={g2['rec_B']:.4f} RBM={g2['rec_RBM']:.4f} "
              f"Δ={g2['delta']:+.4f}", flush=True)
    g3 = out["G3"]
    print(f"G3 GQA nDCG@25% B−RBM = {g3['mean']:+.4f} CI[{g3['ci_lo']:+.4f},"
          f"{g3['ci_hi']:+.4f}] → {out['verdicts']['G3']}", flush=True)
    for b in BENCHES:
        g4 = out["G4"][b]
        print(f"G4 {b:10s} win-pred acc={g4['acc']:.4f} "
              f"CI[{g4['acc_ci_lo']:.4f},{g4['acc_ci_hi']:.4f}] n_test={g4['n_test']}",
              flush=True)
    print(f"SIGNAL-vs-causal mean per-sample Spearman:", flush=True)
    for b in BENCHES:
        s = out["signal_causal_spearman"][b]
        print(f"  {b:10s} pre={s['pre_l2']:+.3f} post={s['post_l2']:+.3f} "
              f"fastv={s['fastv']:+.3f} qsim={s['qsim']:+.3f}", flush=True)
    print(f"GLOBAL: {out['global']}", flush=True)
    print(f"[e1-metrics] -> {os.path.join(args.out, 'metrics.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
