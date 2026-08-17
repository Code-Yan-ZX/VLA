#!/usr/bin/env python3
"""Train the learned boundary scorer (stage-distillation) per benchmark.

Target: POST-stage L2 ranking (the model's full-output view of unit
importance). Features = deployable boundary set (see learned_scorer.py).
Held-out IMAGE split for honest generalization check; in-sample reported
too (both are "can we imitate POST ranking from boundary feats").

Usage: python scripts/train_learned_scorer.py [textvqa docvqa gqa] --out-dir ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learned_scorer import (build_features_np, build_mlp, standardize,
                            LearnedScorer)

ROOT = Path("/media/disk2/YZX/research/vla")
CAP = ROOT / "runs/v3_merger_aware/survival_capture_lrn200"


def split_by_image(offsets, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    n_img = len(offsets) - 1
    perm = rng.permutation(n_img)
    cut = int(n_img * frac)
    tr = set(perm[:cut].tolist())
    tr_u = np.zeros(offsets[-1], bool)
    te_u = np.zeros(offsets[-1], bool)
    for i in range(n_img):
        s, e = offsets[i], offsets[i + 1]
        (tr_u if i in tr else te_u)[s:e] = True
    return tr_u, te_u


def spear(a, b, offs):
    rs = []
    for i in range(len(offs) - 1):
        s, e = offs[i], offs[i + 1]
        if e - s < 4:
            continue
        rs.append(np.corrcoef(np.argsort(np.argsort(a[s:e])),
                              np.argsort(np.argsort(b[s:e])))[0, 1])
    return float(np.mean(rs))


def topk_jaccard(post, pred, offs, k=0.25):
    ov = []
    for i in range(len(offs) - 1):
        s, e = offs[i], offs[i + 1]
        kk = max(1, int(round((e - s) * k)))
        t = set(np.argsort(post[s:e])[-kk:])
        p = set(np.argsort(pred[s:e])[-kk:])
        ov.append(len(t & p) / kk)
    return float(np.mean(ov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["textvqa", "docvqa", "gqa"])
    ap.add_argument("--out-dir", default=str(ROOT / "runs/learned_scorer"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for bench in args.bench:
        npz = np.load(CAP / f"{bench}.npz")
        pre, post, edge, efeat, var = (npz["pre"], npz["post"], npz["edge"],
                                       npz["edge_feat"], npz["var_feat"])
        offsets, h, w = npz["offsets"], npz["h"], npz["w"]
        X, _ = build_features_np(pre, efeat, var, offsets, h, w)
        y = post
        Xs, mu, sd = standardize(X)
        tr_u, te_u = split_by_image(offsets, frac=0.7, seed=args.seed)
        Xtr, ytr, Xte, yte = Xs[tr_u], y[tr_u], Xs[te_u], y[te_u]
        # test offsets (compressed to test units only)
        offs_te = [0]
        for i in range(len(offsets) - 1):
            s, e = offsets[i], offsets[i + 1]
            if te_u[s]:
                offs_te.append(offs_te[-1] + (e - s))
        offs_te = np.array(offs_te)

        Xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor((ytr - ytr.mean()) / (ytr.std() + 1e-9),
                          dtype=torch.float32)
        Xv = torch.tensor(Xte, dtype=torch.float32)
        mlp = build_mlp(X.shape[1])
        opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
        lossf = nn.MSELoss()
        for _ in range(args.epochs):
            opt.zero_grad()
            loss = lossf(mlp(Xt).squeeze(-1), yt)
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            pred_te = mlp(Xv).squeeze(-1).numpy()
            pred_all = mlp(torch.tensor(Xs, dtype=torch.float32)
                           ).squeeze(-1).numpy()

        jac_pre = topk_jaccard(post[te_u], pre[te_u], offs_te)
        jac_mlp = topk_jaccard(post[te_u], pred_te, offs_te)
        # in-sample (all units)
        offs_all = offsets
        jac_pre_all = topk_jaccard(post, pre, offs_all)
        jac_mlp_all = topk_jaccard(post, pred_all, offs_all)
        print(f"[{bench}] n_img={len(offsets)-1} (tr={int(tr_u.sum())}u "
              f"te={int(te_u.sum())}u)")
        print(f"   HELD-OUT top-25% Jaccard vs POST: PRE={jac_pre:.3f} "
              f"MLP={jac_mlp:.3f} (Δ{jac_mlp-jac_pre:+.3f})  "
              f"spear {spear(pre[te_u], post[te_u], offs_te):.3f}->"
              f"{spear(pred_te, post[te_u], offs_te):.3f}")
        print(f"   IN-SAMPLE (all) Jaccard:            PRE={jac_pre_all:.3f} "
              f"MLP={jac_mlp_all:.3f} (Δ{jac_mlp_all-jac_pre_all:+.3f})")

        # save scorer trained on ALL units (full data for deployment)
        mlp2 = build_mlp(X.shape[1])
        Xt_all = torch.tensor(Xs, dtype=torch.float32)
        yt_all = torch.tensor((y - y.mean()) / (y.std() + 1e-9),
                              dtype=torch.float32)
        opt2 = torch.optim.Adam(mlp2.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            opt2.zero_grad()
            loss = lossf(mlp2(Xt_all).squeeze(-1), yt_all)
            loss.backward()
            opt2.step()
        sc = LearnedScorer.__new__(LearnedScorer)
        sc.mu = torch.tensor(mu); sc.sd = torch.tensor(sd); sc.mlp = mlp2
        out_p = out / f"{bench}_scorer.json"
        sc.save(str(out_p))
        print(f"   saved scorer -> {out_p}")
        with open(out / f"{bench}_report.json", "w") as f:
            json.dump({"bench": bench, "jac_heldout": {"pre": jac_pre,
                      "mlp": jac_mlp}, "jac_insample": {"pre": jac_pre_all,
                      "mlp": jac_mlp_all},
                      "spear_heldout": {"pre": spear(pre[te_u], post[te_u],
                                                     offs_te),
                      "mlp": spear(pred_te, post[te_u], offs_te)}}, f,
                      indent=2)


if __name__ == "__main__":
    main()
