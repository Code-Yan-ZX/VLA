#!/usr/bin/env python3
"""Zero-GPU offline prototype: can a learned boundary scorer imitate the
model's POST-stage (full-model output) importance ranking better than the
hand-set PRE L2 heuristic?

Data: runs/v3_merger_aware/survival_capture/{bench}.npz -- per-unit
{pre(L2 at merger input), post(L2 at visual output), edge(Sobel)}, offsets
(unit->image), h/w (grid). Built by scripts/mechanism_token_survival.py.

Design (scientific):
  * Teacher ranking = POST L2 (full model output view).
  * Student features (boundary-only, cheap): pre L2, edge, + position
    (normalized row/col, centeredness) + 2x2-neighborhood mean of pre/edge.
  * Model: linear (Ridge) and a shallow MLP -- both pure numpy/torch, CPU.
  * Metric: top-k kept-set Jaccard vs teacher, and Spearman rank corr,
    per image, k = round(n*0.25) (r=0.75 -> keep 25%). Baseline = PRE L2.
  * Train on a held-out IMAGE split (not unit split) -- generalization is
    the claim (unseen images), avoids leak.
  * NOTE: this validates "we can imitate POST ranking from boundary feats".
    Whether POST ranking is the RIGHT task signal is a separate question
    (correctness-flip labels); if imitation fails there is no D1 at all.

Usage: python scripts/proto_stage_distill.py [textvqa docvqa gqa]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CAP = Path("/media/disk2/YZX/research/vla/runs/v3_merger_aware/survival_capture")


def load(bench: str):
    d = np.load(CAP / f"{bench}.npz")
    pre, post, edge, offsets, h, w = (d["pre"], d["post"], d["edge"],
                                      d["offsets"], d["h"], d["w"])
    return pre, post, edge, offsets, h, w


def featurize(pre, post, edge, offsets, h, w, k_keep):
    """Per-unit feature vectors: [pre_l2, edge, row_n, col_n, rowc, colc,
    pre_neigh, edge_neigh]. Neighbor = mean over the 2x2 merge unit's own
    cells is not available here (units are 1:1 with tokens) -- approximate
    with spatial neighbors via the grid."""
    Xs, ys = [], []
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        n = e - s
        Hp, Wp = int(h[i]), int(w[i])
        Hh, Ww = Hp // 2, Wp // 2          # unit grid = patch grid / merge size 2
        if Hh * Ww != n:
            Ww = int(round(np.sqrt(n * Wp / Hp))) or 1
            Hh = n // Ww
        rows = np.arange(n) // Ww
        cols = np.arange(n) % Ww
        rn = rows / max(1, Hh - 1) if Hh > 1 else np.zeros(n)
        cn = cols / max(1, Ww - 1) if Ww > 1 else np.zeros(n)
        rowc = (rn - 0.5)
        colc = (cn - 0.5)
        p, ed = pre[s:e], edge[s:e]
        # 4-neighbor mean (grid wrap) for pre and edge
        p_neigh = np.empty(n)
        e_neigh = np.empty(n)
        for u in range(n):
            r, c = rows[u], cols[u]
            nb = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < Hh and 0 <= cc < Ww:
                    nb.append(rr * Ww + cc)
            if nb:
                p_neigh[u] = p[nb].mean()
                e_neigh[u] = ed[nb].mean()
            else:
                p_neigh[u] = p[u]
                e_neigh[u] = ed[u]
        X = np.stack([p, ed, rn, cn, rowc, colc, p_neigh, e_neigh], axis=1)
        Xs.append(X)
        ys.append(post[s:e])
    return np.concatenate(Xs), np.concatenate(ys)


def split_by_image(offsets, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    n_img = len(offsets) - 1
    perm = rng.permutation(n_img)
    cut = int(n_img * frac)
    tr = sorted(perm[:cut]); te = sorted(perm[cut:])
    tr_units = np.zeros(offsets[-1], bool)
    te_units = np.zeros(offsets[-1], bool)
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        if i in tr:
            tr_units[s:e] = True
        else:
            te_units[s:e] = True
    return tr_units, te_units


def topk_jaccard(true, pred, k):
    # per-image top-k overlap (both score vectors over the same unit slice)
    ov = []
    for s, e in zip(true["offsets"][:-1], true["offsets"][1:]):
        kk = max(1, int(round((e - s) * k)))
        t = set(np.argsort(true["post"][s:e])[-kk:])
        p = set(np.argsort(pred[s:e])[-kk:])
        ov.append(len(t & p) / kk)
    return float(np.mean(ov))


def main():
    benches = sys.argv[1:] or ["textvqa", "docvqa", "gqa"]
    K = 0.25
    for bench in benches:
        pre, post, edge, offsets, h, w = load(bench)
        X, y = featurize(pre, post, edge, offsets, h, w, K)
        # standardize
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        tr_u, te_u = split_by_image(offsets)
        Xtr, ytr = Xs[tr_u], y[tr_u]
        Xte, yte = Xs[te_u], y[te_u]
        # --- linear (Ridge) ---
        ridge = np.linalg.solve(Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1]),
                                Xtr.T @ ytr)
        pred_lin = Xte @ ridge
        # --- shallow MLP ---
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor((ytr - ytr.mean()) / (ytr.std() + 1e-9), dtype=torch.float32)
        Xv = torch.tensor(Xte, dtype=torch.float32)
        mlp = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        mlp.train()
        for _ in range(300):
            opt.zero_grad()
            loss = lossf(mlp(Xt).squeeze(-1), yt)
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            pred_mlp = mlp(Xv).squeeze(-1).numpy()
        # --- evaluate on held-out images ---
        # recompute offsets for the test-unit mask
        offs_te = [0]
        for i in range(len(offsets) - 1):
            s, e = offsets[i], offsets[i + 1]
            if te_u[s]:
                offs_te.append(offs_te[-1] + (e - s))
        offs_te = np.array(offs_te)
        te = {"post": yte, "offsets": offs_te}
        jac_pre = topk_jaccard(te, pre[te_u], K)
        jac_lin = topk_jaccard(te, pred_lin, K)
        jac_mlp = topk_jaccard(te, pred_mlp, K)
        # spearman (per image, pooled)
        def spear(a, b, offs):
            rs = []
            for i in range(len(offs) - 1):
                s, e = offs[i], offs[i + 1]
                if e - s < 4:
                    continue
                rs.append(np.corrcoef(
                    np.argsort(np.argsort(a[s:e])),
                    np.argsort(np.argsort(b[s:e])))[0, 1])
            return float(np.mean(rs))
        print(f"[{bench}] n_img={len(offsets)-1} train_units={tr_u.sum()} test_units={te_u.sum()}")
        print(f"   top-25% Jaccard vs POST: PRE-l2={jac_pre:.3f}  Ridge={jac_lin:.3f}  "
              f"MLP={jac_mlp:.3f}  (delta MLP-PRE={jac_mlp-jac_pre:+.3f})")
        print(f"   rank corr vs POST:       PRE-l2={spear(pre[te_u], yte, offs_te):.3f}  "
              f"Ridge={spear(pred_lin, yte, offs_te):.3f}  MLP={spear(pred_mlp, yte, offs_te):.3f}")


if __name__ == "__main__":
    main()
