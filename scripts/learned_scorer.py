#!/usr/bin/env python3
"""Shared learned boundary scorer: feature building + MLP, used by BOTH the
training script (numpy) and the runner at hook time (torch).

The scorer is a tiny MLP that maps per-merge-unit boundary features to a
relevance score, trained to imitate the model's POST-stage (full-model output)
importance ranking (stage-distillation). All features are computable at the
runner's pre-merger hook from the [num_units, 4, ctx] tensor + the image grid
-- NO pixels, NO query (query-aware variant is a later step).

Deployable feature vector per unit (must be IDENTICAL train/runtime):
  [l2, edge_feat, var_feat, row_n, col_n, rowc, colc,
   n_l2, n_edge, n_var]     (10 dims)
  where edge_feat/var_feat use the runner's _score_units("edge"/"var")
  formulas (feature-space, not pixels), row_n/col_n are normalized grid
  coordinates, rowc/colc are centered versions, n_* are 4-neighbor means.

Runtime flow (runner slice_input):
  1. feats [num_units, 4, ctx] -> l2/edge/var via _score_units
  2. grids (per-image h,w from visual.forward grid_thw) -> pos + neighbor feats
  3. standardize with saved mu/sd, run MLP -> [num_units] scores -> top-k
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEAT_NAMES = ["l2", "edge", "var", "row_n", "col_n", "rowc", "colc",
              "n_l2", "n_edge", "n_var"]


def build_mlp(n_in: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


# --------------------------------------------------------------------------
# numpy feature builder (training). Recomputes l2/edge/var from per-unit
# score arrays (the capture already stored the runner's exact scores).
# Neighbor = 4-neighbor mean over the unit grid, zero-padded at borders.
# VECTORIZED (per-unit python loop was ~minutes for 700k docvqa units).
# --------------------------------------------------------------------------
def _neigh_mean_np(a, Hh, Ww):
    """4-neighbor mean of a (Hh*Ww,) array, zero-padded borders. Vectorized."""
    a2 = a.reshape(Hh, Ww).astype(np.float64)
    pad = np.zeros((Hh + 2, Ww + 2), dtype=np.float64)
    pad[1:-1, 1:-1] = a2
    s = (pad[2:, 1:-1] + pad[:-2, 1:-1] + pad[1:-1, 2:] + pad[1:-1, :-2])
    cnt = np.zeros((Hh, Ww))
    cnt[:, :] = 4
    cnt[0, :] -= 1; cnt[-1, :] -= 1; cnt[:, 0] -= 1; cnt[:, -1] -= 1
    return (s / cnt).reshape(-1)


def build_features_np(pre, edge_feat, var_feat, offsets, h, w):
    """-> X [n_units, 10], per-unit rows grouped by image (offsets)."""
    Xs = []
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        n = e - s
        Hp, Wp = int(h[i]), int(w[i])
        Hh, Ww = Hp // 2, Wp // 2
        if Hh * Ww != n:
            Ww = int(round(np.sqrt(n * Wp / Hp))) or 1
            Hh = n // Ww
        rows = np.arange(n) // Ww
        cols = np.arange(n) % Ww
        rn = rows / max(1, Hh - 1) if Hh > 1 else np.zeros(n)
        cn = cols / max(1, Ww - 1) if Ww > 1 else np.zeros(n)
        p, ed, v = pre[s:e], edge_feat[s:e], var_feat[s:e]
        p_n = _neigh_mean_np(p, Hh, Ww)
        e_n = _neigh_mean_np(ed, Hh, Ww)
        v_n = _neigh_mean_np(v, Hh, Ww)
        X = np.stack([p, ed, v, rn, cn, rn - 0.5, cn - 0.5,
                      p_n, e_n, v_n], axis=1)
        Xs.append(X)
    return np.concatenate(Xs)


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0)
        sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


# --------------------------------------------------------------------------
# Torch feature builder (runner hook). Feats [num_units, 4, ctx] already
# scored per unit; grids = list of (h_patch, w_patch) per image; counts =
# per-image unit counts (== self.full_units).
# --------------------------------------------------------------------------
def _neigh_mean_torch(a, Hh, Ww, device):
    """4-neighbor mean of a (Hh*Ww,) tensor, zero-padded borders. Vectorized."""
    a2 = a.reshape(Hh, Ww)
    pad = torch.zeros(Hh + 2, Ww + 2, device=device, dtype=a.dtype)
    pad[1:-1, 1:-1] = a2
    s = (pad[2:, 1:-1] + pad[:-2, 1:-1] + pad[1:-1, 2:] + pad[1:-1, :-2])
    cnt = torch.full((Hh, Ww), 4.0, device=device)
    cnt[0, :] -= 1; cnt[-1, :] -= 1; cnt[:, 0] -= 1; cnt[:, -1] -= 1
    return (s / cnt).reshape(-1)


def build_features_torch(l2, edge, var, counts, grids, device):
    """-> torch [n_units, 10] on `device`. counts = per-image unit counts;
    grids = per-image (h_patch, w_patch)."""
    feats = []
    off = 0
    for n, (Hp, Wp) in zip(counts, grids):
        Hh, Ww = Hp // 2, Wp // 2
        if Hh * Ww != n:
            Ww = max(1, int(round(np.sqrt(n * Wp / Hp))))
            Hh = n // Ww
        rows = torch.arange(n, device=device) // Ww
        cols = torch.arange(n, device=device) % Ww
        rn = rows / max(1, Hh - 1) if Hh > 1 else torch.zeros(n, device=device)
        cn = cols / max(1, Ww - 1) if Ww > 1 else torch.zeros(n, device=device)
        p = l2[off:off + n]; ed = edge[off:off + n]; v = var[off:off + n]
        p_n = _neigh_mean_torch(p, Hh, Ww, device)
        e_n = _neigh_mean_torch(ed, Hh, Ww, device)
        v_n = _neigh_mean_torch(v, Hh, Ww, device)
        feats.append(torch.stack([p, ed, v, rn, cn, rn - 0.5, cn - 0.5,
                                  p_n, e_n, v_n], dim=1))
        off += n
    return torch.cat(feats)


class LearnedScorer:
    """Loaded scorer: standardize + MLP forward."""

    def __init__(self, path: str):
        path = str(Path(path))
        with open(path) as f:
            meta = json.load(f)
        self.mu = torch.tensor(meta["mu"], dtype=torch.float32)
        self.sd = torch.tensor(meta["sd"], dtype=torch.float32)
        hidden = meta.get("hidden", 64)
        self.mlp = build_mlp(len(FEAT_NAMES), hidden)
        self.mlp.load_state_dict({
            k: torch.tensor(v) for k, v in meta["state_dict"].items()})

    def to(self, device):
        self.mlp = self.mlp.to(device)
        self.mu = self.mu.to(device)
        self.sd = self.sd.to(device)
        return self

    @torch.no_grad()
    def score(self, l2, edge, var, counts, grids):
        dev = l2.device
        X = build_features_torch(l2, edge, var, counts, grids, dev)
        X = (X - self.mu) / self.sd
        return self.mlp(X).squeeze(-1)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "hidden": self.mlp[0].out_features,
                "feat_names": FEAT_NAMES,
                "state_dict": {k: v.tolist()
                               for k, v in self.mlp.state_dict().items()}},
                f)
