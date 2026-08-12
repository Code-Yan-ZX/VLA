"""Teaser figure: pre-merge vs post-merge importance maps for TextVQA 34863.

Left:  pre-merge (RBM) importance, 32x24, 192 individual cells kept
Right: post-merge (Post-L2) importance, 16x12 padded to 32x24, 48 blocks kept
       derived by 2x2 MEAN-pool of pre-merge (simulates merger averaging)
       then top-25% selection of the mean values.

Both share vmin/vmax, viridis colormap, no axes, dpi=300.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/media/disk2/YZX/research/vla")
OUT_DIR = ROOT / "drafts/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- load pre-merge kept indices ---
with (ROOT / "runs/cascade/gate_pre25_textvqa.json").open() as f:
    d = json.load(f)
for s in d["per_sample"]:
    if str(s.get("id", "")) == "34863":
        pre_block = s["pre"]
        break

kept_indices = pre_block["kept_per_image"][0]
n_units_full = pre_block["n_units_full"]  # 768
n_units_kept = pre_block["n_units_kept"]  # 192
H_pre, W_pre = 24, 32  # 32*24 = 768

# --- pre-merge importance matrix: binary, 1.0 for kept, 0.0 for not kept ---
mat_pre = np.zeros(H_pre * W_pre, dtype=np.float32)
for idx in kept_indices:
    mat_pre[idx] = 1.0
mat_pre = mat_pre.reshape(H_pre, W_pre)

# --- post-merge importance matrix ---
# 2x2 MEAN-pool of pre-merge (simulates merger averaging) → 16x12
H_post, W_post = H_pre // 2, W_pre // 2  # 12, 16
mat_post_pooled = np.zeros((H_post, W_post), dtype=np.float32)
for r in range(H_post):
    for c in range(W_post):
        block = mat_pre[r*2:(r+1)*2, c*2:(c+1)*2]
        mat_post_pooled[r, c] = block.mean()

# select top-25% of post-merge tokens (48 out of 192) by mean value
n_post_full = H_post * W_post  # 192
n_post_kept = round(n_post_full * 0.25)  # 48
# use np.argpartition for proper top-k with tie-breaking
flat = mat_post_pooled.ravel()
threshold_idx = np.argpartition(flat, -n_post_kept)[-n_post_kept:]
threshold = flat[threshold_idx].min()
mat_post_selected = (mat_post_pooled >= threshold).astype(np.float32)
# if ties give more than n_post_kept, zero out the extras at the threshold boundary
if mat_post_selected.sum() > n_post_kept:
    extras = int(mat_post_selected.sum() - n_post_kept)
    boundary = (mat_post_pooled == threshold)
    boundary_idx = np.argwhere(boundary)
    np.random.seed(42)
    np.random.shuffle(boundary_idx)
    for r, c in boundary_idx[:extras]:
        mat_post_selected[r, c] = 0
n_post_selected = int(mat_post_selected.sum())
print(f"Post-merge: {n_post_selected} blocks selected (target {n_post_kept})")

# pad post-merge to same physical size as pre-merge (32x24)
mat_post_padded = np.kron(mat_post_selected, np.ones((2, 2), dtype=np.float32))

# --- shared color scale ---
vmin = 0.0
vmax = 1.0

# --- text region bounding box (LED scoreboard "HAPPY BIRTHDAY") ---
bb_y0, bb_y1 = int(0.58 * H_pre), int(0.88 * H_pre)
bb_x0, bb_x1 = int(0.04 * W_pre), int(0.96 * W_pre)

# --- render ---
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), dpi=300)
for ax, mat in zip(axes, [mat_pre, mat_post_padded]):
    ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.add_patch(plt.Rectangle((bb_x0, bb_y0), bb_x1 - bb_x0, bb_y1 - bb_y0,
                                fill=False, edgecolor="red", linewidth=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)
out_pdf = OUT_DIR / "teaser_pre_vs_post.pdf"
out_png = OUT_DIR / "teaser_pre_vs_post.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
fig.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=300)
print(f"OK: wrote {out_pdf}, {out_png}")
print(f"Pre-merge: {n_units_kept} cells in {H_pre}x{W_pre} grid")
print(f"Post-merge: {n_post_selected} 2x2 blocks in {H_post}x{W_post} grid, padded to {H_pre}x{W_pre}")
