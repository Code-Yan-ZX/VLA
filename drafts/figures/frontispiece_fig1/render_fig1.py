"""Render frontispiece Fig. 1: OCRBench ocr0422, 7.0x3.6 inches, 3 zones."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/media/disk2/YZX/research/vla")
DATA_DIR = ROOT / "drafts/figures/frontispiece_fig1/data"
CAND_DIR = ROOT / "drafts/figures/frontispiece_fig1/candidates"
OUT_FIG = ROOT / "drafts/figs/fig1.pdf"

C_AMBER = "#E8A33D"
C_BLUE = "#4A6FA5"
C_GRAY = "#8A93A3"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_INK3 = "#898781"
C_RED = "#D03B3B"
C_OK = "#e8f5e9"
C_ERR = "#fde8e8"

with (DATA_DIR / "manifest.json").open() as f:
    M = json.load(f)

img = Image.open(DATA_DIR / M["image"]).convert("RGB")
img_w, img_h = img.size
arr = np.array(img)

# Load each method's own mask (Post-L2 and FastV-k3 were re-captured
# 2026-08-12; do not reuse the RBM mask for the other methods).
masks = {}
sides = {}
for m in M["methods"]:
    npz = np.load(DATA_DIR / m["mask"])
    masks[m["key"]] = npz["keep"]
    sides[m["key"]] = int(np.sqrt(npz["keep"].size))

rbm = M["methods"][0]
post = M["methods"][1]
fastv = M["methods"][2]
ev = M["evidence_bbox_source_px"]


def make_overlay(img_arr, keep_mask, side_grid, color=(0.91, 0.64, 0.05, 0.45)):
    h, w = img_arr.shape[:2]
    overlay = img_arr.copy().astype(np.float32) / 255.0
    bh, bw = h / side_grid, w / side_grid
    for idx in range(side_grid * side_grid):
        r, c = divmod(idx, side_grid)
        y0 = int(r * bh); y1 = min(int((r + 1) * bh), h)
        x0 = int(c * bw); x1 = min(int((c + 1) * bw), w)
        if y1 <= y0: y1 = y0 + 1
        if x1 <= x0: x1 = x0 + 1
        if keep_mask[r, c]:
            overlay[y0:y1, x0:x1] = (
                overlay[y0:y1, x0:x1] * (1 - color[3]) +
                np.array(color[:3]) * color[3]
            )
            overlay[y0, x0:x1] = color[:3]
            overlay[min(y1-1, h-1), x0:x1] = color[:3]
            overlay[y0:y1, x0] = color[:3]
            overlay[y0:y1, min(x1-1, w-1)] = color[:3]
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def render(candidate: str, out_path: Path):
    if candidate == "a":
        widths = [1.0, 1.0, 1.1]
        show_labels = True
    elif candidate == "b":
        widths = [1.3, 0.8, 1.2]
        show_labels = True
    else:
        widths = [1.0, 0.9, 1.1]
        show_labels = False

    fig = plt.figure(figsize=(7.0, 3.6), dpi=300)
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.98, top=0.90, bottom=0.10,
                          width_ratios=widths, wspace=0.10)

    # --- Zone 1: Input ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(arr)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_edgecolor(C_INK2); sp.set_linewidth(0.4)
    ax1.add_patch(mpatches.Rectangle((ev[0], ev[1]), ev[2]-ev[0], ev[3]-ev[1],
                                      fill=False, edgecolor=C_RED, linewidth=1.5, zorder=5))
    if show_labels:
        fig.text(ax1.get_position().x0, 0.93, "Input", fontsize=8, color=C_INK3,
                 fontweight="bold", va="bottom")
    fig.text(ax1.get_position().x0, 0.04, f'Q: {M["question"]}', fontsize=7, color=C_INK2)

    # --- Zone 2: Where to rank ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 3.2); ax2.set_ylim(0, 2.2)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    if show_labels:
        fig.text(ax2.get_position().x0, 0.93, "Where to rank", fontsize=8, color=C_INK3,
                 fontweight="bold", va="bottom")

    def draw_lane(y_base, color, labels, lane_name):
        bw, bh = 0.85, 0.42
        for i, label in enumerate(labels):
            x = 0.15 + i * 1.0
            rect = mpatches.FancyBboxPatch((x, y_base), bw, bh,
                                            boxstyle="round,pad=0.02",
                                            facecolor="white", edgecolor=color, linewidth=1.2)
            ax2.add_patch(rect)
            ax2.text(x + bw/2, y_base + bh/2, label, ha="center", va="center",
                      fontsize=7, color=color, fontweight="bold")
        # arrows
        for i in range(len(labels) - 1):
            x_arr = 0.15 + (i + 1) * 1.0 - 0.08
            ax2.annotate("", xy=(x_arr + 0.08, y_base + bh/2),
                         xytext=(x_arr, y_base + bh/2),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        # lane name
        ax2.text(1.65, y_base + bh + 0.08, lane_name, ha="center", fontsize=8.5,
                 color=color, fontweight="bold")

    draw_lane(1.4, C_AMBER, ["Rank\n2x2 units", "Keep 25%", "Native\nmerger"], "RBM (ours)")
    draw_lane(0.3, C_BLUE,  ["Native\nmerger", "Rank\nmerged", "Keep 25%"], "Post-L2")
    ax2.text(1.65, 0.05, "rank before information is mixed", ha="center",
             fontsize=7, color=C_INK3, style="italic")

    # --- Zone 3: Evidence ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_visible(False)
    if show_labels:
        fig.text(ax3.get_position().x0, 0.93, "Evidence", fontsize=8, color=C_INK3,
                 fontweight="bold", va="bottom")

    # crop evidence region
    ev_crop = arr[ev[1]:ev[3], ev[0]:ev[2]]

    # two zoom panels: RBM (left) and Post-L2 (right); each uses its OWN mask
    # (Post-L2 mask was re-captured 2026-08-12; not reused from RBM)
    # RBM zoom
    ax3_rbm = ax3.inset_axes([0.0, 0.55, 0.48, 0.38])
    ax3_rbm.imshow(make_overlay(ev_crop, masks["rbm"], sides["rbm"]))
    ax3_rbm.set_xticks([]); ax3_rbm.set_yticks([])
    for sp in ax3_rbm.spines.values():
        sp.set_edgecolor(C_AMBER); sp.set_linewidth(1.5)
    ax3.text(0.24, 0.96, f"RBM: {rbm.get('display_answer', rbm['short_answer'])}", ha="center", fontsize=7,
             color=C_AMBER, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.24, 0.50, "Correct", ha="center", fontsize=7, color="#0a7a0a",
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=C_OK,
                       edgecolor="#0a7a0a", linewidth=0.5))

    # Post-L2 zoom
    ax3_post = ax3.inset_axes([0.52, 0.55, 0.48, 0.38])
    ax3_post.imshow(make_overlay(ev_crop, masks["post"], sides["post"]))
    ax3_post.set_xticks([]); ax3_post.set_yticks([])
    for sp in ax3_post.spines.values():
        sp.set_edgecolor(C_BLUE); sp.set_linewidth(1.5)
    ax3.text(0.76, 0.96, f"Post-L2: {post.get('display_answer', post['short_answer'])}", ha="center", fontsize=7,
             color=C_BLUE, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.76, 0.50, "Incorrect", ha="center", fontsize=7, color=C_RED,
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=C_ERR,
                       edgecolor=C_RED, linewidth=0.5))

    # GT
    ax3.text(0.0, 0.42, "Ground truth: A105-108", fontsize=7.5, color=C_INK2,
             fontweight="bold", transform=ax3.transAxes)
    # FastV
    ax3.text(0.0, 0.32, f"FastV-k3: {fastv.get('display_answer', fastv['short_answer'])}", fontsize=7,
             color=C_GRAY, transform=ax3.transAxes)
    ax3.text(0.0, 0.24, "Incorrect", fontsize=7, color=C_RED,
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.15", facecolor=C_ERR,
                       edgecolor=C_RED, linewidth=0.5))
    # footer
    ax3.text(0.0, 0.14, f"25% retained | identical final token count: {M['ptid']}",
             fontsize=7, color=C_INK3, style="italic", transform=ax3.transAxes)
    ax3.text(0.0, 0.06, "OCRBench: RBM +16.0 pp vs FastV-k3", fontsize=7,
             color=C_AMBER, fontweight="bold", transform=ax3.transAxes)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"OK: {candidate} -> {out_path}")


def main():
    for cand in ["a", "b", "c"]:
        suffix = {"a": "balanced", "b": "visual", "c": "minimal"}[cand]
        render(cand, CAND_DIR / f"fig1_candidate_{cand}_{suffix}.pdf")
    # choose candidate_b_visual
    render("b", OUT_FIG)
    render("b", OUT_FIG.with_suffix(".png"))
    print(f"OK: final fig1 -> {OUT_FIG}")


if __name__ == "__main__":
    main()
