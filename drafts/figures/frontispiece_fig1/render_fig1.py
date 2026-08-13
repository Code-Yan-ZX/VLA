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

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "drafts/figures/frontispiece_fig1/data"
CAND_DIR = ROOT / "drafts/figures/frontispiece_fig1/candidates"
OUT_FIG = ROOT / "drafts/overleaf_submission/figs/fig1.pdf"

# Restrained editorial palette: terracotta for RBM, deep blue-green for
# controls, and brick red only for the failure signal.
C_AMBER = "#C86B43"
C_BLUE = "#496B78"
C_GRAY = "#86939A"
C_INK = "#202628"
C_INK2 = "#4D575B"
C_INK3 = "#7D878A"
C_RED = "#B94A43"
C_OK = "#F5F0E8"
C_ERR = "#F8EEEB"

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
    gs = fig.add_gridspec(1, 3, left=0.035, right=0.975, top=0.88, bottom=0.11,
                          width_ratios=[1.18, 1.40, 1.32], wspace=0.14)

    # --- Zone 1: Input ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(arr)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_edgecolor("#C5CCCD"); sp.set_linewidth(0.7)
    ax1.add_patch(mpatches.Rectangle((ev[0], ev[1]), ev[2]-ev[0], ev[3]-ev[1],
                                      fill=False, edgecolor=C_RED, linewidth=1.5, zorder=5))
    if show_labels:
        fig.text(ax1.get_position().x0, 0.905, "INPUT", fontsize=7.2, color=C_INK3,
                 fontweight="bold", va="bottom")
    fig.text(ax1.get_position().x0, 0.055, f'Q  {M["question"]}', fontsize=7.2, color=C_INK2)

    # --- Zone 2: Where to rank ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 3.2); ax2.set_ylim(0, 2.2)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    if show_labels:
        fig.text(ax2.get_position().x0, 0.905, "ORDER OF OPERATIONS", fontsize=7.2, color=C_INK3,
                 fontweight="bold", va="bottom")

    def draw_lane(y_base, color, labels, lane_name):
        bw, bh = 0.78, 0.42
        for i, label in enumerate(labels):
            x = 0.18 + i * 0.99
            rect = mpatches.FancyBboxPatch((x, y_base), bw, bh,
                                            boxstyle="round,pad=0.025,rounding_size=0.035",
                                            facecolor="#FCFCFB", edgecolor=color, linewidth=1.05)
            ax2.add_patch(rect)
            ax2.text(x + bw/2, y_base + bh/2, label, ha="center", va="center",
                      fontsize=6.5, color=color, fontweight="bold", linespacing=1.0)
        # arrows
        for i in range(len(labels) - 1):
            x_arr = 0.18 + (i + 1) * 0.99 - 0.08
            ax2.annotate("", xy=(x_arr + 0.08, y_base + bh/2),
                         xytext=(x_arr, y_base + bh/2),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        # lane name
        ax2.text(1.58, y_base + bh + 0.09, lane_name, ha="center", fontsize=8.1,
                 color=color, fontweight="bold")

    draw_lane(1.40, C_AMBER, ["Rank\n2x2", "Keep\n25%", "Merge"], "RBM (ours)")
    draw_lane(0.30, C_BLUE,  ["Merge", "Rank", "Keep\n25%"], "Post-L2")
    ax2.text(1.58, 0.05, "rank before information is mixed", ha="center",
             fontsize=6.9, color=C_INK3, style="italic")

    # --- Zone 3: Evidence ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_visible(False)
    if show_labels:
        fig.text(ax3.get_position().x0, 0.905, "ANSWER-BEARING EVIDENCE", fontsize=7.2, color=C_INK3,
                 fontweight="bold", va="bottom")

    # crop evidence region
    ev_crop = arr[ev[1]:ev[3], ev[0]:ev[2]]

    # Two vertically aligned zoom panels make the same evidence easy to compare.
    # (Post-L2 mask was re-captured 2026-08-12; not reused from RBM)
    # RBM zoom
    ax3_rbm = ax3.inset_axes([0.0, 0.61, 1.0, 0.27])
    ax3_rbm.imshow(make_overlay(ev_crop, masks["rbm"], sides["rbm"]))
    ax3_rbm.set_xticks([]); ax3_rbm.set_yticks([])
    for sp in ax3_rbm.spines.values():
        sp.set_edgecolor(C_AMBER); sp.set_linewidth(1.25)
    ax3.text(0.0, 0.92, f"RBM   {rbm.get('display_answer', rbm['short_answer'])}", ha="left", fontsize=7.1,
             color=C_AMBER, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.98, 0.92, "CORRECT", ha="right", fontsize=6.5, color=C_AMBER,
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.18", facecolor=C_OK,
                       edgecolor=C_AMBER, linewidth=0.55))

    # Post-L2 zoom
    ax3_post = ax3.inset_axes([0.0, 0.27, 1.0, 0.27])
    ax3_post.imshow(make_overlay(ev_crop, masks["post"], sides["post"]))
    ax3_post.set_xticks([]); ax3_post.set_yticks([])
    for sp in ax3_post.spines.values():
        sp.set_edgecolor(C_BLUE); sp.set_linewidth(1.25)
    ax3.text(0.0, 0.58, "Post-L2", ha="left", fontsize=7.1,
             color=C_BLUE, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.98, 0.58, "NOT RECOVERED", ha="right", fontsize=6.5, color=C_RED,
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.18", facecolor=C_ERR,
                       edgecolor=C_RED, linewidth=0.5))

    # GT
    ax3.text(0.0, 0.18, "GT  A105-108", fontsize=7.1, color=C_INK2,
             fontweight="bold", transform=ax3.transAxes)
    # FastV
    ax3.text(0.0, 0.11, f"FastV-k3  {fastv.get('display_answer', fastv['short_answer'])}", fontsize=6.8,
             color=C_GRAY, transform=ax3.transAxes)
    # footer
    ax3.text(0.0, 0.035,
             f"25% retained | {M['ptid']} tokens   ·   OCRBench: +16.0 pp vs FastV-k3",
             fontsize=5.7, color=C_INK3, style="italic", transform=ax3.transAxes)

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
