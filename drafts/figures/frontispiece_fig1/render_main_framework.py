"""Render an insight-driven CVPR-style overview for Rank-Before-Merge.

The image is the measured OCRBench ocr0422 example; every box, label, token,
arrow, and answer card is drawn as editable vector content by matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as P
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "drafts/figures/frontispiece_fig1/data"
OUT = ROOT / "drafts/overleaf_submission/figs"

INK = "#202629"
MUTED = "#657177"
BLUE = "#4b7186"
BLUE_LT = "#eaf3f7"
ORANGE = "#c96b3d"
ORANGE_LT = "#fff1e8"
RED = "#c34e45"
RED_LT = "#fff0ee"
GOLD = "#aa7e11"
GRID = "#bec9cd"

with (DATA / "manifest.json").open(encoding="utf-8") as f:
    M = json.load(f)
img = np.asarray(Image.open(DATA / M["image"]).convert("RGB"))
ev = M["evidence_bbox_source_px"]


def box(ax, x, y, w, h, *, edge=INK, face="white", lw=1.1, dashed=False,
        radius=0.06):
    ax.add_patch(P.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "-", zorder=2))


def arrow(ax, x0, y0, x1, y1, *, color=INK, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=5)


def token_row(ax, x, y, n=8, keep=(), color=ORANGE, edge=ORANGE, size=0.15):
    keep = set(keep)
    for i in range(n):
        fill = color if i in keep else "white"
        ec = edge if i in keep else GRID
        box(ax, x + i * (size + 0.045), y, size, size, edge=ec, face=fill, lw=0.65,
            radius=0.025)
        ax.text(x + i * (size + 0.045) + size / 2, y + size / 2, str(i + 1),
                ha="center", va="center", fontsize=5.2, color=INK, zorder=4)


def masked_crop(key: str, tint=(0.95, 0.67, 0.18)):
    crop = img[ev[1]:ev[3], ev[0]:ev[2]]
    method = next(m for m in M["methods"] if m["key"] == key)
    keep = np.load(DATA / method["mask"])["keep"]
    side = int(np.sqrt(keep.size))
    out = crop.astype(float) / 255.0
    h, w = crop.shape[:2]
    for i in range(side * side):
        r, c = divmod(i, side)
        y0, y1 = int(r * h / side), int((r + 1) * h / side)
        x0, x1 = int(c * w / side), int((c + 1) * w / side)
        if keep[r, c]:
            out[y0:y1, x0:x1] = 0.60 * out[y0:y1, x0:x1] + 0.40 * np.asarray(tint)
            out[y0:y1, x0] = tint; out[y0:y1, x1 - 1] = tint
            out[y0, x0:x1] = tint; out[y1 - 1, x0:x1] = tint
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def step(ax, x, y, w, label, *, edge, face="white", fs=7.1, dashed=False):
    box(ax, x, y, w, 0.47, edge=edge, face=face, lw=1.15, dashed=dashed)
    ax.text(x + w / 2, y + 0.235, label, ha="center", va="center", fontsize=fs,
            color=INK, linespacing=0.95)


def render(path: Path):
    fig = plt.figure(figsize=(11.2, 4.25), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.25); ax.axis("off")

    # Figure-level insight headline.
    ax.text(0.34, 4.03, "Saliency should be measured", fontsize=16, fontweight="bold", color=INK)
    ax.text(4.36, 4.03, "before information mixing", fontsize=16, fontweight="bold", color=ORANGE)
    ax.plot([4.34, 8.08], [3.93, 3.93], color=ORANGE, lw=1.8)

    # Problem: real image and OCR-sensitive region.
    ax.text(0.36, 3.66, "PROBLEM", fontsize=8.2, fontweight="bold", color=MUTED)
    box(ax, 0.33, 0.50, 1.88, 2.95, edge=INK, dashed=True, radius=0.17)
    ax.text(0.52, 3.24, "Query: which counter is", fontsize=8.0, color=INK)
    ax.text(0.52, 3.03, "boarding?", fontsize=8.0, color=INK)
    ax.imshow(img, extent=(0.52, 1.95, 1.47, 2.79), aspect="auto", zorder=3)
    # Highlight answer-bearing OCR rows.
    ex, ey, ew, eh = ev
    ax.add_patch(P.Rectangle((0.52 + 1.43 * ex / img.shape[1],
                              1.47 + 1.32 * (img.shape[0] - ey - eh) / img.shape[0]),
                             1.43 * ew / img.shape[1], 1.32 * eh / img.shape[0],
                             fill=False, edgecolor=RED, linewidth=1.45, zorder=5))
    ax.text(1.235, 1.14, "OCR-sensitive region", ha="center", fontsize=6.5, color=RED)
    arrow(ax, 2.22, 2.05, 2.48, 2.05, color=INK)

    # Insight: same raw units, opposite order.
    ax.text(2.52, 3.66, "INSIGHT", fontsize=8.2, fontweight="bold", color=MUTED)
    box(ax, 2.48, 2.25, 2.65, 1.18, edge=INK, face="white", radius=0.10)
    ax.text(2.65, 3.17, "Raw visual units", fontsize=8.0, fontweight="bold", color=INK)
    token_row(ax, 2.68, 2.82, n=8, keep=range(8), color="#f4c39f", edge=ORANGE)
    ax.text(2.66, 2.48, "each unit contains local text evidence", fontsize=6.5, color=MUTED)
    # Two compact order comparisons.
    ax.text(2.58, 2.01, "FastV", fontsize=7.8, fontweight="bold", color=BLUE)
    step(ax, 3.12, 1.82, 0.80, "Merge", edge=BLUE, face=BLUE_LT, fs=7.2)
    arrow(ax, 3.95, 2.05, 4.22, 2.05, color=BLUE)
    step(ax, 4.25, 1.82, 0.73, "Rank", edge=BLUE, face=BLUE_LT, fs=7.2)
    ax.text(2.58, 1.43, "RBM (ours)", fontsize=7.8, fontweight="bold", color=ORANGE)
    step(ax, 3.12, 1.24, 0.80, "Rank", edge=ORANGE, face=ORANGE_LT, fs=7.2)
    arrow(ax, 3.95, 1.47, 4.22, 1.47, color=ORANGE)
    step(ax, 4.25, 1.24, 0.73, "Merge", edge=ORANGE, face=ORANGE_LT, fs=7.2)
    ax.text(3.83, 0.96, "only the order changes", ha="center", fontsize=6.5, color=MUTED, style="italic")
    arrow(ax, 5.17, 1.82, 5.42, 1.82, color=INK)

    # Method: two symmetric branches; ranking location is the only visual change.
    ax.text(5.47, 3.66, "METHOD", fontsize=8.2, fontweight="bold", color=MUTED)
    ax.text(5.47, 3.38, "Same model path, same budget", fontsize=8.3, color=INK, fontweight="bold")
    ax.text(5.50, 3.05, "RBM", fontsize=8.6, color=ORANGE, fontweight="bold")
    step(ax, 5.95, 2.89, 0.97, "Rank raw\nunits", edge=ORANGE, face=ORANGE_LT, fs=7.0)
    arrow(ax, 6.97, 3.12, 7.18, 3.12, color=ORANGE)
    step(ax, 7.21, 2.89, 1.05, "Keep top-$\\kappa N$", edge=ORANGE, face=ORANGE_LT, fs=6.6)
    arrow(ax, 8.31, 3.12, 8.52, 3.12, color=ORANGE)
    step(ax, 8.55, 2.89, 1.00, "Native $2\\times2$\nmerger", edge=ORANGE, face="white", fs=6.6)
    ax.text(5.50, 2.20, "FastV-style", fontsize=8.6, color=BLUE, fontweight="bold")
    step(ax, 5.95, 2.04, 1.05, "Native $2\\times2$\nmerger", edge=BLUE, face="white", fs=6.6)
    arrow(ax, 7.05, 2.27, 7.18, 2.27, color=BLUE)
    step(ax, 7.21, 2.04, 1.05, "Rank merged\ntokens", edge=BLUE, face=BLUE_LT, fs=6.8)
    arrow(ax, 8.31, 2.27, 8.52, 2.27, color=BLUE)
    step(ax, 8.55, 2.04, 1.00, "Keep top-$\\kappa N$", edge=BLUE, face=BLUE_LT, fs=6.6)
    ax.text(7.32, 1.66, "Same budget  $\\kappa=25\\%$", fontsize=8.1, color=INK, fontweight="bold")
    ax.text(7.32, 1.43, "ranking is the only visual difference", fontsize=6.5, color=MUTED, style="italic")
    arrow(ax, 9.62, 2.55, 9.86, 2.55, color=INK)

    # Evidence: direct answer comparison.
    ax.text(9.89, 3.66, "EVIDENCE", fontsize=8.2, fontweight="bold", color=MUTED)
    box(ax, 9.86, 1.98, 1.06, 1.40, edge=BLUE, face="white", radius=0.08)
    ax.text(10.39, 3.19, "FastV", ha="center", fontsize=7.8, color=BLUE, fontweight="bold")
    ax.imshow(masked_crop("fastv", tint=(0.42, 0.60, 0.70)), extent=(9.96, 10.82, 2.45, 3.08), aspect="auto", zorder=3)
    ax.text(10.39, 2.23, "✗  A115-126", ha="center", fontsize=8.0, color=RED, fontweight="bold")
    box(ax, 9.86, 0.46, 1.06, 1.40, edge=ORANGE, face="white", radius=0.08)
    ax.text(10.39, 1.68, "RBM", ha="center", fontsize=7.8, color=ORANGE, fontweight="bold")
    ax.imshow(masked_crop("rbm"), extent=(9.96, 10.82, 0.94, 1.57), aspect="auto", zorder=3)
    ax.text(10.39, 0.72, "✓  A105-108", ha="center", fontsize=8.0, color=ORANGE, fontweight="bold")

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    render(OUT / "fig1.pdf")
    render(OUT / "fig1.svg")
    render(OUT / "fig1.png")
    print("wrote", OUT / "fig1.pdf", OUT / "fig1.svg")
