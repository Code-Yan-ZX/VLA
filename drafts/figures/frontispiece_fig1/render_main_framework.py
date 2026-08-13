"""Render the Fig. 1 framework schematic in the style of a VLM method overview.

The diagram is intentionally code-native: all labels, arrows, token states, and
method-specific masks are drawn by matplotlib while the airport-board crop is
the measured OCRBench example from the frontispiece bundle.
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

# Reference-like palette: pale blue containers, warm orange visual tokens,
# green instruction/query tokens, and a restrained brick-red method accent.
INK = "#1f2528"
MUTED = "#68757a"
BLUE = "#9ec5e8"
BLUE_D = "#315a78"
ORANGE = "#f2b183"
ORANGE_D = "#c66b3e"
GREEN = "#a9d18e"
GREEN_D = "#547b48"
YELLOW = "#f7dfa0"
YELLOW_D = "#9b7411"
RED = "#c95045"
PANEL = "#f5f9fc"
PANEL2 = "#fcfbf7"
GRID = "#d6dde0"

with (DATA / "manifest.json").open(encoding="utf-8") as f:
    M = json.load(f)
img = np.asarray(Image.open(DATA / M["image"]).convert("RGB"))
ev = M["evidence_bbox_source_px"]


def rounded(ax, xy, w, h, *, fc="white", ec=INK, lw=1.0, radius=0.08,
            ls="-", z=1):
    patch = P.FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                             facecolor=fc, edgecolor=ec, linewidth=lw,
                             linestyle=ls, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.1, z=8, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=z)


def token(ax, x, y, label="", color=ORANGE, edge=None, size=0.18, fs=7.0,
          alpha=1.0):
    rounded(ax, (x, y), size, size, fc=color, ec=edge or color, lw=0.45,
            radius=0.025, z=4)
    if label:
        ax.text(x + size / 2, y + size / 2, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=5)


def token_row(ax, x, y, n, keep=None, gap=0.055, size=0.19):
    keep = set(keep or [])
    for i in range(n):
        c = ORANGE if i in keep else "#e7ecee"
        edge = ORANGE_D if i in keep else "#b9c3c7"
        token(ax, x + i * (size + gap), y, str(i + 1), color=c, edge=edge,
              size=size, fs=5.8)


def crop_with_mask(key: str):
    crop = img[ev[1]:ev[3], ev[0]:ev[2]]
    z = np.load(DATA / M["methods"][[m["key"] for m in M["methods"]].index(key)]["mask"])
    keep = z["keep"]
    side = int(np.sqrt(keep.size))
    out = crop.copy().astype(float) / 255.0
    h, w = crop.shape[:2]
    for i in range(side * side):
        r, c = divmod(i, side)
        y0, y1 = int(r * h / side), int((r + 1) * h / side)
        x0, x1 = int(c * w / side), int((c + 1) * w / side)
        if keep[r, c]:
            tint = np.array([0.95, 0.67, 0.18])
            out[y0:y1, x0:x1] = 0.58 * out[y0:y1, x0:x1] + 0.42 * tint
            out[y0:y1, x0] = tint; out[y0:y1, x1 - 1] = tint
            out[y0, x0:x1] = tint; out[y1 - 1, x0:x1] = tint
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def draw_stage_stack(ax, x, y, w):
    """Right-hand vertical stage accumulation, echoing the reference figure."""
    stages = [
        ("ViT layers $\\times M$", BLUE, BLUE_D),
        ("RBM: rank raw units", ORANGE, ORANGE_D),
        ("Keep top-$\\kappa N$", GREEN, GREEN_D),
        ("Native $2\\times2$ merger", YELLOW, YELLOW_D),
        ("LLM layers $\\times K$", BLUE, BLUE_D),
    ]
    bh = 0.36
    gap = 0.16
    for i, (label, fc, ec) in enumerate(stages):
        yy = y + i * (bh + gap)
        rounded(ax, (x, yy), w, bh, fc=fc, ec=ec, lw=0.8, radius=0.07)
        ax.text(x + w / 2, yy + bh / 2, label, ha="center", va="center",
                fontsize=6.8, color=INK)
        if i:
            arrow(ax, x + w / 2, yy - 0.03, x + w / 2, yy - gap + 0.03,
                  color=INK, lw=0.9)
    ax.text(x + w + 0.08, y + 2.0, "stage-aware\nselection", fontsize=6.0,
            color=MUTED, va="center", ha="left", style="italic")


def render(out_path: Path):
    fig = plt.figure(figsize=(10.0, 4.25), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.25); ax.axis("off")

    # Left: query + input image.
    rounded(ax, (0.28, 1.07), 1.62, 2.66, fc="white", ec=INK, lw=1.0,
            radius=0.18, ls=(0, (3, 2)))
    ax.text(0.46, 3.37, "Query: which counter is\nboarding?", fontsize=8.7,
            color=INK, va="top", linespacing=1.2)
    ax.imshow(img, extent=(0.45, 1.73, 1.52, 2.86), zorder=2, aspect="auto")
    ax.add_patch(P.Rectangle((0.45 + 1.28 * ev[0] / img.shape[1],
                              1.52 + 1.34 * (img.shape[0] - ev[3]) / img.shape[0]),
                             1.28 * (ev[2] - ev[0]) / img.shape[1],
                             1.34 * (ev[3] - ev[1]) / img.shape[0],
                             fill=False, ec=RED, lw=1.4, zorder=5))
    ax.text(1.09, 1.25, "Input image", ha="center", fontsize=7.2, color=MUTED)
    arrow(ax, 1.09, 1.02, 1.09, 0.78, color=INK, lw=1.0)
    ax.text(1.09, 0.61, "Image + text\ntokenize", ha="center", va="top", fontsize=6.8,
            color=INK)

    # Center: method panel with two orderings.
    rounded(ax, (2.14, 0.50), 4.78, 3.23, fc=PANEL, ec=BLUE_D, lw=1.0, radius=0.20)
    ax.text(2.36, 3.47, "Rank-Before-Merge (RBM): change the stage, keep the model", fontsize=10.0,
            color=INK, fontweight="bold")
    # RBM lane.
    ax.text(2.39, 3.14, "a) Pre-merger ranking (ours)", fontsize=8.4, color=ORANGE_D,
            fontweight="bold")
    labels = ["raw $2\\times2$\nunits", "rank by\nL2 score", "keep top-$\\kappa N$", "native\nmerge"]
    xs = [2.42, 3.38, 4.35, 5.48]
    ws = [0.78, 0.78, 0.95, 0.92]
    for i, (x, w, lab) in enumerate(zip(xs, ws, labels)):
        fc = "#fff8f2" if i in (1, 2) else "white"
        ec = ORANGE_D if i in (1, 2) else BLUE_D
        rounded(ax, (x, 2.31), w, 0.48, fc=fc, ec=ec, lw=1.0, radius=0.06)
        ax.text(x + w / 2, 2.55, lab, ha="center", va="center", fontsize=6.6,
                color=INK, linespacing=0.95)
        if i < len(xs) - 1:
            arrow(ax, x + w + 0.05, 2.55, xs[i + 1] - 0.06, 2.55, color=ORANGE_D, lw=1.1)
    token_row(ax, 2.50, 2.88, 7, keep=[0, 2, 4], size=0.17)
    ax.text(2.42, 3.08, "N", fontsize=6.5, color=MUTED)
    ax.text(5.48, 3.08, "$\\kappa N$", fontsize=6.5, color=MUTED)
    ax.text(3.98, 2.15, "saliency is measured BEFORE information is mixed", ha="center",
            fontsize=6.6, color=ORANGE_D, style="italic")
    # Post lane.
    ax.text(2.39, 1.88, "b) Post-merger ranking (control / FastV-style tap)", fontsize=8.2,
            color=BLUE_D, fontweight="bold")
    labels2 = ["raw units", "native\nmerge", "rank merged\ntokens", "keep top-$\\kappa N$"]
    xs2 = [2.42, 3.38, 4.35, 5.48]
    ws2 = [0.78, 0.78, 0.95, 0.92]
    for i, (x, w, lab) in enumerate(zip(xs2, ws2, labels2)):
        fc = "#f6fbff" if i == 2 else "white"
        ec = RED if i == 2 else BLUE_D
        rounded(ax, (x, 1.08), w, 0.48, fc=fc, ec=ec, lw=1.0,
                radius=0.06, ls="--" if i == 2 else "-")
        ax.text(x + w / 2, 1.32, lab, ha="center", va="center", fontsize=6.5,
                color=INK, linespacing=0.95)
        if i < len(xs2) - 1:
            arrow(ax, x + w + 0.05, 1.32, xs2[i + 1] - 0.06, 1.32, color=BLUE_D, lw=1.1)
    token_row(ax, 2.50, 1.63, 7, keep=[1, 5], size=0.17)
    ax.text(3.98, 0.90, "same budget, different representation", ha="center",
            fontsize=6.6, color=BLUE_D, style="italic")
    # Connector from input into method panel.
    arrow(ax, 1.92, 2.34, 2.12, 2.34, color=INK, lw=1.0)

    # Right: ranking summary + measured evidence.
    rounded(ax, (7.17, 2.31), 1.42, 1.42, fc="white", ec=BLUE_D, lw=1.0,
            radius=0.18, ls=(0, (3, 2)))
    ax.text(7.88, 3.48, "Overall ranking", ha="center", fontsize=8.6, fontweight="bold")
    bars = [(0.25, ORANGE), (0.18, BLUE), (0.12, GREEN), (0.08, YELLOW)]
    for i, (v, c) in enumerate(bars):
        x = 7.38 + i * 0.25
        ax.add_patch(P.Rectangle((x, 2.55), 0.14, v * 1.65, facecolor=c, edgecolor="none"))
    ax.plot([7.32, 8.42], [2.55, 2.55], color=INK, lw=0.8)
    ax.text(7.88, 2.40, "retain $\\kappa=25\\%$", ha="center", fontsize=6.3, color=MUTED)
    arrow(ax, 6.93, 2.16, 7.15, 2.16, color=INK, lw=1.0)
    rounded(ax, (7.17, 0.53), 1.42, 1.37, fc="white", ec=INK, lw=1.0,
            radius=0.18, ls=(0, (3, 2)))
    ax.text(7.88, 1.67, "Result", ha="center", fontsize=8.6, fontweight="bold")
    ax.imshow(crop_with_mask("rbm"), extent=(7.34, 8.42, 0.82, 1.49), zorder=2, aspect="auto")
    ax.add_patch(P.Rectangle((7.34, 0.82), 1.08, 0.67, fill=False, ec=ORANGE_D, lw=1.0))
    ax.text(7.88, 0.68, "RBM: A105-108", ha="center", fontsize=6.7, color=ORANGE_D, fontweight="bold")

    # Far right: accumulation/stage stack.
    draw_stage_stack(ax, 8.88, 0.48, 0.92)
    ax.text(9.34, 3.72, "b) Accumulation of\nselection stages", ha="center", va="top",
            fontsize=7.5, color=INK, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    render(OUT / "fig1.pdf")
    render(OUT / "fig1.png")
    print("wrote", OUT / "fig1.pdf")
