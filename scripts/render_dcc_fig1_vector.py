#!/usr/bin/env python3
"""Rebuild DCC Figure 1 with vector text, lines, arrows, and boxes.

The only raster element is the source OCRBench photograph.  The topology and
labels reproduce the method overview embedded in the selected Word manuscript.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "drafts" / "figures" / "server_exports" / "cvpr_figure_data_v1" / "cases" / "ocrbench_ocr0422" / "input.jpg"
OUT = ROOT / "drafts" / "dcc2027_submission_20260914" / "figs" / "fig1_vector.pdf"

INK = "#24303F"
BLUE = "#315B7D"
BLUE_LIGHT = "#EAF1F6"
GOLD = "#E2B714"
GOLD_LIGHT = "#FFF7D0"
GREY = "#AAB4BE"
LIGHT = "#F5F7F9"
RED = "#C45A3C"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "text.color": INK,
})


def box(ax, x, y, w, h, text, edge=INK, face="white", fs=11.5, weight="normal"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.006,rounding_size=0.008",
                           linewidth=1.2, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=INK, linespacing=1.1)
    return patch


def arrow(ax, x1, y1, x2, y2, color=INK, style="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.15,
                                 linestyle=style, color=color,
                                 shrinkA=1.5, shrinkB=1.5))


def tokens(ax, x, y, count=4, kept=2, color=GOLD):
    size, gap = .014, .005
    for i in range(count):
        face = color if i < kept else LIGHT
        edge = color if i < kept else GREY
        ax.add_patch(Rectangle((x + i * (size + gap), y), size, size,
                               linewidth=.85, edgecolor=edge, facecolor=face))


fig, ax = plt.subplots(figsize=(12.0, 4.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

img = Image.open(PHOTO).convert("RGB")
ax.imshow(img, extent=(.012, .185, .29, .76), aspect="auto", zorder=0)
ax.add_patch(Rectangle((.045, .415), .105, .13, linewidth=1.7,
                       edgecolor=GOLD, facecolor="none"))
ax.text(.099, .245, "Input", ha="center", va="top", fontsize=11.5, fontweight="bold")

box(ax, .208, .445, .075, .16, "Vision\nencoder", face=LIGHT, fs=11.3)
arrow(ax, .185, .525, .208, .525)

ax.text(.306, .815, "Post-L2: merge $\\rightarrow$ rank", fontsize=12.0,
        fontweight="bold", color=BLUE)
ax.text(.306, .13, "RBM: rank $\\rightarrow$ merge", fontsize=12.0,
        fontweight="bold", color=BLUE)

# Shared patch stream and native-unit cue.
arrow(ax, .283, .525, .305, .525)
tokens(ax, .305, .518, count=4, kept=4)
ax.text(.338, .565, "native merge unit", ha="center", va="bottom",
        fontsize=9.5, color=INK)

# Post-L2 lane: merger first, then rank.
arrow(ax, .382, .525, .405, .69, color=BLUE)
box(ax, .415, .62, .078, .145, "Native\nmerger", edge=BLUE, face=BLUE_LIGHT, fs=10.7)
arrow(ax, .493, .692, .515, .692, color=BLUE)
box(ax, .522, .62, .087, .145, "Merged-token\nL2 rank", edge=BLUE, face="white", fs=10.2)
arrow(ax, .609, .692, .63, .692, color=BLUE)
box(ax, .637, .635, .057, .115, "Top-$k$", edge=BLUE, face="white", fs=11.0)
arrow(ax, .694, .692, .72, .692, color=BLUE)
tokens(ax, .724, .684, count=4, kept=2, color=BLUE)
ax.text(.758, .632, "selected after\nrepresentation change", ha="center",
        va="top", fontsize=8.7, color=BLUE)

# RBM lane: rank complete units first, then invoke the original merger.
arrow(ax, .382, .525, .405, .34, color=BLUE)
box(ax, .415, .267, .078, .145, "Unit L2\nscore", edge=BLUE, face="white", fs=10.7)
arrow(ax, .493, .34, .515, .34, color=BLUE)
box(ax, .522, .282, .087, .115, "Top-$k$\nunits", edge=BLUE, face="white", fs=10.7)
arrow(ax, .609, .34, .63, .34, color=BLUE)
tokens(ax, .635, .333, count=4, kept=2)
arrow(ax, .708, .34, .728, .34, color=BLUE)
box(ax, .735, .267, .078, .145, "Native\nmerger", edge=BLUE, face=GOLD_LIGHT, fs=10.7)

# Common model interface.
arrow(ax, .785, .692, .835, .555, color=BLUE)
arrow(ax, .813, .34, .835, .495, color=BLUE)
box(ax, .842, .44, .069, .17, "Native\ninterface", face=LIGHT, fs=10.7)
arrow(ax, .911, .525, .93, .525)
box(ax, .937, .455, .05, .14, "LLM", face=LIGHT, fs=11.5)

ax.text(.50, .035,
        "Training-free  ·  same per-image rate $R=k/N$  ·  native merger and model weights unchanged",
        ha="center", va="bottom", fontsize=10.3, color=INK)

fig.subplots_adjust(left=.004, right=.996, top=.98, bottom=.02)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=.02)
print(OUT)

