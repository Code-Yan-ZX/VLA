#!/usr/bin/env python3
"""Retention vs compression depth -- small-multiples figure (CPU only).

One panel per benchmark (TextVQA, DocVQA, OCR-Bench, ChartQA, GQA); a sixth
cell holds the shared legend + a one-paragraph takeaway.  Each panel plots the
*accuracy retention ratio*  acc(keep%) / acc(100%)  against the *visual-token
retention* keep% of merge-units, for two selector stages:

    pre-merger   blue,  solid line,  circle markers
    post-merger  red,   dashed line, square markers

Error bars are the binomial standard error on accuracy, sqrt(p(1-p)/n) with
n = 200, propagated to the ratio by dividing by the (held-fixed) baseline, so
the 100% reference point carries no bar (it is the normaliser, ratio == 1 by
construction).  A thin grey hairline marks ratio = 1.0; each panel annotates
its absolute baseline accuracy.

Style note
----------
The repo's drafts/figures/_style.py enables a y-grid and drops the top/right
spines, but the named style reference for *this* figure, stage_law.png, shows a
thin FULL box with the grid OFF, and the task brief specifies "white grid off,
light spines".  To stay reproducible without a sys.path import and to match the
reference exactly, the academic style is reproduced inline below (white surface,
thin light four-sided box, no gridlines, muted ink).  The ink/surface hex are
taken verbatim from _style.py so the palette family still matches the paper.

Reproducible: all data are inline; no model, no GPU, no network.

Outputs
-------
drafts/figures/retention_curves.png   (300 dpi)
drafts/figures/retention_curves.pdf   (vector, editable text)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

# --------------------------------------------------------------------------- #
# Paths (resolved from this file, so the script is relocatable within repo)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent          # .../scripts
REPO = HERE.parent                              # repo root
OUT_DIR = REPO / "drafts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = OUT_DIR / "retention_curves"

# --------------------------------------------------------------------------- #
# Data  (n = 200, Qwen3-VL-8B).  Format  keep% : accuracy.
# Each benchmark: baseline = acc at keep 100%; pre / post = the two stages.
# --------------------------------------------------------------------------- #
N = 200
DATA = {
    "TextVQA":   dict(tier="scene text",          base=0.820,
                      pre={50: 0.750, 25: 0.695, 12.5: 0.615},
                      post={50: 0.510, 25: 0.255, 12.5: 0.175}),
    "DocVQA":    dict(tier="document",            base=0.770,
                      pre={25: 0.725, 12.5: 0.610},
                      post={25: 0.390, 12.5: 0.135}),
    "OCR-Bench": dict(tier="mixed OCR",           base=0.760,
                      pre={25: 0.580, 12.5: 0.380},
                      post={25: 0.165, 12.5: 0.075}),
    "ChartQA":   dict(tier="chart (budget-bound)", base=0.820,
                      pre={50: 0.390, 25: 0.190, 12.5: 0.150},
                      post={50: 0.335, 25: 0.190, 12.5: 0.095}),
    "GQA":       dict(tier="object",              base=0.415,
                      pre={50: 0.380, 25: 0.320, 12.5: 0.250},
                      post={50: 0.405, 25: 0.380, 12.5: 0.305}),
}
# Panel order across the 2x3 grid (row-major); 6th cell = legend/takeaway.
ORDER = ["TextVQA", "DocVQA", "OCR-Bench", "ChartQA", "GQA"]

# --------------------------------------------------------------------------- #
# Palette / ink  (hex verbatim from drafts/figures/_style.py)
# --------------------------------------------------------------------------- #
SURFACE   = "#fcfcfb"   # plot + figure background (near-white)
INK_PRIM  = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
SPINE     = "#52514e"   # thin "light" box (not pure black, not chartjunk)
REFLINE   = "#c3c2b7"   # hairline at ratio = 1.0

PRE_BLUE  = "#2a78d6"   # pre-merger  (canonical CAT blue)
POST_RED  = "#e34948"   # post-merger (canonical CAT red)

# --------------------------------------------------------------------------- #
# Academic rcParams (sans-serif, no chartjunk, editable vector text)
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 9.0,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": INK_SEC,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "text.color": INK_PRIM,
    "axes.titlecolor": INK_PRIM,
    "legend.frameon": False,
})


def binom_se(p: float, n: int = N) -> float:
    """Binomial standard error on a proportion (Wald)."""
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def series(name: str, stage: str):
    """Return (keep%, ratio, ratio_err) for one stage of one benchmark.

    The 100% baseline point is appended with ratio = 1.0 and zero error (it is
    the fixed normaliser).  ratio_err = binom_se(acc) / baseline.
    """
    rec = DATA[name]
    base = rec["base"]
    pts = dict(rec[stage])         # keep% -> acc
    pts[100] = base                # baseline == keep 100%
    xs, rs, es = [], [], []
    for k in sorted(pts):
        acc = pts[k]
        xs.append(k)
        rs.append(acc / base)
        es.append(0.0 if k == 100 else binom_se(acc) / base)
    return xs, rs, es


def style_panel(ax: plt.Axes) -> None:
    """Thin light four-sided box, NO gridlines (matches stage_law.png)."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(axis="both", which="both", length=3.0, width=0.8,
                   colors=INK_SEC, labelsize=8.5)
    ax.minorticks_off()
    ax.set_axisbelow(True)


def configure_x(ax: plt.Axes) -> None:
    """log2 x-axis: the halving sequence 12.5->25->50->100 spaces evenly."""
    ax.set_xscale("log", base=2)
    ax.set_xlim(10, 128)
    ax.xaxis.set_major_locator(FixedLocator([12.5, 25, 50, 100]))
    ax.xaxis.set_major_formatter(
        FixedFormatter(["12.5%", "25%", "50%", "100%"]))


# --------------------------------------------------------------------------- #
# Build the 2 x 3 grid
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(7.7, 4.9))
flat = axes.ravel()

for idx, name in enumerate(ORDER):
    ax = flat[idx]
    rec = DATA[name]
    base = rec["base"]

    # reference hairline at ratio = 1.0
    ax.axhline(1.0, color=REFLINE, linewidth=0.9, zorder=1)

    # post-merger first so the blue circle sits on top at the shared 100% point
    for stage, color, ls, mk in (
        ("post", POST_RED, (0, (4.0, 2.6)), "s"),
        ("pre",  PRE_BLUE, "solid",         "o"),
    ):
        xs, rs, es = series(name, stage)
        ax.errorbar(xs, rs, yerr=es, fmt=mk, color=color,
                    linestyle=ls, linewidth=1.7, markersize=5.6,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.7, elinewidth=1.0, capsize=2.4,
                    capthick=1.0, zorder=3 if stage == "pre" else 2,
                    label=stage)

    configure_x(ax)
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    style_panel(ax)

    # two-line title drawn above the panel (full style control per line)
    ax.text(0.5, 1.17, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK_PRIM)
    ax.text(0.5, 1.05, rec["tier"], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.8, style="italic", color=INK_MUTED)

    # absolute baseline accuracy, tucked top-left where the curve is empty
    ax.text(0.04, 0.96, f"baseline = {base:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.6, color=INK_MUTED)

    # axis titles only on the outer edges of the small-multiples grid
    if idx % 3 == 0:
        ax.set_ylabel("accuracy retention\n(acc / baseline)", fontsize=9.0,
                      color=INK_SEC)
    if idx >= 3:
        ax.set_xlabel("visual-token retention (keep %)", fontsize=9.0,
                      color=INK_SEC)

# --------------------------------------------------------------------------- #
# Sixth cell: shared legend + structured takeaway
# --------------------------------------------------------------------------- #
leg_ax = flat[5]
leg_ax.set_xticks([])
leg_ax.set_yticks([])
for s in leg_ax.spines.values():
    s.set_visible(False)

handles = [
    Line2D([0], [0], color=PRE_BLUE, linestyle="solid", marker="o",
           markersize=6, markerfacecolor=PRE_BLUE, markeredgecolor="white",
           markeredgewidth=0.7, linewidth=1.7),
    Line2D([0], [0], color=POST_RED, linestyle=(0, (4.0, 2.6)), marker="s",
           markersize=6, markerfacecolor=POST_RED, markeredgecolor="white",
           markeredgewidth=0.7, linewidth=1.7),
]
leg = leg_ax.legend(handles, ["pre-merger", "post-merger"],
                    title="selection stage", loc="upper center",
                    bbox_to_anchor=(0.5, 0.97), fontsize=9.0,
                    title_fontsize=9.3, handlelength=2.6,
                    handletextpad=0.6, labelspacing=0.5)
leg.get_title().set_color(INK_PRIM)

takeaway = (
    "By tier:\n"
    "• Text-dense (TextVQA, DocVQA, OCR-Bench): pre-merger keeps\n"
    "   ~70–90% of baseline at 25% retention; post-merger collapses.\n"
    "• ChartQA (budget-bound): both stages collapse — accuracy is\n"
    "   token-budget limited, not selector-stage limited.\n"
    "• GQA (object): post-merger is marginally stronger — the only\n"
    "   tier where merging helps."
)
leg_ax.text(0.02, 0.50, takeaway, transform=leg_ax.transAxes, ha="left",
            va="top", fontsize=8.0, color=INK_SEC, linespacing=1.35)

# --------------------------------------------------------------------------- #
# Figure-level stats / integrity footnote (part of the figure, per contract)
# --------------------------------------------------------------------------- #
fig.text(0.012, 0.012,
         f"Qwen3-VL-8B, n = {N} per point.  "
         "Error bars = binomial SE sqrt(p(1-p)/n) on accuracy, propagated to "
         "the ratio with the baseline held fixed (100% point = 1.0, no bar).  "
         "x-axis is log2 so each halving of retained tokens is one equal step.",
         fontsize=7.0, color=INK_MUTED, ha="left", va="bottom")

fig.subplots_adjust(left=0.075, right=0.988, top=0.85, bottom=0.135,
                    wspace=0.34, hspace=0.80)

# Figure-level title (the manuscript caption will carry the formal legend)
fig.suptitle("Retention vs compression depth",
             fontsize=12.5, fontweight="bold", color=INK_PRIM, y=0.985)

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
print(f"wrote {OUT_STEM.with_suffix('.png')}")
print(f"wrote {OUT_STEM.with_suffix('.pdf')}")
