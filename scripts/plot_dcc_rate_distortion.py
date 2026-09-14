#!/usr/bin/env python3
"""Render the DCC task rate--distortion operating-point figure.

All values are the frozen Qwen3-VL-8B n=200 diagnostic cells already used by
the retention-depth analysis.  No interpolation or new experimental result is
introduced.  Rate is the retained native-unit fraction R=k/N.  Task distortion
is D(R)=1-S(R)/S(1), where S is the benchmark score.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "drafts" / "dcc2027_submission_20260914" / "figs" / "fig5_rate_distortion.pdf"

DATA = {
    "TextVQA": {
        "base": 0.820,
        "rbm": {12.5: 0.615, 25: 0.695, 50: 0.750},
        "post": {12.5: 0.175, 25: 0.255, 50: 0.510},
    },
    "DocVQA": {
        "base": 0.770,
        "rbm": {12.5: 0.610, 25: 0.725},
        "post": {12.5: 0.135, 25: 0.390},
    },
    "OCRBench": {
        "base": 0.760,
        "rbm": {12.5: 0.380, 25: 0.580},
        "post": {12.5: 0.075, 25: 0.165},
    },
    "GQA": {
        "base": 0.415,
        "rbm": {12.5: 0.250, 25: 0.320, 50: 0.380},
        "post": {12.5: 0.305, 25: 0.380, 50: 0.405},
    },
}

RBM = "#315B7D"
POST = "#C45A3C"
INK = "#20252B"
MUTED = "#687078"
GRID = "#D9DEE3"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "font.size": 8.0,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


def curve(record, key):
    points = dict(record[key])
    points[100] = record["base"]
    x = sorted(points)
    y = [1.0 - points[r] / record["base"] for r in x]
    return x, y


fig, axes = plt.subplots(1, 4, figsize=(7.35, 1.95), sharey=True)
for ax, (name, record) in zip(axes, DATA.items()):
    for key, color, marker, style in (
        ("rbm", RBM, "o", "-"),
        ("post", POST, "s", (0, (4, 2.2))),
    ):
        x, y = curve(record, key)
        ax.plot(x, y, color=color, marker=marker, linestyle=style,
                linewidth=1.45, markersize=4.4, markeredgecolor="white",
                markeredgewidth=0.55, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlim(10, 125)
    ax.set_ylim(-0.035, 0.93)
    ax.xaxis.set_major_locator(FixedLocator([12.5, 25, 50, 100]))
    ax.xaxis.set_major_formatter(FixedFormatter([".125", ".25", ".50", "1"] ))
    ax.set_yticks([0, .25, .50, .75])
    ax.grid(axis="y", color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    ax.set_title(name, fontsize=9.0, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.3, length=2.5, width=.65)
    for spine in ax.spines.values():
        spine.set_linewidth(.7)
    ax.set_xlabel("rate $R=k/N$", fontsize=7.6, labelpad=2)

axes[0].set_ylabel("task distortion  $D=1-S(R)/S(1)$\n(lower is better)", fontsize=7.7)

handles = [
    Line2D([0], [0], color=RBM, marker="o", linewidth=1.45,
           markersize=4.4, markeredgecolor="white", markeredgewidth=.55,
           label="RBM"),
    Line2D([0], [0], color=POST, marker="s", linestyle=(0, (4, 2.2)),
           linewidth=1.45, markersize=4.4, markeredgecolor="white",
           markeredgewidth=.55, label="Post-L2"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.025),
           ncol=2, frameon=False, fontsize=7.8, handlelength=2.2,
           columnspacing=1.8)
fig.subplots_adjust(left=.086, right=.995, top=.76, bottom=.23, wspace=.18)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=.025)
print(OUT)

