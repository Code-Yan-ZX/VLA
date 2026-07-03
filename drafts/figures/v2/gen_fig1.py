#!/usr/bin/env python3
"""Fig 1 (v2) - The served-throughput gap (framework motivation).

Two bars over the 37-method landscape:
  A = 13/37 report some wall-clock-style number (offline CUDA latency / prefill / decode).
  B = 0/37 measure served throughput inside a production serving engine.
Re-cast as framework motivation: the gap is unfilled, and a framework (any boundary
compressor across engines/architectures) - not a single number - is what closes it.

Data: notes/lit-survey.md §2 comparison table (37 methods), parsed via the v1
loader _data.throughput_tally().
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))   # v1 _data.py + _style.py live one level up
sys.path.insert(0, str(HERE))     # _data_v2.py
import matplotlib.pyplot as plt
import _data as D            # v1 loader: parses lit-survey.md
import _style as S

S.apply_rc(fontsize=9)

rows, n_total, n_wall, n_dep = D.throughput_tally()
assert n_total == 37, f"expected 37 surveyed methods, got {n_total}"

fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 2.7))

labels = [
    f"Reports FLOPs /\ntoken-count",
    f"Reports any\nwall-clock",
    f"Measures served\nthroughput (engine)",
]
counts = [n_total, n_wall, n_dep]
# color: ramps from muted to the stark-red zero bar
colors = [S.INK_MUTED, S.CAT["yellow"], S.CAT["red"]]

bars = ax.bar(labels, counts, color=colors, width=0.62, edgecolor=S.SURFACE,
              linewidth=1.2, zorder=3)

# value labels on top of bars
for b, v in zip(bars, counts):
    txt = f"{v}/{n_total}" if v > 0 else "0/37"
    ax.text(b.get_x() + b.get_width() / 2, v + 0.7, txt,
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
            color=S.INK_PRIM if v > 0 else S.CAT["red"])

# the stark 0 callout
ax.annotate("0 of 37 measure served\nthroughput inside a\nserving engine",
            xy=(2, 0.4), xytext=(0.55, 14),
            fontsize=8.2, color=S.CAT["red"], ha="left", va="top",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=S.CAT["red"], lw=1.3,
                            connectionstyle="arc3,rad=-0.18"),
            bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE,
                      ec=S.CAT["red"], lw=0.9))

ax.set_ylim(0, n_total + 4)
ax.set_ylabel("# of surveyed methods  (of 37, 2023-2026)", color=S.INK_SEC)
ax.set_yticks([0, 10, 20, 30, 37])
S.style_axes(ax, grid_axis="y")
ax.margins(x=0.04)

# subtitle / source footnote
ax.text(0.0, -0.30,
        "Survey: 37 visual-token compressors, 2023-2026  "
        "(src: notes/lit-survey.md §2).",
        transform=ax.transAxes, fontsize=6.8, color=S.INK_MUTED, ha="left")

fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.22)
out = HERE / "fig1_gap.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
print(f"  n_total={n_total}  n_wallclock={n_wall}  n_engine={n_dep}")
