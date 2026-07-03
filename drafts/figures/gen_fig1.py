#!/usr/bin/env python3
"""Fig 1 - The served-throughput gap.

Two bars over the 37-method landscape:
  A = 13/37 report *any* wall-clock-style number (offline CUDA / prefill / decode)
  B =  0/37 measure served throughput inside a production serving engine
Bar B (the 0) is the headline novelty and is rendered starkly.

Data: notes/lit-survey.md §2 table (parsed via _data.throughput_tally).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import _data as D
import _style as S

S.apply_rc(fontsize=9)
rows, n_total, n_wall, n_deploy = D.throughput_tally()

fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 2.7))

# reference: the full 37-method landscape
ax.axhline(n_total, color=S.INK_MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
ax.text(1.46, n_total, f"{n_total} methods surveyed (2023-2026)",
        ha="right", va="bottom", fontsize=7.5, color=S.INK_SEC)

# bar A: wall-clock reporters (filled)
blue = S.CAT["blue"]
ax.bar(0, n_wall, width=0.55, color=blue, edgecolor="none", zorder=3)
ax.text(0, n_wall + 0.6, f"{n_wall}/{n_total}", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=S.INK_PRIM)
ax.text(0, -2.6, "Reports a wall-clock\nnumber (offline CUDA /\nprefill / decode)",
        ha="center", va="top", fontsize=7.5, color=S.INK_SEC)

# bar B: serving-engine measurements (0) - stark hollow bar
x1 = 1
bar_h = 0  # literally zero
# draw a hollow ghost bar (full outline at a token height for visibility)
ghost = 0.9
ax.add_patch(Rectangle((x1 - 0.275, 0), 0.55, ghost,
                       facecolor="none", edgecolor=S.CAT["red"],
                       lw=1.6, ls=(0, (4, 2)), zorder=3))
# stark 0 callout
ax.annotate("0", xy=(x1, ghost), xytext=(x1, 12.5),
            ha="center", va="center", fontsize=34, fontweight="bold",
            color=S.CAT["red"],
            arrowprops=dict(arrowstyle="-|>", color=S.CAT["red"], lw=1.4,
                            shrinkA=0, shrinkB=2))
ax.text(x1, 11.0, f"0 / {n_total}", ha="center", va="center",
        fontsize=10, fontweight="bold", color=S.CAT["red"])
ax.text(x1, -2.6, "Measures served throughput\ninside a production\nserving engine",
        ha="center", va="top", fontsize=7.5, color=S.INK_SEC)

ax.set_xticks([])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-6.5, n_total + 6.5)
ax.set_ylabel("# surveyed methods", color=S.INK_SEC)
S.style_axes(ax, grid_axis="y")
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)

fig.subplots_adjust(left=0.16, right=0.97, top=0.80, bottom=0.30)
fig.suptitle("0 of 37 VLM token-compression methods measure served throughput\n"
             "inside a production serving engine",
             x=0.06, ha="left", fontsize=9.5, fontweight="bold",
             color=S.INK_PRIM, y=0.97)

out = Path(__file__).resolve().parent / "fig1_gap.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
