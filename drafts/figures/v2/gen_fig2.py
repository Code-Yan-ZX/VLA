#!/usr/bin/env python3
"""Fig 2 (v2) - Concurrency x prune curve, c1-c64, with ceiling-lift.

req/s (y) vs prune rate (x: r0 / r50 / r75), four curves for c in {1,4,16,64}
on LLaVA-1.5-7B / GQA / vLLM V1. Annotates:
  (a) the r75/r0 amplification 1.19x (c1) -> 2.22x (c64);
  (b) the ceiling-lift: r0 plateaus c16->c64 (+12%) while r75 keeps climbing (+26%).

Concurrency is ordered magnitude -> single-hue (blue) sequential ramp, light->dark.

Data: runs/v2_p2/batch_c{C}_r{R}.json (Table A / paper Table 1).
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt
import _style as S
import _data_v2 as D

S.apply_rc(fontsize=9)
m = D.scale_matrix()

PRUNES = [0.0, 0.50, 0.75]
CONCS = [1, 4, 16, 64]
# single-hue blue sequential ramp (light -> dark), steps 200/350/450/650
BLUE_RAMP = {1: "#9ec5f4", 4: "#5598e7", 16: "#2a78d6", 64: "#184f95"}
MARKERS = {1: "o", 4: "s", 16: "D", 64: "^"}

fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 3.0))

for c in CONCS:
    xs = [p for p in PRUNES if (c, p) in m]
    ys = [m[(c, p)]["req_s"] for p in xs]
    ax.plot(xs, ys, color=BLUE_RAMP[c], marker=MARKERS[c], markersize=6,
            linewidth=1.9, markeredgecolor=S.SURFACE, markeredgewidth=1.0,
            label=f"$c={c}$", zorder=4 if c == 64 else 3)

# ---- headline annotation: c64/r75 = 2.22x ----
c64_r0 = m[(64, 0.0)]["req_s"]; c64_r75 = m[(64, 0.75)]["req_s"]
speedup_64 = c64_r75 / c64_r0
ax.annotate(f"{speedup_64:.2f}$\\times$\nserved req/s",
            xy=(0.75, c64_r75), xytext=(0.32, 19.2),
            fontsize=8.8, fontweight="bold", color=BLUE_RAMP[64],
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=BLUE_RAMP[64], lw=1.3),
            bbox=dict(boxstyle="round,pad=0.28", fc=S.SURFACE,
                      ec=BLUE_RAMP[64], lw=0.9))

# ---- amplification annotation: r75/r0 1.19x (c1) -> 2.22x (c64) ----
s_c1 = m[(1, 0.75)]["req_s"] / m[(1, 0.0)]["req_s"]
ax.text(0.50, 3.6, f"r75/r0 amplification\n$c1$: {s_c1:.2f}$\\times$ "
        f"$\\rightarrow$ $c64$: {speedup_64:.2f}$\\times$",
        fontsize=7.3, color=S.INK_SEC, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc=S.SURFACE,
                  ec=S.GRIDLINE, lw=0.7))

# ---- ceiling-lift annotation: r0 plateaus c16->c64, r75 climbs ----
r0_c16, r0_c64 = m[(16, 0.0)]["req_s"], m[(64, 0.0)]["req_s"]
r75_c16, r75_c64 = m[(16, 0.75)]["req_s"], m[(64, 0.75)]["req_s"]
r0_growth = (r0_c64 / r0_c16 - 1) * 100
r75_growth = (r75_c64 / r75_c16 - 1) * 100
# double-arrow on r0 curve c16->c64 (plateau)
ax.annotate("", xy=(0.0, r0_c64), xytext=(0.0, r0_c16),
            arrowprops=dict(arrowstyle="<->", color=S.CAT["red"], lw=1.0))
ax.text(-0.075, (r0_c16 + r0_c64) / 2,
        f"r0\n+{r0_growth:.0f}%\n(plateau)",
        fontsize=6.8, color=S.CAT["red"], ha="right", va="center")
# double-arrow on r75 curve c16->c64 (climbing)
ax.annotate("", xy=(0.75, r75_c64), xytext=(0.75, r75_c16),
            arrowprops=dict(arrowstyle="<->", color=BLUE_RAMP[64], lw=1.0))
ax.text(0.835, (r75_c16 + r75_c64) / 2,
        f"r75\n+{r75_growth:.0f}%\n(climbing)",
        fontsize=6.8, color=BLUE_RAMP[64], ha="left", va="center")

ax.set_xticks(PRUNES)
ax.set_xticklabels(["$r0$\n(576 tok)", "$r50$\n(288 tok)", "$r75$\n(144 tok)"])
ax.set_xlabel("visual-token prune rate", color=S.INK_SEC)
ax.set_ylabel("served req/s  (LLaVA-1.5-7B, GQA, V1)", color=S.INK_SEC)
ax.set_xlim(-0.18, 0.95)
ax.set_ylim(0, 22.5)
S.style_axes(ax, grid_axis="y")
leg = ax.legend(loc="upper left", title="max\\_num\\_seqs",
                title_fontsize=7.5, ncol=2)
leg.get_title().set_color(S.INK_SEC)

ax.text(0.0, -0.26,
        "c64 = single-A40 r0 ceiling (peak KV $\\approx$41 GB). "
        "src: runs/v2_p2/batch\\_c{1,4,16,64}\\_r{0,50,75}.json",
        transform=ax.transAxes, fontsize=6.6, color=S.INK_MUTED, ha="left")

fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.25)
out = HERE / "fig2_concurrency_ceiling.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
print(f"  r75/r0: c1={s_c1:.2f}x  c64={speedup_64:.2f}x")
print(f"  ceiling-lift: r0 +{r0_growth:.0f}% (c16->c64), r75 +{r75_growth:.0f}%")
