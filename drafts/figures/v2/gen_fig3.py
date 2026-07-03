#!/usr/bin/env python3
"""Fig 3 (v2) - Goodput-Pareto at c64 (the deployment figure / paper's lead).

Main panel: served req/s (x) vs p99-TTFT (y, lower is better, log scale), three
points r0/r50/r75 at c64. r75 sits at the upper-right (high throughput, low tail
latency) -> STRICTLY DOMINATES r0 (lower-left). Annotated "2.22x throughput AND
2.84x lower p99 - no tradeoff, pure win."

Inset: goodput (req/s under SLO) vs SLO threshold (TTFT), three curves, showing
r75's 7.4x goodput at TTFT<=5s.

Data: runs/v2_p2/batch_c64_r{0,50,75}.json (Tables B, C / paper Table 3).
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import _style as S
import _data_v2 as D

S.apply_rc(fontsize=9)
m = D.scale_matrix()

RATES = [(0.0, "r0", S.CAT["red"]), (0.50, "r50", S.CAT["yellow"]),
         (0.75, "r75", S.CAT["aqua"])]
pts = {tag: (m[(64, r)]["req_s"], m[(64, r)]["ttft_p99"])
       for r, tag, _ in RATES}

# ratios for annotation
r0_req, r0_p99 = pts["r0"]
r75_req, r75_p99 = pts["r75"]
r50_req, r50_p99 = pts["r50"]
ratio_tput = r75_req / r0_req
ratio_p99 = r0_p99 / r75_p99

# ---- main panel ----
fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 3.3))

# shade the dominated region (r0 is dominated by r75)
xs = np.array([r0_req, r75_req])
ax.fill_betweenx([r0_p99, 22000], 0, r0_req, color=S.CAT["red"],
                 alpha=0.05, zorder=1)
ax.fill_between([0, r0_req], r0_p99, 22000, color=S.CAT["red"],
                alpha=0.05, zorder=1)

for r, tag, col in RATES:
    x, y = pts[tag]
    ax.scatter([x], [y], s=70, color=col, edgecolor=S.SURFACE,
               linewidth=1.4, zorder=4)
    # log axes: x offset additive, y offset MULTIPLICATIVE (must stay > 0).
    # r0/r50 labels go up-left of their points; r75 label goes down-right.
    cfg = {"r0": (-0.55, 1.18, "right", "bottom"),
           "r50": (-0.55, 1.18, "right", "bottom"),
           "r75": (0.50, 0.82, "left", "top")}
    dx, ymul, ha, va = cfg[tag]
    ax.annotate(f"${tag}$\n{x:.1f} req/s\np99 {y/1000:.1f}s",
                xy=(x, y), xytext=(x + dx, y * ymul),
                fontsize=7.6, color=col, ha=ha, va=va, fontweight="bold")

# the dominance arrow r0 -> r75
arr = FancyArrowPatch((r0_req, r0_p99), (r75_req, r75_p99),
                      arrowstyle="-|>", color=S.INK_MUTED, lw=1.3,
                      connectionstyle="arc3,rad=0.18", zorder=3)
ax.add_patch(arr)

# headline annotation box - placed in the empty lower-left region of the
# log-log Pareto (no point has both low throughput and low tail latency).
ax.text(0.04, 0.30,
        f"r75 STRICTLY DOMINATES r0\n"
        f"{ratio_tput:.2f}$\\times$ throughput AND\n"
        f"{ratio_p99:.2f}$\\times$ lower p99-TTFT\n"
        f"no tradeoff, pure win",
        transform=ax.transAxes, fontsize=7.6, color=S.INK_PRIM,
        ha="left", va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", fc=S.SURFACE,
                  ec=S.INK_SEC, lw=0.8))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("served req/s  (higher is better)", color=S.INK_SEC)
ax.set_ylabel("p99 TTFT  [ms]  (lower is better)", color=S.INK_SEC)
ax.set_xlim(7, 28)
ax.set_ylim(5000, 23000)
ax.set_xticks([8, 10, 15, 20])
ax.set_xticklabels(["8", "10", "15", "20"])
ax.set_yticks([6000, 8000, 10000, 15000, 20000])
ax.set_yticklabels(["6k", "8k", "10k", "15k", "20k"])
S.style_axes(ax, grid_axis="both")
ax.set_title("Goodput-Pareto at $c=64$  (LLaVA-1.5-7B, GQA, V1, n=200)",
             fontsize=9, color=S.INK_PRIM, pad=4)

# ---- inset: goodput vs SLO threshold ----
ax_in = fig.add_axes([0.62, 0.59, 0.35, 0.34])  # [l, b, w, h]
SLOS = list(range(500, 12000, 250))
gp = D.goodput_sweep_c64(slos_ms=SLOS)
for r, tag, col in RATES:
    ys = [gp[(r, s)] for s in SLOS]
    ax_in.plot(SLOS, ys, color=col, linewidth=1.7,
               marker="", label=f"${tag}$", zorder=3)
# 5s SLO marker
ax_in.axvline(5000, color=S.INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
gp5_r0 = gp[(0.0, 5000)]; gp5_r75 = gp[(0.75, 5000)]
ax_in.text(5000, 0.2, "5s", fontsize=6.5, color=S.INK_MUTED, ha="right")
ax_in.text(0.97, 0.92, f"goodput@5s\nr75/r0 = {gp5_r75/gp5_r0:.1f}$\\times$",
           transform=ax_in.transAxes, fontsize=6.6, color=S.INK_PRIM,
           ha="right", va="top",
           bbox=dict(boxstyle="round,pad=0.2", fc=S.SURFACE,
                     ec=S.GRIDLINE, lw=0.6))
ax_in.set_xlabel("TTFT SLO  [s]", fontsize=7, color=S.INK_SEC)
ax_in.set_ylabel("goodput (req/s)", fontsize=7, color=S.INK_SEC)
ax_in.set_title("goodput vs SLO", fontsize=7.2, color=S.INK_PRIM)
ax_in.tick_params(labelsize=6.5)
# relabel x in seconds
xt = [1000, 3000, 5000, 7500, 10000]
ax_in.set_xticks(xt)
ax_in.set_xticklabels(["1", "3", "5", "7.5", "10"])
S.style_axes(ax_in, grid_axis="y")

fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.14)
out = HERE / "fig3_goodput_pareto.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
print(f"  r75/r0 throughput = {ratio_tput:.2f}x; p99 reduction = {ratio_p99:.2f}x")
print(f"  goodput@5s: r0={gp5_r0:.2f}  r75={gp5_r75:.2f}  ratio={gp5_r75/gp5_r0:.1f}x")
