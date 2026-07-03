#!/usr/bin/env python3
"""Fig 4 (v2) - Architecture-conditional amplification (honest boundary).

r75/r0 served-speedup (y) vs concurrency c (x, log scale), two curves:
  - LLaVA-1.5-7B (strong, steep):  c1 1.19x -> c4 1.53x -> c16 1.96x -> c64 2.22x
  - Qwen3-VL-8B  (attenuated):     c1 1.08x -> c12 1.29x -> c64 1.34x (r50/r0)
The native 2x2 merger substitutes for post-hoc pruning -> diminishing returns on
modern architectures. Mechanism callout + ~3x attenuation bracket at c~max.

Data: runs/v2_p2 (LLaVA c1-c64; Qwen3-VL c64) + notes/v2_p1_qwen3vl.md (Qwen3-VL
c1/c12). Qwen3-VL c64 measured only at r0/r50, so c64 uses r50/r0 (annotated).
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
ac = D.arch_conditional_amplification()
llava = ac["LLaVA-1.5-7B"]      # {c: (r75/r0, r0, r75)}  c in {1,4,16,64}
qwen = ac["Qwen3-VL-8B"]        # {c: (ratio, r0, r_top)} c in {1,12,64}
# Qwen3-VL c64 ratio is r50/r0 (r75 not measured); c1/c12 are r75/r0.

fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 3.0))

# LLaVA: full r75/r0 curve
lx = sorted(llava.keys())
ly = [llava[c][0] for c in lx]
ax.plot(lx, ly, color=S.CAT["blue"], marker="o", markersize=7,
        linewidth=2.0, markeredgecolor=S.SURFACE, markeredgewidth=1.1,
        label="LLaVA-1.5-7B  (fixed 576 tok, $r_{75}/r_0$)", zorder=4)
for c in lx:
    ax.annotate(f"{llava[c][0]:.2f}$\\times$", xy=(c, llava[c][0]),
                xytext=(0, 9), textcoords="offset points",
                fontsize=7.0, color=S.CAT["blue"], ha="center",
                fontweight="bold")

# Qwen3-VL: split r75/r0 (c1,c12) and r50/r0 (c64)
qx_full = [c for c in qwen if c in (1, 12)]   # r75/r0
qy_full = [qwen[c][0] for c in qx_full]
ax.plot(qx_full, qy_full, color=S.CAT["violet"], marker="s", markersize=7,
        linewidth=2.0, markeredgecolor=S.SURFACE, markeredgewidth=1.1,
        label="Qwen3-VL-8B  (native 2$\\times$2 merger, $r_{75}/r_0$)", zorder=3)
for c in qx_full:
    ax.annotate(f"{qwen[c][0]:.2f}$\\times$", xy=(c, qwen[c][0]),
                xytext=(0, -14), textcoords="offset points",
                fontsize=7.0, color=S.CAT["violet"], ha="center",
                fontweight="bold")
# Qwen3-VL c64 (r50/r0) - distinct marker, dashed connector
ax.plot([12, 64], [qwen[12][0], qwen[64][0]], color=S.CAT["violet"],
        linewidth=1.4, linestyle="--", zorder=2)
ax.scatter([64], [qwen[64][0]], color=S.CAT["violet"], marker="D", s=48,
           edgecolor=S.SURFACE, linewidth=1.1, zorder=3)
ax.annotate(f"{qwen[64][0]:.2f}$\\times$\n($r_{50}/r_0$)", xy=(64, qwen[64][0]),
            xytext=(0, -16), textcoords="offset points",
            fontsize=6.8, color=S.CAT["violet"], ha="center", fontweight="bold")

# 1.0x reference line (no benefit)
ax.axhline(1.0, color=S.INK_MUTED, linewidth=0.8, linestyle=":", zorder=1)
ax.text(1.05, 1.02, "no speedup", fontsize=6.6, color=S.INK_MUTED, va="bottom")

# attenuation bracket between the two c~max points (c64)
y_top = llava[64][0]; y_bot = qwen[64][0]
ax.annotate("", xy=(80, y_bot), xytext=(80, y_top),
            arrowprops=dict(arrowstyle="<->", color=S.INK_SEC, lw=0.9))
ax.text(95, (y_top + y_bot) / 2,
        f"$\\sim${y_top/y_bot:.1f}$\\times$\nattenuated",
        fontsize=7.0, color=S.INK_SEC, ha="left", va="center")

# mechanism callout
ax.text(0.50, 0.78,
        "Native 2$\\times$2 MLP merger compresses\n"
        "post-encoder tokens BEFORE the pruner.\n"
        "Merger and post-hoc pruner are SUBSTITUTES\n"
        "$\\rightarrow$ diminishing returns on modern arch.",
        transform=ax.transAxes, fontsize=7.0, color=S.INK_PRIM,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE,
                  ec=S.GRIDLINE, lw=0.7))

ax.set_xscale("log")
ax.set_xticks([1, 4, 12, 16, 64])
ax.set_xticklabels(["c1", "c4", "c12", "c16", "c64"])
ax.set_xlabel("max\\_num\\_seqs  (concurrency)", color=S.INK_SEC)
ax.set_ylabel("served-req/s speedup vs $r_0$", color=S.INK_SEC)
ax.set_ylim(0.95, 2.55)
ax.set_xlim(0.85, 130)
S.style_axes(ax, grid_axis="y")
leg = ax.legend(loc="lower right", fontsize=6.8, labelspacing=0.3)
ax.text(0.0, -0.27,
        "LLaVA: runs/v2_p2  |  Qwen3-VL c1,c12: notes/v2_p1\\_qwen3vl.md, "
        "c64: runs/v2_p2/qwen3vl\\_c64\\_r{0,50}.json",
        transform=ax.transAxes, fontsize=6.2, color=S.INK_MUTED, ha="left")

fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.22)
out = HERE / "fig4_arch_conditional.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
print(f"  LLaVA r75/r0: " + ", ".join(f"c{c}={llava[c][0]:.2f}x" for c in lx))
print(f"  Qwen3-VL: " + ", ".join(f"c{c}={qwen[c][0]:.2f}x" for c in sorted(qwen)))
print(f"  attenuation @c64: {y_top/y_bot:.2f}x")
