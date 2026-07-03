#!/usr/bin/env python3
"""Fig 2 - Concurrency x prune curve (GQA, served req/s).

req/s (y) vs prune rate (x: 0 / 0.50 / 0.75), three curves for
max_num_seqs in {1, 4, 12}. Annotates the c12/r75 = 1.75x headline and the
c1->c12 amplification (r50: 1.17x->1.42x; r75: 1.26x->1.75x).

Concurrency is an ordered magnitude, so curves use a single-hue (blue)
sequential ramp light->dark (the dataviz sequential rule).

Data: runs/p2_d/m2_c{1,4,12}_r{0,50,75}.json (Table A).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import _data as D
import _style as S

S.apply_rc(fontsize=9)
m = D.m2_matrix()

# Speedup annotations: displayed values are the paper-stated figures from
# eval/final_results.md Table A (1.17/1.42/1.26/1.75). NOTE: the c12/r75 ratio
# computed from the raw JSON is 1.7546 -> 1.75x at 2 dp; Table A rounds it to
# 1.75x. We display Table A's 1.75x so the figure agrees with the paper text;
# the raw value is logged below for transparency. See digest flag to main.
STATED = {(1, 0.50): 1.17, (12, 0.50): 1.42,
          (1, 0.75): 1.26, (12, 0.75): 1.75}

PRUNES = [0.0, 0.50, 0.75]
CONCS = [1, 4, 12]
# single-hue blue sequential ramp (light -> dark) for c1 -> c12
BLUE_RAMP = {1: "#86b6ef", 4: "#2a78d6", 12: "#184f95"}  # steps 250/450/650
MARKERS = {1: "o", 4: "s", 12: "D"}

fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 2.9))

for c in CONCS:
    xs = [p for p in PRUNES if (c, p) in m]
    ys = [m[(c, p)] for p in xs]
    ax.plot(xs, ys, color=BLUE_RAMP[c], marker=MARKERS[c], markersize=6,
            linewidth=1.8, markeredgecolor=S.SURFACE, markeredgewidth=1.0,
            label=f"$c={c}$" if c != 1 else "$c=1$ (no batching)",
            zorder=3 if c == 12 else 2)

# ---- headline annotation: c12/r75 = 1.75x (per Table A) ----
c12_r0 = m[(12, 0.0)]; c12_r75 = m[(12, 0.75)]
speedup_75 = c12_r75 / c12_r0  # raw: 1.7546
ax.annotate(f"{STATED[(12, 0.75)]:.2f}$\\times$\nserved req/s",
            xy=(0.75, c12_r75), xytext=(0.40, 11.3),
            fontsize=8.5, fontweight="bold", color=BLUE_RAMP[12],
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=BLUE_RAMP[12], lw=1.3),
            bbox=dict(boxstyle="round,pad=0.25", fc=S.SURFACE,
                      ec=BLUE_RAMP[12], lw=0.8))

# ---- amplification annotations (c1 -> c12 speedup growth, Table A values) ----
s_r50_c1, s_r50_c12 = STATED[(1, 0.50)], STATED[(12, 0.50)]
ax.text(0.50, 3.7, f"r50: {s_r50_c1:.2f}$\\times$$\\rightarrow${s_r50_c12:.2f}$\\times$\n($c1\\rightarrow c12$)",
        fontsize=7.3, color=S.INK_SEC, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc=S.SURFACE,
                  ec=S.GRIDLINE, lw=0.7))
# r75 annotation (c1 -> c12)
s_r75_c1, s_r75_c12 = STATED[(1, 0.75)], STATED[(12, 0.75)]
ax.annotate("", xy=(0.75, m[(12, 0.75)]), xytext=(0.75, m[(1, 0.75)]),
            arrowprops=dict(arrowstyle="<->", color=S.INK_MUTED, lw=0.9))
ax.text(0.855, (m[(12, 0.75)] + m[(1, 0.75)]) / 2,
        f"r75\n{s_r75_c1:.2f}$\\rightarrow${s_r75_c12:.2f}$\\times$",
        fontsize=7.0, color=S.INK_SEC, ha="left", va="center")

ax.set_xticks(PRUNES)
ax.set_xticklabels(["$r0$\n(576 tok)", "$r50$\n(288 tok)", "$r75$\n(144 tok)"])
ax.set_xlabel("visual-token prune rate", color=S.INK_SEC)
ax.set_ylabel("served req/s  (GQA, batch-submit)", color=S.INK_SEC)
ax.set_xlim(-0.08, 0.93)
ax.set_ylim(0, 12.5)
S.style_axes(ax, grid_axis="y")
leg = ax.legend(loc="upper left", title="max\\_num\\_seqs", title_fontsize=7.5)
leg.get_title().set_color(S.INK_SEC)

fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.20)
out = Path(__file__).resolve().parent / "fig2_concurrency_prune.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
print(f"  c12/r75 raw ratio = {speedup_75:.4f}x (Table A states 1.75x; "
      f"raw rounds to {speedup_75:.2f}x) -- FLAG to main")
print(f"  displayed (Table A): r50 {s_r50_c1:.2f}->{s_r50_c12:.2f}; "
      f"r75 {s_r75_c1:.2f}->{s_r75_c12:.2f}")
