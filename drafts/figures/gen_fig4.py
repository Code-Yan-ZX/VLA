#!/usr/bin/env python3
"""Fig 4 - Method Pareto frontier at n=500 (honest).

req/s (x) vs accuracy (y), 5 benchmarks x 3 configs = 15 points.
Color = benchmark (fixed categorical order); marker shape = config.
The three configs per benchmark are joined by a thin trajectory line; the
adaptive marker is emphasized. Caption states the honest n=500 verdict:
adaptive dominates r25 on req/s on 3/5 (MMBench/ScienceQA/TextVQA) but does
NOT beat fixed-r50 on accuracy anywhere.

Data: eval/final_results.md Table C-n500 (parsed via _data.pareto_n500).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _data as D
import _style as S

S.apply_rc(fontsize=9)
P = D.pareto_n500()

# benchmark -> categorical slot (fixed order, never cycle)
BENCH_SLOT = {
    "GQA":       "blue",
    "TEXTVQA":   "aqua",
    "MME":       "yellow",
    "MMBENCH":   "violet",
    "SCIENCEQA": "orange",
}
BENCH_LABEL = {"GQA": "GQA", "TEXTVQA": "TextVQA", "MME": "MME",
               "MMBENCH": "MMBench", "SCIENCEQA": "ScienceQA"}
# config -> marker
CFG_MARKER = {"fixed r25": "s", "adaptive": "o", "fixed r50": "^"}
CFG_LABEL = {"fixed r25": "fixed r25 (acc-favoring)",
             "adaptive": "adaptive (ours)",
             "fixed r50": "fixed r50 (tput-favoring)"}
# plot order so adaptive sits on top
CFG_ORDER = ["fixed r50", "fixed r25", "adaptive"]

fig, ax = plt.subplots(figsize=(S.SINGLE_COL, 3.0))

# join the 3 configs per benchmark with a thin trajectory line
for bench, slot in BENCH_SLOT.items():
    if bench not in P:
        continue
    pts = [(P[bench][c]["req_s"], P[bench][c]["acc"]) for c in
           ["fixed r25", "adaptive", "fixed r50"]]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    # order by req/s for a clean monotonic line
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    xs = [xs[k] for k in order]; ys = [ys[k] for k in order]
    ax.plot(xs, ys, color=S.CAT[slot], lw=0.9, alpha=0.45, zorder=2)

# points
for cfg in CFG_ORDER:
    for bench, slot in BENCH_SLOT.items():
        if bench not in P or cfg not in P[bench]:
            continue
        d = P[bench][cfg]
        kw = dict(marker=CFG_MARKER[cfg], s=55 if cfg == "adaptive" else 38,
                  color=S.CAT[slot],
                  edgecolors=S.INK_PRIM if cfg == "adaptive" else S.SURFACE,
                  linewidths=1.4 if cfg == "adaptive" else 0.8,
                  zorder=6 if cfg == "adaptive" else 5)
        ax.scatter(d["req_s"], d["acc"], **kw)

# direct-label benchmarks at their adaptive point (relief rule: aqua/yellow
# are sub-3:1 on light, so labels carry identity alongside color)
for bench, slot in BENCH_SLOT.items():
    if bench not in P:
        continue
    d = P[bench]["adaptive"]
    # nudge label off the point
    dx, dy = (0.02, 0.012)
    ha = "left"
    if bench == "SCIENCEQA":
        dx, dy = (-0.03, 0.012); ha = "right"
    if bench == "TEXTVQA":
        dx, dy = (-0.02, -0.018); ha = "right"
    ax.text(d["req_s"] + dx, d["acc"] + dy, BENCH_LABEL[bench],
            fontsize=7.8, color=S.INK_PRIM, ha=ha, va="center",
            fontweight="bold")

ax.set_xlabel("served req/s  (higher = faster)", color=S.INK_SEC)
ax.set_ylabel("accuracy  (higher = better)", color=S.INK_SEC)
S.style_axes(ax, grid_axis="both")

# two legends: config (marker) + benchmark (color)
cfg_handles = [Line2D([0], [0], marker=CFG_MARKER[c], color="none",
                      markerfacecolor=S.INK_SEC, markeredgecolor=S.INK_PRIM if c == "adaptive" else S.INK_SEC,
                      markersize=7 if c == "adaptive" else 6,
                      markeredgewidth=1.4 if c == "adaptive" else 0.8,
                      label=CFG_LABEL[c]) for c in CFG_ORDER]
leg1 = ax.legend(handles=cfg_handles, loc="lower right",
                 fontsize=7.2, title="config", title_fontsize=7.4)
leg1.get_title().set_color(S.INK_SEC)
ax.add_artist(leg1)

ax.set_xlim(2.15, 3.45)
ax.set_ylim(0.495, 0.80)

fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.17)
out = Path(__file__).resolve().parent / "fig4_pareto.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
