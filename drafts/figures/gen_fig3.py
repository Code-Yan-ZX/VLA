#!/usr/bin/env python3
"""Fig 3 - Controller load-tracking (step profile, GQA).

Plots the controller's per-decision realized prune rate r and the engine-load
signal (concurrency fraction = num_running / max_num_seqs) over decision index,
under the low->high->low step load profile. Both quantities share the [0,1]
unit interval, so they render on a SINGLE shared y-axis (no dual-axis chart).

The load signal rises to 1.0 at the high-batch step; the controller responds
by raising r from r_min (0.25) to r_max (0.50) for the next segment (one-
segment-lag reactive loop), then returns to r_min for the low-load tail.

Data: runs/p2_d/p3s1_gqa_adaptive_step_mt32.json -> controller.realized (Table F2).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import _data as D
import _style as S

S.apply_rc(fontsize=9)
series, meta = D.step_profile()
i = np.array([e["i"] for e in series])
r = np.array([e["r"] for e in series])
cf = np.array([e["conc_frac"] for e in series])
r_min, r_max = series[0]["r_min"], series[0]["r_max"]
conc_lo, conc_hi = meta["conc_lo"], meta["conc_hi"]

# decision phases: low (one-at-a-time), high (batch of 60), tail
# identify the high-load event (where num_running spikes)
high_idx = int(i[np.argmax([e["num_running"] for e in series])])

fig, ax = plt.subplots(figsize=(S.DOUBLE_COL, 2.6))

# phase shading: low | high burst | tail
ax.axvspan(-2, high_idx - 0.5, color=S.GRIDLINE, alpha=0.45, zorder=0)
ax.axvspan(high_idx - 0.5, high_idx + 0.5, color=S.CAT["yellow"], alpha=0.20, zorder=0)
ax.axvspan(high_idx + 0.5, i[-1] + 2, color=S.GRIDLINE, alpha=0.45, zorder=0)
ytop = 1.06
ax.text((high_idx) / 2, ytop, "low load (1-at-a-time)", ha="center", va="bottom",
        fontsize=7.3, color=S.INK_SEC)
ax.text(high_idx, ytop, "high\nbatch", ha="center", va="bottom",
        fontsize=7.0, color=S.CAT["yellow"])
ax.text((high_idx + 1 + i[-1]) / 2, ytop, "low tail", ha="center", va="bottom",
        fontsize=7.3, color=S.INK_SEC)

# load signal: concurrency fraction (input the controller reads)
ax.step(i, cf, where="mid", color=S.CAT["aqua"], linewidth=1.4, alpha=0.85,
        label=f"concurrency fraction  ($n_{{run}}/{meta['max_num_seqs']}$)", zorder=2)

# controller output: realized r (step line)
ax.step(i, r, where="mid", color=S.CAT["blue"], linewidth=2.2,
        label="realized prune rate  $r$", zorder=4)
ax.scatter(i, r, s=9, color=S.CAT["blue"], zorder=5, edgecolor=S.SURFACE, linewidth=0.4)

# r_min / r_max reference lines
ax.axhline(r_min, color=S.INK_MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.axhline(r_max, color=S.INK_MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.text(i[-1] + 1.5, r_min, f"$r_{{min}}={r_min:.2f}$", va="center", fontsize=7,
        color=S.INK_SEC)
ax.text(i[-1] + 1.5, r_max, f"$r_{{max}}={r_max:.2f}$", va="center", fontsize=7,
        color=S.INK_SEC)

# controller breakpoints on the load axis (conc_lo / conc_hi)
for cb, lab in [(conc_lo, "$conc_{lo}$"), (conc_hi, "$conc_{hi}$")]:
    ax.axhline(cb, color=S.INK_MUTED, lw=0.5, ls=":", zorder=1)
    ax.text(-3.5, cb, lab, va="center", ha="right", fontsize=6.5, color=S.INK_SEC)

# annotate the controller response at the high-batch step
ax.annotate("controller raises\n$r$: $r_{min}\\!\\rightarrow\\!r_{max}$",
            xy=(high_idx, r_max), xytext=(high_idx + 18, 0.78),
            fontsize=7.6, color=S.CAT["blue"], ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color=S.CAT["blue"], lw=1.1),
            bbox=dict(boxstyle="round,pad=0.22", fc=S.SURFACE,
                      ec=S.CAT["blue"], lw=0.7))

ax.set_xlim(-5, i[-1] + 6)
ax.set_ylim(-0.03, 1.18)
ax.set_xlabel("decision index (per segment)", color=S.INK_SEC)
ax.set_ylabel("fraction / prune rate  (both in [0, 1])", color=S.INK_SEC)
S.style_axes(ax, grid_axis="y")
ax.legend(loc="upper right", fontsize=7.6)

fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.22)
out = Path(__file__).resolve().parent / "fig3_controller.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}  | n_decisions={meta['n_decisions']}  "
      f"high_idx={high_idx}  (r={r.min()}..{r.max()}, conc_frac={cf.min()}..{cf.max()})")
