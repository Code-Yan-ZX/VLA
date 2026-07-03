#!/usr/bin/env python3
"""Fig 5 (v2) - Cross-compressor panel at c64 (framework generality).

Two panels at c64 (LLaVA-1.5-7B, GQA, V1, n=200):
  (a) req/s vs prune rate r  - 4 compressors OVERLAP at iso-k (framework property:
      served throughput is compressor-invariant; only ToMe's merge-compute shows
      a ~1.5-2% dip).
  (b) accuracy vs r  - ToMe-merge HIGHEST at r75 (info preservation); saliency
      selectors (proxy, true_cls) UNDERPERFORM uniform random at r75 (the honest
      FastV-style red flag).

Data: runs/v2_p3/{proxy,true_cls,tome_merge,random}_c64_r{0,50,75}.json
(paper Tables 4-5 / notes/v2_p3_crosscompressor.md Tables 1, 3).
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
cc = D.cross_compressor_c64()

PRUNES = [0.0, 0.50, 0.75]
PRUNE_LBL = ["$r0$", "$r50$", "$r75$"]
# slot order: proxy blue, true_cls violet, tome aqua (merge - distinct), random muted
CMAP = {
    "proxy":      (S.CAT["blue"],   "o", "proxy (saliency, prune)"),
    "true_cls":   (S.CAT["violet"], "s", "true\\_cls (CLS-attn, prune)"),
    "tome_merge": (S.CAT["aqua"],   "D", "ToMe (cos-sim, MERGE)"),
    "random":     (S.INK_MUTED,     "^", "random (uniform, prune)"),
}

fig, (ax_t, ax_a) = plt.subplots(
    1, 2, figsize=(S.DOUBLE_COL, 2.9), sharex=True)

# ---- panel (a): req/s vs r ----
for sel in D.P3_COMPRESSORS:
    col, mk, lbl = CMAP[sel]
    xs = [r for r in PRUNES if (sel, r) in cc]
    ys = [cc[(sel, r)]["req_s"] for r in xs]
    lw = 1.5 if sel != "tome_merge" else 2.0
    ax_t.plot(xs, ys, color=col, marker=mk, markersize=6.5, linewidth=lw,
              markeredgecolor=S.SURFACE, markeredgewidth=1.0, label=lbl, zorder=3)
    # r75 value label
    ax_t.annotate(f"{cc[(sel, 0.75)]['req_s']:.1f}",
                  xy=(0.75, cc[(sel, 0.75)]["req_s"]),
                  xytext=(8, 0), textcoords="offset points",
                  fontsize=6.4, color=col, va="center")

# iso-k band annotation: spread at r75
r75_vals = [cc[(sel, 0.75)]["req_s"] for sel in D.P3_COMPRESSORS]
spread = (max(r75_vals) - min(r75_vals)) / (sum(r75_vals) / len(r75_vals)) * 100
ax_t.text(0.50, 21.2, f"compressor-invariant\nat iso-$k$\n(spread $\\leq${spread:.0f}% at $r75$)",
          fontsize=6.8, color=S.INK_PRIM, ha="center", va="top",
          bbox=dict(boxstyle="round,pad=0.25", fc=S.SURFACE,
                    ec=S.GRIDLINE, lw=0.7))
ax_t.set_xticks(PRUNES)
ax_t.set_xticklabels(PRUNE_LBL)
ax_t.set_ylim(7, 23)
ax_t.set_ylabel("served req/s  (higher = better)", color=S.INK_SEC)
ax_t.set_title("(a) Throughput vs prune rate", fontsize=8.5,
               color=S.INK_PRIM, pad=4)
S.style_axes(ax_t, grid_axis="y")

# ---- panel (b): accuracy vs r ----
for sel in D.P3_COMPRESSORS:
    col, mk, lbl = CMAP[sel]
    xs = [r for r in PRUNES if (sel, r) in cc]
    ys = [cc[(sel, r)]["acc"] for r in xs]
    lw = 1.5 if sel != "tome_merge" else 2.0
    ax_a.plot(xs, ys, color=col, marker=mk, markersize=6.5, linewidth=lw,
              markeredgecolor=S.SURFACE, markeredgewidth=1.0, label=lbl, zorder=3)
    ax_a.annotate(f"{cc[(sel, 0.75)]['acc']:.3f}",
                  xy=(0.75, cc[(sel, 0.75)]["acc"]),
                  xytext=(8, 0), textcoords="offset points",
                  fontsize=6.4, color=col, va="center")

# red-flag callout: random (muted) beats saliency at r75
rand_r75 = cc[("random", 0.75)]["acc"]
proxy_r75 = cc[("proxy", 0.75)]["acc"]
ax_a.annotate(f"random $>$ saliency\nat $r75$\n({proxy_r75:.3f} vs {rand_r75:.3f})",
              xy=(0.75, proxy_r75), xytext=(0.42, 0.43),
              fontsize=6.8, color=S.CAT["red"], ha="center", va="center",
              fontweight="bold",
              arrowprops=dict(arrowstyle="-|>", color=S.CAT["red"], lw=1.0,
                              connectionstyle="arc3,rad=0.18"),
              bbox=dict(boxstyle="round,pad=0.25", fc=S.SURFACE,
                        ec=S.CAT["red"], lw=0.7))
ax_a.set_xticks(PRUNES)
ax_a.set_xticklabels(PRUNE_LBL)
ax_a.set_ylim(0.40, 0.62)
ax_a.set_ylabel("GQA accuracy", color=S.INK_SEC)
ax_a.set_title("(b) Accuracy vs prune rate", fontsize=8.5,
               color=S.INK_PRIM, pad=4)
S.style_axes(ax_a, grid_axis="y")

# single shared legend below
handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.0,
           bbox_to_anchor=(0.5, -0.015), frameon=False, columnspacing=1.2)

fig.suptitle("Cross-compressor panel at $c=64$  (LLaVA-1.5-7B, GQA, V1, n=200)",
             fontsize=9.5, color=S.INK_PRIM, y=1.02)
fig.text(0.5, -0.04,
         "src: runs/v2_p3/{proxy,true\\_cls,tome\\_merge,random}\\_c64\\_r{0,50,75}.json",
         fontsize=6.4, color=S.INK_MUTED, ha="center")

fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20, wspace=0.22)
out = HERE / "fig5_crosscompressor.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"wrote {out}")
print(f"  r75 req/s spread: {min(r75_vals):.2f}-{max(r75_vals):.2f} ({spread:.1f}%)")
print(f"  r75 acc: proxy={cc[('proxy',0.75)]['acc']:.3f}  "
      f"cls={cc[('true_cls',0.75)]['acc']:.3f}  "
      f"tome={cc[('tome_merge',0.75)]['acc']:.3f}  "
      f"random={cc[('random',0.75)]['acc']:.3f}")
