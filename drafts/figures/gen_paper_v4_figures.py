"""Paper v4 figure set (submission-grade, vector, colorblind-safe).

Fig1 pipeline schematic | Fig2 mechanism (M1 + M3) | Fig3 retention-vs-gap
curves (core) | Fig4 qualitative examples.

Data (read-only): runs/full_matrix/ablations/j8_summary.json,
runs/full_matrix/j7_main_table.json, drafts/mechanism_verification_report.md,
drafts/qualitative_examples.md. Style: _style.py (dataviz validated palette).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import _style as S

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, AQUA, GREEN = S.CAT["blue"], S.CAT["red"], S.CAT["aqua"], S.CAT["green"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Fig 3 — retention-vs-gap curves (CORE)
# ---------------------------------------------------------------------------
def fig3():
    S.apply_rc(fontsize=9)
    # official scores; retention x = [100, 75, 50, 25]
    # 100% = none/full; 75/50% = j8 n=200 subset (L2); 25% = j7 full split (L2)
    panels = [
        dict(title="Qwen3-VL-8B · TextVQA (VQA acc)",
             pre=[0.844, 0.740, 0.670, 0.605],
             post=[0.844, 0.653, 0.380, 0.222],
             gap="+38.4 pp", ylim=(0.12, 0.94), note=None),
        dict(title="Qwen3-VL-8B · DocVQA (ANLS)",
             pre=[None, 0.687, 0.589, 0.481],
             post=[None, 0.700, 0.531, 0.238],
             gap="+24.3 pp", ylim=(0.12, 0.82),
             note="100%: native eval. n/a\n(huge-image skip)"),
        dict(title="Qwen2.5-VL-7B · TextVQA (VQA acc)",
             pre=[0.862, 0.870, 0.813, 0.702],
             post=[0.862, 0.800, 0.660, 0.442],
             gap="+26.1 pp", ylim=(0.32, 0.96), note=None),
        dict(title="Qwen2.5-VL-7B · DocVQA (ANLS)",
             pre=[0.949, 0.946, 0.875, 0.636],
             post=[0.949, 0.967, 0.889, 0.526],
             gap="+11.0 pp", ylim=(0.42, 1.02), note=None),
    ]
    x = np.arange(4)
    xlabels = ["100%", "75%", "50%", "25%"]
    is_full = [True, False, False, True]  # full split vs n=200 subset

    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_COL, 4.7), sharex=True)
    for ax, p in zip(axes.ravel(), panels):
        S.style_axes(ax)
        ax.set_xlim(-0.18, 3.72)  # headroom for the 25%-gap label
        ax.set_ylim(*p["ylim"])
        for vals, color, lab in ((p["pre"], BLUE, "pre-merger (RBM, ours)"),
                                 (p["post"], RED, "post-merger (VZ-style)")):
            xs = [i for i, v in enumerate(vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.plot(xs, ys, "-", color=color, lw=1.6, zorder=3,
                    label=lab if ax is axes[0][0] else None)
            for i, v in zip(xs, ys):
                ax.scatter(i, v, s=34, color=color, zorder=4,
                           edgecolor=color, linewidth=1.3,
                           facecolor=S.SURFACE if is_full[i] else color)
        # gap annotation at 25% retention
        pre25, post25 = p["pre"][3], p["post"][3]
        ax.annotate("", xy=(3, pre25), xytext=(3, post25),
                    arrowprops=dict(arrowstyle="<->", color=S.INK_SEC, lw=1.0),
                    zorder=5)
        ax.text(3.10, (pre25 + post25) / 2, p["gap"], color=S.INK_PRIM,
                fontsize=8.5, fontweight="bold", va="center", ha="left")
        if p["note"]:
            ax.text(0.0, p["ylim"][1] - 0.015 * (p["ylim"][1] - p["ylim"][0]),
                    p["note"], fontsize=7.0, color=S.INK_MUTED, va="top", ha="left")
        ax.set_title(p["title"], fontsize=9, pad=5)
        ax.set_xticks(x)
    for ax in axes[1]:
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("visual-token retention (compression depth  →)",
                      fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("official score")
    h, l = axes[0][0].get_legend_handles_labels()
    h += [plt.Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=S.INK_SEC, markeredgecolor=S.INK_SEC, ms=6),
          plt.Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=S.SURFACE, markeredgecolor=S.INK_SEC, ms=6,
                     markeredgewidth=1.3)]
    l += ["n = 200 subset", "full split"]
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.4, handletextpad=0.4)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.095,
                        hspace=0.42, wspace=0.16)
    save(fig, "fig3_retention_gap")


# ---------------------------------------------------------------------------
# Fig 1 — pipeline schematic
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc="white", ec=S.INK_SEC, ls="-", lw=1.0,
         fs=8.0, tc=None, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.06",
                                fc=fc, ec=ec, ls=ls, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or S.INK_PRIM, weight=weight, linespacing=1.25)


def _arrow(ax, x0, x1, y, color=S.INK_SEC, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def fig1():
    S.apply_rc(fontsize=9)
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COL, 2.75))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 4.35)
    ax.axis("off")

    # column geometry (shared x across both rows; only the box CONTENT swaps)
    COL = {"image": (0.10, 0.95), "vit": (1.30, 1.20),
           "c3": (2.80, 2.05), "c4": (5.15, 2.05), "llm": (7.50, 1.10)}
    h = 1.00
    ya, yb = 2.80, 0.70

    def col(name):
        return COL[name][0], COL[name][1]

    def row(y, score_first, title, title_col, score_fs=6.9):
        ax.text(0.10, y + h + 0.24, title, fontsize=9, weight="bold",
                color=title_col)
        ix, iw = col("image"); _box(ax, ix, y, iw, h, "Image", fs=8.5)
        vx, vw = col("vit")
        _box(ax, vx, y, vw, h, "ViT\n32 px units  (N)", fs=7.4)
        c3x, c3w = col("c3"); c4x, c4w = col("c4")
        lx, lw = col("llm")
        if score_first:
            _box(ax, c3x, y, c3w, h, "Rank RAW units (L2)\n→ keep top-κN",
                 fc="#fdeaea", ec=RED, lw=1.8, fs=score_fs, weight="bold")
            _box(ax, c4x, y, c4w, h, "Native 2×2\nmerger", fs=7.6)
            cx = c3x + c3w / 2
            cap, capcol = "saliency BEFORE averaging", RED
        else:
            _box(ax, c3x, y, c3w, h, "Native 2×2\nmerger", fs=7.6)
            _box(ax, c4x, y, c4w, h, "Rank MERGED tokens\n→ keep top-κN",
                 fc="#f2f2ee", ec=S.INK_MUTED, ls=(0, (4, 2.5)), lw=1.4,
                 fs=score_fs)
            cx = c4x + c4w / 2
            cap, capcol = "saliency AFTER averaging", S.INK_MUTED
        _box(ax, lx, y, lw, h, "LLM", fs=8.5)
        # arrows between columns
        for (a, b) in [("image", "vit"), ("vit", "c3"), ("c3", "c4"),
                       ("c4", "llm")]:
            ax0 = COL[a][0] + COL[a][1]; ax1 = COL[b][0]
            _arrow(ax, ax0, ax1, y + h / 2)
        ax.text(cx, y - 0.18, cap, ha="center", fontsize=7.0, color=capcol,
                style="italic")
        return y + h + 0.10

    # token-count labels over arrows (midpoints)
    def toklabels(y, l34):
        m23 = (COL["vit"][0] + COL["vit"][1] + COL["c3"][0]) / 2
        m34 = (COL["c3"][0] + COL["c3"][1] + COL["c4"][0]) / 2
        m45 = (COL["c4"][0] + COL["c4"][1] + COL["llm"][0]) / 2
        for mx, t in [(m23, "N"), (m34, l34), (m45, "κN")]:
            ax.text(mx, y + h + 0.04, t, ha="center", fontsize=6.9,
                    color=S.INK_MUTED)

    row(ya, True, "(a)  Pre-merger selection — this work", RED)
    toklabels(ya, "κN kept")
    row(yb, False, "(b)  Post-merger selection — published family (e.g. VisionZip)",
        S.INK_SEC)
    toklabels(yb, "N merged")

    ax.text(4.4, 0.12,
            "Native 2×2 merger (+ deepstack mergers on Qwen3-VL) and LLM are "
            "identical in (a) and (b); only the saliency tap point differs.",
            ha="center", fontsize=7.0, color=S.INK_SEC)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------------------
# Fig 2 — mechanism: M1 decorrelation + M3 swap control
# ---------------------------------------------------------------------------
def fig2():
    S.apply_rc(fontsize=9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.DOUBLE_COL, 3.15),
                                   gridspec_kw=dict(wspace=0.30))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.15)

    # ---- panel a: M1 ----
    S.style_axes(ax1)
    benches = ["DocVQA", "TextVQA", "GQA"]
    rho, rho_sd = [0.137, 0.332, 0.360], [0.158, 0.124, 0.091]
    jac, jac_sd = [0.180, 0.243, 0.278], [0.091, 0.070, 0.060]
    xb = np.arange(3)
    w = 0.38
    b1 = ax1.bar(xb - w / 2, rho, w, color=BLUE, edgecolor=BLUE,
                 yerr=rho_sd, capsize=2.5, error_kw=dict(lw=0.9, ecolor=S.INK_SEC),
                 label="Spearman ρ (rank corr.)", zorder=3)
    b2 = ax1.bar(xb + w / 2, jac, w, color=AQUA, edgecolor=AQUA,
                 yerr=jac_sd, capsize=2.5, error_kw=dict(lw=0.9, ecolor=S.INK_SEC),
                 label="Jaccard@25% (kept-set)", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.045,
                     f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.3)
    ax1.axhline(1.0, color=S.INK_MUTED, ls="--", lw=0.9, zorder=2)
    ax1.text(2.46, 0.965, "ρ = 1 ⇒ identical rankings", fontsize=7.0,
             color=S.INK_MUTED, ha="right", va="top")
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(benches)
    ax1.set_ylabel("pre- vs post-merger agreement")
    ax1.legend(fontsize=7.3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, columnspacing=1.2)

    # ---- panel b: M3 ----
    S.style_axes(ax2)
    groups = ["TextVQA (VQA acc)", "DocVQA (ANLS)"]
    vals = {"Uncompressed": [0.858, 0.976],
            "post": [0.215, 0.200],
            "pre (ours)": [0.598, 0.465],
            "swap ≡ pre": [0.603, 0.465]}
    err = {"Uncompressed": [0.025, 0.011], "post": [0.029, 0.028],
           "pre (ours)": [0.035, 0.035], "swap ≡ pre": [0.035, 0.035]}
    colors = {"Uncompressed": S.BASELINE, "post": RED,
              "pre (ours)": BLUE, "swap ≡ pre": BLUE}
    hatches = {"Uncompressed": "", "post": "", "pre (ours)": "", "swap ≡ pre": "////"}
    xg = np.arange(2)
    w = 0.20
    for i, k in enumerate(vals):
        bars = ax2.bar(xg + (i - 1.5) * w, vals[k], w, color=colors[k],
                       edgecolor=S.INK_PRIM if k == "swap ≡ pre" else colors[k],
                       linewidth=0.8 if k == "swap ≡ pre" else 0,
                       hatch=hatches[k], yerr=err[k], capsize=2,
                       error_kw=dict(lw=0.9, ecolor=S.INK_SEC), label=k, zorder=3)
        for r, v in zip(bars, vals[k]):
            ax2.text(r.get_x() + r.get_width() / 2, v + 0.055, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=6.8)
    # equivalence brackets pre <-> swap
    for gi, (vp, vs) in enumerate(zip(vals["pre (ours)"], vals["swap ≡ pre"])):
        xp, xs = xg[gi] + 0.5 * w, xg[gi] + 1.5 * w
        ytop = max(vp, vs) + 0.145
        ax2.plot([xp, xp, xs, xs], [ytop - 0.02, ytop, ytop, ytop - 0.02],
                 color=S.INK_PRIM, lw=0.9, zorder=4)
        ax2.text((xp + xs) / 2, ytop + 0.012,
                 "≡ (Δ ≤ 0.005)", ha="center", va="bottom",
                 fontsize=7.0, weight="bold")
    ax2.set_ylim(0, 1.22)
    ax2.set_xticks(xg)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("official score (n = 200)")
    ax2.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=4, columnspacing=0.9)
    # panel titles (figure-level, above legend band) + footnotes (below axes)
    p1 = ax1.get_position(); p2 = ax2.get_position()
    fig.text((p1.x0 + p1.x1) / 2, 0.94,
             "a | M1: merger reshuffles unit ranks", ha="center", fontsize=9.5,
             weight="bold")
    fig.text((p2.x0 + p2.x1) / 2, 0.94,
             "b | M3: ranking-swap causal control", ha="center", fontsize=9.5,
             weight="bold")
    fig.text((p1.x0 + p1.x1) / 2, 0.035,
             "n = 64 images / bench (Qwen3-VL-8B); bars = mean ± s.d.",
             ha="center", fontsize=6.8, color=S.INK_MUTED)
    fig.text((p2.x0 + p2.x1) / 2, 0.05,
             "post forward path + pre ranking ⇒ pre accuracy;\n"
             "the pre–post gap is 100% a ranking effect",
             ha="center", fontsize=6.8, color=S.INK_SEC, linespacing=1.25)
    save(fig, "fig2_mechanism")


# ---------------------------------------------------------------------------
# Fig 4 — qualitative examples
# ---------------------------------------------------------------------------
def fig4():
    S.apply_rc(fontsize=9)
    examples = [
        dict(head="TextVQA · id 35014", tag="text erased",
             q="Q: What is the date on the right page?",
             gt="GT: 07/10/2012",
             rows=[("Pre (ours)", "07/10/2012", True),
                   ("Post (L2)", "“no visible date”", False)]),
        dict(head="DocVQA · id 58439", tag="unit corrupted — 1000× error",
             q="Q: Amount spent on promotional\nmeetings & events, 1998?",
             gt="GT: $1.3 BILLION",
             rows=[("Pre (ours)", "$1.3 billion", True),
                   ("Post (L2)", "$1.3 million", False),
                   ("VZ-style", "$1.3 million", False)]),
        dict(head="GQA · id 201370409", tag="object-centric trade-off",
             q="Q: What are the scissors on?",
             gt="GT: paper",
             rows=[("Pre (ours)", "“no scissors visible”", False),
                   ("Post (L2)", "paper", True)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE_COL, 4.5))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.935, bottom=0.012, wspace=0.05)
    for ax, ex in zip(axes, examples):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.015, 0.012), 0.97, 0.976,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="white", ec=S.BASELINE, lw=1.0))
        ax.text(0.5, 0.965, ex["head"], ha="center", va="top", fontsize=8.4,
                weight="bold")
        # image placeholder (light hatch + white-backed caption for legibility)
        ax.add_patch(FancyBboxPatch((0.08, 0.760), 0.84, 0.150,
                                    boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=S.SURFACE, ec=S.INK_MUTED, lw=0.8,
                                    ls=(0, (3, 2)), hatch="//"))
        ax.text(0.5, 0.835, "image omitted  (see supplementary)", ha="center",
                va="center", fontsize=7.0, color=S.INK_SEC,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))
        ax.text(0.06, 0.720, ex["q"], ha="left", va="top", fontsize=7.6,
                linespacing=1.2)
        ax.text(0.06, 0.615, ex["gt"], ha="left", va="top", fontsize=7.6,
                weight="bold", color=S.INK_SEC)
        y = 0.540
        rh = 0.128
        for name, ans, ok in ex["rows"]:
            color = GREEN if ok else RED
            mark = "✓" if ok else "✗"
            ax.add_patch(FancyBboxPatch((0.05, y - rh), 0.90, rh,
                                        boxstyle="round,pad=0.004,rounding_size=0.015",
                                        fc=("#eaf5ea" if ok else "#fdeaea"),
                                        ec=color, lw=1.1))
            ax.text(0.085, y - 0.035, f"{name}:", ha="left", va="center",
                    fontsize=7.3, weight="bold", color=color)
            ax.text(0.085, y - 0.092, f"{mark}  {ans}", ha="left", va="center",
                    fontsize=7.3, color=color, weight="bold")
            y -= rh + 0.022
        ax.text(0.5, y + 0.002, ex["tag"], ha="center", va="top", fontsize=7.1,
                color=S.INK_SEC, style="italic")
    fig.text(0.5, 0.985,
             "Qwen3-VL-8B, keep = 25% (r = 0.75), L2 selector; identical 25% "
             "token budget in every condition — contrasts isolate selection order.",
             ha="center", fontsize=7.0, color=S.INK_SEC)
    save(fig, "fig4_qualitative")


if __name__ == "__main__":
    fig3()
    fig1()
    fig2()
    fig4()
    print("done")
