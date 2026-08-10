"""Camera-ready Figure 1: paper-level performance/efficiency overview.

Generates drafts/figures/camera_ready/fig1_overview_base.{svg,pdf,png} from
drafts/figures/camera_ready/fig1_overview_data.json (authoritative values).

Layout (2x2):
  (a) Qwen3-VL-8B retention @ 25% -- radar, 4 axes, 3 methods, mixed scope
  (b) Qwen2.5-VL-7B retention @ 25% -- radar, 4 axes, 3 methods, paired n200
  (c) OCR regime: OCRBench accuracy vs mean visual tokens
  (d) vLLM RBM throughput speedup vs retention

Color palette and style spec are pinned in fig1_overview_data.json under _meta.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ----------------------------- style -----------------------------------------
DATA_PATH = Path(__file__).with_name("fig1_overview_data.json")
OUT_DIR = DATA_PATH.parent
FIG_W, FIG_H = 7.1, 5.2  # inches

# spec colors
C_RBM = "#2F66B0"   # deep blue
C_FV = "#DD8452"    # orange
C_L2 = "#8A93A3"    # cool gray
C_NONE = "#0b0b0b"  # dark
C_OCR = "#F2B701"   # amber accent
C_GUIDE = "#c3c2b7" # thin guide line
C_INK2 = "#52514e"
C_INK3 = "#898781"

mpl.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": C_INK2,
    "axes.labelcolor": C_INK2,
    "xtick.color": C_INK2,
    "ytick.color": C_INK2,
    "text.color": C_NONE,
    "axes.titlecolor": C_NONE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ----------------------------- data load -------------------------------------
with DATA_PATH.open() as f:
    DATA = json.load(f)

AXES = ["OCRBench", "TextVQA", "DocVQA", "GQA"]  # global axis order, identical in a/b


def panel_label(ax, letter, title, scope=None, fig=None, fig_y=None):
    """Place (letter) + title + scope relative to the axes (so tight bbox
    doesn't move them). For polar axes, push well above the outer ring so the
    OCRBench label doesn't collide."""
    # detect polar vs cartesian
    is_polar = getattr(ax, "name", "") == "polar"
    title_y = 1.42 if is_polar else 1.18
    scope_y = 1.32 if is_polar else 1.10
    ax.text(-0.30 if is_polar else -0.22, title_y, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left", color=C_NONE)
    ax.text(-0.20 if is_polar else -0.13, title_y, title, transform=ax.transAxes,
            fontsize=10, va="bottom", ha="left", color=C_NONE)
    if scope:
        ax.text(-0.20 if is_polar else -0.13, scope_y, scope, transform=ax.transAxes,
                fontsize=7.5, style="italic", va="bottom", ha="left", color=C_INK3)


# ----------------------------- radar helper ----------------------------------
def radar_panel(ax, retention_dict, axes_labels, ocr_axis_idx=0, *,
                ymax=100, yticks=(50, 75, 100), ymin=0,
                fill_alpha=0.06, daggers=None, ocr_accent=True):
    """4-axis radar (polar plot). retention_dict: {method: [v0,v1,v2,v3], ...}."""
    n = len(axes_labels)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{v}%" for v in yticks], fontsize=7, color=C_INK3)
    ax.set_rlabel_position(90)

    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=9, color=C_NONE)
    # bring the axis labels CLOSER to the radar so they do not collide with the
    # fig-level panel title at the top.
    ax.tick_params(axis="x", pad=2)

    if ocr_accent:
        # subtle amber outer ring on the OCRBench axis side (12 deg each way).
        # Use a polar bar so it sits on the outer ring without obscuring the
        # axis label.
        ax.bar(angles[0], height=ymax * 0.06, width=math.radians(24),
               bottom=ymax * 0.98, color=C_OCR, alpha=0.35, edgecolor="none",
               zorder=0)

    ax.grid(color=C_INK3, linewidth=0.5, alpha=0.4)
    ax.spines["polar"].set_color(C_INK3)
    ax.spines["polar"].set_linewidth(0.6)

    method_style = {
        "RBM":      dict(color=C_RBM, ls="-",  marker="o", mfc=C_RBM,  mec=C_RBM,  label="RBM (ours)", lw=1.6, ms=6),
        "FastV_k3": dict(color=C_FV,  ls="--", marker="s", mfc=C_FV,   mec=C_FV,   label="FastV-k3",   lw=1.2, ms=5),
        "Post_L2":  dict(color=C_L2,  ls=":",  marker="^", mfc=C_L2,   mec=C_L2,   label="Post-L2",    lw=1.2, ms=5),
    }

    for method, vals in retention_dict.items():
        style = method_style[method]
        proc_vals = [None if v is None or isinstance(v, str) else float(v) for v in vals]
        plot_arr = np.array([v if v is not None else np.nan for v in proc_vals], dtype=float)
        plot_closed = np.concatenate([plot_arr, plot_arr[:1]])
        ax.plot(angles_closed, plot_closed, color=style["color"],
                linestyle=style["ls"], linewidth=style["lw"], label=style["label"],
                zorder=3)
        ax.fill(angles_closed, plot_closed, color=style["color"], alpha=fill_alpha, zorder=2)
        for ang, v in zip(angles, proc_vals):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            ax.plot(ang, v, marker=style["marker"], color=style["color"],
                    mfc=style["mfc"], mec=style["mec"], markersize=style["ms"],
                    zorder=4)
        if daggers and method in daggers:
            for ang, axis_name in zip(angles, axes_labels):
                if axis_name in daggers[method]:
                    # render a dagger at a low fixed radius (50%) so it's visible
                    ax.plot(ang, 50, marker="|", color=style["color"], markersize=10, mew=2,
                            zorder=5)

    return ax


# ----------------------------- panel (a) Qwen3 radar ------------------------
def panel_a(ax):
    a = DATA["panel_a_qwen3vl_retention_25pct"]
    retention = a["retention_pct"]
    rd = {
        m: [None if isinstance(retention[m][ax_name], str) else float(retention[m][ax_name])
            for ax_name in AXES]
        for m in ["RBM", "FastV_k3", "Post_L2"]
    }
    daggers = {
        "RBM":     ["DocVQA"],
        "Post_L2": ["DocVQA"],
    }
    radar_panel(ax, rd, AXES, ocr_axis_idx=0, ymax=100, yticks=(50, 75, 100),
                daggers=daggers, ocr_accent=True)
    panel_label(ax, "a", "Qwen3-VL-8B retention @ 25%",
                scope="mixed scope; dagger = none missing")


# ----------------------------- panel (b) Qwen2.5 radar ----------------------
def panel_b(ax):
    b = DATA["panel_b_qwen25vl_retention_25pct"]
    retention = b["retention_pct"]
    rd = {
        m: [None if isinstance(retention[m][ax_name], str) else float(retention[m][ax_name])
            for ax_name in AXES]
        for m in ["RBM", "FastV_k3", "Post_L2"]
    }
    radar_panel(ax, rd, AXES, ocr_axis_idx=0, ymax=100, yticks=(50, 75, 100),
                daggers=None, ocr_accent=True)
    angles = np.linspace(0, 2 * math.pi, len(AXES), endpoint=False)
    for tgt_ax in ("DocVQA", "GQA"):
        idx = AXES.index(tgt_ax)
        ang = angles[idx]
        rv = retention["RBM"][tgt_ax]
        if isinstance(rv, (int, float)):
            ax.plot(ang, float(rv), marker="o", mfc="none", mec=C_RBM,
                    markersize=10, mew=1.2, zorder=6)
    panel_label(ax, "b", "Qwen2.5-VL-7B retention @ 25%",
                scope="paired n=200 (same skip set)")


# ----------------------------- panel (c) OCR Pareto -------------------------
def panel_c(ax):
    c = DATA["panel_c_ocr_pareto"]
    methods_order = ["none", "RBM", "FastV_k3", "Post_L2"]
    method_label = {"none": "none", "RBM": "RBM", "FastV_k3": "FastV-k3", "Post_L2": "post-L2"}
    method_color = {"none": C_NONE, "RBM": C_RBM, "FastV_k3": C_FV, "Post_L2": C_L2}

    pts = {model: [] for model in ["Qwen3_VL_8B", "Qwen2_5_VL_7B"]}
    for model in pts:
        for m in methods_order:
            d = c["data"][model][m]
            if d["score"] is None or d["tokens"] is None:
                continue
            pts[model].append((m, float(d["tokens"]), float(d["score"])))

    for model, plist in pts.items():
        if len(plist) < 2:
            continue
        xs = [p[1] for p in plist]
        ys = [p[2] for p in plist]
        ax.plot(xs, ys, color=C_GUIDE, lw=0.8, ls="-", zorder=1, alpha=0.9)

    for model, plist in pts.items():
        mshape = "o" if model == "Qwen3_VL_8B" else "s"
        for m, x, y in plist:
            ax.scatter(x, y, marker=mshape, s=70,
                       facecolor=method_color[m], edgecolor="white",
                       linewidth=0.7, zorder=3)
            ax.scatter(x, y, marker=mshape, s=70,
                       facecolor="none", edgecolor=method_color[m],
                       linewidth=0.9, zorder=4)

    for model, mtext, color, x_off, y_off in [
        ("Qwen3_VL_8B",   "+16.0 pp", C_RBM, 5.0,  0.03),
        ("Qwen2_5_VL_7B", "+8.5 pp",  C_RBM, 3.0,  0.03),
    ]:
        d = c["data"][model]
        rbm = d["RBM"]
        fv = d["FastV_k3"]
        if rbm["score"] is None or fv["score"] is None:
            continue
        x_text = rbm["tokens"] * x_off
        y_text = rbm["score"] + y_off
        ax.annotate(
            mtext,
            xy=(rbm["tokens"], rbm["score"]),
            xytext=(x_text, y_text),
            fontsize=9, color=color, fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.7, alpha=0.95),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean visual tokens sent to LLM  (log scale)", fontsize=9, color=C_INK2)
    ax.set_ylabel("OCRBench official score", fontsize=9, color=C_INK2)
    ax.set_ylim(0.10, 0.90)
    ax.set_xlim(80, 6000)

    for model, plist in pts.items():
        for m, x, y in plist:
            ax.text(x * 1.05, y, method_label[m], fontsize=7, color=method_color[m],
                    va="center", ha="left")

    q3_handle = Line2D([], [], marker="o", color="none", mfc=C_INK2, mec=C_INK2, label="Qwen3-VL-8B", markersize=7)
    q25_handle = Line2D([], [], marker="s", color="none", mfc=C_INK2, mec=C_INK2, label="Qwen2.5-VL-7B", markersize=7)
    leg = ax.legend(handles=[q3_handle, q25_handle], loc="lower right", frameon=True,
                    fontsize=8, title="Model", title_fontsize=8)
    leg.get_frame().set_edgecolor(C_INK3)
    leg.get_frame().set_linewidth(0.6)
    ax.grid(True, color=C_INK3, linewidth=0.4, alpha=0.4)

    panel_label(ax, "c", "OCR regime: accuracy vs visual tokens",
                scope="OCRBench; RBM = home regime")


# ----------------------------- panel (d) vLLM efficiency --------------------
def panel_d(ax):
    d = DATA["panel_d_vllm_efficiency"]
    pc = d["primary_curve"]
    ret = pc["retention_points_pct"]
    sp = pc["speedup_over_none"]

    # primary curve: thin line with small hollow-circle markers (N=200 setup)
    ax.plot(ret, sp, color=C_RBM, marker="o", markersize=5,
            mfc="white", mec=C_RBM, lw=1.4, zorder=3,
            label="curve: Qwen3 TextVQA\nN=200, max_seqs=16 (efficiency_summary)")

    cm = d["compact_markers_25pct"]
    label_positions = []
    for model_key, mshape in [("Qwen3_VL_8B", "o"), ("Qwen2_5_VL_7B", "s")]:
        for bench, cell in cm[model_key].items():
            sp_val = cell.get("speedup")
            if sp_val is None:
                continue
            ax.scatter(25, sp_val, marker=mshape, s=80, facecolor=C_RBM,
                       edgecolor="white", linewidth=0.7, zorder=4)
            label_positions.append((model_key, bench, sp_val))

    # labels: place to the LEFT of the 25% line, at the marker's y, sorted by speedup.
    # To avoid overlap when speeds are close, use the marker's exact y but with a
    # very small x offset; rely on the distinct y to separate.
    label_positions.sort(key=lambda t: t[2])
    for model_key, bench, sp_val in label_positions:
        ax.text(23.5, sp_val, f"{bench}", fontsize=7.0, color=C_NONE,
                ha="right", va="center")

    ax.text(0.03, 0.96,
            "25% kept: mean tokens\n"
            "  Qwen3:    97-229\n"
            "  Qwen2.5:  128-1228",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            color=C_INK2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_INK3, linewidth=0.4))

    ax.set_xlim(20, 105)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.invert_xaxis()
    ax.set_xlabel("Visual-token retention (100% → 25%)", fontsize=9, color=C_INK2)
    ax.set_ylabel("Speedup over none  (×)", fontsize=9, color=C_INK2)
    ax.set_ylim(0.8, max(sp) * 1.15 + 0.4)
    ax.axhline(y=1.0, color=C_INK3, lw=0.6, ls="--", zorder=1)
    ax.grid(True, color=C_INK3, linewidth=0.4, alpha=0.4)

    ax.axvline(x=25, color=C_OCR, lw=1.2, ls="-", alpha=0.65, zorder=2)
    # 25% emphasis: place inside the plot at the bottom-left, with a small box
    ax.text(60, ax.get_ylim()[0] + 0.15, "amber line: 25% retention (compressed regime)",
            color=C_OCR, fontsize=7.5, ha="left", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_OCR, linewidth=0.5, alpha=0.9))

    panel_label(ax, "d", "vLLM RBM throughput speedup",
                scope="vLLM 0.19.0; curve = TextVQA N=200")


# ----------------------------- figure assembly -------------------------------
def build_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.82, bottom=0.13,
                          wspace=0.20, hspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0], projection="polar")
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)

    radar_handles = [
        Line2D([], [], color=C_RBM, marker="o", mfc=C_RBM, mec=C_RBM, lw=1.6, ms=6, label="RBM (ours)"),
        Line2D([], [], color=C_FV,  marker="s", mfc=C_FV,  mec=C_FV,  lw=1.2, ms=5, label="FastV-k3"),
        Line2D([], [], color=C_L2,  marker="^", mfc=C_L2,  mec=C_L2,  lw=1.2, ms=5, label="Post-L2"),
        Line2D([], [], color=C_NONE, marker="o", mfc=C_NONE, mec=C_NONE, lw=0, ms=4, label="none (uncompressed)"),
    ]
    extra_handles = [
        Line2D([], [], color="none", marker="o", mfc="none", mec=C_RBM, mew=1.2, ms=8, label="inconclusive (hollow)"),
        mpatches.Patch(facecolor=C_OCR, alpha=0.25, edgecolor="none", label="OCRBench / 25% emphasis"),
    ]
    fig.legend(handles=radar_handles + extra_handles,
               loc="lower center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.005), handletextpad=0.4, columnspacing=1.0)

    return fig


def main():
    fig = build_figure()
    out_svg = OUT_DIR / "fig1_overview_base.svg"
    out_pdf = OUT_DIR / "fig1_overview_base.pdf"
    out_png = OUT_DIR / "fig1_overview_base.png"
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"OK: wrote {out_svg}, {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
