#!/usr/bin/env python3
"""Render the final two-panel operational/matched-boundary Figure 2."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OLD_STATS = ROOT / "experiments" / "paired_metric_statistics.json"
FINAL = ROOT / "results" / "acmmm_final_controls" / "analysis.json"
OUT = ROOT / "drafts" / "overleaf_submission" / "figs"

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
LABELS = ["TextVQA", "DocVQA", "OCRBench", "GQA"]
MODELS = [
    ("Qwen3-VL-8B", "qwen3vl", "#2563EB", "o"),
    ("Qwen2.5-VL-7B", "qwen2vl", "#D1495B", "s"),
    ("InternVL3-8B", "internvl3", "#168C83", "D"),
]


def load_cells():
    old = json.loads(OLD_STATS.read_text(encoding="utf-8"))["stage_law_table1"]
    final = json.loads(FINAL.read_text(encoding="utf-8"))

    operational = {}
    for label, key, _color, _marker in MODELS:
        operational[label] = {}
        for bench in BENCHES:
            cell = old[key]["benchmarks"][bench]["pre_vs_post"]
            operational[label][bench] = (
                float(cell["mean_delta_pp"]),
                float(cell["ci95_pp"][0]),
                float(cell["ci95_pp"][1]),
            )

    # Replace only the stale Qwen2.5 OCRBench operational cell with the final
    # matched-cap campaign. JSON stores a fraction, which maps directly to
    # OCRBench /10 on the shared percentage-point axis.
    q2 = final["P0_2_ocrbench_matched"]["k25"]["paired"]
    operational["Qwen2.5-VL-7B"]["ocrbench"] = tuple(
        100.0 * float(q2[k]) for k in ("mean_delta", "ci95_lo", "ci95_hi")
    )

    matched = {}
    for bench in BENCHES:
        cell = final["P0_1_pure_stage_control"][bench]["pre_final_vs_post"]["paired"]
        matched[bench] = tuple(
            100.0 * float(cell[k]) for k in ("mean_delta", "ci95_lo", "ci95_hi")
        )
    return operational, matched


def draw():
    operational, matched = load_cells()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.72), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.78, bottom=0.22, wspace=0.20)
    centers = [3.0, 2.0, 1.0, 0.0]
    grid = "#E3E8EF"
    ink = "#17202A"

    ax = axes[0]
    offsets = [0.20, 0.0, -0.20]
    for (label, _key, color, marker), off in zip(MODELS, offsets):
        xs, los, his = [], [], []
        for bench in BENCHES:
            x, lo, hi = operational[label][bench]
            xs.append(x)
            los.append(x - lo)
            his.append(hi - x)
        ys = [c + off for c in centers]
        ax.errorbar(
            xs,
            ys,
            xerr=[los, his],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2,
            elinewidth=1.0,
            markersize=4.7,
            markeredgecolor=ink,
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.set_xlim(-7.5, 49)
    ax.set_xticks([-5, 0, 10, 20, 30, 40])
    ax.set_title("(a) Operational RBM vs Post-L2", fontweight="bold", pad=7)

    ax = axes[1]
    xs, los, his = [], [], []
    for bench in BENCHES:
        x, lo, hi = matched[bench]
        xs.append(x)
        los.append(x - lo)
        his.append(hi - x)
    ax.errorbar(
        xs,
        centers,
        xerr=[los, his],
        fmt="o",
        color="#7A3E9D",
        ecolor="#7A3E9D",
        capsize=2.5,
        elinewidth=1.2,
        markersize=5.2,
        markeredgecolor=ink,
        markeredgewidth=0.5,
        zorder=3,
    )
    for x, y in zip(xs, centers):
        ax.text(
            x + (0.75 if x >= 0 else -0.75),
            y,
            f"{x:+.1f}".replace("-", "−"),
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=6.6,
            color="#5F2E78",
        )
    ax.set_xlim(-8.5, 33)
    ax.set_xticks([-5, 0, 10, 20, 30])
    ax.set_title("(b) Qwen3-VL matched boundary", fontweight="bold", pad=7)

    for ax in axes:
        for y in centers:
            ax.axhline(y, color=grid, linewidth=0.55, zorder=0)
        ax.axvline(0, color="#344054", linewidth=1.0, zorder=1)
        ax.grid(axis="x", color=grid, linewidth=0.5, linestyle=(0, (2, 3)))
        ax.set_ylim(-0.45, 3.45)
        ax.set_yticks(centers)
        ax.set_yticklabels(LABELS, fontweight="bold")
        ax.tick_params(length=2.5, pad=2, colors=ink)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#667085")
        ax.set_xlabel("pre − post difference (pp; OCRBench /10)", labelpad=5)

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=ink,
            markeredgewidth=0.45,
            markersize=5.5,
            label=label,
        )
        for label, _key, color, marker in MODELS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#7A3E9D",
            markeredgecolor=ink,
            markersize=5.5,
            label="Qwen3 matched boundary",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.98),
        ncol=4,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig2.pdf")
    fig.savefig(OUT / "fig2.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    draw()
