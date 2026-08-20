#!/usr/bin/env python3
"""Render the mechanism figure from per-image and ranking-swap measurements."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "server_exports" / "cvpr_figure_data_v1"
OUT = Path(__file__).resolve().parent / "outputs"

INK = "#263238"
MUTED = "#667178"
GRID = "#DCE2E3"
DOC = "#B85F45"
OCR = "#C99348"
TEXT = "#4F7890"
GQA = "#6B806B"
RBM = "#C56A3D"
POST = "#3F6672"
SWAP = "#6B806B"
PAPER = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.edgecolor": "#A8B0B3",
        "axes.linewidth": 0.6,
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "axes.facecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

ORDER = ["docvqa", "ocrbench", "textvqa", "gqa"]
LABEL = {"docvqa": "DocVQA", "ocrbench": "OCRBench", "textvqa": "TextVQA", "gqa": "GQA"}
COLOR = {"docvqa": DOC, "ocrbench": OCR, "textvqa": TEXT, "gqa": GQA}


def read_mechanism() -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with (BUNDLE / "mechanism" / "mechanism_per_image.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            benchmark = row["benchmark"]
            for key, value in row.items():
                if key in {"benchmark", "model", "sample_id"} or value in {"", None}:
                    continue
                try:
                    grouped[benchmark][key].append(float(value))
                except ValueError:
                    pass
    return {benchmark: {key: np.asarray(values) for key, values in metrics.items()} for benchmark, metrics in grouped.items()}


def horizontal_distribution(ax, data, metric: str, title: str, xlim: tuple[float, float]) -> None:
    rng = np.random.default_rng(4)
    positions = np.arange(len(ORDER))
    values = [data[name][metric] for name in ORDER]
    box = ax.boxplot(
        values,
        orientation="horizontal",
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.0},
        whiskerprops={"color": MUTED, "linewidth": 0.65},
        capprops={"color": MUTED, "linewidth": 0.65},
    )
    for patch, name in zip(box["boxes"], ORDER):
        patch.set_facecolor(COLOR[name])
        patch.set_alpha(0.28)
        patch.set_edgecolor(COLOR[name])
    for pos, name, series in zip(positions, ORDER, values):
        jitter = rng.normal(0, 0.055, size=len(series))
        ax.scatter(series, pos + jitter, s=4.0, color=COLOR[name], alpha=0.32, linewidth=0)
        ax.text(np.nanmean(series), pos - 0.31, f"{np.nanmean(series):.2f}", color=COLOR[name], ha="center", fontsize=5.7)
    ax.set_yticks(positions, [LABEL[name] for name in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.axvline(0, color=INK, linewidth=0.65)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=2)


def edge_panel(fig: plt.Figure, spec, data) -> None:
    grid = spec.subgridspec(2, 1, hspace=0.52, height_ratios=[1, 1])
    ax = fig.add_subplot(grid[0])
    y = np.arange(len(ORDER))
    dropped = np.array([np.nanmean(data[name]["mean_edge_pre_keep_post_drop"]) for name in ORDER])
    reverse = np.array([np.nanmean(data[name]["mean_edge_post_keep_pre_drop"]) for name in ORDER])
    for idx, name in enumerate(ORDER):
        ax.plot([reverse[idx], dropped[idx]], [idx, idx], color=COLOR[name], linewidth=1.4)
    ax.scatter(reverse, y, facecolors=PAPER, edgecolors=[COLOR[name] for name in ORDER], s=23, linewidth=1.0, label="Post keeps / RBM drops")
    ax.scatter(dropped, y, color=[COLOR[name] for name in ORDER], s=23, linewidth=0, label="RBM keeps / Post drops")
    ax.set_yticks(y, [LABEL[name] for name in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.72, dropped.max() * 1.10))
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.set_xlabel("mean Sobel edge energy / unit")
    ax.set_title("(b) Edge-rich units are demoted", loc="left", fontweight="bold", color=INK, pad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=5.5, handletextpad=0.4)

    lower = fig.add_subplot(grid[1])
    horizontal_distribution(lower, data, "rank_shift_edge_rho", "Rank shift vs edge correlation", (-0.25, 0.8))
    lower.set_xlabel("Spearman rho  (+ = high-edge units demoted)")


def swap_panel(fig: plt.Figure, spec) -> dict:
    swap = json.loads((BUNDLE / "mechanism" / "swap_control.json").read_text(encoding="utf-8"))
    q3 = swap["qwen3vl"]
    benchmarks = ["textvqa", "docvqa"]
    x = np.arange(len(benchmarks))
    width = 0.22
    values = {
        "RBM pre": [q3[name]["pre_accuracy"] for name in benchmarks],
        "Post-L2": [q3[name]["post_accuracy"] for name in benchmarks],
        "Swap": [q3[name]["swap_accuracy"] for name in benchmarks],
    }
    colors = {"RBM pre": RBM, "Post-L2": POST, "Swap": SWAP}
    ax = fig.add_subplot(spec)
    for offset, (label, series) in zip((-width, 0, width), values.items()):
        bars = ax.bar(x + offset, series, width=width * 0.88, color=colors[label], label=label, alpha=0.92)
        for bar, value in zip(bars, series):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.014, f"{value:.3f}", ha="center", va="bottom", fontsize=5.6, color=colors[label], rotation=90)
    ax.set_xticks(x, ["TextVQA", "DocVQA"])
    ax.set_ylim(0, 0.74)
    ax.set_ylabel("official score")
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_title("(c) Ranking-swap control", loc="left", fontweight="bold", color=INK, pad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              fontsize=5.5, ncol=3, columnspacing=0.8, handlelength=1.3)
    ax.text(
        0.5,
        -0.13,
        "same post path; Jaccard=1.000 | Qwen3-VL: n=200 answers, n=30/31 kept sets",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
        fontsize=5.4,
        clip_on=False,
    )
    return q3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_mechanism()
    fig = plt.figure(figsize=(7.25, 3.6))
    outer = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.84, bottom=0.13, width_ratios=[1.08, 1.25, 1.05], wspace=0.40)

    left = outer[0].subgridspec(2, 1, hspace=0.52)
    ax1 = fig.add_subplot(left[0])
    horizontal_distribution(ax1, data, "spearman_pre_post", "(a) Rank correlation across merger", (-0.28, 0.62))
    ax1.set_xlabel("Spearman rho")
    ax2 = fig.add_subplot(left[1])
    horizontal_distribution(ax2, data, "jaccard_topk", "Top-25% kept-set overlap", (0.02, 0.45))
    ax2.set_xlabel("Jaccard")

    edge_panel(fig, outer[1], data)
    q3 = swap_panel(fig, outer[2])

    fig.text(0.06, 0.975, "How the native merger reshapes saliency rankings", color=INK, fontweight="bold", fontsize=10.3, va="top")
    fig.text(0.985, 0.975, "per-image distributions | 25% retained", color=MUTED, ha="right", fontsize=6.4, va="top")

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            OUT / f"fig3_mechanism_candidate.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
            transparent=True,
        )
    plt.close(fig)

    provenance = {
        "mechanism_source": "mechanism/mechanism_per_image.csv",
        "swap_source": "mechanism/swap_control.json",
        "benchmarks": {name: int(len(data[name]["spearman_pre_post"])) for name in ORDER},
        "qwen3_swap": q3,
    }
    (OUT / "fig3_mechanism_candidate_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT / 'fig3_mechanism_candidate.pdf'}")


if __name__ == "__main__":
    main()
