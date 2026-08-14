#!/usr/bin/env python3
"""Render a CVPR-style measured overview without overwriting the paper figure."""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "server_exports" / "cvpr_figure_data_v1"
OUT = Path(__file__).resolve().parent / "outputs"

INK = "#263238"
MUTED = "#667178"
GRID = "#DCE2E3"
RBM = "#C56A3D"
RBM_LIGHT = "#F5DED2"
POST = "#3F6672"
POST_LIGHT = "#D9E6E9"
GOOD = "#4E7A61"
BAD = "#A4483F"
PAPER = "#FBFAF7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.edgecolor": "#A8B0B3",
        "axes.linewidth": 0.6,
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_case(case_id: str) -> tuple[Path, dict]:
    case_dir = BUNDLE / "cases" / case_id
    return case_dir, json.loads((case_dir / "case.json").read_text(encoding="utf-8"))


def load_method(case_dir: Path, method: str) -> tuple[np.ndarray, np.ndarray]:
    spec = {
        "rbm": ("pre_scores.npz", "unit_l2_pre", "kept_indices_pre"),
        "post_l2": ("post_scores.npz", "unit_l2_post", "kept_indices_post"),
    }[method]
    with np.load(case_dir / spec[0], allow_pickle=False) as data:
        return np.asarray(data[spec[1]], dtype=float), np.asarray(data[spec[2]], dtype=int)


def robust_scale(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def compact_answer(answer: str, correct: bool) -> str:
    bold = re.findall(r"\*\*(.*?)\*\*", answer, flags=re.DOTALL)
    if bold:
        text = bold[-1] if correct else bold[0]
    else:
        text = answer.replace("\n", " ").replace("**", " ")
    text = " ".join(text.split())
    return text if len(text) <= 31 else text[:29].rstrip() + "..."


def image_and_prompt(fig: plt.Figure, spec, case_dir: Path, meta: dict, tag: str) -> None:
    grid = spec.subgridspec(2, 1, height_ratios=[4.2, 1.25], hspace=0.06)
    ax = fig.add_subplot(grid[0])
    ax.imshow(Image.open(case_dir / "input.jpg"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#7F898D")
        spine.set_linewidth(0.7)
    ax.set_title(tag, loc="left", color=INK, fontweight="bold", pad=3)

    text_ax = fig.add_subplot(grid[1])
    text_ax.axis("off")
    question_raw = meta["question"].split("\n", 1)[0]
    question_raw = re.split(r"\s+Answer (?:the|this) question", question_raw, maxsplit=1)[0].strip()
    question = textwrap.fill(question_raw, width=30)
    text_ax.text(0.0, 0.92, "Q", color=MUTED, fontweight="bold", va="top")
    text_ax.text(0.12, 0.98, question, color=INK, va="top", wrap=True, fontsize=6.3, linespacing=1.08)
    text_ax.text(0.0, 0.02, "GT", color=MUTED, fontweight="bold", va="bottom")
    text_ax.text(0.18, 0.02, meta["ground_truth"].split(";")[0], color=INK, fontweight="bold", va="bottom")


def score_panel(fig: plt.Figure, spec, case_dir: Path, meta: dict, method: str) -> None:
    color = RBM if method == "rbm" else POST
    light = RBM_LIGHT if method == "rbm" else POST_LIGHT
    title = "Rank before merger (RBM)" if method == "rbm" else "Rank after merger (Post-L2)"
    scores, kept = load_method(case_dir, method)
    norm = robust_scale(scores)
    h, w = map(int, meta["unit_grid_hw"])
    assert h * w == len(scores)
    score_grid = norm.reshape(h, w)
    keep_grid = np.zeros(len(scores), dtype=bool)
    keep_grid[kept] = True
    keep_grid = keep_grid.reshape(h, w)

    grid = spec.subgridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.08)
    curve_ax = fig.add_subplot(grid[0])
    x = np.arange(len(norm))
    curve_ax.vlines(x, 0, norm, color=light, linewidth=0.35, alpha=0.9)
    window = max(5, len(norm) // 45)
    kernel = np.ones(window) / window
    smooth = np.convolve(norm, kernel, mode="same")
    curve_ax.plot(x, smooth, color=color, linewidth=1.1)
    curve_ax.set_xlim(0, len(norm) - 1)
    curve_ax.set_ylim(0, 1.03)
    curve_ax.set_xticks([])
    curve_ax.set_yticks([0, 1], ["low", "high"])
    curve_ax.grid(axis="y", color=GRID, linewidth=0.45)
    curve_ax.set_title(title, color=color, fontweight="bold", pad=3)
    curve_ax.text(0.99, 0.04, "unit index", transform=curve_ax.transAxes, ha="right", color=MUTED, fontsize=5.8)

    map_ax = fig.add_subplot(grid[1])
    image = Image.open(case_dir / "input.jpg")
    map_ax.imshow(image, extent=(0, w, h, 0), alpha=0.42)
    cmap = LinearSegmentedColormap.from_list("method", ["#EEF1F0", light, color])
    map_ax.imshow(score_grid, cmap=cmap, extent=(0, w, h, 0), interpolation="nearest", alpha=0.72, vmin=0, vmax=1)
    yy, xx = np.where(keep_grid)
    map_ax.scatter(xx + 0.5, yy + 0.5, marker="s", s=max(2.2, 92 / len(scores)), facecolors="none", edgecolors=color, linewidths=0.35)
    map_ax.set_xlim(0, w)
    map_ax.set_ylim(h, 0)
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    map_ax.text(
        0.02,
        0.03,
        f"{len(kept)} / {len(scores)} retained",
        transform=map_ax.transAxes,
        color=INK,
        fontsize=6.0,
        bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )


def outcome_panel(fig: plt.Figure, spec, meta: dict) -> None:
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.set_title("Measured outcomes", loc="left", color=INK, fontweight="bold", pad=3)
    rows = [("RBM", "rbm", RBM), ("Post-L2", "post_l2", POST), ("FastV-k3", "fastv_k3", MUTED)]
    y_values = [0.74, 0.45, 0.16]
    for (label, key, color), y in zip(rows, y_values):
        correct = bool(meta["correctness"][key])
        ax.plot([0.0, 0.05], [y + 0.07, y + 0.07], color=color, linewidth=2.4, solid_capstyle="butt")
        ax.text(0.08, y + 0.08, label, color=color, fontweight="bold", va="center")
        ax.text(0.08, y - 0.02, compact_answer(meta["answers"][key], correct), color=INK, va="top", fontsize=6.5, wrap=True)
        ax.text(0.98, y + 0.08, "CORRECT" if correct else "INCORRECT", ha="right", va="center", color=GOOD if correct else BAD, fontweight="bold", fontsize=6.2)
        if y > 0.2:
            ax.axhline(y - 0.14, color=GRID, linewidth=0.55)
    ax.text(0.0, -0.01, "Audited answer excerpts; identical 25% visual-token budget", color=MUTED, fontsize=5.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def aggregate_strip(fig: plt.Figure, spec) -> dict[str, str]:
    with (BUNDLE / "aggregate" / "aggregate_main_results.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if float(row["retention"]) == 0.25]

    labels = []
    for benchmark, display in (("textvqa", "TextVQA"), ("docvqa", "DocVQA"), ("ocrbench", "OCRBench")):
        values = [float(row["delta_pp"]) for row in rows if row["benchmark"] == benchmark]
        if benchmark == "ocrbench":
            values = [value * 10 for value in values]
            labels.append((display, f"+{min(values):.0f} to +{max(values):.0f} pts"))
        else:
            labels.append((display, f"+{min(values):.1f} to +{max(values):.1f} pp"))

    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.axhline(0.98, color="#A8B0B3", linewidth=0.7)
    ax.text(0.0, 0.56, "FULL-SPLIT ANCHOR", color=MUTED, fontweight="bold", va="center", fontsize=6.2)
    x_positions = [0.20, 0.43, 0.68]
    for x, (name, value) in zip(x_positions, labels):
        ax.text(x, 0.68, name, color=INK, fontweight="bold", va="center")
        ax.text(x, 0.28, value, color=RBM, fontweight="bold", va="center")
    ax.text(0.91, 0.68, "GQA", color=INK, fontweight="bold", va="center")
    ax.text(0.91, 0.28, "scope boundary", color=POST, fontweight="bold", va="center")
    for x in (0.17, 0.40, 0.65, 0.88):
        ax.axvline(x, ymin=0.16, ymax=0.82, color=GRID, linewidth=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return {name: value for name, value in labels}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 5.0), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        4,
        left=0.025,
        right=0.985,
        top=0.875,
        bottom=0.045,
        width_ratios=[1.25, 1.42, 1.42, 1.25],
        height_ratios=[1.0, 1.0, 0.20],
        wspace=0.16,
        hspace=0.25,
    )

    selected = [
        ("ocrbench_ocr0804", "(a) Document text preserved"),
        ("textvqa_34982", "(b) Scene text preserved"),
    ]
    provenance = {"bundle": str(BUNDLE), "cases": [], "aggregate_source": "aggregate/aggregate_main_results.csv"}
    for row, (case_id, tag) in enumerate(selected):
        case_dir, meta = load_case(case_id)
        image_and_prompt(fig, grid[row, 0], case_dir, meta, tag)
        score_panel(fig, grid[row, 1], case_dir, meta, "rbm")
        score_panel(fig, grid[row, 2], case_dir, meta, "post_l2")
        outcome_panel(fig, grid[row, 3], meta)
        provenance["cases"].append(
            {
                "case_id": case_id,
                "metadata": str((case_dir / "case.json").relative_to(BUNDLE)),
                "scores": [
                    str((case_dir / "pre_scores.npz").relative_to(BUNDLE)),
                    str((case_dir / "post_scores.npz").relative_to(BUNDLE)),
                ],
            }
        )

    provenance["aggregate_labels"] = aggregate_strip(fig, grid[2, :])
    fig.text(0.025, 0.978, "Saliency changes after information mixing", color=INK, fontweight="bold", fontsize=10.3, va="top")
    fig.text(0.985, 0.978, "Qwen3-VL-8B | greedy | 25% retained", color=MUTED, ha="right", fontsize=6.5, va="top")

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig1_cvpr_candidate.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    (OUT / "fig1_cvpr_candidate_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT / 'fig1_cvpr_candidate.pdf'}")


if __name__ == "__main__":
    main()
