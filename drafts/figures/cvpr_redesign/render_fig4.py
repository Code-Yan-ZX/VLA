#!/usr/bin/env python3
"""Render the workload regime map plus stable qualitative boundary cases."""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "server_exports" / "cvpr_figure_data_v1"
OUT = Path(__file__).resolve().parent / "outputs"
INK = "#263238"
MUTED = "#667178"
GRID = "#DCE2E3"
RBM = "#C56A3D"
FASTV = "#4F7180"
GOOD = "#4E7A61"
BAD = "#A4483F"
UNCERTAIN = "#8B8D7A"
PAPER = "#FBFAF7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.4,
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


def compact_answer(answer: str, correct: bool) -> str:
    bold = re.findall(r"\*\*(.*?)\*\*", answer, flags=re.DOTALL)
    text = (bold[-1] if correct and bold else bold[0] if bold else answer).replace("\n", " ").replace("**", " ")
    text = " ".join(text.split())
    return text if len(text) <= 23 else text[:21].rstrip() + "..."


def load_case(case_id: str) -> tuple[Path, dict]:
    case_dir = BUNDLE / "cases" / case_id
    return case_dir, json.loads((case_dir / "case.json").read_text(encoding="utf-8"))


def regime_map(fig: plt.Figure, spec) -> None:
    rows = []
    with (BUNDLE / "aggregate" / "aggregate_regime_map.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    models = ["qwen3vl", "qwen2vl"]
    benchmarks = ["textvqa", "docvqa", "ocrbench", "gqa"]
    ax = fig.add_subplot(spec)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks([0, 1], ["Qwen3-VL-8B", "Qwen2.5-VL-7B"])
    ax.set_yticks(range(4), ["TextVQA", "DocVQA", "OCRBench", "GQA"])
    ax.invert_yaxis()
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    row_lookup = {(row["model"], row["benchmark"]): row for row in rows}
    for xi, model in enumerate(models):
        for yi, benchmark in enumerate(benchmarks):
            row = row_lookup[(model, benchmark)]
            winner = row["winner"]
            margin = float(row["margin_pp"])
            uncertain = row["ci_low"] and row["ci_high"] and float(row["ci_low"]) <= 0 <= float(row["ci_high"])
            if uncertain:
                face, edge, label = "#EEF0EC", UNCERTAIN, f"{winner[0]}?  {margin:.1f} pp"
            elif winner == "RBM":
                face, edge, label = "#F4DED3", RBM, f"R  +{margin:.1f} pp"
            else:
                face, edge, label = "#DCE8EB", FASTV, f"F  +{margin:.1f} pp"
            rect = mpl.patches.Rectangle((xi - 0.43, yi - 0.36), 0.86, 0.72, facecolor=face, edgecolor=edge, linewidth=1.2)
            ax.add_patch(rect)
            ax.text(xi, yi - 0.04, label, ha="center", va="center", color=edge, fontweight="bold", fontsize=7.3)
            ax.text(xi, yi + 0.20, f"n={int(row['n'])}", ha="center", va="center", color=MUTED, fontsize=5.2)
    ax.set_title("(a) Workload-conditioned regime map", loc="left", color=INK, fontweight="bold", fontsize=8.8, pad=5)
    ax.text(1.0, 1.06, "R = RBM | F = FastV | ? = CI crosses zero", transform=ax.transAxes, color=MUTED, fontsize=5.9, ha="right", va="bottom")
    ax.set_axisbelow(True)


def qualitative_case(fig: plt.Figure, spec, case_id: str, label: str, note: str) -> None:
    case_dir, meta = load_case(case_id)
    grid = spec.subgridspec(2, 1, height_ratios=[2.5, 1.5], hspace=0.08)
    image_ax = fig.add_subplot(grid[0])
    image_ax.imshow(Image.open(case_dir / "input.jpg"))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    image_ax.set_title(label, loc="left", color=INK, fontweight="bold", pad=3)
    for spine in image_ax.spines.values():
        spine.set_color("#7F898D")
        spine.set_linewidth(0.7)

    text_ax = fig.add_subplot(grid[1])
    text_ax.axis("off")
    question_raw = meta["question"].split("\n", 1)[0]
    question_raw = re.split(r"\s+Answer (?:the|this) question", question_raw, maxsplit=1)[0].strip()
    text_ax.text(0.0, 0.98, textwrap.fill(question_raw, width=25), color=INK, va="top", fontsize=6.0, linespacing=1.05)
    y_values = [0.44, 0.10]
    for y, key, method, color in zip(y_values, ("rbm", "fastv_k3"), ("RBM", "FastV-k3"), (RBM, FASTV)):
        correct = bool(meta["correctness"][key])
        text_ax.text(0.0, y, method, color=color, fontweight="bold", va="center", fontsize=6.0)
        text_ax.text(0.30, y, compact_answer(meta["answers"][key], correct), color=INK, va="center", fontsize=5.8)
        text_ax.text(0.99, y, "CORRECT" if correct else "INCORRECT", color=GOOD if correct else BAD, ha="right", va="center", fontweight="bold", fontsize=5.6)
    text_ax.text(0.0, -0.20, note, color=MUTED, fontsize=5.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 4.15))
    outer = fig.add_gridspec(2, 1, left=0.035, right=0.985, top=0.86, bottom=0.08, height_ratios=[1.05, 1.55], hspace=0.36)
    regime_map(fig, outer[0])
    cases = outer[1].subgridspec(1, 3, wspace=0.20)
    qualitative_case(fig, cases[0], "ocrbench_ocr0804", "(b) Dense document", "RBM retains the exact date; FastV selects a wrong date.")
    qualitative_case(fig, cases[1], "textvqa_34982", "(c) Scene text", "FastV recovers the query-local text; RBM also remains correct.")
    qualitative_case(fig, cases[2], "gqa_201056134", "(d) Object-QA boundary", "FastV is correct while RBM is wrong: no universal winner.")
    fig.text(0.035, 0.965, "RBM and FastV occupy different workload regimes", color=INK, fontweight="bold", fontsize=10.3, va="top")
    fig.text(0.985, 0.965, "25% retention | paired scope where available", color=MUTED, ha="right", fontsize=6.3, va="top")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig4_regime_qualitative_candidate.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    provenance = {
        "regime_source": "aggregate/aggregate_regime_map.csv",
        "qualitative_cases": ["ocrbench_ocr0804", "textvqa_34982", "gqa_201056134"],
        "case_sources": {case_id: f"cases/{case_id}/case.json" for case_id in ("ocrbench_ocr0804", "textvqa_34982", "gqa_201056134")},
    }
    (OUT / "fig4_regime_qualitative_candidate_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'fig4_regime_qualitative_candidate.pdf'}")


if __name__ == "__main__":
    main()
