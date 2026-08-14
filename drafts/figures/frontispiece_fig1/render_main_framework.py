"""Render the measured hybrid frontispiece used as paper Figure 1.

The top strip isolates the stage-order intervention. The main matrix uses two
audited, regeneration-stable cases and the exact kept indices for RBM,
Post-L2, and FastV-k3. Aggregate values are read from the official full-split
CSV and normalized to a common 0--100 display scale.
"""
from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "drafts/figures/server_exports/cvpr_figure_data_v1"
OUT = ROOT / "drafts/overleaf_submission/figs"
PROVENANCE_OUT = ROOT / "drafts/figures/frontispiece_fig1/fig1_provenance.json"

INK = "#20282C"
MUTED = "#66747A"
GRID = "#CDD5D8"
PAPER = "#FCFCFA"
RBM = "#C9683B"
RBM_LIGHT = "#FFF0E7"
POST = "#3F6875"
POST_LIGHT = "#EAF2F4"
FASTV = "#66757D"
SELECT = "#1689B8"
GOOD = "#2F7D55"
BAD = "#B55243"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)

METHODS = (
    ("RBM (ours)", "rbm", RBM),
    ("Post-L2", "post_l2", POST),
    ("FastV-k3", "fastv_k3", FASTV),
)


def load_case(case_id: str) -> tuple[Path, dict]:
    case_dir = BUNDLE / "cases" / case_id
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if not meta["iso_token_contract"]["holds"]:
        raise ValueError(f"iso-token contract failed for {case_id}")
    return case_dir, meta


def load_kept_indices(case_dir: Path, method: str) -> np.ndarray:
    filename, key = {
        "rbm": ("pre_scores.npz", "kept_indices_pre"),
        "post_l2": ("post_scores.npz", "kept_indices_post"),
        "fastv_k3": ("fastv_scores.npz", "kept_indices_fastv"),
    }[method]
    with np.load(case_dir / filename, allow_pickle=False) as data:
        kept = np.asarray(data[key], dtype=int)
        n_full = int(data["n_units_full"])
    if len(kept) != len(np.unique(kept)) or np.any(kept < 0) or np.any(kept >= n_full):
        raise ValueError(f"invalid kept indices in {case_dir.name}/{filename}")
    return kept


def compact_answer(answer: str) -> str:
    bold = re.findall(r"\*\*(.*?)\*\*", answer, flags=re.DOTALL)
    if bold:
        answer = bold[-1]
    answer = " ".join(answer.replace("**", "").split())
    if len(answer) > 25:
        answer = answer[:23].rstrip() + "..."
    return answer


def contained_extent(image: Image.Image, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    image_ratio = image.width / image.height
    box_ratio = w / h
    if image_ratio >= box_ratio:
        draw_w = w
        draw_h = w / image_ratio
        draw_x = x
        draw_y = y + (h - draw_h) / 2
    else:
        draw_h = h
        draw_w = h * image_ratio
        draw_x = x + (w - draw_w) / 2
        draw_y = y
    return draw_x, draw_y, draw_w, draw_h


def flow_box(ax, x: float, y: float, w: float, text: str, *, edge: str, face: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            0.25,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.85,
        )
    )
    ax.text(x + w / 2, y + 0.125, text, ha="center", va="center", color=INK, fontsize=6.9)


def flow_arrow(ax, x0: float, y: float, x1: float, *, color: str) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9, "mutation_scale": 7},
    )


def draw_status(ax, x: float, y: float, correct: bool) -> None:
    color = GOOD if correct else BAD
    ax.add_patch(patches.Circle((x, y), 0.095, facecolor=PAPER, edgecolor=color, linewidth=1.15, zorder=8))
    if correct:
        ax.plot([x - 0.043, x - 0.012, x + 0.050], [y - 0.003, y - 0.040, y + 0.043], color=color, lw=1.35, solid_capstyle="round", zorder=9)
    else:
        ax.plot([x - 0.040, x + 0.040], [y - 0.040, y + 0.040], color=color, lw=1.35, solid_capstyle="round", zorder=9)
        ax.plot([x - 0.040, x + 0.040], [y + 0.040, y - 0.040], color=color, lw=1.35, solid_capstyle="round", zorder=9)


def draw_masked_image(
    ax,
    image: Image.Image,
    kept: np.ndarray,
    grid_hw: tuple[int, int],
    box: tuple[float, float, float, float],
    correct: bool,
) -> None:
    x, y, w, h = contained_extent(image, *box)
    ax.imshow(image, extent=(x, x + w, y, y + h), origin="upper", aspect="auto", zorder=1)
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none", edgecolor="#7E8A8F", linewidth=0.55, zorder=5))

    gh, gw = grid_hw
    cell_w, cell_h = w / gw, h / gh
    for index in kept:
        row, col = divmod(int(index), gw)
        cell_x = x + col * cell_w
        cell_y = y + h - (row + 1) * cell_h
        ax.add_patch(
            patches.Rectangle(
                (cell_x, cell_y),
                cell_w,
                cell_h,
                facecolor=SELECT,
                edgecolor=SELECT,
                linewidth=0.30,
                alpha=0.14,
                zorder=4,
            )
        )
    draw_status(ax, x + 0.11, y + h - 0.11, correct)


def normalized_qwen3_rows() -> list[dict[str, object]]:
    csv_path = BUNDLE / "aggregate" / "aggregate_main_results.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["model"] == "qwen3vl" and float(row["retention"]) == 0.25
        ]
    by_benchmark = {row["benchmark"]: row for row in rows}
    expected = ("textvqa", "docvqa", "ocrbench", "gqa")
    if set(by_benchmark) != set(expected):
        raise ValueError(f"unexpected Qwen3 aggregate rows: {sorted(by_benchmark)}")

    result = []
    for benchmark in expected:
        row = by_benchmark[benchmark]
        result.append(
            {
                "benchmark": benchmark,
                "rbm": round(100 * float(row["rbm"]), 3),
                "post_l2": round(100 * float(row["post_l2"]), 3),
                "delta": round(float(row["delta_pp"]), 3),
                "n": int(row["n"]),
                "source": row["source"],
                "native_unit": row["unit"],
            }
        )
    return result


def render(path: Path) -> dict[str, object]:
    cases = [
        ("ocrbench_ocr0804", "(a) Document OCR", "Value of CIRCULATION DATES?"),
        ("textvqa_34982", "(b) Scene text", "Name written on the black shirt?"),
    ]
    loaded_cases = [(case_id, label, short_q, *load_case(case_id)) for case_id, label, short_q in cases]
    aggregates = normalized_qwen3_rows()

    fig = plt.figure(figsize=(7.25, 5.10), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.25)
    ax.set_ylim(0, 5.10)
    ax.axis("off")

    # Headline and compact stage-order intervention.
    ax.text(0.16, 4.94, "Saliency should be measured", fontsize=14.0, fontweight="bold", color=INK, va="top")
    ax.text(3.77, 4.94, "before information mixing", fontsize=14.0, fontweight="bold", color=RBM, va="top")
    ax.text(0.17, 4.53, "CORE INTERVENTION", fontsize=7.0, fontweight="bold", color=MUTED, va="center")
    ax.text(1.27, 4.53, "same model path  |  same L2 selector  |  same final visual-token budget", fontsize=7.2, color=INK, va="center")

    labels_x = 0.20
    box_x = 1.42
    box_w = 1.45
    gap = 0.31
    ax.text(labels_x, 4.20, "RBM", fontsize=8.2, fontweight="bold", color=RBM, va="center")
    rbm_steps = ((r"Rank raw $2\times2$ units", RBM_LIGHT), (r"Keep top-$\kappa N$", RBM_LIGHT), (r"Native $2\times2$ merger", "white"))
    for i, (label, face) in enumerate(rbm_steps):
        x = box_x + i * (box_w + gap)
        flow_box(ax, x, 4.07, box_w, label, edge=RBM, face=face)
        if i < 2:
            flow_arrow(ax, x + box_w + 0.04, 4.195, x + box_w + gap - 0.04, color=RBM)

    ax.text(labels_x, 3.82, "Post-L2", fontsize=8.2, fontweight="bold", color=POST, va="center")
    post_steps = ((r"Native $2\times2$ merger", "white"), ("Rank merged rows", POST_LIGHT), (r"Keep top-$\kappa N$", POST_LIGHT))
    for i, (label, face) in enumerate(post_steps):
        x = box_x + i * (box_w + gap)
        flow_box(ax, x, 3.69, box_w, label, edge=POST, face=face)
        if i < 2:
            flow_arrow(ax, x + box_w + 0.04, 3.815, x + box_w + gap - 0.04, color=POST)

    ax.plot([0.16, 7.09], [3.51, 3.51], color=GRID, lw=0.7)
    ax.text(0.17, 3.37, "MEASURED EVIDENCE", fontsize=7.0, fontweight="bold", color=MUTED, va="center")
    ax.text(1.46, 3.37, "blue cells = retained visual units", fontsize=6.9, color=MUTED, va="center")
    ax.text(7.08, 3.37, "audited answers  |  25% retained", fontsize=6.9, color=MUTED, ha="right", va="center")

    # Equal method columns; case metadata remains outside the image grid.
    label_x = 0.17
    method_start = 1.26
    method_w = 1.84
    method_gap = 0.105
    method_centers = [method_start + i * (method_w + method_gap) + method_w / 2 for i in range(3)]
    for (display, _, color), center in zip(METHODS, method_centers):
        ax.text(center, 3.18, display, fontsize=8.5, fontweight="bold", color=color, ha="center", va="center")

    row_specs = [(2.05, 1.02), (0.91, 1.02)]
    provenance_cases = []
    for (case_id, case_label, short_q, case_dir, meta), (row_y, row_h) in zip(loaded_cases, row_specs):
        gt = meta["ground_truth"].split(";")[0]
        kept_count = int(meta["final_visual_tokens"]["rbm"])
        ax.text(label_x, row_y + row_h - 0.03, case_label, fontsize=7.8, fontweight="bold", color=INK, va="top")
        ax.text(label_x, row_y + row_h - 0.27, textwrap.fill(short_q, width=19), fontsize=6.9, color=INK, va="top", linespacing=1.05)
        ax.text(label_x, row_y + 0.25, "GT", fontsize=6.8, fontweight="bold", color=MUTED, va="bottom")
        ax.text(label_x + 0.22, row_y + 0.25, gt, fontsize=7.2, fontweight="bold", color=INK, va="bottom")
        ax.text(label_x, row_y + 0.06, f"{kept_count} / {meta['n_units_full']} units", fontsize=6.7, color=MUTED, va="bottom")

        image = Image.open(case_dir / "input.jpg").convert("RGB")
        method_sources = []
        for col, ((_, method, _), center) in enumerate(zip(METHODS, method_centers)):
            kept = load_kept_indices(case_dir, method)
            if len(kept) != kept_count:
                raise ValueError(f"budget mismatch for {case_id}/{method}")
            cell_x = method_start + col * (method_w + method_gap)
            draw_masked_image(
                ax,
                image,
                kept,
                tuple(map(int, meta["unit_grid_hw"])),
                (cell_x + 0.035, row_y + 0.24, method_w - 0.07, row_h - 0.27),
                bool(meta["correctness"][method]),
            )
            answer = compact_answer(meta["answers"][method])
            answer_color = GOOD if meta["correctness"][method] else BAD
            ax.text(center, row_y + 0.075, answer, fontsize=7.1, fontweight="bold", color=answer_color, ha="center", va="bottom")
            filename, key = {
                "rbm": ("pre_scores.npz", "kept_indices_pre"),
                "post_l2": ("post_scores.npz", "kept_indices_post"),
                "fastv_k3": ("fastv_scores.npz", "kept_indices_fastv"),
            }[method]
            method_sources.append({"method": method, "file": str((case_dir / filename).relative_to(BUNDLE)), "key": key})

        provenance_cases.append(
            {
                "case_id": case_id,
                "metadata": str((case_dir / "case.json").relative_to(BUNDLE)),
                "image": str((case_dir / "input.jpg").relative_to(BUNDLE)),
                "methods": method_sources,
            }
        )

    # One-model quantitative anchor; every benchmark is on a common 0--100 scale.
    ax.plot([0.16, 7.09], [0.78, 0.78], color="#AAB5B9", lw=0.75)
    ax.text(0.17, 0.63, "QWEN3-VL-8B FULL-SPLIT ANCHOR", fontsize=6.9, fontweight="bold", color=MUTED, va="center")
    ax.text(0.17, 0.40, "RBM - Post-L2", fontsize=7.4, fontweight="bold", color=INK, va="center")
    ax.text(0.17, 0.20, "official score, 0-100 scale", fontsize=6.7, color=MUTED, va="center")

    names = {"textvqa": "TextVQA", "docvqa": "DocVQA", "ocrbench": "OCRBench /10", "gqa": "GQA"}
    metric_centers = [2.28, 3.55, 4.82, 6.09]
    for row, center in zip(aggregates, metric_centers):
        delta = float(row["delta"])
        color = RBM if delta > 0 else POST
        sign = "+" if delta > 0 else ""
        ax.text(center, 0.62, names[str(row["benchmark"])], fontsize=7.2, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(center, 0.37, f"{sign}{delta:.1f} pts", fontsize=9.5, fontweight="bold", color=color, ha="center", va="center")
        ax.text(center, 0.17, f"{row['rbm']:.2f} vs {row['post_l2']:.2f}", fontsize=6.8, color=MUTED, ha="center", va="center")

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "cases": provenance_cases,
        "aggregate_source": "aggregate/aggregate_main_results.csv",
        "aggregate_filter": {"model": "qwen3vl", "retention": 0.25},
        "aggregate_display": aggregates,
        "notes": [
            "All mask rectangles are reconstructed from native-grid kept_indices, not square-packed keep masks.",
            "Source images are rendered at full opacity; selected cells use a 0.14-alpha blue overlay.",
            "OCRBench is normalized from /1000 to the same 0--100 display scale as the other official metrics.",
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = None
    for suffix in ("pdf", "svg", "png"):
        output_path = OUT / f"fig1.{suffix}"
        current = render(output_path)
        if suffix == "svg":
            lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        if provenance is None:
            provenance = current
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("wrote", OUT / "fig1.pdf", OUT / "fig1.svg", OUT / "fig1.png")


if __name__ == "__main__":
    main()
