"""Render qualitative Fig. 1: 3 examples x 3 methods (RBM, FastV, Post-L2)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/media/disk2/YZX/research/vla")
DATA_DIR = ROOT / "drafts/figures/qualitative_fig1/data"
OUT_DIR = ROOT / "drafts/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_RBM = "#E8A33D"
C_FASTV = "#4A6FA5"
C_POST = "#8A93A3"
C_EVIDENCE = "#D03B3B"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_INK3 = "#898781"
C_OK_BG = "#e8f5e9"
C_ERR_BG = "#fde8e8"

# Evidence box coordinates (left, top, right, bottom) in source image pixels
# Manually annotated, verified to contain GT text
EVIDENCE_BBOX = {
    "textvqa_34863": (0.05, 0.60, 0.95, 0.85),    # HAPPY BIRTHDAY on scoreboard
    "docvqa_52297":  (0.20, 0.02, 0.65, 0.12),    # July 17 and 18, 1996 at top
    "ocrbench_ocr0804": (0.45, 0.30, 0.90, 0.40),  # CIRCULATION DATES row
}

mpl = matplotlib
mpl.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": C_INK2,
    "axes.labelcolor": C_INK2, "xtick.color": C_INK2,
    "ytick.color": C_INK2, "text.color": C_INK,
    "axes.titlecolor": C_INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

with (DATA_DIR / "manifest.json").open() as f:
    MANIFEST = json.load(f)


def make_overlay(img_arr, keep_mask, side_grid, color=(0.91, 0.64, 0.05, 0.45)):
    h, w = img_arr.shape[:2]
    overlay = img_arr.copy().astype(np.float32) / 255.0
    block_h = h / side_grid
    block_w = w / side_grid
    for idx in range(side_grid * side_grid):
        r, c = divmod(idx, side_grid)
        y0, y1 = int(r * block_h), int((r + 1) * block_h)
        x0, x1 = int(c * block_w), int((c + 1) * block_w)
        if keep_mask[r, c]:
            overlay[y0:y1, x0:x1] = (
                overlay[y0:y1, x0:x1] * (1 - color[3]) +
                np.array(color[:3]) * color[3]
            )
            overlay[y0, x0:x1] = color[:3]
            overlay[max(y1-1, y0), x0:x1] = color[:3]
            overlay[y0:y1, x0] = color[:3]
            overlay[y0:y1, max(x1-1, x0)] = color[:3]
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def render_example(ax_left, ax_right, ex):
    img = Image.open(DATA_DIR / ex["image"]).convert("RGB")
    w, h = img.size
    arr = np.array(img)

    # load mask
    mask_npz = np.load(DATA_DIR / ex["methods"][0]["mask"])
    keep = mask_npz["keep"]
    side = int(mask_npz["unit_grid_hw"][0])

    # evidence box
    bbox = EVIDENCE_BBOX.get(f'{ex["benchmark"].lower()}_{ex["id"]}', None)

    # left: source image with evidence box + question
    ax_left.imshow(arr)
    ax_left.set_xticks([]); ax_left.set_yticks([])
    for sp in ax_left.spines.values():
        sp.set_edgecolor(C_INK2); sp.set_linewidth(0.6)
    if bbox:
        x0, y0, x1, y1 = bbox
        rect = mpatches.Rectangle(
            (x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
            linewidth=2, edgecolor=C_EVIDENCE, facecolor="none", zorder=5)
        ax_left.add_patch(rect)

    bench_id = f'{ex["benchmark"]} / {ex["id"]}'
    ax_left.set_title(bench_id, fontsize=9, fontweight="bold", loc="left", pad=2)
    q = ex["question"].replace("\n", " ")
    ax_left.text(0, h + 4, f'Q: {q}', fontsize=6.5, color=C_INK2, va="top")

    # right: 3 method rows
    ax_right.set_xticks([]); ax_right.set_yticks([])
    for sp in ax_right.spines.values():
        sp.set_edgecolor(C_INK2); sp.set_linewidth(0.6)

    n_methods = len(ex["methods"])
    row_h = 1.0 / n_methods
    for i, m in enumerate(ex["methods"]):
        y0 = 1.0 - (i + 1) * row_h
        ax_m = ax_right.inset_axes([0.0, y0, 1.0, row_h * 0.95])
        ax_m.set_xticks([]); ax_m.set_yticks([])
        for sp in ax_m.spines.values():
            sp.set_visible(False)

        overlay = make_overlay(arr, keep, side)
        ax_m.imshow(overlay, aspect="auto")

        if bbox:
            x0, y0_b, x1, y1 = bbox
            rect = mpatches.Rectangle(
                (x0 * w, y0_b * h), (x1 - x0) * w, (y1 - y0_b) * h,
                linewidth=1.5, edgecolor=C_EVIDENCE, facecolor="none", zorder=5)
            ax_m.add_patch(rect)

        color = {"rbm": C_RBM, "fastv": C_FASTV, "post": C_POST}[m["key"]]
        ax_m.text(0.01, 0.92, m["label"], transform=ax_m.transAxes,
                  fontsize=7.5, fontweight="bold", color=color, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor=color, linewidth=0.5, alpha=0.9))

        correct = m["correct"]
        ans = m["answer"]
        if len(ans) > 60:
            ans = ans[:57] + "..."
        marker = "Correct" if correct else "Incorrect"
        bg = C_OK_BG if correct else C_ERR_BG
        ax_m.text(0.99, 0.85, f"Answer: {ans}", transform=ax_m.transAxes,
                  fontsize=6.5, color=C_INK, va="top", ha="right",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor=bg,
                            edgecolor=color, linewidth=0.5, alpha=0.95))
        ax_m.text(0.99, 0.08, f"[{marker}]  ptid={m['ptid']}",
                  transform=ax_m.transAxes, fontsize=6, color=C_INK2,
                  va="bottom", ha="right")

    # GT line
    gt_str = "; ".join(ex["ground_truth"])
    ax_right.text(0.5, 1.01, f'GT: {gt_str}  |  25% retained; ptid={ex["ptid"]}',
                  transform=ax_right.transAxes, fontsize=6.5, color=C_INK3,
                  va="bottom", ha="center", style="italic")


def render():
    fig, axes = plt.subplots(3, 2, figsize=(7.09, 9.0),
                             gridspec_kw={"width_ratios": [1.0, 1.2], "wspace": 0.04, "hspace": 0.12})
    for i, ex in enumerate(MANIFEST["examples"]):
        render_example(axes[i, 0], axes[i, 1], ex)

    fig.suptitle(
        "Selected cases where RBM preserves text that baselines miss\n"
        "(Qwen3-VL-8B-Instruct, greedy, 25% visual-token retention; identical final token count per row; mask = pre-merger RBM retained units)",
        fontsize=9.5, fontweight="bold", y=0.995)

    out_pdf = OUT_DIR / "fig1.pdf"
    out_png = OUT_DIR / "fig1.png"
    out_pdf3 = OUT_DIR / "fig1_three_examples.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(out_pdf3, bbox_inches="tight", facecolor="white")
    print(f"OK: wrote {out_pdf}, {out_png}, {out_pdf3}")


if __name__ == "__main__":
    render()
