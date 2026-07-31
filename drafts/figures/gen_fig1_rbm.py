#!/usr/bin/env python3
"""Generate Fig. 1: Rank-Before-Merge versus post-merger ranking.

The image overlays are deterministic layout proxies. They illustrate where a
score is read and which spatial units survive; they are not model activations,
attention maps, query-conditioned signals, or measured feature L2 scores.
Replace the proxy overlays with measured merger-input/output L2 scores before
submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw


# Restrained palette synthesized from FastV, TokenPacker, VisionZip,
# PyramidDrop, and LLaVA-PruMerge method figures.
INK = "#1F2937"
INK_MUTED = "#667085"
LINE = "#CBD2DC"
SURFACE = "#FFFFFF"
BACKBONE = "#2F5597"
BACKBONE_LIGHT = "#91ACE0"
RBM = "#F2BA02"
RBM_LIGHT = "#FFF4CC"
POST = "#EE822F"
POST_LIGHT = "#FFF0E5"
KEPT = "#3A7EC7"
DROPPED = "#D9DEE7"
MERGED = "#FFF2CC"

GRID_COLS = 14
GRID_ROWS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the RBM pre/post-merger method schematic."
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="Input example image used in all overlay views.")
    parser.add_argument("--output-pdf", type=Path, required=True,
                        help="Vector PDF output path.")
    parser.add_argument("--output-png", type=Path, required=True,
                        help="Raster PNG output path.")
    parser.add_argument("--keep-ratio", type=float, default=0.25,
                        help="Fraction of spatial units retained (default: 0.25).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution (default: 300 dpi).")
    return parser.parse_args()


def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _bounds(length: int, count: int) -> np.ndarray:
    return np.rint(np.linspace(0, length, count + 1)).astype(int)


def _proxy_scores(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic pre/post layout proxies, never model scores."""
    height, width, _ = image.shape
    ys = _bounds(height, GRID_ROWS)
    xs = _bounds(width, GRID_COLS)

    rgb = image.astype(float) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    global_mean = rgb.mean(axis=(0, 1))

    pre = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
    post = np.zeros_like(pre)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y0, y1 = ys[row], ys[row + 1]
            x0, x1 = xs[col], xs[col + 1]
            tile_gray = gray[y0:y1, x0:x1]
            tile_edge = edge[y0:y1, x0:x1]
            tile_rgb = rgb[y0:y1, x0:x1]

            # A local-detail proxy makes the raw/pre view spatially structured.
            pre[row, col] = 0.55 * tile_gray.std() + 0.45 * tile_edge.mean()

            # A tile-average proxy intentionally changes the ranking after the
            # visual averaging operation. It is not a feature-space merger.
            mean_rgb = tile_rgb.mean(axis=(0, 1))
            post[row, col] = np.linalg.norm(mean_rgb - global_mean)

    return _normalize(pre), _normalize(post)


def _top_mask(scores: np.ndarray, keep_ratio: float) -> np.ndarray:
    count = max(1, int(round(scores.size * keep_ratio)))
    chosen = np.argpartition(scores.ravel(), -count)[-count:]
    mask = np.zeros(scores.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(scores.shape)


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _score_view(image: np.ndarray, scores: np.ndarray, color: str,
                base: np.ndarray | None = None) -> np.ndarray:
    canvas = image.copy() if base is None else base.copy()
    height, width, _ = canvas.shape
    ys, xs = _bounds(height, GRID_ROWS), _bounds(width, GRID_COLS)
    tint = _rgb(color)
    output = canvas.astype(float)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y0, y1 = ys[row], ys[row + 1]
            x0, x1 = xs[col], xs[col + 1]
            alpha = 0.08 + 0.56 * scores[row, col]
            output[y0:y1, x0:x1] = (
                (1.0 - alpha) * output[y0:y1, x0:x1] + alpha * tint
            )
    rendered = Image.fromarray(np.uint8(np.clip(output, 0, 255)))
    draw = ImageDraw.Draw(rendered)
    for x in xs[1:-1]:
        draw.line((x, 0, x, height), fill=(255, 255, 255, 150), width=1)
    for y in ys[1:-1]:
        draw.line((0, y, width, y), fill=(255, 255, 255, 150), width=1)
    return np.asarray(rendered)


def _keep_view(image: np.ndarray, keep: np.ndarray,
               color: str = KEPT) -> np.ndarray:
    height, width, _ = image.shape
    ys, xs = _bounds(height, GRID_ROWS), _bounds(width, GRID_COLS)
    output = image.astype(float) * 0.28 + 255.0 * 0.72
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if keep[row, col]:
                y0, y1 = ys[row], ys[row + 1]
                x0, x1 = xs[col], xs[col + 1]
                output[y0:y1, x0:x1] = image[y0:y1, x0:x1]

    rendered = Image.fromarray(np.uint8(np.clip(output, 0, 255)))
    draw = ImageDraw.Draw(rendered)
    kept_rgb = tuple(_rgb(color).astype(int))
    dropped_rgb = tuple(_rgb(DROPPED).astype(int))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y0, y1 = ys[row], ys[row + 1]
            x0, x1 = xs[col], xs[col + 1]
            if keep[row, col]:
                draw.rectangle((x0 + 1, y0 + 1, x1 - 2, y1 - 2),
                               outline=kept_rgb, width=3)
            else:
                draw.line((x0 + 2, y1 - 2, x1 - 2, y0 + 2),
                          fill=dropped_rgb, width=2)
    return np.asarray(rendered)


def _merged_view(image: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    height, width, _ = image.shape
    ys, xs = _bounds(height, GRID_ROWS), _bounds(width, GRID_COLS)
    output = np.full_like(image, 248)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y0, y1 = ys[row], ys[row + 1]
            x0, x1 = xs[col], xs[col + 1]
            mean_rgb = image[y0:y1, x0:x1].mean(axis=(0, 1))
            if keep is not None and not keep[row, col]:
                mean_rgb = 0.18 * mean_rgb + 0.82 * np.array([245, 246, 248])
            output[y0:y1, x0:x1] = mean_rgb

    rendered = Image.fromarray(output)
    draw = ImageDraw.Draw(rendered)
    merged_rgb = tuple(_rgb(BACKBONE_LIGHT).astype(int))
    kept_rgb = tuple(_rgb(KEPT).astype(int))
    dropped_rgb = tuple(_rgb(DROPPED).astype(int))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y0, y1 = ys[row], ys[row + 1]
            x0, x1 = xs[col], xs[col + 1]
            outline = kept_rgb if keep is not None and keep[row, col] else merged_rgb
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=outline, width=2)
            if keep is not None and not keep[row, col]:
                draw.line((x0 + 2, y1 - 2, x1 - 2, y0 + 2),
                          fill=dropped_rgb, width=2)
    return np.asarray(rendered)


def _add_image_axis(fig: plt.Figure, rect: tuple[float, float, float, float],
                    image: np.ndarray) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(LINE)
        spine.set_linewidth(0.8)
    return ax


def _figure_arrow(fig: plt.Figure, start: tuple[float, float],
                  end: tuple[float, float], color: str = INK_MUTED,
                  style: str = "-|>", linewidth: float = 1.0) -> None:
    arrow = FancyArrowPatch(start, end, transform=fig.transFigure,
                            arrowstyle=style, mutation_scale=9,
                            linewidth=linewidth, color=color,
                            shrinkA=0, shrinkB=0, clip_on=False)
    fig.add_artist(arrow)


def _stage_label(fig: plt.Figure, x: float, y: float, number: str,
                 title: str, subtitle: str, color: str) -> None:
    fig.text(x, y, number, ha="left", va="center", fontsize=6.8,
             weight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc=color, ec=color, lw=0))
    fig.text(x + 0.029, y + 0.004, title, ha="left", va="center",
             fontsize=6.7, weight="bold", color=INK)
    fig.text(x + 0.029, y - 0.026, subtitle, ha="left", va="center",
             fontsize=5.3, color=INK_MUTED)


def generate(image_path: Path, output_pdf: Path, output_png: Path,
             keep_ratio: float, dpi: int) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0, 1].")

    pil_image = Image.open(image_path).convert("RGB")
    # A fixed working width makes proxy generation and output deterministic.
    target_width = 980
    target_height = round(pil_image.height * target_width / pil_image.width)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    pil_image = pil_image.resize((target_width, target_height), resampling)
    image = np.asarray(pil_image)

    pre_scores, post_scores = _proxy_scores(image)
    pre_keep = _top_mask(pre_scores, keep_ratio)
    post_keep = _top_mask(post_scores, keep_ratio)

    full_merged = _merged_view(image)
    top_views = [
        _score_view(image, pre_scores, RBM),
        _keep_view(image, pre_keep),
        _merged_view(image, pre_keep),
    ]
    bottom_views = [
        full_merged,
        _score_view(image, post_scores, RBM, base=full_merged),
        _keep_view(full_merged, post_keep),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.09, 3.85), facecolor=SURFACE)

    # Header and restrained row bands.
    fig.text(0.035, 0.960,
             "Rank-Before-Merge isolates the ranking stage",
             ha="left", va="top", fontsize=9.3, weight="bold", color=INK)
    fig.text(0.965, 0.925, "Same input, budget, L2 selector, and native merger",
             ha="right", va="top", fontsize=5.9, color=INK_MUTED)
    fig.add_artist(Rectangle((0.287, 0.557), 0.686, 0.315,
                             transform=fig.transFigure, fc=RBM_LIGHT,
                             ec="none", alpha=0.42, zorder=-10))
    fig.add_artist(Rectangle((0.287, 0.220), 0.686, 0.316,
                             transform=fig.transFigure, fc="#F5F7FA",
                             ec="none", zorder=-10))
    fig.add_artist(Line2D([0.287, 0.973], [0.548, 0.548],
                          transform=fig.transFigure, color=LINE, lw=0.8))

    # Shared input and shared ViT feature source.
    _add_image_axis(fig, (0.038, 0.420, 0.218, 0.239), image)
    fig.text(0.038, 0.682, "Shared visual input", fontsize=7.8,
             weight="bold", ha="left", va="bottom", color=INK)
    fig.text(0.038, 0.397, "One image, two controlled paths",
             fontsize=5.8, ha="left", va="top", color=INK_MUTED)
    vit_box = FancyBboxPatch((0.055, 0.286), 0.184, 0.068,
                             transform=fig.transFigure,
                             boxstyle="round,pad=0.008,rounding_size=0.010",
                             fc=BACKBONE, ec=BACKBONE, lw=0.8)
    fig.add_artist(vit_box)
    fig.text(0.147, 0.320, "ViT units (N)\nmerger input", ha="center",
             va="center", fontsize=6.1, weight="bold", color="white",
             linespacing=1.05)
    _figure_arrow(fig, (0.147, 0.416), (0.147, 0.358), color=BACKBONE)

    # One shared bus branches to the pre/post paths.
    fig.add_artist(Line2D([0.239, 0.272], [0.320, 0.320],
                          transform=fig.transFigure, color=BACKBONE, lw=1.2))
    fig.add_artist(Line2D([0.272, 0.272], [0.320, 0.684],
                          transform=fig.transFigure, color=BACKBONE, lw=1.2))
    _figure_arrow(fig, (0.272, 0.684), (0.307, 0.684), color=BACKBONE)
    _figure_arrow(fig, (0.272, 0.336), (0.307, 0.336), color=BACKBONE)

    # Row labels.
    fig.text(0.297, 0.848, "a", ha="left", va="center", fontsize=8.0,
             weight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc=RBM, ec=RBM, lw=0))
    fig.text(0.322, 0.848, "RBM (ours): rank before the native merger",
             ha="left", va="center", fontsize=8.2, weight="bold", color=INK)
    fig.text(0.297, 0.514, "b", ha="left", va="center", fontsize=8.0,
             weight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc=POST, ec=POST, lw=0))
    fig.text(0.322, 0.514, "Post-merger control: merge first, then rank",
             ha="left", va="center", fontsize=8.2, weight="bold", color=INK)

    xs = [0.315, 0.535, 0.755]
    width, height = 0.195, 0.202
    top_y, bottom_y = 0.558, 0.235
    for x, view in zip(xs, top_views):
        _add_image_axis(fig, (x, top_y, width, height), view)
    for x, view in zip(xs, bottom_views):
        _add_image_axis(fig, (x, bottom_y, width, height), view)

    # Arrows preserve the exact operation order in each row.
    for y in (top_y + height / 2, bottom_y + height / 2):
        _figure_arrow(fig, (xs[0] + width + 0.004, y),
                      (xs[1] - 0.006, y), color=INK_MUTED)
        _figure_arrow(fig, (xs[1] + width + 0.004, y),
                      (xs[2] - 0.006, y), color=INK_MUTED)

    _stage_label(fig, xs[0], 0.813, "1", "L2 RANK (PROXY)",
                 "on merger-input 2x2 units", RBM)
    _stage_label(fig, xs[1], 0.813, "2", r"KEEP TOP-$\kappa N$",
                 rf"$\kappa$ = {keep_ratio:.0%} of N units", RBM)
    _stage_label(fig, xs[2], 0.813, "3", "NATIVE MERGER",
                 "applied only to survivors", BACKBONE)

    _stage_label(fig, xs[0], 0.485, "1", "NATIVE MERGER",
                 "applied to all N units", BACKBONE)
    _stage_label(fig, xs[1], 0.485, "2", "L2 RANK (PROXY)",
                 "on merged outputs", RBM)
    _stage_label(fig, xs[2], 0.485, "3", r"KEEP TOP-$\kappa N$",
                 rf"$\kappa$ = {keep_ratio:.0%} of merged tokens", POST)

    # Compact token-state legend and explicit fairness/disclaimer notes.
    legend_y = 0.178
    legend_items = [
        (KEPT, "solid border: kept"),
        (DROPPED, "diagonal mark: discarded"),
        (MERGED, "outlined block: merged token"),
    ]
    legend_xs = [0.315, 0.515, 0.735]
    for x, (color, label) in zip(legend_xs, legend_items):
        fig.add_artist(Rectangle((x, legend_y - 0.012), 0.017, 0.020,
                                 transform=fig.transFigure, fc=color,
                                 ec=INK_MUTED, lw=0.6))
        fig.text(x + 0.024, legend_y, label, ha="left", va="center",
                 fontsize=6.2, color=INK_MUTED)

    fig.text(0.500, 0.096,
             r"Controlled comparison: same image, model path, $\kappa$, L2 selector, "
             "and native merger; only the score tap point changes.",
             ha="center", va="center", fontsize=5.9, color=INK, weight="bold")
    fig.text(0.500, 0.038,
             "layout proxy - replace with measured L2 scores before submission. "
             "No attention or query signal is depicted.",
             ha="center", va="center", fontsize=5.8, color="#9A6700",
             weight="bold")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_png, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"wrote {output_pdf}")
    print(f"wrote {output_png}")


def main() -> None:
    args = parse_args()
    generate(args.image, args.output_pdf, args.output_png,
             args.keep_ratio, args.dpi)


if __name__ == "__main__":
    main()
