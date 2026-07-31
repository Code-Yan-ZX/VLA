#!/usr/bin/env python3
"""Generate Fig. 1: Rank-Before-Merge versus post-merger ranking.

Two render modes
----------------
1. MEASURED (default intent): overlays come ONLY from measured merger features
   stored in a capture NPZ (pre_l2/post_l2/pre_keep/post_keep/unit_grid_hw).
   Scores are NEVER recomputed from image pixels; keep masks are taken directly
   from the NPZ keep arrays (not re-thresholded). Pre and post scores are
   min-max normalized SEPARATELY and ONLY to drive the color/alpha display; the
   selection shown is the raw measured-L2 top-k recorded in the NPZ.

2. LAYOUT PROXY (legacy): deterministic edge/color layout proxies that only
   illustrate *where* a score is read and *which* units survive. They are not
   model activations, attention, query-conditioned signals, or measured L2.
   This path is retained for layout work and is clearly labeled "PROXY".
   It is only reachable with --allow-layout-proxy (or when --scores-npz is
   absent and --allow-layout-proxy is given). Without measured data and without
   --allow-layout-proxy the script refuses to render and exits nonzero.

The native merger is a LEARNED NONLINEAR merger. The "NATIVE MERGER" panels are
a schematic blocky depiction (per-unit color tiles) for layout only; the merger
itself is not simulated here and must never be described as an average.

No attention, CLIP/SigLIP, or question-conditioned signal is depicted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Reproducible PDF metadata (matplotlib honors SOURCE_DATE_EPOCH for CreationDate).
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

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
WARN = "#9A6700"

# Proxy-mode working grid (coarse, illustrative only).
GRID_COLS = 14
GRID_ROWS = 8

# Display pixels per measured unit. The input image is resized to
# (cols*UNIT_PX, rows*UNIT_PX) so every unit is an exact UNIT_PX x UNIT_PX
# block; this keeps the displayed image aspect-true to the processor geometry
# (h_px:w_px == rows:cols) and keeps mask overlays unit-aligned with no drift.
UNIT_PX = 6

PIPELINE_DIR = Path(__file__).resolve().parent / "real_data_pipeline"
DEFAULT_DATA_DIR = PIPELINE_DIR / "data"
DEFAULT_INPUTS_DIR = PIPELINE_DIR / "inputs"
DEFAULT_OUTDIR = PIPELINE_DIR / "outputs"

PROVENANCE_LABEL = "Qwen3-VL-8B | measured merger features | 25% kept"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the RBM pre/post-merger method schematic "
                    "(measured NPZ mode or clearly-labeled layout proxy).")
    # single-image / proxy outputs
    parser.add_argument("--image", type=Path, default=None,
                        help="Input image (proxy mode, or measured override).")
    parser.add_argument("--output-pdf", type=Path, default=None,
                        help="Vector PDF output path (single-image mode).")
    parser.add_argument("--output-png", type=Path, default=None,
                        help="Raster PNG output path (single-image mode).")
    parser.add_argument("--keep-ratio", type=float, default=0.25,
                        help="Fraction of spatial units retained (default 0.25).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution (default 300 dpi).")
    # measured mode
    parser.add_argument("--scores-npz", type=Path, default=None,
                        help="Measured capture NPZ (pre/post L2 + keep masks).")
    parser.add_argument("--metadata-json", type=Path, default=None,
                        help="Per-sample sidecar JSON (geometry + provenance).")
    parser.add_argument("--validation-report", type=Path, default=None,
                        help="data/validation_report.json (optional gate check).")
    parser.add_argument("--inputs-dir", type=Path, default=None,
                        help="Directory holding source images (default: "
                             "real_data_pipeline/inputs).")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory for per-sample + contact sheet.")
    # guards
    parser.add_argument("--allow-layout-proxy", action="store_true",
                        help="Permit the clearly-labeled layout-proxy path when "
                             "no measured NPZ is supplied.")
    # batch
    parser.add_argument("--batch-dir", type=Path, default=None,
                        help="Directory of <id>.npz + <id>.json; render every "
                             "sample plus a contact sheet in one invocation.")
    parser.add_argument("--sample-id", default=None,
                        help="(batch) render only this sample id.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _minmax(values: np.ndarray) -> np.ndarray:
    """Per-arm min-max to [0,1] for COLOR DISPLAY ONLY (never for selection)."""
    v = values.astype(float)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def _bounds(length: int, count: int) -> np.ndarray:
    return np.rint(np.linspace(0, length, count + 1)).astype(int)


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


# --------------------------------------------------------------------------- #
# PROXY-mode score synthesis (illustrative only; never model/measured scores)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# PROXY-mode views (coarse grid, PIL decorated) -- accepted layout, preserved
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# MEASURED-mode views (real unit grid; arrays are unit-aligned, no pixel scores)
# --------------------------------------------------------------------------- #
def _display_image(image_path: Path, rows: int, cols: int,
                   unit_px: int = UNIT_PX) -> np.ndarray:
    """Resize the source image onto the measured unit grid for DISPLAY.

    Mapping (documented): the capture scored the processor-resized image whose
    pixel geometry is h_px = rows*32, w_px = cols*32 (patch 16px, merge 2 -> unit
    32px), so its aspect ratio is exactly rows:cols. We resize the source image
    directly to (cols*unit_px, rows*unit_px), which preserves that aspect ratio
    and lands every unit on an exact unit_px x unit_px block. Scores/masks are
    NOT derived from these pixels -- they come from the NPZ.
    """
    pil = Image.open(image_path).convert("RGB")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    target = (cols * unit_px, rows * unit_px)  # (width, height)
    pil = pil.resize(target, resampling)
    return np.asarray(pil)


def _m_score_view(image: np.ndarray, norm_scores: np.ndarray, color: str,
                  base: np.ndarray | None = None,
                  grid: bool = False) -> np.ndarray:
    """Per-unit amber tint, alpha = 0.08 + 0.56 * normalized score.

    `norm_scores` is a (rows, cols) array already min-max normalized for COLOR
    only. Selection is never read from here.
    """
    canvas = image if base is None else base
    rows, cols = norm_scores.shape
    u = canvas.shape[0] // rows
    alpha = 0.08 + 0.56 * norm_scores
    alpha_big = np.repeat(np.repeat(alpha, u, axis=0), u, axis=1)[..., None]
    tint = _rgb(color)
    out = (1.0 - alpha_big) * canvas.astype(float) + alpha_big * tint
    out = np.uint8(np.clip(out, 0, 255))
    if grid:
        out = _draw_unit_grid(out, rows, cols, u, (255, 255, 255), alpha=0.5)
    return out


def _m_keep_view(image: np.ndarray, keep: np.ndarray, color: str = KEPT,
                 base: np.ndarray | None = None) -> np.ndarray:
    """Fade everything; restore kept units to full color; amber/kept outline.

    `keep` is the measured NPZ keep mask reshaped to (rows, cols) -- used
    verbatim, never re-thresholded.
    """
    canvas = image if base is None else base
    rows, cols = keep.shape
    u = canvas.shape[0] // rows
    faded = canvas.astype(float) * 0.42 + 255.0 * 0.58
    keep_big = np.repeat(np.repeat(keep, u, axis=0), u, axis=1)
    out = faded.copy()
    out[keep_big] = canvas[keep_big].astype(float)
    rendered = Image.fromarray(np.uint8(np.clip(out, 0, 255)))
    draw = ImageDraw.Draw(rendered)
    rgb = tuple(_rgb(color).astype(int))
    # Outline kept units; at fine grids this reads as a crisp kept-region edge.
    for r in range(rows):
        for c in range(cols):
            if keep[r, c]:
                x0, y0 = c * u, r * u
                draw.rectangle((x0, y0, x0 + u - 1, y0 + u - 1),
                               outline=rgb, width=1)
    return np.asarray(rendered)


def _m_overlay_mask(image: np.ndarray, keep: np.ndarray,
                    color: str = RBM) -> np.ndarray:
    """Contact-sheet overlay: amber mask on kept units over a visible input.

    Dropped units are only lightly paled (image stays readable as context); kept
    units get a translucent amber mask + outline so the 25% survival pattern is
    instantly legible at fine grids. `keep` is the measured NPZ mask (verbatim).
    This helper is for the chooser contact sheet, not the publication figure.
    """
    rows, cols = keep.shape
    u = image.shape[0] // rows
    keep_big = np.repeat(np.repeat(keep, u, axis=0), u, axis=1)
    out = image.astype(float).copy()
    out[~keep_big] = 0.45 * out[~keep_big] + 0.55 * 255.0
    tint = _rgb(color)
    out[keep_big] = 0.42 * out[keep_big] + 0.58 * tint
    rendered = Image.fromarray(np.uint8(np.clip(out, 0, 255)))
    draw = ImageDraw.Draw(rendered)
    rgb = tuple(_rgb(color).astype(int))
    for r in range(rows):
        for c in range(cols):
            if keep[r, c]:
                x0, y0 = c * u, r * u
                draw.rectangle((x0, y0, x0 + u - 1, y0 + u - 1),
                               outline=rgb, width=1)
    return np.asarray(rendered)


def _m_merged_view(image: np.ndarray, keep: np.ndarray | None,
                   unit_px: int = UNIT_PX) -> np.ndarray:
    """SCHEMATIC blocky depiction of merged tokens (per-unit color tiles).

    This is a layout aid ONLY: it visualizes the native merger as blocky tokens
    with dropped units desaturated. The real native merger is a learned
    nonlinear merger and is NOT simulated or averaged here.
    """
    rows, cols = (keep.shape if keep is not None
                  else (image.shape[0] // unit_px, image.shape[1] // unit_px))
    u = image.shape[0] // rows
    img = image.astype(float)
    unit_mean = img.reshape(rows, u, cols, u, 3).mean(axis=(1, 3))
    if keep is not None:
        drop = ~keep
        unit_mean[drop] = 0.18 * unit_mean[drop] + 0.82 * np.array([245, 246, 248])
    out = np.repeat(np.repeat(unit_mean, u, axis=0), u, axis=1)
    out = np.uint8(np.clip(out, 0, 255))
    # subtle merged-token grid
    grid_color = tuple(_rgb(BACKBONE_LIGHT).astype(int))
    out = _draw_unit_grid(out, rows, cols, u, grid_color, alpha=0.55)
    if keep is not None:
        rendered = Image.fromarray(out)
        draw = ImageDraw.Draw(rendered)
        kept_rgb = tuple(_rgb(KEPT).astype(int))
        for r in range(rows):
            for c in range(cols):
                if keep[r, c]:
                    x0, y0 = c * u, r * u
                    draw.rectangle((x0, y0, x0 + u - 1, y0 + u - 1),
                                   outline=kept_rgb, width=1)
        out = np.asarray(rendered)
    return out


def _draw_unit_grid(arr: np.ndarray, rows: int, cols: int, u: int,
                    color: tuple, alpha: float = 0.5) -> np.ndarray:
    """Blend thin grid lines at unit boundaries (interior only)."""
    out = arr.astype(float)
    line = np.array(color, dtype=float)
    h, w = out.shape[:2]
    for r in range(1, rows):
        y = min(r * u, h - 1)
        out[y, :, :] = (1 - alpha) * out[y, :, :] + alpha * line
    for c in range(1, cols):
        x = min(c * u, w - 1)
        out[:, x, :] = (1 - alpha) * out[:, x, :] + alpha * line
    return np.uint8(np.clip(out, 0, 255))


# --------------------------------------------------------------------------- #
# measured data loading + validation gate
# --------------------------------------------------------------------------- #
def load_measured(npz_path: Path) -> dict:
    z = np.load(npz_path)
    unit_grid_hw = z["unit_grid_hw"]
    uh, uw = int(unit_grid_hw[0]), int(unit_grid_hw[1])
    pre_l2 = z["pre_l2"].astype(float)
    post_l2 = z["post_l2"].astype(float)
    if pre_l2.size != uh * uw or post_l2.size != uh * uw:
        raise ValueError(f"{npz_path}: score length {pre_l2.size} != "
                         f"uh*uw={uh * uw}")
    return {
        "uh": uh, "uw": uw, "N": int(pre_l2.size),
        "pre_l2": pre_l2, "post_l2": post_l2,
        "pre_keep": z["pre_keep"].astype(bool),
        "post_keep": z["post_keep"].astype(bool),
        "k": int(z["pre_keep"].sum()),
    }


def validation_passed(sample_id: str, report_path: Path | None) -> bool:
    """True only if a validation report is supplied AND this sample fully passed."""
    if report_path is None or not Path(report_path).is_file():
        return False
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    for s in report.get("samples", []):
        if s.get("sample_id") == sample_id:
            gates_ok = all(g.get("pass") for g in s.get("gates", {}).values())
            return bool(s.get("sample_passed") and gates_ok)
    return False


def resolve_image(metadata: dict | None, inputs_dir: Path | None,
                  image_arg: Path | None) -> Path:
    if image_arg is not None:
        return Path(image_arg)
    if metadata is not None and metadata.get("original_filename"):
        base = inputs_dir or DEFAULT_INPUTS_DIR
        return Path(base) / metadata["original_filename"]
    raise SystemExit("measured mode needs --image or --metadata-json with "
                     "original_filename (plus --inputs-dir).")


# --------------------------------------------------------------------------- #
# shared figure composition (identical accepted layout for both modes)
# --------------------------------------------------------------------------- #
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


def _compose_fig1(top_views, bottom_views, keep_ratio: float, mode: str,
                  val_passed: bool, out_pdf: Path, out_png: Path,
                  dpi: int) -> None:
    measured = (mode == "measured")
    rank_label = "MEASURED L2 RANK" if measured else "L2 RANK (PROXY)"

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

    # Shared input and shared ViT feature source. Callers pass the plain display
    # image as the LAST element of top_views; the first three are the row-a views.
    input_img = top_views[-1]
    _add_image_axis(fig, (0.038, 0.420, 0.218, 0.239), input_img)
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

    # Row labels. Orange accents the post-control row only; the L2 selector is
    # identical in both rows (amber L2-rank visual language in BOTH rows).
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
    for x, view in zip(xs, top_views[:3]):
        _add_image_axis(fig, (x, top_y, width, height), view)
    for x, view in zip(xs, bottom_views[:3]):
        _add_image_axis(fig, (x, bottom_y, width, height), view)

    # Arrows preserve the exact operation order in each row.
    for y in (top_y + height / 2, bottom_y + height / 2):
        _figure_arrow(fig, (xs[0] + width + 0.004, y),
                      (xs[1] - 0.006, y), color=INK_MUTED)
        _figure_arrow(fig, (xs[1] + width + 0.004, y),
                      (xs[2] - 0.006, y), color=INK_MUTED)

    _stage_label(fig, xs[0], 0.813, "1", rank_label,
                 "on merger-input 2x2 units", RBM)
    _stage_label(fig, xs[1], 0.813, "2", r"KEEP TOP-$\kappa N$",
                 rf"$\kappa$ = {keep_ratio:.0%} of N units", RBM)
    _stage_label(fig, xs[2], 0.813, "3", "NATIVE MERGER",
                 "applied only to survivors", BACKBONE)

    _stage_label(fig, xs[0], 0.485, "1", "NATIVE MERGER",
                 "applied to all N units", BACKBONE)
    _stage_label(fig, xs[1], 0.485, "2", rank_label,
                 "on merged outputs", RBM)
    _stage_label(fig, xs[2], 0.485, "3", r"KEEP TOP-$\kappa N$",
                 rf"$\kappa$ = {keep_ratio:.0%} of merged tokens", POST)

    # Compact token-state legend (labels match what is actually drawn per mode).
    legend_y = 0.178
    if measured:
        legend_items = [
            (KEPT, "solid border: kept"),
            ("#EDF0F5", "faded: discarded"),
            (MERGED, "blocky tile: merged token"),
        ]
    else:
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

    # Footer (two non-colliding lines). Provenance left + status right on the
    # upper line; the color/selection rule centered on the lower line.
    if measured:
        fig.text(0.035, 0.046, PROVENANCE_LABEL, ha="left", va="center",
                 fontsize=5.8, color=INK_MUTED, weight="bold")
        if val_passed:
            fig.text(0.965, 0.046,
                     "No attention or query signal is depicted.",
                     ha="right", va="center", fontsize=5.4, color=INK_MUTED)
        else:
            fig.text(0.965, 0.046,
                     "WARNING: measured L2, sample validation NOT confirmed.",
                     ha="right", va="center", fontsize=5.4, color=WARN,
                     weight="bold")
        fig.text(0.500, 0.016,
                 "color: per-arm min-max   |   selection: raw measured L2 (NPZ keep)",
                 ha="center", va="center", fontsize=5.6, color=INK_MUTED)
    else:
        fig.text(0.500, 0.030,
                 "layout proxy - replace with measured L2 scores before submission. "
                 "No attention or query signal is depicted.",
                 ha="center", va="center", fontsize=5.8, color=WARN,
                 weight="bold")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Fixed canvas (no bbox_inches='tight') -> PDF page is exactly 7.09 x 3.85 in.
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# per-sample renderers
# --------------------------------------------------------------------------- #
def render_measured(sample_id: str, npz_path: Path, metadata: dict,
                    inputs_dir: Path | None, image_arg: Path | None,
                    report_path: Path | None, keep_ratio: float, dpi: int,
                    outdir: Path) -> dict:
    m = load_measured(npz_path)
    uh, uw = m["uh"], m["uw"]
    image_path = resolve_image(metadata, inputs_dir, image_arg)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    image = _display_image(Path(image_path), uh, uw)

    # COLOR normalization is per-arm and display-only; SELECTION uses NPZ masks.
    pre_norm = _minmax(m["pre_l2"]).reshape(uh, uw)
    post_norm = _minmax(m["post_l2"]).reshape(uh, uw)
    pre_keep = m["pre_keep"].reshape(uh, uw)
    post_keep = m["post_keep"].reshape(uh, uw)

    full_merged = _m_merged_view(image, None)
    top_views = [
        _m_score_view(image, pre_norm, RBM),
        _m_keep_view(image, pre_keep),
        _m_merged_view(image, pre_keep),
        image,
    ]
    bottom_views = [
        full_merged,
        _m_score_view(image, post_norm, RBM, base=full_merged),
        _m_keep_view(image, post_keep, base=full_merged),
    ]

    val_ok = validation_passed(sample_id, report_path)
    out_pdf = outdir / f"fig1_{sample_id}.pdf"
    out_png = outdir / f"fig1_{sample_id}.png"
    _compose_fig1(top_views, bottom_views, keep_ratio, "measured", val_ok,
                  out_pdf, out_png, dpi)
    return {"sample_id": sample_id, "pdf": out_pdf, "png": out_png,
            "validation_passed": val_ok, "N": m["N"], "k": m["k"]}


def render_proxy(image_path: Path, out_pdf: Path, out_png: Path,
                 keep_ratio: float, dpi: int) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0, 1].")

    pil_image = Image.open(image_path).convert("RGB")
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
        image,
    ]
    bottom_views = [
        full_merged,
        _score_view(image, post_scores, RBM, base=full_merged),
        _keep_view(full_merged, post_keep),
    ]
    _compose_fig1(top_views, bottom_views, keep_ratio, "proxy", False,
                  out_pdf, out_png, dpi)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------- #
# contact sheet (all candidates as small multiples; author chooses, not us)
# --------------------------------------------------------------------------- #
def render_contact_sheet(items: list[dict], outdir: Path, dpi: int) -> tuple:
    """items: list of {sample_id, image (np), pre_keep (2d bool), jaccard, rho}."""
    n = len(items)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.09, 0.62 + 1.30 * nrows), facecolor=SURFACE)
    fig.text(0.5, 0.985, "FIG:1 candidates - RBM keep overlay (measured L2, 25% kept)",
             ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
    fig.text(0.5, 0.958,
             "Author selects the example; panels are not pre-ranked. "
             "color: per-arm min-max; selection: raw L2.",
             ha="center", va="top", fontsize=5.8, color=INK_MUTED)
    gs = fig.add_gridspec(nrows, ncols, left=0.015, right=0.985,
                          top=0.885, bottom=0.02, hspace=0.42, wspace=0.10)
    for idx in range(nrows * ncols):
        ax = fig.add_subplot(gs[idx])
        ax.set_xticks([]); ax.set_yticks([])
        if idx >= n:
            ax.axis("off")
            continue
        it = items[idx]
        view = _m_overlay_mask(it["image"], it["pre_keep"], RBM)
        ax.imshow(view)
        for spine in ax.spines.values():
            spine.set_color(LINE); spine.set_linewidth(0.8)
        ax.set_title(
            f"#{idx} {it['short']}  J={it['jaccard']:.3f} "
            f"ρ={it['rho']:.3f}", fontsize=6.0, color=INK, pad=2)
        # label below the image (as xlabel) so tall panels never collide with the
        # figure-level subtitle at the top.
        ax.set_xlabel(f"kept {it['n_keep']}/{it['N']} units",
                      fontsize=5.0, color=INK_MUTED)
    out_pdf = outdir / "fig1_contact_sheet.pdf"
    out_png = outdir / "fig1_contact_sheet.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=dpi)
    plt.close(fig)
    return out_pdf, out_png


# --------------------------------------------------------------------------- #
# batch driver
# --------------------------------------------------------------------------- #
def _short_id(sample_id: str) -> str:
    # ASCII-safe short label (sample ids may contain non-ASCII / be long UUIDs).
    if len(sample_id) > 20 and "-" in sample_id:
        return sample_id.split("-")[0]
    tail = sample_id.split("_")
    if len(tail) >= 2 and tail[-2].isdigit():
        return tail[-2]
    return sample_id[-6:]


def _diag(npz_path: Path) -> tuple[float, float]:
    from scipy.stats import spearmanr
    z = np.load(npz_path)
    pk, ok = z["pre_keep"].astype(bool), z["post_keep"].astype(bool)
    inter = np.logical_and(pk, ok).sum()
    union = np.logical_or(pk, ok).sum()
    j = float(inter / union) if union else float("nan")
    rho = float(spearmanr(z["pre_rank"].astype(float),
                          z["post_rank"].astype(float))[0])
    return j, rho


def run_batch(batch_dir: Path, inputs_dir: Path, report_path: Path | None,
              keep_ratio: float, dpi: int, outdir: Path,
              only: str | None) -> None:
    batch_dir = Path(batch_dir)
    npzs = sorted(batch_dir.glob("*.npz"))
    if only:
        npzs = [p for p in npzs if p.stem == only]
    if not npzs:
        raise SystemExit(f"no .npz found in {batch_dir}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results, failures, sheet_items = [], [], []
    for npz in npzs:
        sid = npz.stem
        meta_path = batch_dir / f"{sid}.json"
        metadata = {}
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        try:
            res = render_measured(sid, npz, metadata, inputs_dir, None,
                                  report_path, keep_ratio, dpi, outdir)
            j, rho = _diag(npz)
            m = load_measured(npz)
            img = _display_image(resolve_image(metadata, inputs_dir, None),
                                 m["uh"], m["uw"])
            sheet_items.append({
                "sample_id": sid, "short": _short_id(sid),
                "image": img,
                "pre_keep": m["pre_keep"].reshape(m["uh"], m["uw"]),
                "jaccard": j, "rho": rho, "n_keep": m["k"], "N": m["N"],
            })
            results.append(res)
            print(f"[batch] {sid}: OK (validation_passed={res['validation_passed']}, "
                  f"J={j:.3f}, rho={rho:.3f}) -> {res['pdf'].name}")
        except Exception as exc:  # per-sample failure must not abort the batch
            failures.append({"sample_id": sid, "error": str(exc)})
            print(f"[batch] {sid}: FAILED -> {exc}")

    sheet = None
    if sheet_items:
        sheet = render_contact_sheet(sheet_items, outdir, dpi)
        print(f"[batch] contact sheet -> {sheet[0].name}")

    print(f"[batch] done: {len(results)} rendered, {len(failures)} failed.")
    if failures:
        for f in failures:
            print(f"        FAIL {f['sample_id']}: {f['error']}")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # ---- batch mode ---- #
    if args.batch_dir is not None:
        inputs_dir = args.inputs_dir or DEFAULT_INPUTS_DIR
        report = args.validation_report
        if report is None:
            cand = Path(args.batch_dir) / "validation_report.json"
            report = cand if cand.is_file() else None
        outdir = args.outdir or DEFAULT_OUTDIR
        run_batch(Path(args.batch_dir), Path(inputs_dir), report,
                  args.keep_ratio, args.dpi, Path(outdir), args.sample_id)
        return

    # ---- single measured mode ---- #
    if args.scores_npz is not None:
        npz = Path(args.scores_npz)
        if not npz.is_file():
            raise SystemExit(f"scores NPZ not found: {npz}")
        sample_id = npz.stem
        metadata = {}
        if args.metadata_json is not None:
            if not Path(args.metadata_json).is_file():
                raise SystemExit(f"metadata JSON not found: {args.metadata_json}")
            with open(args.metadata_json, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        else:
            guess = npz.with_suffix(".json")
            if guess.is_file():
                with open(guess, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
        inputs_dir = args.inputs_dir
        outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
        outdir.mkdir(parents=True, exist_ok=True)
        res = render_measured(sample_id, npz, metadata, inputs_dir, args.image,
                              args.validation_report, args.keep_ratio,
                              args.dpi, outdir)
        print(f"wrote {res['pdf']}")
        print(f"wrote {res['png']}")
        print(f"validation_passed={res['validation_passed']} N={res['N']} k={res['k']}")
        return

    # ---- no measured data: proxy only with explicit consent, else refuse ---- #
    if args.allow_layout_proxy:
        if not (args.image and args.output_pdf and args.output_png):
            raise SystemExit("proxy mode requires --image, --output-pdf, "
                             "--output-png.")
        render_proxy(Path(args.image), Path(args.output_pdf),
                     Path(args.output_png), args.keep_ratio, args.dpi)
        return

    raise SystemExit(
        "Refusing to render: measured merger-feature NPZ is required for Fig.1.\n"
        "  Provide --scores-npz <id>.npz --metadata-json <id>.json "
        "(+ --validation-report data/validation_report.json), or run\n"
        "  --batch-dir data/ to render every captured sample.\n"
        "  The clearly-labeled layout proxy is available only with "
        "--allow-layout-proxy (layout work only, not for submission).")


if __name__ == "__main__":
    main()
