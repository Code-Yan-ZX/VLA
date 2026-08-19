#!/usr/bin/env python3
"""Render FIG:2 — forest plot of pre-minus-post deltas and paired 95% CIs.

Canonical data: drafts/figures/real_data_pipeline/data/fig2_values.json
Spec: drafts/figs_spec_for_user.md (FIG:2 L16-26; palette/canvas L64-80; FIG:2 prompt L98-106)

OCRBench's /1000-point delta is divided by 10 so every row uses percentage
points. Confidence intervals are loaded from the audited paired-metric report.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

# Reproducible PDF metadata (matplotlib honors SOURCE_DATE_EPOCH for CreationDate).
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[2]

EXPECTED_MODELS = ["Qwen3-VL-8B", "Qwen2.5-VL-7B", "InternVL3-8B"]
EXPECTED_PANELS = ["TextVQA", "DocVQA", "OCRBench", "GQA"]
EXPECTED_UNITS = {"TextVQA": "pp", "DocVQA": "pp", "OCRBench": "pts", "GQA": "pp"}

# High-contrast, print-safe palette. Distinct marker shapes preserve the model
# mapping in grayscale and for readers with color-vision deficiencies.
COLORS = {
    "Qwen3-VL-8B": "#2563EB",
    "Qwen2.5-VL-7B": "#D1495B",
    "InternVL3-8B": "#168C83",
}
MARKERS = {"Qwen3-VL-8B": "o", "Qwen2.5-VL-7B": "s", "InternVL3-8B": "D"}
INK, AXIS_GRAY, GRID_GRAY, ZERO_GRAY = "#17202A", "#667085", "#E3E8EF", "#344054"
MINUS = "−"  # true minus sign


def resolve_path(arg: str) -> Path:
    """Accept CWD-relative or repo-root-relative paths; prefer an existing file."""
    p = Path(arg)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = REPO_ROOT / arg
    if candidate.exists():
        return candidate
    return p  # let the caller raise with a clear message


def darken(hexcolor: str, factor: float = 0.68) -> str:
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(
        int(r * factor), int(g * factor), int(b * factor))


def signed(v: float, unit: str) -> str:
    if unit == "pts":
        s = "{:+.0f}".format(v)
    else:
        s = "{:+.1f}".format(v)
    return s.replace("-", MINUS) + " " + unit


def validate(data: dict, path: Path) -> None:
    """Fail loudly on any schema/unit/key mismatch before plotting."""
    errs = []

    def check(cond, msg):
        if not cond:
            errs.append(msg)

    check(isinstance(data, dict), "top-level JSON is not an object")
    check(data.get("figure") == "FIG:2",
          "expected figure == 'FIG:2', got {!r}".format(data.get("figure")))
    check(data.get("retention_pct") == 25,
          "expected retention_pct == 25, got {!r}".format(data.get("retention_pct")))
    check(data.get("split") == "full",
          "expected split == 'full', got {!r}".format(data.get("split")))

    models = data.get("models")
    check(models == EXPECTED_MODELS,
          "models must be exactly {} in this order, got {!r}".format(EXPECTED_MODELS, models))

    panels = data.get("panel_order")
    check(panels == EXPECTED_PANELS,
          "panel_order must be exactly {}, got {!r}".format(EXPECTED_PANELS, panels))

    benches = data.get("benchmarks")
    check(isinstance(benches, dict) and set(benches) == set(EXPECTED_PANELS),
          "benchmarks keys must be exactly {}, got {!r}".format(
              sorted(EXPECTED_PANELS), sorted(benches) if isinstance(benches, dict) else benches))
    if isinstance(benches, dict):
        for b, meta in benches.items():
            unit = meta.get("unit") if isinstance(meta, dict) else None
            check(unit == EXPECTED_UNITS.get(b),
                  "benchmark {!r}: expected unit {!r}, got {!r}".format(
                      b, EXPECTED_UNITS.get(b), unit))
            yr = meta.get("suggested_y_range") if isinstance(meta, dict) else None
            check(isinstance(yr, list) and len(yr) == 2 and
                  all(isinstance(x, (int, float)) for x in yr) and yr[0] < yr[1],
                  "benchmark {!r}: suggested_y_range must be [lo, hi] with lo < hi, got {!r}".format(b, yr))

    values = data.get("values")
    check(isinstance(values, dict) and set(values) == set(EXPECTED_MODELS),
          "values keys must be exactly {}, got {!r}".format(
              EXPECTED_MODELS, list(values) if isinstance(values, dict) else values))
    if isinstance(values, dict):
        for m in EXPECTED_MODELS:
            row = values.get(m)
            check(isinstance(row, dict) and set(row) == set(EXPECTED_PANELS),
                  "values[{}]: benchmarks must be exactly {}, got {!r}".format(
                      m, sorted(EXPECTED_PANELS), sorted(row) if isinstance(row, dict) else row))
            if isinstance(row, dict):
                for b in EXPECTED_PANELS:
                    v = row.get(b)
                    check(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
                          "values[{}][{}]: expected a finite number, got {!r}".format(m, b, v))

    prov = data.get("provenance")
    check(isinstance(prov, dict) and prov.get("audited") is True,
          "provenance.audited must be true")

    if errs:
        sys.exit("VALIDATION FAILED for {}:\n  - {}".format(path, "\n  - ".join(errs)))


def render(data: dict, outdir: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.edgecolor": AXIS_GRAY,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,  # editable TrueType text
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })

    stats_path = REPO_ROOT / "experiments" / "paired_metric_statistics.json"
    with open(stats_path, "r", encoding="utf-8") as fh:
        stats = json.load(fh)["stage_law_table1"]
    stat_keys = ["qwen3vl", "qwen2vl", "internvl3"]
    bench_keys = ["textvqa", "docvqa", "ocrbench", "gqa"]

    # Minimalist Cleveland-style forest plot. The benchmark rows are the
    # visual structure; all decoration is removed so the effect sizes and CIs
    # remain the only strong marks on the page.
    fig, ax = plt.subplots(figsize=(7.09, 2.85))
    fig.subplots_adjust(left=0.145, right=0.985, top=0.80, bottom=0.22)

    centers = [3.0, 2.0, 1.0, 0.0]
    offsets = [0.20, 0.0, -0.20]
    for center in centers:
        ax.hlines(center, -6, 50, color=GRID_GRAY, linewidth=0.55,
                  zorder=0)
    ax.axvline(0, color=ZERO_GRAY, linewidth=1.15, zorder=1)
    ax.xaxis.grid(True, color=GRID_GRAY, linewidth=0.55, linestyle=(0, (2, 3)),
                  zorder=0)
    ax.set_axisbelow(True)
    for mi, (model, skey) in enumerate(zip(EXPECTED_MODELS, stat_keys)):
        xs, ys, loerr, hierr = [], [], [], []
        for center, bench, bkey in zip(centers, EXPECTED_PANELS, bench_keys):
            cell = stats[skey]["benchmarks"][bkey]["pre_vs_post"]
            xval = cell["mean_delta_pp"]
            lo, hi = cell["ci95_pp"]
            # Canonical figure JSON stores OCRBench on /1000 scale; paired
            # statistics already express the same delta on a 0--100 scale.
            if bench != "OCRBench":
                plotted = data["values"][model][bench]
                if abs(plotted - xval) > 0.15:
                    raise RuntimeError("figure/statistics mismatch: {} {}".format(model, bench))
            xs.append(xval); ys.append(center + offsets[mi])
            loerr.append(xval - lo); hierr.append(hi - xval)
        ax.errorbar(xs, ys, xerr=[loerr, hierr], fmt=MARKERS[model],
                    color=COLORS[model], ecolor=COLORS[model],
                    elinewidth=1.15, capsize=2.25, markersize=5.4,
                    markeredgecolor=darken(COLORS[model]), markeredgewidth=0.7,
                    label=model, zorder=4)
        for xval, yval in zip(xs, ys):
            # Keep labels just outside the marker while preserving a stable
            # model-specific vertical lane within each benchmark row.
            delta = 0.7 if xval >= 0 else -0.7
            ax.text(xval + delta, yval, "{:+.1f}".format(xval).replace("-", MINUS),
                    ha="left" if xval >= 0 else "right", va="center",
                    fontsize=6.5, color=darken(COLORS[model]), zorder=5)

    ax.set_xlim(-6, 50)
    ax.set_ylim(-0.48, 3.48)
    ax.set_yticks(centers)
    ax.set_yticklabels(["TextVQA", "DocVQA", "OCRBench", "GQA"],
                       fontsize=8.2, fontweight="bold")
    ax.set_xticks([-5, 0, 10, 20, 30, 40, 50])
    ax.set_xlabel("pre-merger − post-merger score difference  (pp; OCRBench /10)",
                  labelpad=6, color=INK)
    ax.tick_params(axis="both", colors=INK, length=2.5, pad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_GRAY)
    ax.spines["bottom"].set_linewidth(0.75)

    # Marker-only legend avoids the visually heavy errorbar samples produced
    # by the default legend handler and remains legible in grayscale.
    legend_handles = [Line2D([0], [0], marker=MARKERS[m], linestyle="None",
                             markerfacecolor=COLORS[m],
                             markeredgecolor=darken(COLORS[m]),
                             markeredgewidth=0.7, markersize=6.0, label=m)
                      for m in EXPECTED_MODELS]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.57, 0.965), ncol=3, frameon=False,
               fontsize=7.0, handletextpad=0.42, columnspacing=1.35)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "fig2.pdf"
    png_path = outdir / "fig2.png"
    # fixed canvas (no bbox_inches='tight'): PDF page is exactly 7.09 x 2.8 in
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print("wrote {} and {}".format(pdf_path, png_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--values", default="drafts/figures/real_data_pipeline/data/fig2_values.json",
                    help="canonical values JSON (CWD- or repo-root-relative)")
    ap.add_argument("--outdir", default="drafts/figures/real_data_pipeline/outputs",
                    help="output directory for fig2.pdf / fig2.png")
    args = ap.parse_args()

    vpath = resolve_path(args.values)
    if not vpath.is_file():
        sys.exit("values file not found: {} (also tried {})".format(args.values, REPO_ROOT / args.values))
    with open(vpath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    validate(data, vpath)

    out = Path(args.outdir)
    if not out.is_absolute() and not out.exists() and (REPO_ROOT / args.outdir).exists():
        out = REPO_ROOT / args.outdir
    render(data, out)


if __name__ == "__main__":
    main()
