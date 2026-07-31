#!/usr/bin/env python3
"""Render FIG:2 — three-family pre-minus-post deltas (25% retention, full split).

Canonical data: drafts/figures/real_data_pipeline/data/fig2_values.json
Spec: drafts/figs_spec_for_user.md (FIG:2 L16-26; palette/canvas L64-80; FIG:2 prompt L98-106)

Layout: 1x4 small-multiple grouped-bar panels (TextVQA -> DocVQA -> OCRBench -> GQA),
independent y axes per panel (OCRBench pts NEVER shares an axis with pp), dark-gray
y=0 reference line in every panel, single shared legend above, signed value labels
with units, ACM double-column width 7.09 in. matplotlib only; deterministic.
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

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[3]

EXPECTED_MODELS = ["Qwen3-VL-8B", "Qwen2.5-VL-7B", "InternVL3-8B"]
EXPECTED_PANELS = ["TextVQA", "DocVQA", "OCRBench", "GQA"]
EXPECTED_UNITS = {"TextVQA": "pp", "DocVQA": "pp", "OCRBench": "pts", "GQA": "pp"}

COLORS = {"Qwen3-VL-8B": "#4C72B0", "Qwen2.5-VL-7B": "#DD8452", "InternVL3-8B": "#55A868"}
INK, AXIS_GRAY, GRID_GRAY, ZERO_GRAY = "#222222", "#333333", "#D9D9D9", "#333333"
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

    fig, axes = plt.subplots(1, 4, figsize=(7.09, 2.8))
    fig.subplots_adjust(left=0.052, right=0.988, top=0.735, bottom=0.135, wspace=0.205)

    benches = data["benchmarks"]
    values = data["values"]
    x = [0.0, 1.0, 2.0]
    bar_w = 0.74
    letters = "(a)", "(b)", "(c)", "(d)"
    tick_sets = {
        "TextVQA": [0, 15, 30, 45],
        "DocVQA": [0, 10, 20, 30, 40],
        "OCRBench": [0, 120, 240, 360, 480],
        "GQA": [-3, -2, -1, 0],
    }

    for ax, bench, letter in zip(axes, EXPECTED_PANELS, letters):
        meta = benches[bench]
        unit = meta["unit"]
        lo, hi = meta["suggested_y_range"]
        span = hi - lo

        for i, model in enumerate(EXPECTED_MODELS):
            v = values[model][bench]
            ax.bar(x[i], v, width=bar_w, color=COLORS[model],
                   edgecolor=darken(COLORS[model]), linewidth=0.7, zorder=3)
            # signed value label: above positive bars, below negative bars
            if v >= 0:
                ax.text(x[i], v + 0.018 * span, signed(v, unit),
                        ha="center", va="bottom", fontsize=7, color=INK, zorder=5)
            else:
                ax.text(x[i], v - 0.018 * span, signed(v, unit),
                        ha="center", va="top", fontsize=7, color=INK, zorder=5)

        ax.set_ylim(lo, hi)
        ax.set_yticks(tick_sets[bench])
        ax.tick_params(axis="y", labelsize=7, length=2.5, colors=INK, pad=2)
        ax.xaxis.set_visible(False)
        ax.yaxis.grid(True, color=GRID_GRAY, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(0.0, color=ZERO_GRAY, linewidth=0.9, zorder=4)  # y=0 reference

        title_unit = "pts" if unit == "pts" else "pp"
        ax.set_title("{} (Δ {})".format(bench, title_unit),
                     fontsize=9, fontweight="bold", color=INK, pad=5)
        ax.text(0.0, 1.04, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)

    # single shared legend above the panels (inside the fixed canvas)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], ec=darken(COLORS[m]), lw=0.7)
               for m in EXPECTED_MODELS]
    fig.legend(handles, EXPECTED_MODELS, loc="center",
               bbox_to_anchor=(0.5, 0.895), ncol=3, frameon=False,
               fontsize=7.5, handlelength=1.3, handletextpad=0.5, columnspacing=1.6)

    fig.text(0.5, 0.035,
             "Δ = pre {} post; 25% retained; full split".format(MINUS),
             ha="center", fontsize=7, color=INK)

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
