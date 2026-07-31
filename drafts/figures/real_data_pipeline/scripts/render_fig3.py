#!/usr/bin/env python3
"""Render FIG:3 — retention x stage-gap curves (compression-activated effect, n=200).

Canonical data: drafts/figures/real_data_pipeline/data/fig3_values.json
Spec: drafts/figs_spec_for_user.md (FIG:3 L28-39; palette/canvas L64-80; FIG:3 prompt L108-116)

Layout: 2x2 small multiples; columns = TextVQA / DocVQA (headers above), rows =
Qwen3-VL-8B / Qwen2.5-VL-7B (labels outside left margin); x = visual-token retention
{75, 50, 25}% left->right (deeper compression); shared y = pre - post (pp) over
[-5, 42]; dark-gray y=0 reference line; signed value labels on the 25% endpoints;
negative DocVQA points plotted below zero (never clipped). No confidence bands
(none available per spec). ACM double-column width 7.09 in. Deterministic.
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

EXPECTED_MODELS = ["Qwen3-VL-8B", "Qwen2.5-VL-7B"]
EXPECTED_BENCHES = ["TextVQA", "DocVQA"]
EXPECTED_KEEP = [75, 50, 25]
EXPECTED_N = 200
Y_RANGE = (-5.0, 42.0)
Y_TICKS = [0, 10, 20, 30, 40]

ROW_COLOR = {"Qwen3-VL-8B": "#4C72B0", "Qwen2.5-VL-7B": "#DD8452"}
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


def signed1(v: float) -> str:
    return "{:+.1f}".format(v).replace("-", MINUS)


def validate(data: dict, path: Path) -> None:
    """Fail loudly on any schema/unit/key mismatch before plotting."""
    errs = []

    def check(cond, msg):
        if not cond:
            errs.append(msg)

    check(isinstance(data, dict), "top-level JSON is not an object")
    check(data.get("figure") == "FIG:3",
          "expected figure == 'FIG:3', got {!r}".format(data.get("figure")))
    check(data.get("n_per_benchmark") == EXPECTED_N,
          "expected n_per_benchmark == {}, got {!r}".format(
              EXPECTED_N, data.get("n_per_benchmark")))
    check(data.get("keep_ratios_pct") == EXPECTED_KEEP,
          "keep_ratios_pct must be exactly {}, got {!r}".format(
              EXPECTED_KEEP, data.get("keep_ratios_pct")))
    check(data.get("models") == EXPECTED_MODELS,
          "models must be exactly {} in this order, got {!r}".format(
              EXPECTED_MODELS, data.get("models")))
    check(data.get("benchmarks") == EXPECTED_BENCHES,
          "benchmarks must be exactly {} in this order, got {!r}".format(
              EXPECTED_BENCHES, data.get("benchmarks")))

    units = data.get("units")
    check(isinstance(units, dict) and units.get("delta_pp") == "percentage points (pp)",
          "units.delta_pp must be 'percentage points (pp)', got {!r}".format(
              units.get("delta_pp") if isinstance(units, dict) else units))

    values = data.get("values")
    check(isinstance(values, dict) and set(values) == set(EXPECTED_MODELS),
          "values keys must be exactly {}, got {!r}".format(
              EXPECTED_MODELS, list(values) if isinstance(values, dict) else values))
    if isinstance(values, dict):
        for m in EXPECTED_MODELS:
            mrow = values.get(m)
            check(isinstance(mrow, dict) and set(mrow) == set(EXPECTED_BENCHES),
                  "values[{}]: benchmarks must be exactly {}, got {!r}".format(
                      m, EXPECTED_BENCHES, list(mrow) if isinstance(mrow, dict) else mrow))
            if not isinstance(mrow, dict):
                continue
            for b in EXPECTED_BENCHES:
                cell = mrow.get(b)
                check(isinstance(cell, dict),
                      "values[{}][{}]: expected an object with pre/post/delta_pp".format(m, b))
                if not isinstance(cell, dict):
                    continue
                for arm in ("pre", "post", "delta_pp"):
                    arr = cell.get(arm)
                    ok = (isinstance(arr, list) and len(arr) == len(EXPECTED_KEEP)
                          and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                  and math.isfinite(x) for x in arr))
                    check(ok, "values[{}][{}].{}: expected {} finite numbers, got {!r}".format(
                        m, b, arm, len(EXPECTED_KEEP), arr))
                frac = cell.get("pre", []) + cell.get("post", []) if isinstance(cell.get("pre"), list) \
                    and isinstance(cell.get("post"), list) else []
                if frac and all(isinstance(x, (int, float)) for x in frac):
                    check(all(0.0 <= x <= 1.0 for x in frac),
                          "values[{}][{}]: pre/post must be accuracy fractions in [0,1]".format(m, b))
                deltas = cell.get("delta_pp")
                if isinstance(deltas, list) and all(isinstance(x, (int, float)) for x in deltas):
                    check(all(Y_RANGE[0] <= x <= Y_RANGE[1] for x in deltas),
                          "values[{}][{}].delta_pp outside plottable y range {}".format(m, b, Y_RANGE))

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

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.098, right=0.978, top=0.845, bottom=0.125,
                        wspace=0.085, hspace=0.16)

    values = data["values"]
    x = [0.0, 1.0, 2.0]
    letters = [["(a)", "(b)"], ["(c)", "(d)"]]

    for i, model in enumerate(EXPECTED_MODELS):
        color = ROW_COLOR[model]
        for j, bench in enumerate(EXPECTED_BENCHES):
            ax = axes[i][j]
            y = values[model][bench]["delta_pp"]
            ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=4.6,
                    markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=0.7, zorder=3, clip_on=False)
            # signed label on the 25% endpoint (rightmost) only
            ax.annotate(signed1(y[-1]), xy=(x[-1], y[-1]),
                        xytext=(6.0, 0.0), textcoords="offset points",
                        ha="left", va="center", fontsize=7, fontweight="bold",
                        color=color, zorder=5, clip_on=False)

            ax.set_ylim(*Y_RANGE)
            ax.set_xlim(-0.22, 2.62)
            ax.set_yticks(Y_TICKS)
            ax.set_xticks(x)
            ax.tick_params(axis="both", labelsize=7, length=2.5, colors=INK, pad=2)
            ax.yaxis.grid(True, color=GRID_GRAY, linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.axhline(0.0, color=ZERO_GRAY, linewidth=0.9, zorder=4)  # y=0 reference
            ax.text(0.014, 0.962, letters[i][j], transform=ax.transAxes,
                    fontsize=8, fontweight="bold", va="top", color=INK)
            if i == 0:  # column headers above the top row only
                ax.set_title(bench, fontsize=9.5, fontweight="bold", color=INK, pad=7)
            if j == 1:  # tick labels on left column only
                ax.tick_params(axis="y", labelleft=False)
            if i == 0:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xticklabels(["{}%".format(k) for k in data["keep_ratios_pct"]])

    # single shared legend (model color), top center
    handles = [plt.Line2D([0], [0], color=ROW_COLOR[m], linewidth=1.8, marker="o",
                          markersize=4.6, markerfacecolor=ROW_COLOR[m],
                          markeredgecolor=ROW_COLOR[m]) for m in EXPECTED_MODELS]
    fig.legend(handles, EXPECTED_MODELS, loc="center",
               bbox_to_anchor=(0.5, 0.962), ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.6, handletextpad=0.5, columnspacing=1.8)

    # row labels outside the left margin
    row_y = [0.685, 0.305]
    for model, yc in zip(EXPECTED_MODELS, row_y):
        fig.text(0.045, yc, model, rotation=90, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=ROW_COLOR[model])

    # shared y-axis label
    fig.text(0.0145, 0.495, "pre {} post (pp)".format(MINUS), rotation=90,
             ha="center", va="center", fontsize=8, color=INK)

    fig.text(0.535, 0.060, "Visual-token retention", ha="center", fontsize=8, color=INK)
    fig.text(0.535, 0.031, "deeper compression  →", ha="center", fontsize=7, color=AXIS_GRAY)
    fig.text(0.535, 0.006, "n = {} per benchmark".format(data["n_per_benchmark"]),
             ha="center", fontsize=7, color=AXIS_GRAY)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "fig3.pdf"
    png_path = outdir / "fig3.png"
    # fixed canvas (no bbox_inches='tight'): PDF page is exactly 7.09 x 4.6 in
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print("wrote {} and {}".format(pdf_path, png_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--values", default="drafts/figures/real_data_pipeline/data/fig3_values.json",
                    help="canonical values JSON (CWD- or repo-root-relative)")
    ap.add_argument("--outdir", default="drafts/figures/real_data_pipeline/outputs",
                    help="output directory for fig3.pdf / fig3.png")
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
