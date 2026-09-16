#!/usr/bin/env python3
"""Plot audited Qwen3 full-split scores; never mixes diagnostic subsets.

Source: results/acmmm_final_controls/analysis.json, P0_1_table1_audit and
P0_1_pure_stage_control. Protocol: reports/acmmm_final_controls.md:24-45.
The early-tap RBM vs pre-final gap is not a pure feature-depth estimate:
the recorded implementations also differ in deepstack execution.
"""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/acmmm_final_controls/analysis.json"
OUT = ROOT / "drafts/dcc2027_submission_20260916/figs/fig3_fullsplit_controls"
AUDIT = json.loads(SOURCE.read_text(encoding="utf-8"))
METHODS = ["Early-tap RBM", "Final-input pre-final", "Post-L2"]
COLORS = ["#315B7D", "#67968B", "#C45A3C"]
HATCHES = ["", "///", "..."]
BENCHMARKS = [
    ("textvqa", "TextVQA", "VQA accuracy", 5000),
    ("docvqa", "DocVQA", "ANLS", 5349),
    ("ocrbench", "OCRBench", "points / 1000", 1000),
    ("gqa", "GQA", "exact match", 12578),
]


def score(record, key):
    # OCRBench total1000 is divided by the nominal full split, never answered.
    return record["total1000"] / 1000 if key == "ocrbench" else record["official"]


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.labelcolor": "#27313A", "text.color": "#27313A",
    "axes.edgecolor": "#8E969E", "xtick.color": "#59636C",
    "ytick.color": "#59636C", "axes.linewidth": .6,
})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55), sharey=True)
for ax, (key, title, metric, n) in zip(axes, BENCHMARKS):
    headline = AUDIT["P0_1_table1_audit"][key]
    control = AUDIT["P0_1_pure_stage_control"][key]["pre_final_vs_post"]
    assert control["iso"]["sample_ids_equal"]
    assert control["iso"]["token_counts_equal"]
    assert control["paired"]["n_paired"] == n
    assert score(headline["post"], key) == score(control["metric_B"], key)
    values = [score(headline["pre"], key), score(control["metric_A"], key),
              score(control["metric_B"], key)]
    assert all(0 <= v <= 1 for v in values)
    for j, (value, color, hatch) in enumerate(zip(values, COLORS, HATCHES)):
        ax.bar(j, value, width=.68, color=color, edgecolor="white",
               linewidth=.55, hatch=hatch, zorder=3)
        label = str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        ax.text(j, value + .035, label, ha="center", va="bottom",
                fontsize=7.2, color="#27313A")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=17)
    ax.text(.5, 1.045, f"{metric} | n = {n:,}", transform=ax.transAxes,
            ha="center", fontsize=6.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(-.65, 2.65)
    ax.set_xticks([0, 1, 2], ["Early", "Pre-final", "Post"])
    ax.tick_params(axis="x", length=0, labelsize=6.9, pad=5)
    ax.tick_params(axis="y", length=2.5, labelsize=7)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(axis="y", color="#DCE1E5", linewidth=.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    print(key, dict(zip(METHODS, values)))
axes[0].set_ylabel("Benchmark score (higher is better)", fontsize=7.5)
fig.legend(handles=[Patch(facecolor=c, edgecolor="white", hatch=h, label=m)
                    for m, c, h in zip(METHODS, COLORS, HATCHES)],
           loc="lower center", bbox_to_anchor=(.54, .008), ncol=3,
           frameon=False, fontsize=7.5, handlelength=1.8, columnspacing=1.8)
fig.subplots_adjust(left=.077, right=.995, bottom=.26, top=.76, wspace=.22)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04)
fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=.04)
plt.close(fig)
print(OUT.with_suffix(".pdf"))
