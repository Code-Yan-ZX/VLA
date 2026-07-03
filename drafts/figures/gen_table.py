#!/usr/bin/env python3
"""Related-work throughput tally (evidence for Fig 1 + main novelty).

Two outputs:
  drafts/figures/table_rw_throughput.md   (authoritative markdown)
  drafts/figures/table_rw_throughput.png  (rendered image)

Columns: method | year/venue | reports FLOPs/token (Y, all 37) | reports any
wall-clock (13/37) | measured inside a serving engine (0/37).

Rows: all 13 wall-clock reporters (full detail) + one "...and 24 others" row
for the FLOPs-only majority. Data: notes/lit-survey.md §2 table.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import re
import matplotlib.pyplot as plt
import _data as D
import _style as S

S.apply_rc(fontsize=8)
rows, n_total, n_wall, n_deploy = D.throughput_tally()
wall_rows = [r for r in rows if r["wall_clock"]]
flopsonly_rows = [r for r in rows if not r["wall_clock"]]

# friendly method short names (strip the quoted subtitle)
def short_name(m):
    m = re.sub(r"\*\*", "", m)
    # take text before a double-quote subtitle
    m = re.split(r'["“]', m)[0].strip()
    return m

# year/venue tidy
def short_year(y):
    y = y.replace("2024-12", "2024").replace("2024-07", "2024").replace("2024-10", "2024")
    y = y.replace("2024-11", "2024").replace("2025-01", "2025").replace("2025-03", "2025")
    y = y.replace("2025-08", "2025").replace("2025-11", "2025").replace("2025-10", "2025")
    y = y.replace("2025-07", "2025").replace("2025-05", "2025").replace("2025-02", "2025")
    y = y.replace("2025-12", "2025").replace("2026-03", "2026").replace("2026-01", "2026")
    y = y.replace("2026 / arXiv", "2026").replace("2023-11", "2023")
    return y.strip()

# ---------- markdown ----------
md = []
md.append("# Related-work throughput-reporting tally (37 VLM token-compression methods)\n")
md.append("Evidence for Fig. 1 and the main novelty claim. Source: "
          "`notes/lit-survey.md` §2 (arXiv-verified 2026-07-01). All 37 methods "
          "report FLOPs/token-count; **13/37** report some offline wall-clock number; "
          "**0/37** measure served throughput inside a production serving engine "
          "(vLLM/SGLang/lmdeploy/TRT-LLM).\n")
md.append("| # | Method | Year / Venue | FLOPs / token-count | Any wall-clock | In a serving engine |")
md.append("|--:|---|---|:---:|:---:|:---:|")
i = 1
for r in wall_rows:
    md.append(f"| {i} | {short_name(r['method'])} | {short_year(r['year'])} "
              f"| Y | Y | **0** |")
    i += 1
md.append(f"| {i} | *…and {len(flopsonly_rows)} FLOPs-only others* "
          f"(e.g. FastV, PyramidDrop, VTC-CLS, TokenPacker, "
          f"LLaVA-PruMerge, G-Prune, GlimpsePrune, AgilePruner, VisionTrim, "
          f"METEOR, AdaReTaKe, PPE, RedundancyLens, …) | 2023-2026 | Y | — | **0** |")
md.append("")
md.append(f"**Totals:** 37 surveyed · **13** report any wall-clock (offline CUDA "
          f"latency, prefill, or decode speedup on the authors' own harness) · "
          f"**0** measure served throughput (req/s, tok/s, TTFT, KV-MB) inside a "
          f"production serving engine. The closest, SparseVILA, reports 4.0× "
          f"prefill / 2.6× end-to-end but on its own AWQ pipeline, not vLLM/"
          f"SGLang/lmdeploy/TRT-LLM. The only serving-engine artifact, vLLM RFC "
          f"#45098 (`--image-pruning-rate`), is unfinished infrastructure with no "
          f"benchmarks.")
md.append("")
md.append("*Two independent sources corroborate the gap is open: the Westlake "
          "survey (arXiv 2507.20198) §6.5.3-6.5.4 names the FlashAttention-score "
          "root cause that blocks in-LLM pruning from engine integration and "
          "calls TTFT/per-token latency \"missing\"; the Eval-Framework (arXiv "
          "2510.07143) explicitly demands this evaluation.*")

md_path = Path(__file__).resolve().parent / "table_rw_throughput.md"
md_path.write_text("\n".join(md))
print(f"wrote {md_path}")

# ---------- PNG (rendered table) ----------
fig, ax = plt.subplots(figsize=(S.DOUBLE_COL, 4.6))
ax.axis("off")

# header + rows
header = ["#", "Method", "Year / Venue", "FLOPs /\ntoken-count",
          "Any\nwall-clock", "In a serving\nengine"]
table_rows = []
for k, r in enumerate(wall_rows, 1):
    table_rows.append([str(k), short_name(r["method"]), short_year(r["year"]),
                       "Y", "Y", "0"])
table_rows.append([str(len(wall_rows) + 1),
                   f"…and {len(flopsonly_rows)} FLOPs-only others "
                   f"(FastV, PyramidDrop, VTC-CLS, G-Prune, GlimpsePrune, …)",
                   "2023-26", "Y", "—", "0"])

tbl = ax.table(cellText=table_rows, colLabels=header, loc="center",
               cellLoc="center", colLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.18)

# column widths
widths = [0.04, 0.30, 0.13, 0.13, 0.13, 0.13]
for col, w in enumerate(widths):
    for row in range(len(table_rows) + 1):
        tbl[(row, col)].set_width(w)

# style header
for col in range(len(header)):
    c = tbl[(0, col)]
    c.set_facecolor(S.INK_PRIM); c.set_edgecolor(S.SURFACE)
    c.get_text().set_color(S.SURFACE); c.get_text().set_weight("bold")
    c.get_text().set_fontsize(8)

# style body: zebra + right-most "0" column in red, left-align method
for row in range(1, len(table_rows) + 1):
    for col in range(len(header)):
        c = tbl[(row, col)]
        c.set_edgecolor(S.GRIDLINE); c.set_linewidth(0.5)
        zebra = S.SURFACE if row % 2 else "#f4f3ee"
        c.set_facecolor(zebra)
        txt = c.get_text()
        if col == 1:
            txt.set_ha("left"); c.PAD = 0.02
        txt.set_color(S.INK_PRIM)
        if col == 5:  # "In a serving engine" column = the 0/37 headline
            txt.set_color(S.CAT["red"]); txt.set_weight("bold")
            txt.set_fontsize(9)
        if col == 4:  # "Any wall-clock"
            txt.set_color(S.CAT["blue"]); txt.set_weight("bold")
    # last "...and N others" row in italic
    if row == len(table_rows):
        for col in range(len(header)):
            tbl[(row, col)].get_text().set_fontstyle("italic")
            tbl[(row, col)].get_text().set_color(S.INK_SEC)

fig.suptitle("37 VLM token-compression methods (2023-2026): throughput-reporting tally",
             x=0.5, ha="center", fontsize=10, fontweight="bold",
             color=S.INK_PRIM, y=0.965)
ax.text(0.5, 0.04,
        "13 / 37 report any wall-clock number  ·  "
        "0 / 37 measure served throughput inside a production serving engine",
        ha="center", va="center", fontsize=8.5, color=S.CAT["red"],
        transform=ax.transAxes, fontweight="bold")

fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08)
out = Path(__file__).resolve().parent / "table_rw_throughput.png"
fig.savefig(out, dpi=300)
print(f"wrote {out}")
