#!/usr/bin/env python3
"""Re-layout recorded token-survival maps without recomputing any mask.

The source PDF contains the audited vector overlays. This script clips only the
three measured image axes for each example, removes debug text/colorbars, and
adds a compact common annotation layer suitable for double-column rendering.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "drafts" / "overleaf_submission" / "figs" / "token_survival_combined.pdf"
OUTPUT = ROOT / "drafts" / "overleaf_submission" / "figs" / "fig3_token_survival.pdf"

PAGE_W, PAGE_H = 510.5, 292.0  # 7.09 x 4.06 inches
MARGIN, GAP = 8.0, 5.0
COL_W = (PAGE_W - 2 * MARGIN - 2 * GAP) / 3

# Coordinates are clips of the existing audited vector PDF (top-left origin).
# They contain only image axes and recorded mask overlays: no debug boxes,
# duplicate colorbars, titles, or aggregate statistics.
CLIPS = {
    "textvqa": [
        pymupdf.Rect(5, 160, 310, 307),
        pymupdf.Rect(381, 160, 689, 307),
        pymupdf.Rect(758, 160, 1068, 307),
    ],
    "docvqa": [
        pymupdf.Rect(168, 570, 325, 768),
        pymupdf.Rect(384, 570, 542, 768),
        pymupdf.Rect(600, 570, 753, 768),
    ],
}


def fit_rect(box: pymupdf.Rect, aspect: float) -> pymupdf.Rect:
    """Fit an aspect-ratio rectangle inside box, centered."""
    if box.width / box.height > aspect:
        h = box.height
        w = h * aspect
    else:
        w = box.width
        h = w / aspect
    x0 = box.x0 + (box.width - w) / 2
    y0 = box.y0 + (box.height - h) / 2
    return pymupdf.Rect(x0, y0, x0 + w, y0 + h)


def put_text(page, box, text, size=5.6, color=(0.10, 0.13, 0.17), align=0, font="helv"):
    rc = page.insert_textbox(
        box,
        text,
        fontsize=size,
        fontname=font,
        color=color,
        align=align,
        lineheight=1.05,
    )
    if rc < 0:
        raise RuntimeError(f"annotation overflow ({rc:.2f}): {text}")


def main() -> None:
    src = pymupdf.open(SOURCE)
    if len(src) != 1:
        raise RuntimeError(f"expected one source page, found {len(src)}")

    # Recolor the recorded vector outlines in-memory with the Okabe--Ito
    # blue/vermillion palette. Geometry and masks are untouched. The original
    # green/red strokes and orange disagreement fill are exact PDF operators in
    # the source figure; replacements affect color only.
    replacements = {
        b"0 1 0 RG": b"0 0.447 0.698 RG",
        b"1 0 0 RG": b"0.835 0.369 0 RG",
        b"1 0.646484 0 RG": b"0.8 0.475 0.655 RG",
        b"1 0.646484 0 rg": b"0.8 0.475 0.655 rg",
    }
    for xref in range(1, src.xref_length()):
        if not src.xref_is_stream(xref):
            continue
        data = src.xref_stream(xref)
        changed = data
        for old, new in replacements.items():
            changed = changed.replace(old, new)
        if changed != data:
            src.update_stream(xref, changed)
    out = pymupdf.open()
    page = out.new_page(width=PAGE_W, height=PAGE_H)

    headers = ["RBM (pre-merger L2)", "Post-L2", "Selection difference"]
    header_colors = [(0.10, 0.35, 0.65), (0.85, 0.37, 0.10), (0.30, 0.30, 0.34)]
    for i, (header, color) in enumerate(zip(headers, header_colors)):
        x0 = MARGIN + i * (COL_W + GAP)
        put_text(page, pymupdf.Rect(x0, 5, x0 + COL_W, 18), header, 7.2, color, 1, "hebo")

    rows = [
        {
            "key": "textvqa",
            "name": "TextVQA 35174",
            "meta": 'Q: "What is the name of this business?"  GT: Midas / auto service experts',
            "answers": 'RBM: "Auto Service Experts" (correct)  |  Post: "Krispy Kreme" (wrong)  |  Jaccard 0.257',
            "text_y": (19, 35),
            "answer_y": (34, 48),
            "panel_box": (48, 119),
        },
        {
            "key": "docvqa",
            "name": "DocVQA 58439",
            "meta": 'Q: "Amount spent on promotional meetings and events, 1998?"  GT: $1.3 billion',
            "answers": 'RBM: "$1.3 billion" (correct)  |  Post: "$1.3 million" (wrong)  |  Jaccard 0.079',
            "text_y": (137, 153),
            "answer_y": (152, 166),
            "panel_box": (166, 282),
        },
    ]

    for row in rows:
        put_text(
            page,
            pymupdf.Rect(MARGIN, row["text_y"][0], PAGE_W - MARGIN, row["text_y"][1]),
            f'{row["name"]}  —  {row["meta"]}',
            5.8,
            (0.08, 0.10, 0.13),
            0,
            "hebo",
        )
        put_text(
            page,
            pymupdf.Rect(MARGIN, row["answer_y"][0], PAGE_W - MARGIN, row["answer_y"][1]),
            row["answers"],
            5.5,
            (0.22, 0.25, 0.29),
        )
        y0, y1 = row["panel_box"]
        for i, clip in enumerate(CLIPS[row["key"]]):
            x0 = MARGIN + i * (COL_W + GAP)
            cell = pymupdf.Rect(x0, y0, x0 + COL_W, y1)
            target = fit_rect(cell, clip.width / clip.height)
            page.show_pdf_page(target, src, 0, clip=clip, keep_proportion=True)
            page.draw_rect(target, color=(0.65, 0.68, 0.72), width=0.35, overlay=True)

    # A compact key supplements the solid/dashed outlines and remains
    # distinguishable in grayscale.
    put_text(
        page,
        pymupdf.Rect(MARGIN, 282.5, PAGE_W - MARGIN, 291),
        "Difference panels: RBM-kept units use solid outlines; Post-L2-kept units use dashed outlines.",
        5.2,
        (0.28, 0.30, 0.34),
        1,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUTPUT, garbage=4, deflate=True)
    out.close()
    src.close()


if __name__ == "__main__":
    main()
