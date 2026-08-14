"""Repair the legacy TextVQA annotation placement and combine both deep dives.

The original GPU captures are intentionally not tracked. This utility moves the
Matplotlib annotation group in the existing vector PDF without rasterizing it,
then stacks the repaired TextVQA and DocVQA pages on one large PDF page.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject


ANNOTATION_START = b"/A4 gs 0 J 0 g 0 w 0 G 0 g"
ANNOTATION_END = b"\n0 g 0 G 0 g\nq\n1 0 -0 1"


def move_statistics_annotation(source: Path, destination: Path, dy: float = 70.0) -> None:
    writer = PdfWriter(clone_from=source)
    if len(writer.pages) != 1:
        raise ValueError(f"expected one page in {source}, found {len(writer.pages)}")

    page = writer.pages[0]
    data = page.get_contents().get_data()
    start = data.find(ANNOTATION_START)
    end = data.find(ANNOTATION_END, start)
    if start < 0 or end < 0 or data.count(b"Jaccard") != 1:
        raise ValueError(f"could not uniquely locate the statistics annotation in {source}")

    translation = f"q 1 0 0 1 0 {dy:g} cm\n".encode("ascii")
    if not data[max(0, start - len(translation)):start] == translation:
        moved = (
            data[:start]
            + translation
            + data[start:end]
            + b"\nQ"
            + data[end:]
        )
        stream = DecodedStreamObject()
        stream.set_data(moved)
        page.replace_contents(stream)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)


def stack_pages(textvqa: Path, docvqa: Path, destination: Path, gap: float = 18.0) -> None:
    text_page = PdfReader(textvqa).pages[0]
    doc_page = PdfReader(docvqa).pages[0]
    text_w, text_h = float(text_page.mediabox.width), float(text_page.mediabox.height)
    doc_w, doc_h = float(doc_page.mediabox.width), float(doc_page.mediabox.height)

    canvas_w = max(text_w, doc_w)
    canvas_h = text_h + gap + doc_h
    writer = PdfWriter()
    canvas = writer.add_blank_page(width=canvas_w, height=canvas_h)

    text_x = (canvas_w - text_w) / 2.0
    doc_x = (canvas_w - doc_w) / 2.0
    canvas.merge_transformed_page(
        text_page, Transformation().translate(tx=text_x, ty=doc_h + gap)
    )
    canvas.merge_transformed_page(
        doc_page, Transformation().translate(tx=doc_x, ty=0)
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)


def compact_docvqa_overlap_title(source: Path, destination: Path) -> None:
    """Shorten the portrait panel title so it clears the adjacent colorbar."""
    writer = PdfWriter(clone_from=source)
    page = writer.pages[0]
    data = page.get_contents().get_data()
    title = data.find(b"Overlap")
    if title < 0:
        raise ValueError(f"could not locate the overlap title in {source}")

    pattern = re.compile(
        rb"q\n1 0 -0 1 ([0-9.]+) ([0-9.]+) cm\nBT\n/F1 11 Tf"
    )
    matches = [match for match in pattern.finditer(data, max(0, title - 400), title)]
    if len(matches) != 1:
        raise ValueError(f"could not uniquely locate the overlap title transform in {source}")
    match = matches[0]
    replacement = (
        b"q\n1 0 -0 1 449.7 " + match.group(2) + b" cm\nBT\n/F1 8.5 Tf"
    )
    stream = DecodedStreamObject()
    stream.set_data(data[:match.start()] + replacement + data[match.end():])
    page.replace_contents(stream)

    with destination.open("wb") as handle:
        writer.write(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--textvqa", type=Path, required=True)
    parser.add_argument("--docvqa", type=Path, required=True)
    parser.add_argument("--fixed-textvqa", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    move_statistics_annotation(args.textvqa, args.fixed_textvqa)
    with tempfile.TemporaryDirectory() as temp_dir:
        fixed_docvqa = Path(temp_dir) / "token_survival_docvqa_fixed.pdf"
        compact_docvqa = Path(temp_dir) / "token_survival_docvqa_compact.pdf"
        move_statistics_annotation(args.docvqa, fixed_docvqa)
        compact_docvqa_overlap_title(fixed_docvqa, compact_docvqa)
        stack_pages(args.fixed_textvqa, compact_docvqa, args.combined)


if __name__ == "__main__":
    main()
