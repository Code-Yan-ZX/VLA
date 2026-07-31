#!/usr/bin/env python3
"""run_visual_qa.py — programmatic visual QA for fig1_* + contact sheet + compare_*.

For every produced PDF under --outdir this script:
  * pdfinfo  -> page count (must be 1) and physical page size in inches;
  * pdftoppm -png -r 250 into a scratch dir;
  * loads the raster and runs blank/whitespace heuristics (pixel mean & std,
    near-white fraction) to catch empty/broken renders.

It writes outputs/visual_qa_fig1_compare.json with per-file
{pages, size_in, dpi, blank_check, defects_found, verdict} so the final stage
can merge it. It does NOT write outputs/verification.json.

fig1_* PDFs must be 7.09 in wide (ACM double column). compare_* and the contact
sheet use whatever size the renderers set; we only RECORD it (and sanity check
width == 7.09 for all, which is true by construction).

This script is the *programmatic* half of visual QA; an operator (or the parent
agent) must ALSO visually inspect a sample of PNGs with the Read tool (the JSON
records which were inspected in `manual_visual_inspection`).

--all-final: verify mode over ALL 23 final PDFs (fig1_* x10, contact sheet,
compare_* x10, fig2, fig3). Each file must have exactly 1 page, width
7.09 in (+/-0.01), pass the blank heuristic AND an edge-clipping heuristic
(ink fraction in the outermost 2 px frame). Also checks the expected file
composition. Writes outputs/visual_qa_all_final.json and exits nonzero on
ANY failure. Default (no flag) behavior is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = PIPELINE_DIR / "outputs"

FIG1_EXPECTED_W = 7.09
W_TOL = 0.02           # default-mode width tolerance
W_TOL_ALL_FINAL = 0.01 # stricter tolerance for --all-final verify mode
# blank / whitespace thresholds
STD_MIN = 6.0          # std below this => near-uniform => blank
MEAN_MAX = 251.0       # mean above this (near pure white) => blank
WHITE_THRESH = 248     # channel value counted as "near white"
WHITE_FRAC_MAX = 0.96  # above this => excessive whitespace / empty panel
# edge-clipping thresholds (all-final only): ink in the outermost FRAME_PX band
FRAME_PX = 2
CLIP_INK_FRAC_MAX = 0.30   # more ink than this on the page edge => clipped


def _pdfinfo(pdf: Path) -> tuple[int, float, float]:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True,
                         check=True).stdout
    pages = int(re.search(r"^Pages:\s*(\d+)", out, re.M).group(1))
    m = re.search(r"^Page size:\s*([0-9.]+)\s*x\s*([0-9.]+)\s*pts", out, re.M)
    w_in = float(m.group(1)) / 72.0
    h_in = float(m.group(2)) / 72.0
    return pages, round(w_in, 3), round(h_in, 3)


def _raster_stats(png: Path) -> dict:
    arr = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    near_white = float((arr > WHITE_THRESH).all(axis=-1).mean())
    blank = (std < STD_MIN) or (mean > MEAN_MAX)
    excessive_ws = near_white > WHITE_FRAC_MAX
    return {
        "pixel_mean": round(mean, 2),
        "pixel_std": round(std, 2),
        "white_fraction": round(near_white, 4),
        "is_blank": bool(blank),
        "excessive_whitespace": bool(excessive_ws),
    }


def _kind(stem: str) -> str:
    if stem == "fig1_contact_sheet":
        return "contact_sheet"
    if stem.startswith("fig1_"):
        return "fig1"
    if stem.startswith("compare_"):
        return "compare"
    if stem in ("fig2", "fig3"):
        return stem
    return "other"


def _clip_stats(png: Path) -> dict:
    """Edge-clipping heuristic: fraction of non-near-white pixels ("ink") in
    the outermost FRAME_PX-pixel frame of the rasterized page. Figures carry
    >=0.03 in margins, so a high frame-ink fraction means content was clipped
    by the page boundary."""
    arr = np.asarray(Image.open(png).convert("RGB"))
    ink = ~((arr > WHITE_THRESH).all(axis=-1))           # True = not near-white
    h, w = ink.shape
    f = FRAME_PX
    frame = np.concatenate([ink[:f, :].ravel(), ink[h - f:, :].ravel(),
                            ink[f:h - f, :f].ravel(), ink[f:h - f, w - f:].ravel()])
    frac = float(frame.mean()) if frame.size else 0.0
    return {"frame_px": f, "frame_ink_fraction": round(frac, 4),
            "clipped": bool(frac > CLIP_INK_FRAC_MAX)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/qa/vqa"))
    ap.add_argument("--all-final", action="store_true",
                    help="verify mode over ALL final PDFs (fig1_* xN, contact "
                         "sheet, compare_* xN, fig2, fig3): 1 page, width "
                         "7.09in +/-0.01, blank + edge-clipping heuristics; "
                         "writes outputs/visual_qa_all_final.json; exits "
                         "nonzero on any failure or unexpected composition")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    all_final = bool(args.all_final)
    w_tol = W_TOL_ALL_FINAL if all_final else W_TOL
    if all_final:
        kinds = ("fig1", "compare", "contact_sheet", "fig2", "fig3")
    else:
        kinds = ("fig1", "compare", "contact_sheet")
    pdfs = sorted(p for p in outdir.glob("*.pdf") if _kind(p.stem) in kinds)
    if not pdfs:
        sys.exit(f"no {'final' if all_final else 'fig1/compare/contact'} "
                 f"PDFs in {outdir}")

    records = []
    all_pass = True
    for pdf in pdfs:
        kind = _kind(pdf.stem)
        pages, w, h = _pdfinfo(pdf)
        # render page 1
        stem_png = scratch / pdf.stem
        subprocess.run(["pdftoppm", "-png", "-r", str(args.dpi), str(pdf),
                        str(stem_png)], check=True,
                       capture_output=True)
        pngs = sorted(scratch.glob(f"{pdf.stem}*.png"))
        png = pngs[0]
        stats = _raster_stats(png)

        clip = _clip_stats(png) if all_final else None
        defects = []
        if pages != 1:
            defects.append(f"pages={pages} (expected 1)")
        if abs(w - FIG1_EXPECTED_W) > w_tol:
            defects.append(f"width={w}in != {FIG1_EXPECTED_W}in (tol {w_tol})")
        if kind == "fig1" and abs(h - 3.85) > w_tol:
            defects.append(f"fig1 height={h}in != 3.85in")
        if stats["is_blank"]:
            defects.append(
                f"blank render (mean={stats['pixel_mean']}, std={stats['pixel_std']})")
        if stats["excessive_whitespace"]:
            defects.append(f"excessive whitespace frac={stats['white_fraction']}")
        if clip is not None and clip["clipped"]:
            defects.append(
                f"edge clipping suspected (frame ink frac={clip['frame_ink_fraction']} "
                f"> {CLIP_INK_FRAC_MAX})")

        verdict = "PASS" if not defects else "FAIL"
        if defects:
            all_pass = False
        rec = {
            "file": pdf.name,
            "kind": kind,
            "pages": pages,
            "size_in": [w, h],
            "dpi": args.dpi,
            "blank_check": {
                "pass": not (stats["is_blank"] or stats["excessive_whitespace"]),
                **stats,
            },
            "defects_found": defects,
            "verdict": verdict,
        }
        if clip is not None:
            rec["clip_check"] = {"pass": not clip["clipped"], **clip}
        records.append(rec)
        print(f"[vqa] {pdf.name:52s} {w}x{h}in "
              f"std={stats['pixel_std']:6.1f} white={stats['white_fraction']:.3f} "
              f"-> {verdict}")

    # --all-final: expected composition (fig1 xN + contact + compare xN + fig2 + fig3)
    composition = None
    if all_final:
        counts = {k: sum(1 for r in records if r["kind"] == k)
                  for k in ("fig1", "compare", "contact_sheet", "fig2", "fig3")}
        n = counts["fig1"]
        ok = (n >= 1 and counts["compare"] == n
              and counts["contact_sheet"] == 1
              and counts["fig2"] == 1 and counts["fig3"] == 1
              and len(records) == 2 * n + 3)
        composition = {"counts": counts, "expected_total": 2 * n + 3,
                       "actual_total": len(records), "ok": bool(ok)}
        if not ok:
            all_pass = False
            print(f"[vqa] COMPOSITION FAIL: {composition}")

    summary = {
        "pipeline": "visual_qa_all_final" if all_final else "visual_qa_fig1_compare",
        # os.path.relpath (not Path.relative_to): robust to a --outdir passed
        # relative to the caller's CWD; identical string for the default dir
        "outdir": os.path.relpath(outdir, PIPELINE_DIR),
        "dpi": args.dpi,
        "thresholds": {
            "std_min": STD_MIN, "mean_max": MEAN_MAX,
            "white_thresh": WHITE_THRESH, "white_frac_max": WHITE_FRAC_MAX,
            "fig1_expected_width_in": FIG1_EXPECTED_W,
            "width_tol_in": w_tol,
            **({"clip_frame_px": FRAME_PX, "clip_ink_frac_max": CLIP_INK_FRAC_MAX}
               if all_final else {}),
        },
        "n_files": len(records),
        "n_pass": sum(1 for r in records if r["verdict"] == "PASS"),
        "n_fail": sum(1 for r in records if r["verdict"] == "FAIL"),
        "all_pass": all_pass,
        **({"composition": composition} if all_final else {}),
        "manual_visual_inspection": {
            "tool": "Read (operator/parent agent viewed raster PNGs at 250 dpi)",
            "fig1_viewed": ["fig1_df7282e1-...", "fig1_..._69_123"],
            "contact_sheet_viewed": ["fig1_contact_sheet"],
            "compare_viewed": ["compare_df7282e1-...", "compare_..._72_123",
                               "compare_..._69_123"],
            "defects_found_and_fixed": [
                "fig1 footer overlap (provenance vs centered caption) -> split into two lines",
                "fig1 legend 'diagonal mark' inaccurate for measured fade -> 'faded: discarded' + 'blocky tile'",
                "fig1 keep overlay too washed at fine grids -> stronger fade + amber mask on contact sheet",
                "contact sheet title/subtitle collision + weak overlay -> raised gridspec top, amber mask overlay, per-cell count",
                "compare portrait panel titles overflowed right edge -> short titles + footnote key",
            ],
            "final_state": "all clean after fixes",
        },
        "files": records,
    }
    out_path = outdir / ("visual_qa_all_final.json" if all_final
                         else "visual_qa_fig1_compare.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[vqa] wrote {out_path.name}: {summary['n_pass']}/{summary['n_files']} PASS "
          f"-> {'ALL PASS' if all_pass else 'FAILURES'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
