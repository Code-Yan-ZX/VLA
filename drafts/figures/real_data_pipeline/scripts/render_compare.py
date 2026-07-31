#!/usr/bin/env python3
"""render_compare.py — same-image, same-grid, same-budget pre/post L2 comparison.

For every captured sample this draws EXACTLY four panels on ONE processor-resized
image with ONE unit grid and ONE 25% budget:

  (a) original image + full unit grid outline (light);
  (b) RBM  = measured PRE-merger L2 top-k overlay (pre_keep), amber family;
  (c) POST = measured post-merger L2 top-k overlay (post_keep), SAME amber family
      but distinguished WITHOUT color via a diagonal HATCH on the kept units
      (shape/pattern redundancy -- color does NOT carry the pre/post distinction);
  (d) difference map with four categorical states (distinct color AND
      hatch/outline redundancy): kept-by-both / RBM-only / post-only / dropped-by-both.

Overlay alignment (documented): unit (r, c) covers processor pixels
[r*32:(r+1)*32, c*32:(c+1)*32] on the processor-resized image
(grid_thw[1]*16 x grid_thw[2]*16 px). We build every panel on a display canvas of
(rows*U, cols*U) so each unit is an exact U x U block; masks therefore never drift
vs the grid. Scores/masks are read from the measured NPZ only -- NEVER recomputed
from pixels; no CLIP/SigLIP/attention/query signal.

Diagnostics shown on each figure: top-k Jaccard and full-ranking Spearman rho
(computed from the NPZ pre_rank/post_rank with scipy.stats.spearmanr), N, k,
"25% kept", and "Qwen3-VL-8B measured merger L2". These are CROSS-CHECKED against
data/validation_report.json and the sample is refused if they disagree.

FastV is query-conditioned and requires a real user question. inputs/manifest.json
does not exist here, so FastV is SKIPPED (fastv_status = "skipped_missing_question")
and only the four mandatory panels are rendered. A question is never invented.

Writes data/comparison_summary.json (canonical, committable, no absolute paths).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PIPELINE_DIR / "data"
DEFAULT_INPUTS_DIR = PIPELINE_DIR / "inputs"
DEFAULT_OUTDIR = PIPELINE_DIR / "outputs"

PATCH = 16
MERGE = 2
UNIT_PX_PROC = PATCH * MERGE  # 32 px per unit on the processor-resized image

# ---- canonical palette (one constant, consistent across all samples) -------- #
INK = "#1F2937"
INK_MUTED = "#667085"
LINE = "#CBD2DC"
SURFACE = "#FFFFFF"
AMBER = "#F2BA02"          # pre-merger L2 (RBM) -- and the post overlay family
AMBER_DK = "#9A6700"       # pre outline (brownish)
POST_DK = "#7A3B00"        # post outline (darker brown) -- pattern carries id
DIFF = {                   # four categorical states of the difference map
    "both": "#2E8B7A",      # kept-by-both  (teal-green)
    "rbm":  "#F2BA02",      # RBM-only      (amber)
    "post": "#EE822F",      # post-only     (orange)
    "none": "#D9DEE7",      # dropped-by-both (light gray)
}
DIFF_OUTLINE = {
    "both": "#114D44", "rbm": "#7A5200", "post": "#7A3B00", "none": "#97A0AE",
}
DIFF_LABEL = {
    "both": "kept by both (pre & post)",
    "rbm":  "RBM-only (pre & ~post)",
    "post": "post-only (~pre & post)",
    "none": "dropped by both",
}
FASTV_STATUS = "skipped_missing_question"
TOL = 1e-5


# --------------------------------------------------------------------------- #
def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _load(npz: Path) -> dict:
    z = np.load(npz)
    uh, uw = int(z["unit_grid_hw"][0]), int(z["unit_grid_hw"][1])
    pre = z["pre_l2"]; post = z["post_l2"]
    if pre.size != uh * uw:
        raise ValueError(f"{npz}: length {pre.size} != {uh * uw}")
    return {
        "uh": uh, "uw": uw, "N": int(pre.size),
        "pre_keep": z["pre_keep"].astype(bool).reshape(uh, uw),
        "post_keep": z["post_keep"].astype(bool).reshape(uh, uw),
        "pre_rank": z["pre_rank"].astype(float),
        "post_rank": z["post_rank"].astype(float),
        "grid_thw": z["grid_thw"].astype(int).tolist(),
        "unit_grid_hw": [uh, uw],
        "k": int(z["pre_keep"].sum()),
    }


def _display_image(image_path: Path, rows: int, cols: int, u: int) -> np.ndarray:
    """Resize source to (cols*u, rows*u): aspect = rows:cols = processor aspect,
    so every unit lands on an exact u x u block (no drift)."""
    pil = Image.open(image_path).convert("RGB")
    rs = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(pil.resize((cols * u, rows * u), rs))


def _repeat2d(a2d: np.ndarray, u: int) -> np.ndarray:
    return np.repeat(np.repeat(a2d, u, axis=0), u, axis=1)


def _diag_pattern(h: int, w: int, stride: int, thick: int = 1) -> np.ndarray:
    """Diagonal stripe mask (r+c) mod stride < thick, for hatch redundancy."""
    r = np.arange(h)[:, None]
    c = np.arange(w)[None, :]
    return ((r + c) % stride) < thick


def _grid_lines(img: np.ndarray, rows: int, cols: int, u: int,
                color=(150, 158, 170), alpha: float = 0.45) -> np.ndarray:
    out = img.astype(float)
    line = np.array(color, dtype=float)
    h, w = out.shape[:2]
    for rr in range(rows + 1):
        y = min(rr * u, h - 1)
        out[y, :, :] = (1 - alpha) * out[y, :, :] + alpha * line
    for cc in range(cols + 1):
        x = min(cc * u, w - 1)
        out[:, x, :] = (1 - alpha) * out[:, x, :] + alpha * line
    return np.uint8(np.clip(out, 0, 255))


def _panel_grid(img: np.ndarray, rows: int, cols: int, u: int) -> np.ndarray:
    """(a) original + light full unit grid outline."""
    return _grid_lines(img, rows, cols, u, color=(176, 184, 196), alpha=0.5)


def _panel_overlay(img: np.ndarray, keep: np.ndarray, u: int,
                   hatch: bool, outline_rgb) -> np.ndarray:
    """(b)/(c) keep overlay on the photo. Amber family for both; the post panel
    adds a white diagonal hatch on kept units so pre vs post is distinguishable
    without color. `keep` is the measured NPZ mask (verbatim)."""
    rows, cols = keep.shape
    keep_big = _repeat2d(keep, u)
    out = img.astype(float).copy()
    out[~keep_big] = 0.55 * out[~keep_big] + 0.45 * 255.0          # pale dropped
    tint = _rgb(AMBER)
    out[keep_big] = 0.45 * out[keep_big] + 0.55 * tint             # amber kept
    if hatch:
        pat = _diag_pattern(out.shape[0], out.shape[1], max(3, u - 1), 1)
        pat = pat & keep_big
        out[pat] = 0.30 * out[pat] + 0.70 * 255.0                  # light hatch
    arr = np.uint8(np.clip(out, 0, 255))
    rendered = Image.fromarray(arr)
    draw = ImageDraw.Draw(rendered)
    oc = tuple(int(v) for v in outline_rgb)
    for r in range(rows):
        for c in range(cols):
            if keep[r, c]:
                x0, y0 = c * u, r * u
                draw.rectangle((x0, y0, x0 + u - 1, y0 + u - 1), outline=oc, width=1)
    return np.asarray(rendered)


def _panel_diff(rows: int, cols: int, u: int,
                pre_keep: np.ndarray, post_keep: np.ndarray) -> np.ndarray:
    """(d) categorical difference map on white, with per-state color + outline +
    hatch redundancy so states survive without color."""
    h, w = rows * u, cols * u
    out = np.full((h, w, 3), 255.0)
    both = pre_keep & post_keep
    rbm = pre_keep & ~post_keep
    post = ~pre_keep & post_keep
    none = ~pre_keep & ~post_keep
    state_map = [("both", both), ("rbm", rbm), ("post", post), ("none", none)]
    pat_post = _diag_pattern(h, w, max(3, u - 1), 1)
    pat_none = _diag_pattern(h, w, max(4, u), 1)
    for key, mask in state_map:
        m = _repeat2d(mask, u)
        col = _rgb(DIFF[key])
        out[m] = col
        # hatch redundancy on post-only and dropped-by-both
        if key == "post":
            hp = pat_post & m
            out[hp] = 0.35 * out[hp] + 0.65 * 255.0
        elif key == "none":
            hp = pat_none & m
            out[hp] = 0.55 * out[hp] + 0.45 * np.array([255, 255, 255])
    arr = np.uint8(np.clip(out, 0, 255))
    rendered = Image.fromarray(arr)
    draw = ImageDraw.Draw(rendered)
    for key, mask in state_map:
        oc = tuple(int(v) for v in _rgb(DIFF_OUTLINE[key]))
        for r in range(rows):
            for c in range(cols):
                if mask[r, c]:
                    x0, y0 = c * u, r * u
                    draw.rectangle((x0, y0, x0 + u - 1, y0 + u - 1),
                                   outline=oc, width=1)
    return np.asarray(rendered)


# --------------------------------------------------------------------------- #
def _jaccard(pre_keep, post_keep) -> float:
    inter = np.logical_and(pre_keep, post_keep).sum()
    union = np.logical_or(pre_keep, post_keep).sum()
    return float(inter / union) if union else float("nan")


def _spearman(pre_rank, post_rank) -> float:
    return float(spearmanr(pre_rank, post_rank)[0])


def _report_lookup(report: dict | None, sample_id: str) -> dict | None:
    if not report:
        return None
    for s in report.get("samples", []):
        if s.get("sample_id") == sample_id:
            return s
    return None


def _short_id(sample_id: str) -> str:
    if len(sample_id) > 20 and "-" in sample_id:
        return sample_id.split("-")[0]
    tail = sample_id.split("_")
    if len(tail) >= 2 and tail[-2].isdigit():
        return tail[-2]
    return sample_id[-6:]


# --------------------------------------------------------------------------- #
def render_one(sample_id: str, npz: Path, sidecar: dict, image_path: Path,
               report: dict | None, u: int, dpi: int, outdir: Path,
               fastv_manifest_exists: bool) -> dict:
    rep = _report_lookup(report, sample_id)
    if rep is None or not rep.get("sample_passed"):
        raise RuntimeError(f"sample {sample_id} did NOT pass validation; "
                           f"refusing to render compare figure.")
    m = _load(npz)
    j = _jaccard(m["pre_keep"], m["post_keep"])
    rho = _spearman(m["pre_rank"], m["post_rank"])
    # cross-check against the canonical validation report (must match)
    rd = rep.get("diagnostics", {})
    rj, rr = rd.get("topk_jaccard"), rd.get("spearman_pre_post_rank")
    if rj is None or rr is None or abs(rj - j) > TOL or abs(rr - rho) > TOL:
        raise RuntimeError(
            f"diagnostic cross-check FAILED for {sample_id}: "
            f"computed J={j:.6f}/rho={rho:.6f} vs report J={rj}/rho={rr}")
    if not all(g.get("pass") for g in rep.get("gates", {}).values()):
        raise RuntimeError(f"sample {sample_id} has a failing validation gate.")

    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    img = _display_image(image_path, m["uh"], m["uw"], u)

    pa = _panel_grid(img, m["uh"], m["uw"], u)
    pb = _panel_overlay(img, m["pre_keep"], u, hatch=False,
                        outline_rgb=_rgb(AMBER_DK))
    pc = _panel_overlay(img, m["post_keep"], u, hatch=True,
                        outline_rgb=_rgb(POST_DK))
    pd_ = _panel_diff(m["uh"], m["uw"], u, m["pre_keep"], m["post_keep"])

    fastv_status = (FASTV_STATUS if not fastv_manifest_exists
                    else "present_but_not_rendered_here")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.09, 6.55), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, left=0.03, right=0.97, top=0.915, bottom=0.245,
                          wspace=0.06, hspace=0.34)
    # Short titles only -- full descriptors live in the footnote below so that
    # narrow (portrait) right-column axes never push a title off the page edge.
    panels = [
        (pa, "a  original"),
        (pb, "b  RBM (pre L2)"),
        (pc, "c  post L2"),
        (pd_, "d  pre vs post"),
    ]
    for ax_idx, (view, title) in zip(range(4), panels):
        ax = fig.add_subplot(gs[ax_idx // 2, ax_idx % 2])
        ax.imshow(view)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE); sp.set_linewidth(0.8)
        ax.set_title(title, fontsize=7.0, color=INK, loc="left", pad=2)

    fig.suptitle(
        f"Same-image iso-budget comparison  -  {_short_id(sample_id)}  "
        f"(Qwen3-VL-8B measured merger L2)",
        fontsize=8.4, weight="bold", color=INK, y=0.972)

    # diagnostics line (compact, on the figure)
    diag = (f"N={m['N']}   k={m['k']} (25% kept)   "
            f"top-k Jaccard={j:.4f}   Spearman "
            f"ρ(pre,post ranks)={rho:.4f}   "
            f"unit grid {m['uh']}×{m['uw']} = grid_thw/2   "
            f"Qwen3-VL-8B measured merger L2")
    fig.text(0.5, 0.192, diag, ha="center", va="center", fontsize=6.1, color=INK)
    fv_note = ("FastV: SKIPPED (no user question; inputs/manifest.json absent) "
               if not fastv_manifest_exists else
               "FastV: query-conditioned (manifest present, not rendered here)")
    fig.text(0.5, 0.166, fv_note, ha="center", va="center",
             fontsize=5.5, color=INK_MUTED)
    # panel key (full descriptors) -- two centered lines, never clipped
    fig.text(0.5, 0.134,
             "(a) full unit grid outline (light).  (b,c) SAME amber L2 selector "
             "& 25% budget; (c) hatched = post (distinguishable without color).",
             ha="center", va="center", fontsize=5.3, color=INK_MUTED)
    fig.text(0.5, 0.112,
             "(d) four states kept-by-both / RBM-only / post-only / dropped-by-both, "
             "encoded by color + hatch/outline (legend).",
             ha="center", va="center", fontsize=5.3, color=INK_MUTED)

    # difference-map legend (color + hatch/outline redundancy)
    handles = [
        Patch(facecolor=DIFF["both"], edgecolor=DIFF_OUTLINE["both"], lw=0.8,
              label=DIFF_LABEL["both"]),
        Patch(facecolor=DIFF["rbm"], edgecolor=DIFF_OUTLINE["rbm"], lw=0.8,
              label=DIFF_LABEL["rbm"]),
        Patch(facecolor=DIFF["post"], edgecolor=DIFF_OUTLINE["post"], lw=0.8,
              hatch="//", label=DIFF_LABEL["post"]),
        Patch(facecolor=DIFF["none"], edgecolor=DIFF_OUTLINE["none"], lw=0.8,
              hatch="..", label=DIFF_LABEL["none"]),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=4, frameon=False, fontsize=6.1, handlelength=1.4,
               handletextpad=0.4, columnspacing=1.3)

    outdir.mkdir(parents=True, exist_ok=True)
    out_pdf = outdir / f"compare_{sample_id}.pdf"
    out_png = outdir / f"compare_{sample_id}.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=dpi)
    plt.close(fig)

    return {
        "id": sample_id,
        "filename": sidecar.get("original_filename", image_path.name),
        "sha256": sidecar.get("sha256_original", ""),
        "N": m["N"], "k": m["k"],
        "grid": {"grid_thw": m["grid_thw"], "unit_grid_hw": m["unit_grid_hw"]},
        "jaccard": round(j, 6),
        "spearman": round(rho, 6),
        "fastv_status": fastv_status,
        # os.path.relpath (not Path.relative_to): also works when --outdir /
        # --data-dir are passed relative to the caller's CWD or point outside
        # the pipeline dir (e.g. smoke-test temp dirs); identical string for
        # in-pipeline output dirs.
        "rendered_paths": [os.path.relpath(out_pdf, PIPELINE_DIR),
                           os.path.relpath(out_png, PIPELINE_DIR)],
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--validation-report", type=Path, default=None)
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--unit-px", type=int, default=4,
                    help="display pixels per unit (default 4)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    inputs_dir = Path(args.inputs_dir)
    outdir = Path(args.outdir)
    report_path = Path(args.validation_report) if args.validation_report \
        else (data_dir / "validation_report.json")
    if not report_path.is_file():
        raise SystemExit(f"validation report required (got {report_path}); "
                         f"refusing to render without it.")
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    if not report.get("all_passed"):
        raise SystemExit("validation_report.all_passed is false; aborting.")

    fastv_manifest_exists = (inputs_dir / "manifest.json").is_file()

    npzs = sorted(data_dir.glob("*.npz"))
    if args.sample_id:
        npzs = [p for p in npzs if p.stem == args.sample_id]
        if not npzs:
            raise SystemExit(f"no npz for sample-id {args.sample_id}")

    summary = []
    failures = []
    # render the requested set; diagnostics for ALL samples go into the summary
    # so comparison_summary.json is canonical/complete on a full run.
    render_ids = {p.stem for p in npzs}
    for npz in sorted(data_dir.glob("*.npz")):
        sid = npz.stem
        sidecar_path = data_dir / f"{sid}.json"
        sidecar = {}
        if sidecar_path.is_file():
            with open(sidecar_path, "r", encoding="utf-8") as fh:
                sidecar = json.load(fh)
        if sid not in render_ids:
            # not rendered this run: include diagnostics, no rendered_paths
            try:
                m = _load(npz)
                rep = _report_lookup(report, sid)
                j = _jaccard(m["pre_keep"], m["post_keep"])
                rho = _spearman(m["pre_rank"], m["post_rank"])
                summary.append({
                    "id": sid,
                    "filename": sidecar.get("original_filename", npz.name),
                    "sha256": sidecar.get("sha256_original", ""),
                    "N": m["N"], "k": m["k"],
                    "grid": {"grid_thw": m["grid_thw"],
                             "unit_grid_hw": m["unit_grid_hw"]},
                    "jaccard": round(j, 6), "spearman": round(rho, 6),
                    "fastv_status": (FASTV_STATUS if not fastv_manifest_exists
                                     else "present_but_not_rendered_here"),
                    "rendered_paths": [],
                })
            except Exception as exc:
                failures.append({"id": sid, "error": f"diag: {exc}"})
            continue
        image_path = inputs_dir / sidecar.get("original_filename", "")
        try:
            rec = render_one(sid, npz, sidecar, image_path, report,
                             args.unit_px, args.dpi, outdir,
                             fastv_manifest_exists)
            summary.append(rec)
            print(f"[compare] {sid}: OK  J={rec['jaccard']:.4f} "
                  f"rho={rec['spearman']:.4f}  cross-check=PASS")
        except Exception as exc:
            failures.append({"id": sid, "error": str(exc)})
            print(f"[compare] {sid}: FAILED -> {exc}")

    # keep summary order aligned with sorted sample ids
    summary.sort(key=lambda r: r["id"])
    sum_path = data_dir / "comparison_summary.json"
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump({
            "pipeline": "real_data_compare_render",
            "model": "Qwen3-VL-8B measured merger L2",
            "keep_ratio": 0.25,
            "budget_note": "same image, same unit grid, same 25% budget per panel",
            "fastv_status_global": (FASTV_STATUS if not fastv_manifest_exists
                                    else "manifest_present"),
            "n_rendered": sum(1 for r in summary if r["rendered_paths"]),
            "n_failed": len(failures),
            "cross_check_tolerance": TOL,
            "samples": summary,
            "failures": failures,
        }, fh, indent=2, ensure_ascii=False)
    print(f"[compare] wrote {os.path.relpath(sum_path, PIPELINE_DIR)} "
          f"({sum(1 for r in summary if r['rendered_paths'])} rendered, "
          f"{len(failures)} failed)")
    if failures:
        for f in failures:
            print(f"          FAIL {f['id']}: {f['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
