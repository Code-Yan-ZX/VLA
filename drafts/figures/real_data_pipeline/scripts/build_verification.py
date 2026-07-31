#!/usr/bin/env python3
"""build_verification.py — merge the final verification report.

Reads the artifacts produced by run_all.sh and writes
drafts/figures/real_data_pipeline/outputs/verification.json, merging:

  * per-PDF QA for ALL 23 final PDFs   (outputs/visual_qa_all_final.json,
    from run_visual_qa.py --all-final: {file, pages, width_in, height_in,
    dpi_checked, blank_check, clip_check, verdict})
  * capture summary                    (data/capture_manifest.json)
  * validation summary                 (data/validation_report.json:
    all_passed, tolerance 0.0, tie-break rule, gates 8/8)
  * render provenance                  (data/run_all_manifest.json: git HEAD,
    SOURCE_DATE_EPOCH, package versions, exact commands)
  * FastV status                       (data/comparison_summary.json:
    skipped_missing_question)

Machine-readable; relative paths only (no absolute private paths, no secrets).
Exits nonzero if any input is missing or any QA verdict is FAIL.

CLI:  python drafts/figures/real_data_pipeline/scripts/build_verification.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
DATA = PIPELINE_DIR / "data"
OUT = PIPELINE_DIR / "outputs"

GATE_IDS = ["1_finite_1d", "2_geometry_count", "3_k_counts", "4_topk_tiebreak",
            "5_rank_perm_consistent", "6_unit_identity", "7_repeatability",
            "8_hooks_no_hidden_states"]


def load(p: Path) -> dict:
    if not p.is_file():
        sys.exit(f"required input missing: {p} — run ./run_all.sh first")
    return json.loads(p.read_text())


def strip_private(obj):
    """Recursively drop keys that carry absolute/host-specific paths."""
    bad = re.compile(r"(temp_output_dir|recapture_log|log|path)$", re.I)
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items()
                if not (isinstance(v, str) and v.startswith(("/", "~")))
                and not bad.search(k)}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def main() -> int:
    vqa = load(OUT / "visual_qa_all_final.json")
    manifest = load(DATA / "capture_manifest.json")
    val = load(DATA / "validation_report.json")
    cmp = load(DATA / "comparison_summary.json")
    runman = load(DATA / "run_all_manifest.json")

    # ---- per-PDF QA (23 final PDFs) ---- #
    pdf_qa_files = []
    for r in vqa["files"]:
        pdf_qa_files.append({
            "file": r["file"],
            "kind": r["kind"],
            "pages": r["pages"],
            "width_in": r["size_in"][0],
            "height_in": r["size_in"][1],
            "dpi_checked": r["dpi"],
            "blank_check": r["blank_check"],
            "clip_check": r.get("clip_check"),
            "verdict": r["verdict"],
        })
    pdf_qa = {
        "tool": "run_visual_qa.py --all-final",
        "thresholds": vqa["thresholds"],
        "composition": vqa.get("composition"),
        "n_files": vqa["n_files"],
        "n_pass": vqa["n_pass"],
        "all_pass": vqa["all_pass"],
        "files": pdf_qa_files,
    }

    # ---- capture summary ---- #
    capture = {
        "n_samples": manifest["n_samples"],
        "model_family": manifest["model_family"],
        "model_id": manifest["model_id"],
        "resolved_revision": manifest["resolved_revision"],
        "model_class": manifest["model_class"],
        "visual_class": manifest["visual_class"],
        "processor_class": manifest["processor_class"],
        "keep_ratio": manifest["keep_ratio"],
        "k_formula": manifest["k_formula"],
        "seed": manifest["seed"],
        "max_pixels": manifest["max_pixels"],
        "dtype": manifest["dtype"],
        "patch_size": manifest["patch_size"],
        "spatial_merge_size": manifest["spatial_merge_size"],
        "pre_hook_module": manifest["pre_hook_module"],
        "post_hook_definition": manifest["post_hook_definition"],
        "score_definitions": manifest["score_definitions"],
        "tie_break_rule": manifest["tie_break_rule"],
        "capture_prompt": manifest["capture_prompt"],
        "max_generated_tokens": 1,
        "samples": [dict(sample_id=s["sample_id"],
                         original_filename=s["original_filename"],
                         n_units=s["n_units"], k=s["k"],
                         grid_thw=s["grid_thw"],
                         unit_grid_hw=s["unit_grid_hw"],
                         pre_post_topk_jaccard=s["pre_post_topk_jaccard"])
                    for s in manifest["samples"]],
    }

    # ---- validation summary (gates 8/8) ---- #
    per_sample = []
    for s in val["samples"]:
        passed = sum(1 for g in s["gates"].values() if g["pass"])
        per_sample.append({"sample_id": s["sample_id"],
                           "gates_passed": f"{passed}/{len(s['gates'])}",
                           "sample_passed": s["sample_passed"],
                           "topk_jaccard": s["diagnostics"]["topk_jaccard"],
                           "spearman_pre_post_rank":
                               s["diagnostics"]["spearman_pre_post_rank"]})
    rep = val.get("repeatability", {})
    hook = val.get("hook_check", {})
    gates8 = (val.get("all_passed") and rep.get("pass") and hook.get("pass")
              and all(s["sample_passed"] for s in val["samples"]))
    validation = {
        "all_passed": val["all_passed"],
        "gates_passed": "8/8" if gates8 else "INCOMPLETE",
        "gate_ids": GATE_IDS,
        "n_samples": val["n_samples"],
        "tolerance_recorded": val["tolerance_recorded"],
        "k_formula": val["k_formula"],
        "tie_break_rule": manifest["tie_break_rule"],
        "hook_check": {"pass": hook.get("pass"),
                       "detail": strip_private(hook.get("detail", ""))},
        "repeatability_gate7": {
            "pass": rep.get("pass"),
            "bit_identical": rep.get("bit_identical"),
            "observed_max_abs_diff": rep.get("observed_max_abs_diff"),
            "tolerance_recorded": rep.get("tolerance_recorded"),
            "detail": strip_private(rep.get("detail", "")),
        },
        "per_sample": per_sample,
    }

    # ---- render provenance ---- #
    render_provenance = {
        "git_head": runman.get("git_head"),
        "source_date_epoch": runman.get("source_date_epoch"),
        "conda_env": runman.get("conda_env"),
        "python": runman.get("python"),
        "package_versions": runman.get("package_versions"),
        "poppler": runman.get("poppler"),
        "stage_status": runman.get("stage_status"),
        "commands": strip_private(runman.get("commands", [])),
        "determinism_note": "renderers pin SOURCE_DATE_EPOCH for PDF metadata; "
                            "fixed-canvas savefig (no bbox_inches='tight'); "
                            "fig2/compare/fig1 PNG md5 verified stable across "
                            "full reruns on this host",
    }

    # ---- FastV + figure values ---- #
    fastv = {
        "status": cmp.get("fastv_status_global", "unknown"),
        "reason": "no real question accompanies the input images "
                  "(inputs/manifest.json absent) -> FastV comparison "
                  "deliberately not rendered",
    }
    fig_vals = {}
    for fig in ("fig2", "fig3"):
        vp = DATA / f"{fig}_values.json"
        if vp.is_file():
            d = json.loads(vp.read_text())
            fig_vals[fig] = {"values_file": f"data/{fig}_values.json",
                             "figure": d.get("figure"),
                             "provenance_audited":
                                 bool(d.get("provenance", {}).get("audited"))}

    verification = {
        "schema": "real_data_pipeline_verification/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_dir": "drafts/figures/real_data_pipeline",
        "pdf_qa": pdf_qa,
        "capture": capture,
        "validation": validation,
        "render_provenance": render_provenance,
        "fastv": fastv,
        "figure_values": fig_vals,
        "overall": {
            "pdf_qa_all_pass": pdf_qa["all_pass"],
            "pdf_qa_n_pass": f"{pdf_qa['n_pass']}/{pdf_qa['n_files']}",
            "validation_all_passed": validation["all_passed"],
            "capture_n_samples": capture["n_samples"],
        },
    }

    out_path = OUT / "verification.json"
    out_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False))
    ok = pdf_qa["all_pass"] and validation["all_passed"] and gates8
    print(f"[verify-json] wrote {out_path}: "
          f"pdf_qa {pdf_qa['n_pass']}/{pdf_qa['n_files']} PASS, "
          f"validation all_passed={validation['all_passed']} "
          f"({validation['gates_passed']}), "
          f"fastv={fastv['status']} -> {'OK' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
