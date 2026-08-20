#!/usr/bin/env python3
"""Generate the reproducibility manifest for the acmmm_final_controls campaign:
  * results/acmmm_final_controls/cell_summary.csv   (one row per experiment cell)
  * results/acmmm_final_controls/environment.txt    (env / model / commit / scorer)
  * results/acmmm_final_controls/MANIFEST.sha256    (sha256 + size of every
      source file: analysis.json, raw per-sample JSONs, per-item JSONs, run
      scripts, analysis script, verification report)

CPU-only; reads the existing result JSONs and the official scorers.
Every number in cell_summary.csv is recomputed here from the per-sample JSONs
with the SAME rescore path as scripts/analyze_acmmm_final_controls.py (official
scorers + OCRBench question_type for the HME branch).

Usage: python scripts/gen_acmmm_manifest.py
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
import platform

REPO = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from analyze_acmmm_final_controls import per_sample_scores, official_metric, token_stats

ROOT = os.path.join(REPO, "results", "acmmm_final_controls")

ENGINE = {
    "P0-1": "vllm", "P0-2": "vllm", "P1": "hf",
}
RETENTION = {0.75: 0.25, 0.875: 0.125}

MODEL_NAME = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}
BENCH_METRIC = {
    "textvqa": "VQA-acc", "docvqa": "ANLS", "ocrbench": "OCRBench /1000",
    "gqa": "exact-match",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_rows():
    """Yield dicts for the cell_summary rows."""
    rows = []
    # ---- P0-1: Qwen3 pre-final / post x 4 benches (native, r=0.75) ----
    for b in ("textvqa", "docvqa", "ocrbench", "gqa"):
        for m in ("pre-final", "post"):
            p = os.path.join(ROOT, "p0_1", f"p0_1_qwen3_{m}_{b}_r0.750_full.json")
            rows.append(make_row("P0-1", "qwen3vl", b, m, 0.25, 0, p))
    # P0-1 none anchors (reused verified cells, not re-run)
    anchors = {
        "textvqa": os.path.join(REPO, "runs/full_matrix/j7_qwen3vl_none_textvqa_r0.000_full.json"),
        "docvqa": os.path.join(REPO, "runs/r2_same_scope/r2_qwen3vl_none_docvqa_full5349.json"),
        "ocrbench": os.path.join(REPO, "runs/full_matrix/j7_qwen3vl_none_ocrbench_r0.000_full.json"),
        "gqa": os.path.join(REPO, "runs/full_matrix/j7_qwen3vl_none_gqa_r0.000_full.json"),
    }
    for b, p in anchors.items():
        r = make_row("P0-1", "qwen3vl", b, "none", None, 0, p)
        r["note"] = "anchor (reused verified cell, not re-run)"
        rows.append(r)
    # ---- P0-2: Qwen2.5 OCRBench pre/post x k=0.25/0.125 (4M) ----
    for r in (0.75, 0.875):
        for m in ("pre", "post"):
            p = os.path.join(ROOT, "p0_2", f"p0_2_qwen2_{m}_ocrbench_r{r:.3f}_full.json")
            rows.append(make_row("P0-2", "qwen2vl", "ocrbench", m, RETENTION[r], 4000000, p))
    # ---- P1: RBM vs FastV-k3 (HF harness, 4M, OCRBench full) ----
    for fam in ("qwen3vl", "qwen2vl"):
        rows.append(make_row("P1", fam, "ocrbench", "rbm", 0.25, 4000000,
                             os.path.join(ROOT, "p1", f"p1_{fam}_pre_ocrbench_r25_full.json")))
        rows.append(make_row("P1", fam, "ocrbench", "fastv-k3", 0.25, 4000000,
                             os.path.join(ROOT, "p1", f"p1_{fam}_fastv_ocrbench_k3_full.json")))
    return rows


def make_row(campaign, fam, bench, mode, retention, pixel_cap, path):
    d = json.load(open(path))
    scores = per_sample_scores(d)
    m = official_metric(d, scores)
    tok = token_stats(scores)
    if bench == "ocrbench":
        metric, score = "OCRBench /1000", m["total1000"]
    else:
        metric, score = BENCH_METRIC[bench], m["official"]
    ps = d.get("per_sample", [])
    n_skip = d.get("n_skipped", sum(1 for p in ps if p.get("skipped")))
    return {
        "campaign": campaign,
        "model": MODEL_NAME[fam],
        "benchmark": bench,
        "mode": mode,
        "retention": retention if retention is not None else "N/A (none)",
        "pixel_cap": pixel_cap if pixel_cap else "native(0)",
        "n_attempted": len(ps),
        "n_completed": len(ps) - n_skip,
        "n_skipped": n_skip,
        "official_metric": metric,
        "official_score": score,
        "mean_visual_tokens": tok["mean"],
        "engine": ENGINE[campaign],
        "source_json": path.replace(REPO + "/", ""),
        "sha256": sha256(path),
    }


def main():
    rows = cell_rows()
    # cell_summary.csv
    csv_path = os.path.join(ROOT, "cell_summary.csv")
    fields = ["campaign", "model", "benchmark", "mode", "retention", "pixel_cap",
              "n_attempted", "n_completed", "n_skipped", "official_metric",
              "official_score", "mean_visual_tokens", "engine", "source_json",
              "sha256", "note"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"cell_summary.csv: {len(rows)} rows -> {csv_path}")

    # environment.txt
    import torch, transformers, vllm, PIL
    env_lines = [
        "acmmm_final_controls environment (captured 2026-08-19)",
        "=" * 60,
        f"host          : {platform.node()} ({platform.platform()})",
        f"python        : {platform.python_version()} ({sys.executable})",
        f"torch         : {torch.__version__}  cuda={torch.version.cuda}",
        f"transformers  : {transformers.__version__}",
        f"vllm          : {vllm.__version__}",
        f"PIL           : {PIL.__version__}",
        f"gpu           : A40 46GB (1x), driver nvidia 560.35.05, CUDA 12.6",
        f"git commit    : {os.popen('cd ' + REPO + ' && git rev-parse HEAD').read().strip()}",
        f"git branch    : {os.popen('cd ' + REPO + ' && git rev-parse --abbrev-ref HEAD').read().strip()}",
        "model qwen3   : Qwen/Qwen3-VL-8B-Instruct (HF hub cache, sha 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b)",
        "model qwen2.5 : Qwen/Qwen2.5-VL-7B-Instruct (HF hub cache, sha cc594898137f460bfe9f0759e9844b3ce807cfb5)",
        "env vars      : HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_MODELSCOPE=False VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "official scorer: src/v3_premerger/official_scorers.py (VQA-acc / ANLS / exact-match / OCRBench containment+HME)",
        "dataset       : eval/full_splits/{textvqa_val,docvqa_val,ocrbench,gqa_testdev}.jsonl",
    ]
    env_path = os.path.join(ROOT, "environment.txt")
    with open(env_path, "w") as f:
        f.write("\n".join(env_lines) + "\n")
    print(f"environment.txt -> {env_path}")

    # MANIFEST.sha256 : all source files (committed or not)
    files = []
    files.append(os.path.join(ROOT, "analysis.json"))
    files.append(os.path.join(ROOT, "cell_summary.csv"))
    files.append(os.path.join(ROOT, "environment.txt"))
    files += [os.path.join(ROOT, "analysis", f)
              for f in sorted(os.listdir(os.path.join(ROOT, "analysis")))]
    files += [r["source_json"] for r in rows]   # includes none anchors (uncommitted raw)
    files += [os.path.join(REPO, "scripts", "run_p0_1_full_split.sh"),
              os.path.join(REPO, "scripts", "run_p0_2_ocrbench_matched.sh"),
              os.path.join(REPO, "scripts", "run_p1_fastv_hf_ocrbench.sh"),
              os.path.join(REPO, "scripts", "run_acmmm_chain.sh"),
              os.path.join(REPO, "scripts", "analyze_acmmm_final_controls.py"),
              os.path.join(REPO, "scripts", "gen_acmmm_manifest.py"),
              os.path.join(REPO, "reports", "acmmm_final_controls.md"),
              os.path.join(REPO, "reports", "acmmm_final_controls_artifacts.md")]
    # committed gzip artifacts (derived; sha of the .gz bytes)
    for g in sorted(glob.glob(os.path.join(REPO, "artifacts", "acmmm_final_controls", "**", "*.gz"),
                              recursive=True)):
        files.append(g)
    manifest = []
    seen = set()
    for p in files:
        full = p if p.startswith("/") else os.path.join(REPO, p)
        if not os.path.exists(full) or full in seen:
            continue
        seen.add(full)
        manifest.append(f"{sha256(full)}  {os.path.getsize(full):>10}  {full.replace(REPO + '/', '')}")
    manifest_path = os.path.join(ROOT, "MANIFEST.sha256")
    with open(manifest_path, "w") as f:
        f.write("# acmmm_final_controls reproducibility manifest\n")
        f.write("# <sha256>  <bytes>  <path relative to repo root>\n")
        f.write("\n".join(sorted(manifest)) + "\n")
    print(f"MANIFEST.sha256: {len(manifest)} files -> {manifest_path}")

    # quick console summary of the key cells
    print("\nkey cells:")
    for r in rows:
        if r.get("note"):
            continue
        print(f"  {r['campaign']:4s} {r['model'].split('/')[1][:12]:12s} "
              f"{r['benchmark']:8s} {r['mode']:9s} score={r['official_score']} "
              f"skip={r['n_skipped']}/{r['n_attempted']} tokens={r['mean_visual_tokens']}")


if __name__ == "__main__":
    main()
