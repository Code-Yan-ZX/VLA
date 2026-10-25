#!/usr/bin/env python3
"""Bounded CPU audit of all manuscript OCRBench cells against pinned lmms-eval.

Executes the upstream OCRBench per-item scorer directly, including fullwidth
ASCII folding. Uses the existing complete 1000-item dataset/question_type map
and saved GT variants; original unjoined answer arrays are not independently
recovered. No model loading, GPU use, or historical-result mutation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

from analyze_cvm_expB_official import ROOT, import_file, load_json, paired

COMMIT = "8f40097532a7bf4db6123713f1a905e02c8f5971"
URL = f"https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/{COMMIT}/lmms_eval/tasks/ocrbench/utils.py"
SHA256 = "0baaa04eed6ba5c610b7d2260a5a44c9fa76cd06331cbd86817064d882ac3557"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/cvm_cpu_rescore_20261025")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/cvm2027_ocrbench_rescore_20261025.json")
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    reference = args.cache / "ocrbench_utils_reference.py"
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    raw = reference.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
             and node.name in ["_fold_fullwidth_ascii", "ocrbench_process_results"]]
    assert len(nodes) == 2
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), "exec"), namespace)
    scorer = namespace["ocrbench_process_results"]
    # Test the newly inspected branch and distinct HME semantics.
    def score(answer, prediction, dataset="IIIT5K"):
        return scorer({"answer": [answer], "dataset": dataset,
                       "question_type": "probe"}, [prediction])["ocrbench_accuracy"]["score"]
    assert score("ABC", "ＡＢＣ") == 1
    assert score("ABC", "ABC", "HME100k") == 1
    assert score("A B C", "ABC", "HME100k") == 1
    assert score("ABC", "abc", "HME100k") == 0
    assert score("ABC", "ＡＢＣ", "HME100k") == 0
    annotation_path = ROOT / "eval/full_splits/ocrbench.jsonl"
    annotation_bytes = annotation_path.read_bytes()
    annotations = {str(row["id"]): row for row in map(json.loads, annotation_bytes.decode("utf8").splitlines())}
    assert len(annotations) == 1000
    assert all(row.get("dataset") and row.get("question_type") for row in annotations.values())
    old = import_file("cvm_ocr_legacy", ROOT / "src/v3_premerger/official_scorers.py")
    artifact = ROOT / "experiments/artifact_anonymous_tcsvt_20260904/results"
    controls = ROOT / "artifacts/acmmm_final_controls"
    sources = {
        "qwen3vl_none": args.cache / "matrix/j7_qwen3vl_none_ocrbench_r0.000_full.json",
        "qwen3vl_pre": args.cache / "matrix/j7_qwen3vl_pre_ocrbench_r0.750_full.json",
        "qwen3vl_post": controls / "p0_1/p0_1_qwen3_post_ocrbench_r0.750_full.json.gz",
        "qwen3vl_pre_final": controls / "p0_1/p0_1_qwen3_pre-final_ocrbench_r0.750_full.json.gz",
        "qwen3vl_post_main": args.cache / "expA/full_ocrbench_main.json",
        "qwen2vl_none": args.cache / "matrix/j7_qwen2vl_none_ocrbench_r0.000_full.json",
        "qwen2vl_pre": controls / "p0_2/p0_2_qwen2_pre_ocrbench_r0.750_full.json.gz",
        "qwen2vl_post": controls / "p0_2/p0_2_qwen2_post_ocrbench_r0.750_full.json.gz",
    }
    for mode in ["none", "pre", "post"]:
        ratio = "0.000" if mode == "none" else "0.750"
        sources[f"internvl3_{mode}"] = artifact / f"internvl3_{mode}_ocrbench_r{ratio}_full.json"
    for family in ["qwen3vl", "qwen2vl"]:
        for mode, suffix in [("pre", "r25"), ("fastv", "k3")]:
            sources[f"{family}_hf_{mode}"] = controls / "p1" / f"p1_{family}_{mode}_ocrbench_{suffix}_full.json.gz"
    output = {"reference_commit": COMMIT, "reference_url": URL, "reference_sha256": SHA256,
              "annotation_source": str(annotation_path), "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
              "ground_truth_source": "Saved GT variants checked against complete local annotation map; original unjoined answer arrays not independently recovered",
              "annotation_n": len(annotations), "hme_n": sum(row["dataset"] == "HME100k" for row in annotations.values()),
              "denominator": "1000 attempted; skips score zero", "cells": {}, "pairs": {}}
    memory = {}
    for label, path in sources.items():
        data, digest = load_json(path)
        rows = {str(row["id"]): row for row in data["per_sample"]}
        assert len(rows) == len(data["per_sample"]) == 1000
        assert set(rows) == set(annotations)
        scores, old_scores = {}, {}
        for sid, row in rows.items():
            metadata = annotations[sid]
            assert str(row.get("gt", "")) == metadata["gt"], f"GT mismatch {label}/{sid}"
            if row.get("skipped"):
                scores[sid] = old_scores[sid] = 0.0
                continue
            doc = {"answer": str(row["gt"]).split(";"), "dataset": metadata["dataset"],
                   "question_type": metadata["question_type"]}
            prediction = row.get("answer", "")
            scores[sid] = float(scorer(doc, [prediction])["ocrbench_accuracy"]["score"])
            old_scores[sid] = float(old.score_ocrbench(prediction, row["gt"], question_type=metadata["question_type"]))
        summary = {"source": str(path), "uncompressed_sha256": digest,
                   "attempted": len(rows), "skipped": sum(bool(row.get("skipped")) for row in rows.values()),
                   "score_out_of_1000": sum(scores.values()), "legacy_score_out_of_1000": sum(old_scores.values()),
                   "changed_samples": sum(scores[sid] != old_scores[sid] for sid in rows)}
        output["cells"][label] = summary
        memory[label] = scores, rows
        print(label, summary["score_out_of_1000"], "changed", summary["changed_samples"], flush=True)
    pairs = [("qwen3vl_pre", "qwen3vl_post"), ("qwen2vl_pre", "qwen2vl_post"),
             ("internvl3_pre", "internvl3_post"), ("qwen3vl_pre_final", "qwen3vl_post"),
             ("qwen3vl_pre_final", "qwen3vl_post_main"),
             ("qwen3vl_hf_pre", "qwen3vl_hf_fastv"), ("qwen2vl_hf_pre", "qwen2vl_hf_fastv")]
    for index, (a, b) in enumerate(pairs):
        value = paired(*memory[a], *memory[b], args.resamples, 20261120 + index, include_skips=True)
        value["delta_points_out_of_1000"] = value["delta_pp"] * 10
        value["ci95_points_out_of_1000"] = [v * 10 for v in value["ci95_pp"]]
        output["pairs"][a + "_minus_" + b] = value
        print(a, "minus", b, value["delta_points_out_of_1000"], value["ci95_points_out_of_1000"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    md = ["# OCRBench pinned lmms-eval CPU audit (2026-10-25)", "", "Source: " + URL, "",
          "All 1000 attempted IDs; shared skips score zero. Includes fullwidth ASCII folding and HME-specific rules.", "",
          "| Cell | Score /1000 | Legacy /1000 | Changed items |", "|---|---:|---:|---:|"]
    for label, value in output["cells"].items():
        md.append(f"| {label} | {value['score_out_of_1000']:.0f} | {value['legacy_score_out_of_1000']:.0f} | {value['changed_samples']} |")
    md.extend(["", "| A minus B | Delta /1000 | 95% CI /1000 |", "|---|---:|---:|"])
    for label, value in output["pairs"].items():
        lo, hi = value["ci95_points_out_of_1000"]
        md.append(f"| {label} | {value['delta_points_out_of_1000']:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
