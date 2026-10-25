#!/usr/bin/env python3
"""Rescore saved DocVQA predictions with a pinned lmms-eval ANLS function.

CPU only. Keeps historical results untouched. Uses saved GT variants; the
original unjoined DocVQA annotation arrays are not independently recovered.
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
URL = f"https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/{COMMIT}/lmms_eval/api/metrics.py"
SHA256 = "eaf48195af80a46d22e545a03a1bd7e4cad4fcbadeb3d2d817ac7d148390d8f4"


def load_reference(cache):
    path = cache / "metrics_reference.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    nodes = []
    for node in ast.parse(raw).body:
        if isinstance(node, ast.FunctionDef) and node.name in ["anls", "levenshtein_distance"]:
            node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == 2
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    anls = namespace["anls"]
    assert anls(["CAT"], ["cat"])["anls"] == 1.0
    assert anls(["abcd"], ["abxx"])["anls"] == 0.5
    assert anls(["abcd"], ["xxxx"])["anls"] == 0.0
    assert anls(["a  b"], ["a b"])["anls"] == 1.0
    return anls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/cvm_cpu_rescore_20261025")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/cvm2027_docvqa_rescore_20261025.json")
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    anls = load_reference(args.cache)
    old = import_file("cvm_doc_legacy", ROOT / "src/v3_premerger/official_scorers.py")
    artifact = ROOT / "experiments/artifact_anonymous_tcsvt_20260904/results"
    sources = {
        "qwen3vl_none": args.cache / "matrix/r2_qwen3vl_none_docvqa_full5349.json",
        "qwen3vl_pre": args.cache / "matrix/j7_qwen3vl_pre_docvqa_r0.750_full.json",
        "qwen3vl_post": ROOT / "artifacts/acmmm_final_controls/p0_1/p0_1_qwen3_post_docvqa_r0.750_full.json.gz",
        "qwen3vl_pre_final": ROOT / "artifacts/acmmm_final_controls/p0_1/p0_1_qwen3_pre-final_docvqa_r0.750_full.json.gz",
        "qwen3vl_post_main": args.cache / "expA/full_docvqa_main.json",
    }
    for mode in ["none", "pre", "post"]:
        ratio = "0.000" if mode == "none" else "0.750"
        sources[f"qwen2vl_{mode}"] = args.cache / "matrix" / f"j7_qwen2vl_{mode}_docvqa_r{ratio}_full.json"
        sources[f"internvl3_{mode}"] = artifact / f"internvl3_{mode}_docvqa_r{ratio}_full.json"
    output = {"reference_commit": COMMIT, "reference_url": URL, "reference_sha256": SHA256,
              "method": "Execute pinned upstream anls and levenshtein_distance functions directly",
              "ground_truth_source": "Saved semicolon-delimited GT variants; original unjoined DocVQA arrays not independently recovered",
              "denominator": "5349 attempted IDs; skipped score zero", "cells": {}, "pairs": {}}
    memory = {}
    for label, path in sources.items():
        data, digest = load_json(path)
        rows = {str(row["id"]): row for row in data["per_sample"]}
        assert len(rows) == len(data["per_sample"]) == 5349
        scores, old_scores = {}, {}
        for sid, row in rows.items():
            if row.get("skipped"):
                scores[sid] = old_scores[sid] = 0.0
            else:
                prediction, gt = row.get("answer", ""), str(row.get("gt", ""))
                assert gt.strip(), f"Empty DocVQA reference {sid}"
                scores[sid] = anls(gt.split(";"), [prediction])["anls"]
                old_scores[sid] = old.score_docvqa_anls(prediction, gt)
        delta = [scores[sid] - old_scores[sid] for sid in scores]
        summary = {"source": str(path), "uncompressed_sha256": digest,
                   "attempted": len(rows), "skipped": sum(bool(row.get("skipped")) for row in rows.values()),
                   "score_attempted": sum(scores.values()) / len(rows),
                   "legacy_score_attempted": sum(old_scores.values()) / len(rows),
                   "affected_samples": sum(abs(value) > 1e-12 for value in delta),
                   "max_abs_per_sample_change": max(abs(value) for value in delta)}
        output["cells"][label] = summary
        memory[label] = scores, rows
        print(label, summary["score_attempted"], "changed", summary["affected_samples"], flush=True)
    pairs = [("qwen3vl_pre", "qwen3vl_post"), ("qwen2vl_pre", "qwen2vl_post"),
             ("internvl3_pre", "internvl3_post"), ("qwen3vl_pre_final", "qwen3vl_post"),
             ("qwen3vl_pre_final", "qwen3vl_post_main")]
    for index, (a, b) in enumerate(pairs):
        value = paired(*memory[a], *memory[b], args.resamples, 20261110 + index, include_skips=True)
        output["pairs"][a + "_minus_" + b] = value
        print(a, "minus", b, value["delta_pp"], value["ci95_pp"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    md = ["# DocVQA pinned lmms-eval ANLS CPU rescore (2026-10-25)", "",
          "Source: " + URL, "", "Saved GT variants; no new inference; all 5349 attempted IDs included.", "",
          "| Cell | Score | Legacy score | Changed items |", "|---|---:|---:|---:|"]
    for label, value in output["cells"].items():
        md.append(f"| {label} | {value['score_attempted']:.8f} | {value['legacy_score_attempted']:.8f} | {value['affected_samples']} |")
    md.extend(["", "| A minus B | Delta pp | 95% CI pp |", "|---|---:|---:|"])
    for label, value in output["pairs"].items():
        lo, hi = value["ci95_pp"]
        md.append(f"| {label} | {value['delta_pp']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
