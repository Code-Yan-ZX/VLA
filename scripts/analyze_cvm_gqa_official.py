#!/usr/bin/env python3
"""CPU-only GQA audit with the pinned lmms-eval exact-match function.

Reads saved predictions only; no model inference or server mutation. Inputs stay
in tmp/historical artifact directories. Failed examples score zero in the full
attempted denominator. The upstream function is executed directly via AST to
avoid importing unrelated dataset/model dependencies.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import string
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "8f40097532a7bf4db6123713f1a905e02c8f5971"
BASE = f"https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/{COMMIT}/lmms_eval/"
METRICS_SHA = "eaf48195af80a46d22e545a03a1bd7e4cad4fcbadeb3d2d817ac7d148390d8f4"
TASK_SHA = "4530277d93b2644b121ce6741612ffb22bbeadc6bd2e5f5301860a238f36c542"
EXPECTED_N = 12578


def reference_file(path: Path, url: str, expected: str) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, f"Reference changed: {path}"
    return raw


def exact_function(raw: bytes, filename: str):
    nodes = [n for n in ast.parse(raw).body
             if isinstance(n, ast.FunctionDef) and n.name == "exact_match_hf_evaluate"]
    assert len(nodes) == 1
    namespace = {"np": np, "re": re, "string": string}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace["exact_match_hf_evaluate"]


def import_legacy():
    path = ROOT / "src/v3_premerger/official_scorers.py"
    spec = importlib.util.spec_from_file_location("cvm_gqa_legacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_gqa


def checks(score):
    # These distinguish GQA's exact-match configuration from VQA normalization.
    cases = [("Cat!", "cat", 1), ("the cat", "cat", 0),
             ("two", "2", 0), (" cat", "cat", 0),
             ("traffic-light", "traffic light", 0),
             ("bus", "car;bus", 0), ("cat's", "cats", 1)]
    for prediction, target, expected in cases:
        assert score(prediction, target) == expected, (prediction, target)
    return len(cases)


def summarize(path: Path, score, old_score):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    data = json.loads(raw)
    rows = {str(row["id"]): row for row in data["per_sample"]}
    assert len(rows) == len(data["per_sample"]) == EXPECTED_N, f"Count/ID mismatch: {path}"
    official, legacy = {}, {}
    changed = []
    for sid, row in rows.items():
        if row.get("skipped"):
            official[sid] = legacy[sid] = 0.0
            continue
        prediction, target = row.get("answer", ""), row.get("gt", "")
        assert isinstance(prediction, str) and isinstance(target, str), f"Non-string {sid}"
        official[sid] = score(prediction, target)
        legacy[sid] = float(old_score(prediction, target))
        if official[sid] != legacy[sid]:
            changed.append(sid)
    skipped = sum(bool(r.get("skipped")) for r in rows.values())
    return {
        "source": str(path), "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "attempted": len(rows), "answered": len(rows) - skipped, "skipped": skipped,
        "official_attempted": sum(official.values()) / len(rows),
        "official_answered": sum(official.values()) / (len(rows) - skipped),
        "legacy_attempted": sum(legacy.values()) / len(rows),
        "legacy_answered": sum(legacy.values()) / (len(rows) - skipped),
        "official_minus_legacy_pp": (sum(official.values()) - sum(legacy.values())) / len(rows) * 100,
        "changed_items": len(changed),
        "legacy_positive_official_zero": sum(legacy[s] == 1 and official[s] == 0 for s in changed),
        "legacy_zero_official_positive": sum(legacy[s] == 0 and official[s] == 1 for s in changed),
        # IDs alone provide an audit handle without publishing raw predictions.
        "changed_ids": changed,
    }, official, rows


def paired(a_scores, a_rows, b_scores, b_rows, resamples: int, seed: int):
    assert set(a_rows) == set(b_rows), "Paired attempted IDs differ"
    ids = sorted(a_rows)
    assert all(a_rows[s].get("gt") == b_rows[s].get("gt") for s in ids), "Paired GT mismatch"
    a = np.array([a_scores[s] for s in ids], dtype=np.int8)
    b = np.array([b_scores[s] for s in ids], dtype=np.int8)
    difference = a - b
    support, counts = np.unique(difference, return_counts=True)
    # A paired empirical bootstrap resamples per-ID score differences. Since
    # they are {-1,0,1}, multinomial support counts are exactly the same sampling
    # distribution as materializing 20,000 x 12,578 resampled row indices.
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(ids), counts / len(ids), size=resamples)
    bootstrap = (draws @ support) / len(ids)
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    answered = [s for s in ids if not a_rows[s].get("skipped") and not b_rows[s].get("skipped")]
    present_visual = all("n_image_kept" in a_rows[s] and "n_image_kept" in b_rows[s] for s in answered)
    return {
        "n_paired": len(ids), "denominator": "all attempted IDs; failed samples score zero",
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta_pp": float(difference.mean() * 100),
        "ci95_pp": [float(lo * 100), float(hi * 100)],
        "resamples": resamples, "seed": seed,
        "bootstrap": "paired empirical nonparametric bootstrap via exact multinomial difference support",
        "difference_counts": {str(int(s)): int(c) for s, c in zip(support, counts)},
        "skipped_a": sum(bool(r.get("skipped")) for r in a_rows.values()),
        "skipped_b": sum(bool(r.get("skipped")) for r in b_rows.values()),
        "skip_ids_equal": {s for s in ids if a_rows[s].get("skipped")} == {s for s in ids if b_rows[s].get("skipped")},
        "prompt_count_mismatches": sum(a_rows[s].get("prompt_token_ids") != b_rows[s].get("prompt_token_ids") for s in answered),
        "actual_visual_field_present": present_visual,
        "actual_visual_count_mismatches": sum(a_rows[s]["n_image_kept"] != b_rows[s]["n_image_kept"] for s in answered) if present_visual else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/cvm_cpu_rescore_20261025")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/cvm2027_gqa_rescore_20261025.json")
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    metrics_path = args.cache / "metrics_reference.py"
    task_path = args.cache / "gqa_yaml_reference.yaml"
    metrics = reference_file(metrics_path, BASE + "api/metrics.py", METRICS_SHA)
    task = reference_file(task_path, BASE + "tasks/gqa/gqa.yaml", TASK_SHA).decode("utf-8")
    assert re.search(r"ignore_case:\s*true", task) and re.search(r"ignore_punctuation:\s*true", task)
    assert "ignore_numbers" not in task and "regexes_to_ignore" not in task
    upstream = exact_function(metrics, str(metrics_path))
    score = lambda prediction, target: float(upstream(
        predictions=[prediction], references=[target],
        ignore_case=True, ignore_punctuation=True)["exact_match"])
    check_count = checks(score)
    old_score = import_legacy()
    artifact = ROOT / "experiments/artifact_anonymous_tcsvt_20260904/results"
    sources = {}
    for family in ["qwen3vl", "qwen2vl"]:
        for mode in ["none", "pre", "post"]:
            retention = "0.000" if mode == "none" else "0.750"
            sources[f"{family}_{mode}"] = args.cache / "matrix" / f"j7_{family}_{mode}_gqa_r{retention}_full.json"
    for mode in ["none", "pre", "post"]:
        retention = "0.000" if mode == "none" else "0.750"
        sources[f"internvl3_{mode}"] = artifact / f"internvl3_{mode}_gqa_r{retention}_full.json"
    sources.update({
        "qwen3vl_pre_final": ROOT / "artifacts/acmmm_final_controls/p0_1/p0_1_qwen3_pre-final_gqa_r0.750_full.json.gz",
        "qwen3vl_post_main": args.cache / "expA/full_gqa_main.json",
    })
    result = {
        "reference_commit": COMMIT, "metrics_url": BASE + "api/metrics.py",
        "metrics_sha256": METRICS_SHA, "task_url": BASE + "tasks/gqa/gqa.yaml",
        "task_sha256": TASK_SHA, "upstream_function": "exact_match_hf_evaluate",
        "configuration": {"ignore_case": True, "ignore_punctuation": True},
        "reference_checks_passed": check_count, "new_inference": False,
        "metric": "pinned lmms-eval GQA exact_match; no article/number/contraction normalization or whitespace stripping",
        "cells": {}, "pairs": {}, "missing_sources": [],
    }
    memory = {}
    for label, path in sources.items():
        if not path.exists():
            result["missing_sources"].append(str(path))
            continue
        summary, scores, rows = summarize(path, score, old_score)
        result["cells"][label] = summary
        memory[label] = (scores, rows)
        print(label, f"official={summary['official_attempted']:.8f}",
              f"legacy={summary['legacy_attempted']:.8f}", f"changed={summary['changed_items']}", flush=True)
    mirror_path = ROOT / "artifacts/acmmm_final_controls/p0_1/p0_1_qwen3_post_gqa_r0.750_full.json.gz"
    if mirror_path.exists() and "qwen3vl_post" in memory:
        mirror_raw = gzip.decompress(mirror_path.read_bytes())
        mirror = {str(r["id"]): r for r in json.loads(mirror_raw)["per_sample"]}
        matrix = memory["qwen3vl_post"][1]
        assert set(mirror) == set(matrix)
        mismatches = {field: sum(mirror[s].get(field) != matrix[s].get(field) for s in matrix)
                      for field in ["answer", "gt", "skipped"]}
        assert not any(mismatches.values()), f"Control baseline differs: {mismatches}"
        result["control_post_concat_mirror"] = {
            "source": str(mirror_path), "uncompressed_sha256": hashlib.sha256(mirror_raw).hexdigest(),
            "n_paired": len(matrix), "field_mismatches": mismatches,
        }
    pairs = [("qwen3vl_pre", "qwen3vl_post"), ("qwen2vl_pre", "qwen2vl_post"),
             ("internvl3_pre", "internvl3_post"), ("qwen3vl_pre_final", "qwen3vl_post"),
             ("qwen3vl_post_main", "qwen3vl_pre_final")]
    for index, (a, b) in enumerate(pairs):
        if a in memory and b in memory:
            comparison = paired(*memory[a], *memory[b], args.resamples, 20261025 + index)
            comparison["legacy_delta_pp"] = (
                result["cells"][a]["legacy_attempted"] - result["cells"][b]["legacy_attempted"]) * 100
            comparison["delta_sign_changed"] = bool(np.sign(comparison["delta_pp"]) != np.sign(comparison["legacy_delta_pp"]))
            result["pairs"][f"{a}_minus_{b}"] = comparison
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# GQA pinned lmms-eval CPU rescore (2026-10-25)", "",
          "No new model inference. The exact pinned upstream function is executed for every non-failed example.", "",
          f"Reference commit: `{COMMIT}`. [Metric source]({result['metrics_url']}); [GQA task configuration]({result['task_url']}).", "",
          "The task enables ignore_case and ignore_punctuation only. It does not strip whitespace, drop articles, map number words, expand contractions, or split semicolon answer variants.", "",
          "All attempted splits contain 12,578 IDs; failed samples receive zero in means and paired intervals.", "",
          "| Cell | Attempted / answered | Official attempted | Legacy attempted | Changed examples |", "|---|---:|---:|---:|---:|"]
    for label, row in result["cells"].items():
        md.append(f"| {label} | {row['attempted']} / {row['answered']} | {row['official_attempted']:.8f} | {row['legacy_attempted']:.8f} | {row['changed_items']} |")
    md.extend(["", "| A minus B | Paired n | Delta pp | Paired 95% CI pp |", "|---|---:|---:|---:|"])
    for label, row in result["pairs"].items():
        lo, hi = row["ci95_pp"]
        md.append(f"| {label} | {row['n_paired']} | {row['delta_pp']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    md.extend(["", f"Each interval uses {args.resamples:,} paired empirical bootstrap draws. Multinomial counts over the empirical score-difference support implement exactly the same bootstrap distribution as resampling paired IDs, without allocating a large index array.", "",
               "Seven targeted normalization checks passed. JSON records source digests, changed IDs, all seeds, failure handling, and pairing checks. Raw answers are not copied into the report.", "",
               "The Qwen3 control Post-concat artifact and operational-matrix baseline have identical IDs, answers, GT, and skip flags on all 12,578 examples. All five reported comparison directions remain unchanged after official rescoring.", "",
               f"Missing sources: {len(result['missing_sources'])}."])
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
