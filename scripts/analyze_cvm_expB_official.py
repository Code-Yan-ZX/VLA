#!/usr/bin/env python3
"""CPU-only CVM audit: rescore saved TextVQA predictions using lmms-eval.

Does not load a model, alter historical results, or run GPU inference.
The pinned upstream normalizer is cached under --cache; its digest is checked.
Run from any directory; predictions must already exist in --cache or the
historical local artifact directories. Output includes both attempted and
answered denominators, source SHA256s, and paired bootstrap intervals.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "8f40097532a7bf4db6123713f1a905e02c8f5971"
REFERENCE_URL = (
    "https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/"
    + REFERENCE_COMMIT + "/lmms_eval/tasks/_task_utils/vqa_eval_metric.py"
)
REFERENCE_SHA256 = "4e4154c6c7e4657e6c44aa08e184ce5ae0e872665744c6cb684cf645e4647ce0"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def vqa_score(prediction, gt, normalize):
    """Exact leave-one-annotator-out loop in lmms_eval/tasks/textvqa/utils.py."""
    answers = [normalize(str(s)) for s in (gt if isinstance(gt, list) else str(gt).split(";"))]
    assert len(answers) == 10, f"Expected ten TextVQA answers, got {len(answers)}"
    prediction = normalize(str(prediction or ""))
    return sum(min(1.0, sum(a == prediction for j, a in enumerate(answers) if j != i) / 3.0)
               for i in range(len(answers))) / len(answers)


def reference_checks(normalize):
    for matches, expected in enumerate([0.0, 0.3, 0.6, 0.9, 1.0]):
        gt = ";".join(["cat"] * matches + ["dog"] * (10 - matches))
        assert abs(vqa_score("cat", gt, normalize) - expected) < 1e-12
    for prediction, gold in [("The TWO!", "2"), ("an apple", "apple"),
                             ("a book", "book"), ("cant", "can't"),
                             ("John's", "John 's")]:
        assert abs(vqa_score(prediction, ";".join([gold] * 10), normalize) - 1) < 1e-12
    assert vqa_score("a cat and a dog", ";".join(["cat"] * 10), normalize) == 0


def summarize(path, normalize, old, official_answers):
    data, digest = load_json(path)
    rows = data.get("per_sample", [])
    assert len({str(row["id"]) for row in rows}) == len(rows), f"Duplicate ID in {path}"
    scores, old_scores, online_scores = {}, {}, {}
    records = {}
    for row in rows:
        sid = str(row["id"])
        records[sid] = row
        if row.get("skipped"):
            scores[sid] = old_scores[sid] = online_scores[sid] = 0.0
        else:
            answer, gt = row.get("answer", ""), row.get("gt", "")
            assert ";".join(a.strip() for a in official_answers[sid]) == gt, f"Saved GT differs from official annotation: {sid}"
            scores[sid] = vqa_score(answer, official_answers[sid], normalize)
            old_scores[sid] = old.score_textvqa_vqaacc(answer, gt)
            online_scores[sid] = float(row.get("correct", 0.0))
    n = len(rows)
    answered = [sid for sid in scores if not records[sid].get("skipped")]
    return {
        "source": str(path), "uncompressed_sha256": digest,
        "attempted": n, "answered": len(answered), "skipped": n - len(answered),
        "score_attempted": sum(scores.values()) / n,
        "score_answered": sum(scores[sid] for sid in answered) / len(answered),
        "old_local_scorer_answered": sum(old_scores[sid] for sid in answered) / len(answered),
        "online_binary_correct_answered": sum(online_scores[sid] for sid in answered) / len(answered),
        "normalizer_or_consensus_changed_items": sum(abs(scores[sid] - old_scores[sid]) > 1e-12 for sid in answered),
        "ambiguous_semicolon_gt_items": sum(len(str(records[sid].get("gt", "")).split(";")) != 10 for sid in answered),
    }, scores, records


def paired(scores_a, rows_a, scores_b, rows_b, resamples, seed, include_skips=False):
    assert set(rows_a) == set(rows_b), "Attempted ID sets differ"
    common = sorted(sid for sid in rows_a if include_skips or
                    (not rows_a[sid].get("skipped") and not rows_b[sid].get("skipped")))
    for sid in common:
        assert rows_a[sid].get("gt") == rows_b[sid].get("gt"), f"GT mismatch {sid}"
    a = np.array([scores_a[sid] for sid in common])
    b = np.array([scores_b[sid] for sid in common])
    diff = a - b
    rng = np.random.default_rng(seed)
    bootstrap, permutation = [], []
    for start in range(0, resamples, 128):
        batch = min(128, resamples - start)
        indices = rng.integers(0, len(diff), (batch, len(diff)))
        bootstrap.extend(diff[indices].mean(axis=1).tolist())
        signs = rng.integers(0, 2, (batch, len(diff)), dtype=np.int8) * 2 - 1
        permutation.extend((signs * diff).mean(axis=1).tolist())
    observed = float(diff.mean())
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    budget_ids = [sid for sid in common if not rows_a[sid].get("skipped") and not rows_b[sid].get("skipped")]
    present_visual = all("n_image_kept" in rows_a[sid] and "n_image_kept" in rows_b[sid] for sid in budget_ids)
    return {
        "n_paired": len(common), "n_paired_answered": len(budget_ids), "mean_a": float(a.mean()),
        "denominator": "all attempted; skipped score zero" if include_skips else "common answered",
        "mean_b": float(b.mean()), "delta_pp": observed * 100,
        "ci95_pp": [float(lo) * 100, float(hi) * 100],
        "sign_flip_p_two_sided": (1 + sum(abs(v) >= abs(observed) - 1e-12 for v in permutation)) / (1 + resamples),
        "skipped_a": sum(bool(row.get("skipped")) for row in rows_a.values()),
        "skipped_b": sum(bool(row.get("skipped")) for row in rows_b.values()),
        "skip_ids_equal": {sid for sid, row in rows_a.items() if row.get("skipped")} == {sid for sid, row in rows_b.items() if row.get("skipped")},
        "prompt_token_count_mismatches": sum(rows_a[sid].get("prompt_token_ids") != rows_b[sid].get("prompt_token_ids") for sid in common),
        "actual_visual_count_field_present": present_visual,
        "actual_visual_count_check_n": len(budget_ids),
        "actual_visual_count_mismatches": sum(rows_a[sid]["n_image_kept"] != rows_b[sid]["n_image_kept"] for sid in budget_ids) if present_visual else None,
        "resamples": resamples, "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/cvm_cpu_rescore_20261025")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/cvm2027_textvqa_rescore_20261025.json")
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--annotations", type=Path, help="Official TextVQA v0.5.1 val JSON; digest is verified")
    args = parser.parse_args()
    reference = args.cache / "vqa_eval_metric_reference.py"
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(urllib.request.urlopen(REFERENCE_URL, timeout=30).read())
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == REFERENCE_SHA256, "Reference digest mismatch"
    normalizer = import_file("cvm_pinned_vqa_reference", reference).EvalAIAnswerProcessor()
    old = import_file("cvm_legacy_scorer", ROOT / "src/v3_premerger/official_scorers.py")
    reference_checks(normalizer)
    annotation_path = args.annotations or args.cache / "TextVQA_0.5.1_val.json"
    annotation_url = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
    if not annotation_path.exists():
        annotation_path.write_bytes(urllib.request.urlopen(annotation_url, timeout=30).read())
    annotation_bytes = annotation_path.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    assert annotation_sha256 == "4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d"
    official_answers = {str(row["question_id"]): row["answers"] for row in json.loads(annotation_bytes)["data"]}
    assert len(official_answers) == 5000
    # Independently execute the upstream task function, with unrelated task IO
    # imports excluded via AST. Compare against our helper on every HF sample.
    task_path = args.cache / "textvqa_utils_reference.py"
    task_url = f"https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/{REFERENCE_COMMIT}/lmms_eval/tasks/textvqa/utils.py"
    if not task_path.exists():
        task_path.write_bytes(urllib.request.urlopen(task_url, timeout=30).read())
    task_raw = task_path.read_bytes()
    task_sha256 = hashlib.sha256(task_raw).hexdigest()
    assert task_sha256 == "4d1e10d2a72136e2fb11486827f29d955f4782b51e0cafe7228b1d53da14d405"
    task_nodes = [node for node in ast.parse(task_raw).body if isinstance(node, ast.FunctionDef) and node.name == "textvqa_process_results"]
    assert len(task_nodes) == 1
    task_namespace = {"EvalAIAnswerProcessor": type(normalizer), "statistics": statistics}
    exec(compile(ast.Module(body=task_nodes, type_ignores=[]), str(task_path), "exec"), task_namespace)
    task_function = task_namespace["textvqa_process_results"]
    independent_checks = 0
    for row in load_json(args.cache / "expB/full_textvqa_rbm_pre_r25.json")[0]["per_sample"]:
        if row.get("skipped"):
            continue
        answers = official_answers[str(row["id"])]
        direct = task_function({"answers": list(answers), "question_id": str(row["id"])}, [row.get("answer", "")])["exact_match"]
        assert abs(direct - vqa_score(row.get("answer", ""), answers, normalizer)) < 1e-12
        independent_checks += 1
    artifact = ROOT / "experiments/artifact_anonymous_tcsvt_20260904/results"
    sources = {}
    for family in ["qwen3vl", "qwen2vl"]:
        for mode in ["none", "pre", "post"]:
            retention = "0.000" if mode == "none" else "0.750"
            sources[f"{family}_{mode}"] = args.cache / "matrix" / f"j7_{family}_{mode}_textvqa_r{retention}_full.json"
    for mode in ["none", "pre", "post"]:
        retention = "0.000" if mode == "none" else "0.750"
        sources[f"internvl3_{mode}"] = artifact / f"internvl3_{mode}_textvqa_r{retention}_full.json"
    sources.update({
        "qwen3vl_pre_final": ROOT / "artifacts/acmmm_final_controls/p0_1/p0_1_qwen3_pre-final_textvqa_r0.750_full.json.gz",
        "qwen3vl_post_main": args.cache / "expA/full_textvqa_main.json",
        "qwen3vl_hf_rbm": args.cache / "expB/full_textvqa_rbm_pre_r25.json",
        "qwen3vl_hf_fastv": args.cache / "expB/full_textvqa_fastv_k3.json",
        "qwen3vl_hf_old_fastv": artifact / "r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000.json",
    })
    result = {"reference_commit": REFERENCE_COMMIT, "reference_url": REFERENCE_URL,
              "reference_sha256": REFERENCE_SHA256, "reference_checks": "passed",
              "official_annotation_url": annotation_url, "official_annotation_sha256": annotation_sha256,
              "task_reference_url": task_url, "task_reference_sha256": task_sha256,
              "independent_upstream_function_checks": independent_checks,
              "metric": "lmms-eval TextVQA: EvalAIAnswerProcessor + leave-one-annotator-out consensus",
              "cells": {}, "pairs": {}, "missing_sources": []}
    memory = {}
    for label, path in sources.items():
        if not path.exists():
            result["missing_sources"].append(str(path))
            continue
        summary, scores, rows = summarize(path, normalizer, old, official_answers)
        result["cells"][label] = summary
        memory[label] = (scores, rows)
        print(label, f"attempted={summary['score_attempted']:.8f}", f"answered={summary['score_answered']:.8f}", flush=True)
    pairs = [("qwen3vl_pre", "qwen3vl_post"), ("qwen2vl_pre", "qwen2vl_post"),
             ("internvl3_pre", "internvl3_post"), ("qwen3vl_pre_final", "qwen3vl_post"),
             ("qwen3vl_pre_final", "qwen3vl_post_main"), ("qwen3vl_hf_rbm", "qwen3vl_hf_fastv")]
    for index, (a, b) in enumerate(pairs):
        if a in memory and b in memory:
            pair = paired(*memory[a], *memory[b], args.resamples, 20261025 + index)
            result["pairs"][f"{a}_minus_{b}"] = pair
            print(a, "minus", b, pair["delta_pp"], pair["ci95_pp"], flush=True)
    a, b = "qwen3vl_hf_rbm", "qwen3vl_hf_fastv"
    result["pairs"][f"{a}_minus_{b}_all_attempted"] = paired(*memory[a], *memory[b], args.resamples, 20261101, include_skips=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# TextVQA official metric CPU rescore (2026-10-25)", "",
          "Reference: " + REFERENCE_URL, "", "No new model inference. Skips receive zero in attempted scores.", "",
          "| Cell | Attempted/answered | Official attempted | Official answered | Legacy scorer answered |",
          "|---|---:|---:|---:|---:|"]
    for label, row in result["cells"].items():
        md.append(f"| {label} | {row['attempted']}/{row['answered']} | {row['score_attempted']:.6f} | {row['score_answered']:.6f} | {row['old_local_scorer_answered']:.6f} |")
    md.extend(["", "Paired deltas use common answered IDs, except the explicitly labelled all-attempted row (shared skips score zero).", "",
               "| A minus B | Paired n | Delta pp | 95% CI pp |",
               "|---|---:|---:|---:|"])
    for label, row in result["pairs"].items():
        lo, hi = row["ci95_pp"]
        md.append(f"| {label} | {row['n_paired']} | {row['delta_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
