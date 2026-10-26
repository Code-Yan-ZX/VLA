#!/usr/bin/env python
"""CPU-only validation of completed corrected P3 cells (no legacy-file fallback)."""
import argparse
import json
import math
from pathlib import Path
import statistics


def validate(paths):
    report = {"files": [], "cells": {}, "warnings": []}
    shared_ids = None
    full_counts = {}
    fingerprints = None
    all_tags = set()
    for path in paths:
        x = json.loads(path.read_text())
        assert x["schema_version"] == 2, path
        assert x["attention"] == {"vision_config": "sdpa", "text_config": "eager"}, path
        assert x["n_requests"] == 200 and x["repeats"] == 3, (path, "incomplete workload")
        fp = (x["script_sha256"], x["timing_ids_sha256"], x["model"], x["dtype"],
              x["max_new_tokens"], x["max_pixels"], x["torch"], x["transformers"])
        if fingerprints is None:
            fingerprints = fp
        assert fp == fingerprints, (path, "protocol mismatch")
        report["files"].append(str(path))
        if x["engine"] == "hilo":
            noop = x["noop"]
            assert noop["n"] == 32 and noop["selection_explicitly_bypassed"] and noop["fusion_explicitly_disabled"]
            assert all(r["same_greedy_tokens"] for r in noop["comparisons"])
            report["noop"] = {"n": 32,
                               "max_logits_abs": max(r["logits_max_abs"] for r in noop["comparisons"]),
                               "all_inputs_positions_identical": all(
                                   z["same_byte_hash"] for r in noop["comparisons"]
                                   for k in ("input_ids", "position_ids") for z in r["prepared"][k])}
        if x.get("partial"):
            report["warnings"].append({"file": str(path), "partial_cells": list(x["partial"])})
        for tag, cell in x["results"].items():
            assert tag not in all_tags, (tag, "duplicate cell")
            all_tags.add(tag)
            rows = cell["requests"]
            ids = [(r["benchmark"], str(r["id"])) for r in rows]
            assert len(ids) == 200 and len(set(ids)) == 200
            assert sum(b == "textvqa" for b, _ in ids) == 100
            assert sum(b == "ocrbench" for b, _ in ids) == 100
            if shared_ids is None:
                shared_ids = ids
            assert ids == shared_ids, (tag, "ID/order mismatch")
            succeeded = [r for r in rows if r["status"] == "ok"]
            assert len(succeeded) == cell["n_success"]
            assert len(rows) - len(succeeded) == cell["n_failed"]
            assert cell["n_attempted"] == 200 and cell["actual_k_verified"]
            completed_time = 0.
            inconsistent = []
            for row in succeeded:
                raw = row["repeats_raw"]
                assert len(raw) == 3, (tag, row["id"], "missing repeats")
                if not row["answers_consistent"]:
                    inconsistent.append(row["id"])
                for rep in raw:
                    n, k = int(rep["n_image_full"]), int(rep["n_image_kept"])
                    expected = n if tag == "none" else max(1, round(n * cell["keep_fraction"]))
                    assert k == expected and rep["actual_k_verified"], (tag, row["id"], n, k, expected)
                    key = (row["benchmark"], row["id"])
                    if key not in full_counts:
                        full_counts[key] = n
                    assert full_counts[key] == n, (tag, row["id"], "different original input budget")
                    for layer in rep["layer_trace"]:
                        idx = layer["layer"]
                        active = n if tag == "none" else k
                        if tag.startswith("fastv") and idx <= 3:
                            active = n
                        if tag.startswith("hilo") and idx <= 2:
                            active = n
                        assert layer["n_image_active"] == active, (tag, row["id"], layer, active)
                    assert 0 < rep["ttft_s"] <= rep["total_s"]
                    assert rep["processor_cpu_s"] + rep["transfer_s"] <= rep["ttft_s"]
                    assert rep["out_len"] == len(rep["generated_token_ids"])
                    completed_time += rep["total_s"]
                for field in ("total_s", "ttft_s", "out_len", "n_image_kept"):
                    assert math.isclose(row[field], statistics.median(r[field] for r in raw), rel_tol=1e-12), (tag, row["id"], field)
            assert completed_time <= cell["workload_wall_s"] + .01, (tag, "workload wall too short")
            report["cells"][tag] = {"attempted": 200, "success": len(succeeded),
                                    "failed": cell["n_failed"], "actual_K_and_layer_trace": "PASS",
                                    "raw_repeats_and_medians": "PASS",
                                    "inconsistent_answer_ids": inconsistent}
    expected_tags = {"none"} | {f"{m}_{k}" for m in ("pre", "fastv", "hilo") for k in (.25, .5, .75)}
    report["missing_cells"] = sorted(expected_tags - all_tags)
    report["complete"] = not report["missing_cells"] and all(
        c["failed"] == 0 and not c["inconsistent_answer_ids"] for c in report["cells"].values())
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    report = validate(a.paths)
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
