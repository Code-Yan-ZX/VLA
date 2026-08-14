#!/usr/bin/env python3
"""Validate the transferred CVPR figure-data bundle before rendering."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "server_exports" / "cvpr_figure_data_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf_normalized(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_checksums() -> int:
    checked = 0
    mismatches: list[str] = []
    for line in (BUNDLE / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = BUNDLE / relative
        assert path.is_file(), f"missing checksummed file: {relative}"
        actual = sha256(path)
        if actual != expected and sha256_lf_normalized(path) != expected:
            mismatches.append(f"{relative}: expected {expected}, got {actual}")
        checked += 1
    assert not mismatches, "checksum mismatches: " + "; ".join(mismatches)
    return checked


def check_case(case_dir: Path) -> dict[str, object]:
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    required = {
        "benchmark",
        "sample_id",
        "question",
        "ground_truth",
        "answers",
        "correctness",
        "keep_ratio",
        "n_units_full",
        "final_visual_tokens",
        "iso_token_contract",
    }
    assert required <= meta.keys(), f"{case_dir.name}: missing case fields"
    assert meta["iso_token_contract"]["holds"] is True

    arrays: dict[str, dict[str, tuple[int, ...]]] = {}
    for filename in ("pre_scores.npz", "post_scores.npz", "fastv_scores.npz", "aux.npz"):
        with np.load(case_dir / filename, allow_pickle=False) as bundle:
            arrays[filename] = {key: tuple(bundle[key].shape) for key in bundle.files}

    n_units = int(meta["n_units_full"])
    expected_kept = round(float(meta["keep_ratio"]) * n_units)
    for method, filename in (
        ("rbm", "pre_scores.npz"),
        ("post_l2", "post_scores.npz"),
        ("fastv_k3", "fastv_scores.npz"),
    ):
        with np.load(case_dir / filename, allow_pickle=False) as bundle:
            keep_keys = [
                key for key in bundle.files if key.startswith("keep") and "indices" not in key
            ]
            index_keys = [key for key in bundle.files if "kept_indices" in key]
            assert keep_keys, f"{case_dir.name}/{filename}: missing keep array"
            assert index_keys, f"{case_dir.name}/{filename}: missing kept indices"
            keep = np.asarray(bundle[keep_keys[0]], dtype=bool).reshape(-1)
            indices = np.asarray(bundle[index_keys[0]], dtype=int).reshape(-1)
            assert keep.size >= n_units, (
                f"{case_dir.name}/{method}: padded mask shorter than unit count"
            )
            assert not keep[n_units:].any(), f"{case_dir.name}/{method}: nonzero padded cells"
            keep = keep[:n_units]
            assert int(keep.sum()) == expected_kept, f"{case_dir.name}/{method}: wrong kept count"
            assert np.array_equal(np.flatnonzero(keep), np.sort(indices)), (
                f"{case_dir.name}/{method}: mask/index disagreement"
            )
            assert np.all((indices >= 0) & (indices < n_units)), (
                f"{case_dir.name}/{method}: kept index out of range"
            )
            assert int(meta["final_visual_tokens"][method]) == expected_kept

    return {
        "case": case_dir.name,
        "n_units": n_units,
        "kept": expected_kept,
        "grid": tuple(meta["unit_grid_hw"]),
        "arrays": arrays,
    }


def check_aggregates() -> dict[str, int]:
    counts = {}
    for path in sorted((BUNDLE / "aggregate").glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows, f"empty aggregate: {path.name}"
        counts[path.name] = len(rows)
    return counts


def main() -> None:
    assert BUNDLE.is_dir(), f"bundle not found: {BUNDLE}"
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    gates = manifest["validation_gates"]
    assert all(item["pass"] for item in gates.values()), "server validation gate failed"

    checked = check_checksums()
    case_summaries = [check_case(path) for path in sorted((BUNDLE / "cases").iterdir()) if path.is_dir()]
    assert len(case_summaries) == 6, "expected six selected cases"
    aggregate_counts = check_aggregates()

    mechanism_rows = sum(1 for _ in (BUNDLE / "mechanism" / "mechanism_per_image.csv").open("r", encoding="utf-8")) - 1
    assert mechanism_rows == 391, f"expected 391 mechanism rows, found {mechanism_rows}"

    print(f"PASS: {checked} checksums; {len(case_summaries)} cases; {mechanism_rows} mechanism rows")
    print(f"aggregate rows: {aggregate_counts}")
    for summary in case_summaries:
        print(f"{summary['case']}: grid={summary['grid']} units={summary['n_units']} kept={summary['kept']}")
        for filename, schema in summary["arrays"].items():
            print(f"  {filename}: {schema}")


if __name__ == "__main__":
    main()
