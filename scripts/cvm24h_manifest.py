#!/usr/bin/env python3
"""Hash local CVM artifacts and export compact provenance, never raw datasets."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/cvm2027_supplement_20261026"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(path):
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    item = {"path": label, "bytes": path.stat().st_size,
            "sha256": digest(path)}
    if path.suffix == ".json":
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A hard-stopped end-of-run writer may leave an incomplete JSON.
            # Preserve its exact bytes and mark it unusable, never count it as
            # a completed prediction or omit it from the provenance record.
            item.update({"json_parse_valid": False,
                         "incomplete_or_invalid_json": str(exc)})
            return item
        if isinstance(d, dict) and "per_sample" in d:
            rows = d["per_sample"]
            ids = [str(r["id"]) for r in rows]
            item.update({"n_records": len(rows), "n_unique_ids": len(set(ids)),
                         "ordered_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                         "protocol": {k: d[k] for k in
                                      ("model", "mode", "engine", "benchmark", "seed", "max_pixels",
                                       "max_tokens", "mrope", "r", "r_pre", "fastv_k", "keep_ratio",
                                       "prune_layer", "candidate_ratio", "use_fusion", "attn", "patches")
                                      if k in d},
                         "failed_ids": [str(r["id"]) for r in rows
                                        if r.get("skipped") or r.get("error")],
                         "missing_kept_counts": sum(r.get("n_image_kept") is None for r in rows)})
    return item


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/cvm2027_artifact_manifest.json")
    args = ap.parse_args()
    paths = set((DATA / "predictions").glob("*.json"))
    paths.update((DATA / "matched_main").rglob("*.json"))
    paths.update((ROOT / "eval/full_splits").glob("*.jsonl"))
    score_cache = ROOT / "tmp/cvm_cpu_rescore_20261025"
    if score_cache.exists():
        paths.update(score_cache.glob("*reference.py"))
        annotation = score_cache / "TextVQA_0.5.1_val.json"
        if annotation.exists():
            paths.add(annotation)
    paths.update(p for p in (DATA / "plan").rglob("*")
                 if p.is_file() and p.suffix in (".json", ".jsonl", ".txt"))
    paths.update((ROOT / "scripts").glob("cvm24h*.py"))
    paths.update((ROOT / "scripts").glob("cvm24h*.sh"))
    paths.update((ROOT / "scripts").glob("analyze_cvm_*_official.py"))
    # Include original reused predictions, not only their curve slices/copies.
    for score in (DATA / "plan/scores").glob("*.json"):
        record = json.loads(score.read_text())
        for arm in ("A", "B"):
            source = record.get(arm, {}).get("source")
            if source and Path(source).is_file():
                paths.add(Path(source).resolve())
    snapshot = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
                / "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    if snapshot.exists():
        paths.update(snapshot.glob("*.json"))
        paths.update(snapshot.glob("*.jinja"))
    paths.add(ROOT / "src/v3_premerger/baselines_hf.py")
    paths.add(ROOT / "notes/cvm2027_server_experiment_prompt_20261025.txt")
    output = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "git_head_at_generation": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "model_snapshot": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "note": "Raw predictions, timing repeats and annotations remain local and ignored by Git. "
                "File hashes describe exact inputs/outputs; they do not prove protocol equivalence. "
                "Legacy timing_harness.json/timing_hilo.json are retained for audit and invalid for cost claims.",
        "artifacts": [describe(p) for p in sorted(paths)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.out}: {len(paths)} artifacts")


if __name__ == "__main__":
    main()
