#!/usr/bin/env python3
"""Reproduce the pinned Hi-Lo checkout and the one documented upstream fix.

No environment installation or algorithm replacement is performed. Existing
checkouts with another revision or unexpected edits are rejected, not reset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN = "29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6"
URL = "https://github.com/sealost/Hi-Lo_Prune"
MODEL_FILE = "hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-dir", type=Path, default=Path("/tmp/Hi-Lo_Prune"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    repo = args.repo_dir.resolve()
    if not repo.exists():
        subprocess.run(["git", "clone", URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", PIN], check=True)
    if git(repo, "rev-parse", "HEAD").decode().strip() != PIN:
        raise SystemExit("Existing Hi-Lo checkout has another revision; use a new directory.")
    original = git(repo, "show", f"{PIN}:{MODEL_FILE}")
    # This block reads an undefined variable before the actual survivor gather.
    block = (b"\n\n                if top_index_1.numel() == 0:\n"
             b"                    keep_image_indices = keep_image_indices.to(torch.long)\n\n")
    if original.count(block) != 1:
        raise SystemExit("Pinned source does not contain the expected unique dead block.")
    patched = original.replace(block, b"")
    path = repo / MODEL_FILE
    actual = path.read_bytes()
    if actual not in (original, patched):
        raise SystemExit("Existing model file contains unexpected edits; refusing to overwrite.")
    changed = git(repo, "diff", "--name-only", "HEAD").decode().splitlines()
    if set(changed) - {MODEL_FILE}:
        raise SystemExit("Existing checkout contains other tracked edits; inspect them first.")
    path.write_bytes(patched)
    record = {
        "upstream": URL, "commit": PIN, "file": MODEL_FILE,
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "fix": "Remove undefined top_index_1 guard before survivor-index gather",
        "configuration_shim": "rope_parameters is supplied by cvm24h_hilo_eval.py",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
