#!/usr/bin/env python3
"""Build HELD-OUT eval slices for the D1 learned-scorer validation.

Training/capture uses eval/subsets/{bench}_200.jsonl (subset A). This builds
a disjoint 200-sample slice from the full split (subset B) with the same
fixed seed, so the deployed learned scorer is evaluated on images it never
saw (honest generalization). Also writes the image list in serve_bench
load_subset format: {"id","image","question","gt",[choices]}.

Usage: python scripts/build_learned_heldout.py [n]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path("/media/disk2/YZX/research/vla")
OUT = ROOT / "eval/learned_heldout"
NAME = {"textvqa": "textvqa_val", "docvqa": "docvqa_val",
        "gqa": "gqa_testdev"}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for bench, fname in NAME.items():
        ids_a = {json.loads(ln)["id"]
                 for ln in open(ROOT / "eval/subsets" / f"{bench}_200.jsonl")}
        cands = [json.loads(ln)
                 for ln in open(ROOT / "eval/full_splits" / f"{fname}.jsonl")
                 if json.loads(ln)["id"] not in ids_a]
        rng = random.Random(seed)
        rng.shuffle(cands)
        picked = cands[:n]
        with open(OUT / f"{bench}_{n}.jsonl", "w") as f:
            for s in picked:
                f.write(json.dumps(s) + "\n")
        print(f"{bench}: held-out {len(picked)} (from {len(cands)} candidates), "
              f"ids={[s['id'] for s in picked[:3]]}...")


if __name__ == "__main__":
    main()
