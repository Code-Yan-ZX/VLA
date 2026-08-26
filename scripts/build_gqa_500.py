#!/usr/bin/env python
"""Build the n=500 GQA subset (testdev pool, already downloaded).

Reuses build_subsets.build_gqa (the testdev_balanced loader) with N=500 and
points GQA_OUT at gqa_500.jsonl. Seed=0, identical sampler -> the n=200 ->
n=500 GQA comparison is clean (the n=200 set is the first 200 of the same
shuffled draw, modulo the per-row image check).

CPU only (testdev images already on disk). Output: eval/subsets/gqa_500.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_subsets as bs  # noqa: E402

bs.N = 500
bs.GQA_OUT = bs.ROOT / "eval/subsets/gqa_500.jsonl"
recs = bs.build_gqa()
bs.verify(recs, "gqa500")
print("DONE")
