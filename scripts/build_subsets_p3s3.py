#!/usr/bin/env python
"""Build the n=500 P3-step-3 subsets for accuracy tightening.

Same lmms-lab sources, seed=0, and MC-cleaning convention as the n=200
subsets (build_subsets_p3s2.py) -- we simply override N and the SCAN_CAP and
point the *_OUT paths at the *_500.jsonl files. The n=200 -> n=500 comparison
is therefore clean (identical sampler, same source order, deterministic seed).

Outputs (serve_bench.load_subset format):
  eval/subsets/mme_500.jsonl
  eval/subsets/mmbench_500.jsonl
  eval/subsets/scienceqa_500.jsonl

Images -> runs/data/{mme,mmbench,scienceqa}/<id>.jpg (shared with the n=200
subset dirs; new rows add new files, existing ones are untouched). CPU/network
only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_subsets_p3s2 as bs  # noqa: E402

bs.N = 500
bs.SCAN_CAP = 2000  # scan enough to collect 500 good rows (200 needed 800)
bs.MME_OUT = bs.ROOT / "eval/subsets/mme_500.jsonl"
bs.MMB_OUT = bs.ROOT / "eval/subsets/mmbench_500.jsonl"
bs.SQA_OUT = bs.ROOT / "eval/subsets/scienceqa_500.jsonl"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "mme", "mmbench", "scienceqa"])
    args = ap.parse_args()
    do = args.only
    if do in {"all", "mme"}:
        m = bs.build_mme(); bs.verify(m, "mme500")
    if do in {"all", "mmbench"}:
        b = bs.build_mmbench(); bs.verify(b, "mmbench500")
    if do in {"all", "scienceqa"}:
        s = bs.build_scienceqa(); bs.verify(s, "scienceqa500")
    print("DONE")
