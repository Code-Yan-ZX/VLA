#!/usr/bin/env python
"""Build the 500-sample TextVQA subset for P3-step-2 acc tightening.

Reuses build_subsets.build_textvqa (the streaming val loader) but overrides
the module-level N / TVQA_SCAN_CAP / TVQA_OUT so we get 500 deterministic
samples instead of 200. Same format (semicolon-joined GT answers), same
images dir (runs/data/textvqa/, shared with the 200 subset), seed=0.

CPU/network only. Output: eval/subsets/textvqa_500.jsonl
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_subsets as bs  # noqa: E402

bs.N = 500
bs.TVQA_SCAN_CAP = 1500
bs.TVQA_OUT = bs.ROOT / "eval/subsets/textvqa_500.jsonl"
recs = bs.build_textvqa()
bs.verify(recs, "textvqa500")
print("DONE")
