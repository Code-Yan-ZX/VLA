"""OCRBench mechanism capture for Export B (mechanism_per_image.csv).

Runs the SAME capture machinery as scripts/mechanism_token_survival.py
(`--mode capture --bench ocrbench`) into the bundle tmp dir, keeping the
identical conditions as the existing docvqa/textvqa/gqa survival captures:
  * model Qwen/Qwen3-VL-8B-Instruct, bf16, enforce_eager
  * mm_processor_kwargs max_pixels=1500000  (same as the 3 existing benches)
  * fixed seed, subset eval/subsets/ocrbench_200.jsonl, r=0.75 (keep 25%)
  * capture-only prefill (max_tokens=1, no generation metric)
  * per-unit PRE (deepstack[0] input) L2, POST (merged row) L2, Sobel edge
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_token_survival import (  # noqa: E402
    wrap_capture, reset, capture_bench, sample_indices,
)

ROOT = Path("/media/disk2/YZX/research/vla")
OUT = ROOT / "drafts/figures/server_exports/tmp/survival_ocrbench"


def main():
    import os
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vllm import LLM, SamplingParams
    from mechanism_token_survival import MODEL, MAX_PIXELS, R
    llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.90, max_model_len=32768,
              trust_remote_code=False, enforce_eager=True,
              limit_mm_per_prompt={"image": 1},
              allowed_local_media_path=str(ROOT), max_num_seqs=4,
              enable_prefix_caching=False, seed=0,
              max_num_batched_tokens=32768,
              mm_processor_kwargs={"max_pixels": MAX_PIXELS})
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    cap = wrap_capture(model.visual)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    meta = capture_bench(llm, cap, sp, "ocrbench", 200, 0, R, OUT)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"DONE -> {OUT}/ocrbench.npz")


if __name__ == "__main__":
    main()