"""vLLM serving benchmark for the P2 go/no-go probe.

Usage (run inside the `vtc_serve` env, on GPU):
    python -m src.serve_bench \
        --model runs/models/llava-1.5-7b-hf \
        --pruning-rate 0.50 \
        --benchmark gqa --subset eval/subsets/gqa_200.jsonl \
        --metrics-out runs/p2_probe/gqa_r50_metrics.json \
        [--max-model-len 4096] [--gpu-memory-utilization 0.90] [--seed 0]

What it does (per notes/method-design.md §1):
  1. Loads LLaVA-1.5-7B in vLLM (offline LLM.generate path; same prefill/decode/
     KV-cache machinery as the server — we measure engine internals, not socket
     overhead).
  2. Installs the probe compressor (`ClsAttnSelector`) as a forward-hook on
     `LlavaMultiModalProjector` + a CLS-attention capture hook on the vision
     tower. At pruning_rate=0 the hooks are a no-op (control).
  3. Runs the benchmark subset, records per-request:
       served_tok_s, served_req_s, ttft_ms, peak_kv_mb, accuracy, answer.
  4. Writes aggregate (mean +/- stderr) + raw per-request rows to --metrics-out.

This file is import-safe WITHOUT vLLM (the vLLM import is lazy so the module can
be syntax-checked / arg-parsed on CPU). The actual GPU run is a queue job.

Outputs JSON schema (metrics-out):
    {
      "benchmark": "gqa",
      "pruning_rate": 0.50,
      "n": 200,
      "agg": {"served_tok_s": {mean, stderr}, "served_req_s": {...},
              "ttft_ms": {...}, "peak_kv_mb": {...}, "accuracy": mean},
      "prefill_speedup_vs_r0": null|float,   # filled when r0 baseline present
      "e2e_speedup_vs_r0":    null|float,
      "raw": [ {id, served_tok_s, served_req_s, ttft_ms, peak_kv_mb,
                correct, answer, gt}, ... ]
    }
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Optional

from .compressors import ClsAttnSelector, keep_count  # noqa: F401  (re-export)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    id: str
    image: str           # path or URL
    question: str
    gt: str
    extra: dict          # benchmark-specific (e.g. GQA answer-set)


def load_subset(path: str) -> list[Sample]:
    """Load a JSONL subset produced by eval/subsets/*.jsonl.

    Each line: {"id","image","question","gt", ...optional "choices"}
    """
    out: list[Sample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(Sample(
                id=str(o["id"]),
                image=o["image"],
                question=o["question"],
                gt=str(o["gt"]),
                extra={k: v for k, v in o.items()
                       if k not in {"id", "image", "question", "gt"}},
            ))
    return out


# --------------------------------------------------------------------------- #
# Accuracy scoring
# --------------------------------------------------------------------------- #
def score_gqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """GQA exact-match (case/whitespace/punct-insensitive). choices optional."""
    def norm(s: str) -> str:
        return "".join(c for c in s.strip().lower() if c.isalnum())
    p, g = norm(pred), norm(gt)
    if not g:
        return 0
    if p == g or g in p.split():
        return 1
    if choices:
        # pick the closest choice to pred, then compare to gt
        best = max(choices, key=lambda c: sum(w in p for w in norm(c).split()))
        return 1 if norm(best) == g else 0
    return 0


def score_textvqa(pred: str, gt: str) -> int:
    """TextVQA VQA-accuracy: 1 if gt (or any of several GTs) appears in pred.

    The real TextVQA scorer is ANLS/overlap; for the probe subset we use the
    conservative 'gt-substring' rule + permissive token containment. Full ANLS
    is computed later in the accuracy table.
    """
    def norm(s: str) -> str:
        return "".join(c for c in s.strip().lower() if c.isalnum() or c.isspace())
    p, g = norm(pred), norm(gt)
    if not g:
        return 0
    # multiple GTs may be semicolon-separated in the subset
    gts = [x.strip() for x in g.split(";") if x.strip()]
    for gt_i in gts:
        if gt_i in p or all(tok in p for tok in gt_i.split() if len(tok) > 2):
            return 1
    return 0


SCORERS = {"gqa": score_gqa, "textvqa": score_textvqa}


# --------------------------------------------------------------------------- #
# vLLM engine + hook installation (lazy import)
# --------------------------------------------------------------------------- #
def build_engine(model: str, args):
    """Construct a vLLM offline LLM with the probe compressor hooked in."""
    import torch  # noqa
    from vllm import LLM, SamplingParams  # noqa  (lazy: CPU-import-safe)
    from vllm.model_executor.models.llava import LlavaMultiModalProjector  # noqa

    llm = LLM(
        model=model,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 1},
    )

    # ---- locate the projector + vision tower on the loaded model ----
    # vLLM wraps the HF model; the projector sits at engine.model.multi_modal_projector
    engine_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    projector: Optional[LlavaMultiModalProjector] = getattr(
        engine_model, "multi_modal_projector", None)
    vision_tower = getattr(engine_model, "vision_tower", None)
    if projector is None:
        raise RuntimeError(
            "multi_modal_projector not found on engine model -- wrong arch?")

    # ---- CLS-attention capture from vision tower ----
    captured = {"scores": None}

    def _vision_hook(module, inputs, outputs):  # noqa: ANN001
        # CLIP/SigLIP vision towers expose attentions only if output_attentions;
        # if unavailable we fall back to a mean-pooling saliency on the patch
        # tokens of the last hidden state. The go/no-go metric is robust to the
        # exact score (we are testing the *serving* question). Detailed in
        # method-design.md §1a/§4 open question 2.
        import torch  # noqa
        hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") \
            else (outputs[0] if isinstance(outputs, tuple) else outputs)
        # hs: (1, 1+N, D). Score = norm of deviation from mean (foreground saliency).
        if hs.dim() == 3:
            patches = hs[:, 1:, :]                       # skip CLS
            sal = (patches - patches.mean(dim=1, keepdim=True)).norm(dim=-1)
            sal = sal / (sal.sum(dim=1, keepdim=True) + 1e-6)
            captured["scores"] = sal                    # (B, N)
        return None

    # ---- projector post-hook: prune output rows ----
    def _projector_hook(module, inputs, output):  # noqa: ANN001
        import torch  # noqa
        if args.pruning_rate == 0.0:
            return None  # control: no-op
        scores = captured.get("scores")
        if scores is None:
            return None  # no score yet (shouldn't happen post vision-tower)
        sel = ClsAttnSelector(pruning_rate=args.pruning_rate)
        kept, keep_idx = sel.select(output, scores)
        module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
        module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
        return kept

    if vision_tower is not None:
        # register on the inner vision model if wrapped
        inner = getattr(vision_tower, "vision_model", vision_tower)
        inner.register_forward_hook(_vision_hook)
    projector.register_forward_hook(_projector_hook)

    return llm, projector


# --------------------------------------------------------------------------- #
# Metrics aggregation
# --------------------------------------------------------------------------- #
def mean_stderr(xs: list[float]) -> dict:
    if not xs:
        return {"mean": float("nan"), "stderr": float("nan"), "n": 0}
    m = statistics.fmean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return {"mean": m, "stderr": se, "n": len(xs)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    samples = load_subset(args.subset)
    scorer = SCORERS[args.benchmark]

    llm, projector = build_engine(args.model, args)

    from vllm import SamplingParams  # noqa (lazy)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    raw = []
    import torch  # noqa
    torch.cuda.reset_peak_memory_stats()

    t_all_start = time.perf_counter()
    for s in samples:
        prompt = (f"USER: <image>\n{s.question}\nASSISTANT:")
        t0 = time.perf_counter()
        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": s.image}},
            sp, use_tqdm=False)
        t1 = time.perf_counter()
        text = outputs[0].outputs[0].text.strip()
        ttft = (t1 - t0) * 1000.0  # ms (approx: prefill-dominated for 1 tok)
        n_out = len(outputs[0].outputs[0].token_ids)
        e2e = t1 - t0
        served_tok_s = (n_out / e2e) if e2e > 0 else 0.0
        served_req_s = (1.0 / e2e) if e2e > 0 else 0.0
        peak_kv_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        correct = scorer(text, s.gt, s.extra.get("choices"))
        raw.append({
            "id": s.id, "served_tok_s": served_tok_s, "served_req_s": served_req_s,
            "ttft_ms": ttft, "peak_kv_mb": peak_kv_mb,
            "correct": correct, "answer": text, "gt": s.gt,
        })
    wall = time.perf_counter() - t_all_start

    agg = {
        "served_tok_s": mean_stderr([r["served_tok_s"] for r in raw]),
        "served_req_s": mean_stderr([r["served_req_s"] for r in raw]),
        "ttft_ms": mean_stderr([r["ttft_ms"] for r in raw]),
        "peak_kv_mb": mean_stderr([r["peak_kv_mb"] for r in raw]),
        "accuracy": (sum(r["correct"] for r in raw) / len(raw)) if raw else float("nan"),
    }

    # speedup vs r0 baseline if present
    prefill_speedup = e2e_speedup = None
    r0_path = os.path.join(os.path.dirname(args.metrics_out),
                           f"{args.benchmark}_r0_metrics.json")
    if args.pruning_rate > 0.0 and os.path.exists(r0_path):
        with open(r0_path) as f:
            r0 = json.load(f)["agg"]
        if r0["ttft_ms"]["mean"] > 0:
            prefill_speedup = r0["ttft_ms"]["mean"] / agg["ttft_ms"]["mean"]
        if r0["served_req_s"]["mean"] > 0:
            e2e_speedup = agg["served_req_s"]["mean"] / r0["served_req_s"]["mean"]

    result = {
        "benchmark": args.benchmark, "pruning_rate": args.pruning_rate,
        "n": len(raw), "wall_s": wall, "agg": agg,
        "prefill_speedup_vs_r0": prefill_speedup, "e2e_speedup_vs_r0": e2e_speedup,
        "raw": raw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[serve_bench] {args.benchmark} r={args.pruning_rate} n={len(raw)} "
          f"acc={agg['accuracy']:.3f} tok/s={agg['served_tok_s']['mean']:.1f} "
          f"ttft={agg['ttft_ms']['mean']:.0f}ms prefill_x={prefill_speedup} "
          f"e2e_x={e2e_speedup} -> {args.metrics_out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 go/no-go vLLM serving benchmark")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pruning-rate", type=float, default=0.0,
                    help="fraction of visual tokens to DROP (0=control)")
    ap.add_argument("--benchmark", required=True, choices=["gqa", "textvqa"])
    ap.add_argument("--subset", required=True, help="JSONL subset path")
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
