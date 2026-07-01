"""vLLM serving benchmark for the P2 go/no-go probe.

Usage (run inside the `vtc_serve` env, on GPU):
    python -m src.serve_bench \
        --model runs/models/llava-1.5-7b-hf \
        --pruning-rate 0.50 \
        --benchmark gqa --subset eval/subsets/gqa_200.jsonl \
        --metrics-out runs/p2_probe/gqa_r50_metrics.json \
        [--max-model-len 4096] [--gpu-memory-utilization 0.90] [--seed 0] \
        [--limit N]            # first N samples only (0=all; for quick validation)

What it does (per notes/method-design.md §1):
  1. Forces the V0 vLLM engine (VLLM_USE_V1=0, set at module import) so the
     model runs in-process and PyTorch forward hooks can reach it.
  2. Loads LLaVA-1.5-7B in vLLM (offline LLM.chat path; same prefill/decode/
     KV-cache machinery as the server — we measure engine internals, not socket
     overhead).
  3. Installs the probe compressor (`ClsAttnSelector`) as a forward-hook on
     `LlavaMultiModalProjector` + a saliency-capture hook on the vision tower.
     At pruning_rate=0 the projector hook is a no-op (control) but still logs
     full token counts.
  4. Runs the benchmark subset, records per-request:
       served_tok_s, served_req_s, ttft_ms, peak_kv_mb, accuracy, answer.
  5. Writes aggregate (mean +/- stderr) + raw rows + hook fire stats to --metrics-out.

This file is import-safe WITHOUT vLLM (the vLLM import is lazy so the module can
be syntax-checked / arg-parsed on CPU). The actual GPU run is a queue job.
"""
from __future__ import annotations

# === FORCE V0 ENGINE (must run BEFORE any `import vllm`) =====================
# vLLM 0.10.2 defaults to the V1 engine, which runs the model in a SPAWNED
# subprocess -- main-process PyTorch forward hooks cannot reach it (the model
# never exists in the main process). V0 runs the model in-process, so the
# attribute chain llm_engine.model_executor.driver_worker.model_runner.model
# resolves and our hooks attach. V0 is still a valid continuous-batching serving
# engine (PagedAttention); if compression yields no wall-clock gain in V0 it
# won't in V1 either (V1 is more optimized -> less headroom), so V0 is a sound,
# slightly-favorable go/no-go testbed.
import os as _os
_os.environ.setdefault("VLLM_USE_V1", "0")

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
    """Construct a vLLM offline LLM (V0 engine, in-process) with the probe
    compressor hooked in. V0 is forced via VLLM_USE_V1=0 at module top."""
    import torch  # noqa
    import vllm  # noqa
    from vllm import LLM  # noqa  (lazy: CPU-import-safe)
    from vllm.model_executor.models.llava import LlavaMultiModalProjector  # noqa

    # V0 engine check (VLLM_USE_V1=0 set at module top before vllm import)
    from vllm.envs import VLLM_USE_V1
    print(f"[serve_bench] vllm={vllm.__version__} VLLM_USE_V1={VLLM_USE_V1} "
          f"(must be 0 / V0 for in-process hooks)", flush=True)
    if VLLM_USE_V1:
        raise RuntimeError(
            "VLLM_USE_V1 is True -- hooks cannot reach the spawned-subprocess "
            "model. Set os.environ['VLLM_USE_V1']='0' BEFORE importing vllm.")

    # allow loading subset images from local paths (file:// or bare path).
    # Subset JSONLs reference absolute paths under <repo>/runs/data/{gqa,textvqa}/;
    # anchor allowed_local_media_path at the repo root so all are covered.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    llm = LLM(
        model=model,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=_repo_root,
    )

    # ---- locate the projector + vision tower on the loaded model (V0 chain) ----
    # V0 runs the model in-process: llm_engine.model_executor.driver_worker.model_runner.model
    engine_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    projector: Optional[LlavaMultiModalProjector] = getattr(
        engine_model, "multi_modal_projector", None)
    vision_tower = getattr(engine_model, "vision_tower", None)
    if projector is None:
        raise RuntimeError(
            "multi_modal_projector not found on engine model -- wrong arch?")
    print(f"[serve_bench] hooks: projector={type(projector).__name__} "
          f"vision_tower={type(vision_tower).__name__}", flush=True)

    # ---- CLS-attention capture from vision tower ----
    captured = {"scores": None, "n_vision_calls": 0}

    def _vision_hook(module, inputs, outputs):  # noqa: ANN001
        import torch  # noqa
        hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") \
            else (outputs[0] if isinstance(outputs, tuple) else outputs)
        # hs: (B, 1+N, D). Score = norm of deviation from mean (foreground saliency).
        # (CLS-attention needs output_attentions=True which vLLM disables for speed;
        #  saliency-on-hidden-states is the cheap proxy. See method-design §1a/§4.)
        if hs.dim() == 3:
            patches = hs[:, 1:, :]                       # skip CLS
            sal = (patches - patches.mean(dim=1, keepdim=True)).norm(dim=-1)
            sal = sal / (sal.sum(dim=1, keepdim=True) + 1e-6)
            captured["scores"] = sal                    # (B, N)
            captured["n_vision_calls"] += 1
        return None

    # ---- projector post-hook: prune output rows ----
    hook_state = {"n_calls": 0, "kept_counts": []}

    def _projector_hook(module, inputs, output):  # noqa: ANN001
        hook_state["n_calls"] += 1
        if args.pruning_rate == 0.0:
            hook_state["kept_counts"].append(output.shape[1])
            return None  # control: no-op, but still log full token count
        scores = captured.get("scores")
        if scores is None:
            hook_state["kept_counts"].append(output.shape[1])
            return None  # no score yet (shouldn't happen post vision-tower)
        sel = ClsAttnSelector(pruning_rate=args.pruning_rate)
        kept, keep_idx = sel.select(output, scores)
        module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
        module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
        hook_state["kept_counts"].append(kept.shape[1])
        # log first few calls so the validation run visibly confirms pruning
        if hook_state["n_calls"] <= 5:
            print(f"[serve_bench] projector hook fire #{hook_state['n_calls']}: "
                  f"in={output.shape[1]} -> kept={kept.shape[1]} "
                  f"(prune_rate={args.pruning_rate})", flush=True)
        return kept

    if vision_tower is not None:
        # register on the inner vision model if wrapped
        inner = getattr(vision_tower, "vision_model", vision_tower)
        inner.register_forward_hook(_vision_hook)
    projector.register_forward_hook(_projector_hook)

    # stash hook_state on the projector for run() to read into the metrics file
    projector._vtc_hook_state = hook_state  # type: ignore[attr-defined]
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
    if getattr(args, "limit", None) and args.limit > 0:
        samples = samples[:args.limit]
    scorer = SCORERS[args.benchmark]

    llm, projector = build_engine(args.model, args)

    from vllm import SamplingParams  # noqa (lazy)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    raw = []
    import torch  # noqa
    torch.cuda.reset_peak_memory_stats()

    t_all_start = time.perf_counter()
    for s in samples:
        # Use llm.chat() with the OpenAI-style message format: the processor
        # applies the correct chat template and counts exactly one image (raw
        # "<image>\n..." prompts were double-counted by the multimodal validator
        # in some vLLM versions; chat() is the robust path, proven in the smoke test).
        # Local paths must be file:// URLs for vLLM's image loader.
        img_url = s.image
        if os.path.exists(img_url):
            img_url = "file://" + os.path.abspath(img_url)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": s.question},
            ],
        }]
        t0 = time.perf_counter()
        outputs = llm.chat(messages, sp, use_tqdm=False)
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

    hook_state = getattr(projector, "_vtc_hook_state", {})

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
        "hook": {
            "n_projector_calls": hook_state.get("n_calls", 0),
            "kept_counts_head": hook_state.get("kept_counts", [])[:10],
        },
        "raw": raw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(result, f, indent=2)
    kept_head = hook_state.get("kept_counts", [])[:5]
    print(f"[serve_bench] {args.benchmark} r={args.pruning_rate} n={len(raw)} "
          f"acc={agg['accuracy']:.3f} tok/s={agg['served_tok_s']['mean']:.1f} "
          f"ttft={agg['ttft_ms']['mean']:.0f}ms prefill_x={prefill_speedup} "
          f"e2e_x={e2e_speedup}", flush=True)
    print(f"[serve_bench] hook fired {hook_state.get('n_calls', 0)}x; "
          f"kept_counts(head)={kept_head} -> {args.metrics_out}", flush=True)
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
    ap.add_argument("--limit", type=int, default=0,
                    help="use only first N subset samples (0=all; for quick validation)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
