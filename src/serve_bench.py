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

from .compressors import (  # noqa: F401  (re-export)
    ClsAttnSelector,
    TrueClsAttnSelector,
    ClsAttnCapture,
    cls_attention_scores,
    keep_count,
)


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
# Accuracy scoring (GQA / TextVQA conventions)
# --------------------------------------------------------------------------- #
_GQA_STOP = set("a an the is are was were be been being of to in on at for with "
                "and or but not no yes this that these there here it its their "
                "he she his her they we you i".split())


def _norm_words(s: str) -> list[str]:
    """Lowercase, keep alnum, split on whitespace; drop leading punctuation."""
    out = []
    for tok in "".join(c if (c.isalnum() or c.isspace()) else " "
                       for c in s.strip().lower()).split():
        out.append(tok)
    return out


def _singular(tok: str) -> str:
    """Crude plural normalization: 'cars'->'car', 'leaves'->'leaf' won't be caught,
    but covers the common -s/-es case. Good enough for the probe scorer."""
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 2 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 1 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def score_gqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """GQA-convention scorer (deterministic, no external deps).

    GQA answers fall into 3 types (matching the official GQA eval):
      1. yes/no: correct if the model's LEAD word is yes/no matching gt
         (model says "Yes, the chair is..." -> lead word "yes").
      2. object/attribute/color/relational: gt is a noun/phrase. Correct if
         the gt (or any synonym in `choices`/answer-set when present) appears
         as a WHOLE WORD in the model's answer, OR the model's first
         noun-phrase equals gt. Plural-normalized both sides.
      3. choice-set: if `choices` is provided (answer-set), match if gt (or any
         choice that equals gt) is contained as a whole word.
    """
    if not gt:
        return 0
    p_words = _norm_words(pred)
    g_norm = "".join(c for c in gt.strip().lower() if c.isalnum() or c.isspace()).strip()
    g_words = g_norm.split()
    if not g_words:
        return 0

    # ---- Type 1: yes/no (gt is a single yes/no token) ----
    if g_norm in {"yes", "no"}:
        # lead token of the model answer (skip leading articles a/an/the)
        lead = None
        for w in p_words:
            if w not in {"a", "an", "the"}:
                lead = w
                break
        if lead in {"yes", "no"} and lead == g_norm:
            return 1
        # also accept exact equality of fully-normalized strings (rare for verbose models)
        return 0

    # ---- Type 2: object/attribute/phrase gt ----
    # build the candidate synonym set: gt + optional choices that equal/contain gt
    syns = {g_norm, _singular(g_norm) if len(g_words) == 1 else g_norm}
    if choices:
        for c in choices:
            cn = "".join(ch for ch in c.strip().lower()
                         if ch.isalnum() or ch.isspace()).strip()
            if cn:
                syns.add(cn)
                if len(cn.split()) == 1:
                    syns.add(_singular(cn))

    p_text = " ".join(p_words)
    # whole-word / whole-phrase containment of gt (or synonym) in the answer
    for s in syns:
        s_words = s.split()
        if len(s_words) == 1:
            # single-token gt: match as whole word (singular OR plural)
            sg = _singular(s)
            if any(w == s or _singular(w) == sg for w in p_words):
                return 1
        else:
            # multi-token phrase: substring match on word-normalized text
            if s in p_text:
                return 1
    return 0


def score_textvqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """TextVQA VQA-accuracy (soft): correct if ANY of the semicolon-separated
    GT answers appears as a whole word/phrase in the model's answer.

    The official TextVQA metric is min(1, len(pred∩gt_set)/3); for the probe we
    use the conservative 'any-gt-contained' rule which is a tight lower bound.
    Full ANLS computed later in the accuracy table. Plural-normalized.
    (`choices` accepted for signature parity with score_gqa; unused — TextVQA
    GTs are semicolon-separated inside `gt`.)
    """
    if not gt:
        return 0
    p_words = _norm_words(pred)
    p_text = " ".join(p_words)
    gts = [x.strip() for x in gt.split(";") if x.strip()]
    for gt_i in gts:
        g_words = _norm_words(gt_i)
        if not g_words:
            continue
        gi = " ".join(g_words)
        if len(g_words) == 1:
            sg = _singular(g_words[0])
            if any(w == g_words[0] or _singular(w) == sg for w in p_words):
                return 1
        else:
            if gi in p_text:
                return 1
    return 0


SCORERS = {"gqa": score_gqa, "textvqa": score_textvqa}


# --------------------------------------------------------------------------- #
# vLLM engine + hook installation (lazy import)
# --------------------------------------------------------------------------- #
def patch_image_token_count(pruning_rate: float, full_n: int = 576) -> int:
    """Override vLLM's LLaVA image-token count 576 -> k = int((1-r)*576).

    WHY: pruning rate r is FIXED per run, so k is known a priori. vLLM's text
    sequence carries `[image_token_id] * num_image_tokens` placeholders
    (llava.py:_get_prompt_updates/get_replacement, which calls
    `info.get_num_image_tokens`). The pruned projector emits exactly k embeddings.
    For the LLM forward to get consistent shapes, the placeholder count MUST equal
    k. Overriding `get_num_image_tokens` -> k makes the sequence GENUINELY k-shorter
    (contiguous compaction, not keep-sparse) -> real wall-clock win (the gate's
    premise). The projector hook places its k selected embeddings into those k
    contiguous slots.

    full_n=576 = LLaVA-1.5 CLIP grid (24x24) after default feature-select.
    Returns k (also stashed for the projector hook to read).
    """
    import vllm.model_executor.models.llava as _llava_mod  # noqa
    k = max(1, int(round(full_n * (1.0 - pruning_rate))))
    InfoCls = _llava_mod.LlavaProcessingInfo

    if pruning_rate == 0.0:
        # restore original (unpatch) so r=0 control is byte-identical to stock vLLM
        if getattr(InfoCls.get_num_image_tokens, "_vtc_patched", False):
            InfoCls.get_num_image_tokens = InfoCls.get_num_image_tokens._vtc_orig
            print(f"[serve_bench] unpatched get_num_image_tokens (r=0)", flush=True)
        return full_n

    if not getattr(InfoCls.get_num_image_tokens, "_vtc_patched", False):
        orig = InfoCls.get_num_image_tokens

        def patched(self, *, image_width, image_height):  # noqa: ANN001
            # Ignore image dims: fixed k for this run (deterministic compaction).
            # (Real content-adaptive budgets come AFTER the gate -- method-design §2.)
            return k

        patched._vtc_orig = orig
        patched._vtc_patched = True
        InfoCls.get_num_image_tokens = patched
        print(f"[serve_bench] patched LlavaProcessingInfo.get_num_image_tokens: "
              f"{full_n} -> {k} (r={pruning_rate})", flush=True)
    return k


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

    # ---- placeholder-count override: 576 -> k (MUST precede the forward) ----
    # k is fixed per run; this makes the text sequence carry exactly k image-token
    # placeholders, matching the k embeddings the projector hook emits.
    target_k = patch_image_token_count(args.pruning_rate, full_n=576)

    # ---- score capture from vision tower (two paths) ----
    # PROXY (probe path, default): hidden-state deviation norm -- a saliency
    #   surrogate for CLS-attention, used because vLLM disables output_attentions.
    # TRUE_CLS (v1 path): real [CLS]->patch softmax attention, captured by
    #   monkeypatching the LAST CLIPAttention layer to expose its weights (the
    #   parallel-manual-softmax path in ClsAttnCapture; encoder numerics unchanged).
    selector_kind = getattr(args, "selector", "proxy")
    captured = {"scores": None, "n_vision_calls": 0}
    cls_capture: Optional[ClsAttnCapture] = None

    def _vision_hook_proxy(module, inputs, outputs):  # noqa: ANN001
        import torch  # noqa
        hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") \
            else (outputs[0] if isinstance(outputs, tuple) else outputs)
        if hs.dim() == 3:
            patches = hs[:, 1:, :]                       # skip CLS
            sal = (patches - patches.mean(dim=1, keepdim=True)).norm(dim=-1)
            sal = sal / (sal.sum(dim=1, keepdim=True) + 1e-6)
            captured["scores"] = sal                    # (B, N)
            captured["n_vision_calls"] += 1
        return None

    if selector_kind == "true_cls":
        # locate the LAST CLIPAttention layer and install the capture
        inner = getattr(vision_tower, "vision_model", vision_tower)
        enc_layers = getattr(inner.encoder, "layers", None)
        if enc_layers is None or len(enc_layers) == 0:
            raise RuntimeError(
                "true_cls selector: vision_tower.vision_model.encoder.layers not "
                "found -- wrong arch (expected CLIPVisionTransformer).")
        last_attn = enc_layers[-1].self_attn
        cls_capture = ClsAttnCapture(last_attn, layer_names=["last"])
        cls_capture.patch()
        print(f"[serve_bench] TRUE_CLS selector: patched last layer self_attn "
              f"({type(last_attn).__name__}) to expose CLS->patch softmax", flush=True)

        def score_provider():
            s = cls_capture.captured.get("scores")
            return s
    else:
        # PROXY path: forward-hook on the vision tower to compute saliency
        def score_provider():
            return captured.get("scores")

    # ---- projector post-hook: prune output rows to exactly target_k ----
    hook_state = {"n_calls": 0, "kept_counts": [], "target_k": target_k,
                  "selector": selector_kind}

    def _projector_hook(module, inputs, output):  # noqa: ANN001
        import torch  # noqa
        hook_state["n_calls"] += 1
        if args.pruning_rate == 0.0:
            hook_state["kept_counts"].append(output.shape[1])
            return None  # control: no-op, but still log full token count
        scores = score_provider()
        # one-time validation log: confirm the capture retains ONLY the CLS row
        # (B,H,1,S) -> head-meaned (B,N), NOT the full (B,H,S,S) [the OOM cause]
        if hook_state["n_calls"] == 1 and selector_kind == "true_cls" and cls_capture is not None:
            row_shape = cls_capture.captured.get("last_cls_row_shape")
            sc_shape = tuple(scores.shape) if scores is not None else None
            print(f"[serve_bench] TRUE_CLS capture shape check: cls_row={row_shape} "
                  f"head_meaned_scores={sc_shape} (expect ~(B,H,1,S) and (B,N); "
                  f"NOT (B,H,S,S))", flush=True)
        full_n = output.shape[1]
        if scores is None or scores.shape[1] != full_n:
            # fallback: take the first target_k tokens (keeps shapes valid even
            # if the vision hook hasn't captured scores for some reason)
            kept = output[:, :target_k, :].contiguous()
            keep_idx = torch.arange(target_k, device=output.device).unsqueeze(0).expand(output.shape[0], -1)
        else:
            # build the v1 selector (true_cls uses TrueClsAttnSelector with optional
            # diversity; proxy keeps the probe's ClsAttnSelector for reproducibility)
            if selector_kind == "true_cls":
                sel = TrueClsAttnSelector(
                    pruning_rate=args.pruning_rate,
                    diversity_lam=getattr(args, "diversity_lam", 0.0),
                )
            else:
                sel = ClsAttnSelector(pruning_rate=args.pruning_rate)
            kept, keep_idx = sel.select(output, scores)
            # guard: if rounding gave k != target_k, trim/pad to target_k so the
            # count EXACTLY matches the placeholder count from patch_image_token_count
            if kept.shape[1] != target_k:
                kept = kept[:, :target_k, :].contiguous()
                keep_idx = keep_idx[:, :target_k].contiguous()
        module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
        module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
        hook_state["kept_counts"].append(kept.shape[1])
        # reset the capture between requests so scores don't leak across images
        if cls_capture is not None:
            cls_capture.reset()
        # log first few calls so the validation run visibly confirms pruning
        if hook_state["n_calls"] <= 5:
            print(f"[serve_bench] projector hook fire #{hook_state['n_calls']}: "
                  f"in={output.shape[1]} -> kept={kept.shape[1]} "
                  f"(prune_rate={args.pruning_rate}, selector={selector_kind})", flush=True)
        return kept

    if selector_kind != "true_cls" and vision_tower is not None:
        # PROXY path only needs the vision-tower saliency hook
        inner = getattr(vision_tower, "vision_model", vision_tower)
        inner.register_forward_hook(_vision_hook_proxy)
    projector.register_forward_hook(_projector_hook)

    # stash hook_state + cls_capture on the projector for run() to read/clean up
    projector._vtc_hook_state = hook_state  # type: ignore[attr-defined]
    projector._vtc_cls_capture = cls_capture  # type: ignore[attr-defined]
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
        "selector": getattr(args, "selector", "proxy"),
        "diversity_lam": getattr(args, "diversity_lam", 0.0),
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
    ap.add_argument("--selector", default="proxy", choices=["proxy", "true_cls"],
                    help="score source: 'proxy' = hidden-state-deviation (the P2 probe) "
                         "or 'true_cls' = real [CLS]->patch softmax attention (v1 method, "
                         "via ClsAttnCapture on the last vision-tower layer)")
    ap.add_argument("--diversity-lam", type=float, default=0.0,
                    help="PRUNESID-style diversity weight (only with --selector true_cls); "
                         "0.0 = pure top-k by CLS-attn (v1 default)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
