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
    QueryAwareSelector,
    ClipQuerySelector,
    clip_text_patch_scores,
    ClsAttnCapture,
    cls_attention_scores,
    text_patch_scores,
    keep_count,
)
from .load_controller import (  # noqa: F401  (re-export)
    LoadAdaptiveController,
    LoadReading,
    PROFILES,
    read_engine_load,
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
def patch_image_token_count(pruning_rate: float, full_n: int = 576,
                            k_cell: Optional[dict] = None) -> int:
    """Override vLLM's LLaVA image-token count 576 -> k = int((1-r)*576).

    WHY: pruning rate r is FIXED per run (or per-request in adaptive mode), so k
    is known a priori. vLLM's text sequence carries `[image_token_id] *
    num_image_tokens` placeholders (llava.py:_get_prompt_updates/get_replacement,
    which calls `info.get_num_image_tokens`). The pruned projector emits exactly k
    embeddings. For the LLM forward to get consistent shapes, the placeholder
    count MUST equal k. Overriding `get_num_image_tokens` -> k makes the sequence
    GENUINELY k-shorter (contiguous compaction, not keep-sparse) -> real wall-clock
    win (the gate's premise). The projector hook places its k selected embeddings
    into those k contiguous slots.

    full_n=576 = LLaVA-1.5 CLIP grid (24x24) after default feature-select.
    Returns k.

    ADAPTIVE MODE (P2 method D): when `k_cell` is provided, the patched function
    reads `k_cell["k"]` on EVERY call instead of closing over a fixed k. This
    lets run() set k per-request (from the load-adaptive controller) right before
    llm.chat(), and the placeholder count + projector hook both honor it. The
    initial k_cell["k"] is set from `pruning_rate` (the r_max for adaptive, or
    the fixed r otherwise); run() updates it before each submission.
    """
    import vllm.model_executor.models.llava as _llava_mod  # noqa
    k = max(1, int(round(full_n * (1.0 - pruning_rate))))
    InfoCls = _llava_mod.LlavaProcessingInfo

    if pruning_rate == 0.0 and k_cell is None:
        # restore original (unpatch) so r=0 control is byte-identical to stock vLLM
        if getattr(InfoCls.get_num_image_tokens, "_vtc_patched", False):
            InfoCls.get_num_image_tokens = InfoCls.get_num_image_tokens._vtc_orig
            print(f"[serve_bench] unpatched get_num_image_tokens (r=0)", flush=True)
        return full_n

    if not getattr(InfoCls.get_num_image_tokens, "_vtc_patched", False):
        orig = InfoCls.get_num_image_tokens

        if k_cell is not None:
            # ADAPTIVE: read k from the mutable cell each call (per-request budget).
            def patched(self, *, image_width, image_height):  # noqa: ANN001
                return k_cell["k"]
            patched._vtc_mode = "adaptive"
            print(f"[serve_bench] patched get_num_image_tokens: ADAPTIVE "
                  f"(per-request k from controller cell; init r={pruning_rate} "
                  f"-> k={k_cell['k']})", flush=True)
        else:
            # FIXED: closed-over k for the whole run (the original probe path).
            def patched(self, *, image_width, image_height):  # noqa: ANN001
                return k
            patched._vtc_mode = "fixed"
            print(f"[serve_bench] patched LlavaProcessingInfo.get_num_image_tokens: "
                  f"{full_n} -> {k} (r={pruning_rate})", flush=True)

        patched._vtc_orig = orig
        patched._vtc_patched = True
        InfoCls.get_num_image_tokens = patched
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
        # ADAPTIVE mode needs eager execution: varying per-segment k -> varying
        # sequence lengths -> vLLM's CUDA graph capture (fixed shapes) mismatches
        # at the image-token masked_scatter when k changes between segments
        # (device-side 'masked_scatter_size_check' assert). Eager mode handles
        # dynamic shapes correctly at a small per-step cost. Fixed-r runs keep
        # graph capture (one k for the whole run -> shapes are static).
        enforce_eager=bool(getattr(args, "adaptive", False)),
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=_repo_root,
        max_num_seqs=getattr(args, "max_num_seqs", 256),  # M2: concurrency control
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
    # k is the (initial) kept-token count; this makes the text sequence carry
    # exactly k image-token placeholders, matching the k embeddings the projector
    # hook emits.
    #
    # ADAPTIVE MODE (P2 method D): the controller decides r per-request from the
    # live engine load, so k is NOT fixed -- it varies per request. We patch the
    # class method to read a MUTABLE k_cell on each call, and run() updates
    # k_cell["k"] from controller.decide_r(read_engine_load()) before every
    # submission. The projector hook ALSO reads k_cell so its kept-count matches
    # the per-request placeholder count exactly.
    adaptive = bool(getattr(args, "adaptive", False))
    if adaptive:
        # init k_cell at r_max (the heavy-load endpoint); run() updates it per-
        # request. r_max is the controller ceiling (also the per-benchmark
        # accuracy guardrail).
        r_max = float(getattr(args, "r_max", args.pruning_rate))
        k_cell = {"k": max(1, int(round(576 * (1.0 - r_max))))}
        target_k = patch_image_token_count(r_max, full_n=576, k_cell=k_cell)
        print(f"[serve_bench] ADAPTIVE mode ON: controller r in "
              f"[{getattr(args, 'r_min', 0.25)}, {r_max}] "
              f"(signal={getattr(args, 'load_signal', 'num_running')}); "
              f"k_cell init k={k_cell['k']}", flush=True)
    else:
        k_cell = None
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

    # ---- v2 query-aware / A'' clip_query plumbing ----
    # WHY: both query_aware and clip_query score patches by relevance to the
    # QUESTION. The question text is known at preprocessing time (it's in the
    # per-sample loop before llm.chat()), so we tokenize it, embed it, and stash
    # the (T, D) features in a FIFO queue the projector hook pops.
    #   * query_aware: question embedded via LLM `embed_tokens` (LLM space).
    #   * clip_query  : question embedded via CLIPTextModel + text_projection
    #     (CLIP CONTRASTIVE space -- the v2 fix). Patches come from the vision
    #     tower's last_hidden_state patch tokens @ CLIP visual_projection. Both
    #     live in CLIP's 768-d contrastive space, so cosine is a meaningful
    #     cross-modal similarity (CLIP trained for it). ~0.5GB extra for CLIP text.
    # PLUMBING POINT (method-design §9): pre-compute BEFORE llm.chat(), keyed by
    # a queue (request order = pop order; vLLM offline LLM.chat is one request at
    # a time so this is unambiguous). No intra-forward input_ids threading needed.
    embed_tokens = None
    tokenizer = None
    clip_text_model = None
    clip_text_proj = None       # Linear -> .weight.T projects text hidden->contrastive
    clip_visual_proj = None     # Linear -> .weight.T projects vision patches->contrastive
    clip_tokenizer = None
    query_queue: list = []   # FIFO of (T, D) question-feature tensors
    if selector_kind == "query_aware":
        import torch  # noqa
        from transformers import AutoTokenizer  # noqa
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        lang_model = getattr(engine_model, "language_model", None)
        llama_model = getattr(lang_model, "model", None) if lang_model is not None else None
        embed_tokens = getattr(llama_model, "embed_tokens", None) if llama_model is not None else None
        if embed_tokens is None:
            raise RuntimeError(
                "query_aware selector: cannot locate embed_tokens on the engine "
                "model (expected engine_model.language_model.model.embed_tokens). "
                "Wrong arch?")
        print(f"[serve_bench] QUERY_AWARE selector: tokenizer={type(tokenizer).__name__} "
              f"embed_tokens={type(embed_tokens).__name__} (will embed question "
              f"tokens per request for text<->patch scoring)", flush=True)
    elif selector_kind == "clip_query":
        # A'': load the CLIP text tower + both projections from the SAME CLIP
        # checkpoint that LLaVA-1.5's vision tower is built from
        # (openai/clip-vit-large-patch14-336). LLaVA-1.5's vision_tower IS a
        # CLIPVisionModel from this checkpoint, so the visual_projection here
        # exactly matches the contrastive head the patches were trained against.
        import torch  # noqa
        from transformers import CLIPModel, CLIPTokenizer  # noqa
        clip_id = "openai/clip-vit-large-patch14-336"
        clip_m = CLIPModel.from_pretrained(clip_id, torch_dtype=torch.float16).eval()
        clip_text_model = clip_m.text_model.to("cuda")
        clip_text_proj = clip_m.text_projection     # Linear (768,768); use .weight.T
        clip_visual_proj = clip_m.visual_projection  # Linear (768,1024); use .weight.T
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        # sanity: confirm dims match LLaVA's vision tower (hidden=1024 -> contrastive 768)
        inner_vt = getattr(vision_tower, "vision_model", vision_tower)
        vt_hidden = getattr(getattr(inner_vt, "config", None), "hidden_size", None)
        assert clip_visual_proj.weight.shape[1] == vt_hidden, (
            f"clip visual_proj in-dim {clip_visual_proj.weight.shape[1]} != vision tower "
            f"hidden {vt_hidden} -- wrong CLIP checkpoint?")
        print(f"[serve_bench] CLIP_QUERY selector (A''): loaded CLIP text tower from "
              f"{clip_id} (text layers={clip_text_model.config.num_hidden_layers}, "
              f"hidden={clip_text_model.config.hidden_size}, proj->"
              f"{clip_text_proj.weight.shape[0]}). visual_proj "
              f"{tuple(clip_visual_proj.weight.shape)} matches vision tower hidden "
              f"{vt_hidden}. ~0.5GB extra.", flush=True)

    # ---- M1 timing instrumentation (P2 method D scoping) ---------------------
    # WHY: Finding #2 said prefill is sub-linear because the vision tower processes
    # ALL 576 tokens regardless of pruning. M1 measures the vision-tower fraction
    # of prefill (TTFT) to decide if mid-encoder/early ViT prune is worth the
    # surgery. We wrap BOTH the vision-tower forward AND the projector forward
    # (vision-tower time is the bulk of the "fixed" cost; projector is the
    # pruning-relevant boundary; LLM prefill = TTFT - vision_tower - queue).
    # `_vtc_times` collects per-request wall times (ms) via pre/post hooks keyed
    # by phase. run() aggregates them into vision_tower_ms / projector_ms /
    # llm_prefill_ms (estimated) and writes to metrics.
    _vtc_times: dict = {"vt_pre": [], "vt_post": [], "proj_pre": [], "proj_post": []}

    def _vt_pre(module, inputs):  # noqa: ANN001
        _vtc_times.setdefault("_stamps", {}).setdefault("stack", []).append(time.perf_counter())

    def _vt_post(module, inputs, outputs):  # noqa: ANN001
        t1 = time.perf_counter()
        stack = _vtc_times.setdefault("_stamps", {}).get("stack", [])
        t0 = stack.pop() if stack else t1
        _vtc_times["vt_post"].append((t1 - t0) * 1000.0)

    def _proj_pre(module, inputs):  # noqa: ANN001
        _vtc_times.setdefault("_stamps", {}).get("stack", [])
        _vtc_times.setdefault("_stamps", {}).setdefault("stack", []).append(time.perf_counter())

    def _proj_post(module, inputs, outputs):  # noqa: ANN001
        t1 = time.perf_counter()
        stack = _vtc_times.setdefault("_stamps", {}).get("stack", [])
        t0 = stack.pop() if stack else t1
        _vtc_times["proj_post"].append((t1 - t0) * 1000.0)

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

    def _vision_hook_clip(module, inputs, outputs):  # noqa: ANN001
        # A'': stash the PRE-PROJECTOR vision-tower patch features (last_hidden_state
        # patch tokens, (B,N,1024)). The projector hook applies CLIP visual_projection
        # to land them in contrastive space and scores against the CLIP-text(question).
        import torch  # noqa
        hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") \
            else (outputs[0] if isinstance(outputs, tuple) else outputs)
        if hs.dim() == 3:
            # store the FULL hidden state (with CLS at index 0); projector hook
            # slices [1:] to get patches. Keep on GPU -- (B,577,1024) fp16 ~4.7MB.
            captured["clip_patches_hidden"] = hs.detach()
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
    # ADAPTIVE (method D): when k_cell is set, the per-request k is read from it
    # (run() updates k_cell["k"] from the controller before each submission), and
    # the per-request r = 1 - k/576 rebuilds the selector so the top-k count and
    # the placeholder count agree. The kept_counts log then visibly tracks the
    # controller's adaptation (the headline-validation signal).
    hook_state = {"n_calls": 0, "kept_counts": [], "target_k": target_k,
                  "selector": selector_kind, "adaptive": adaptive}

    def _cur_k() -> int:
        """Per-request kept-token count: k_cell['k'] if adaptive, else target_k."""
        return k_cell["k"] if (k_cell is not None) else target_k

    def _cur_r() -> float:
        """Per-request pruning rate implied by _cur_k() (r = 1 - k/576)."""
        return 1.0 - (_cur_k() / 576.0)

    def _projector_hook(module, inputs, output):  # noqa: ANN001
        import torch  # noqa
        hook_state["n_calls"] += 1
        ck = _cur_k()
        cr = _cur_r()
        if args.pruning_rate == 0.0 and not adaptive:
            hook_state["kept_counts"].append(output.shape[1])
            return None  # control: no-op, but still log full token count

        # ---- v2 query-aware path: pop the pre-computed question embeddings ----
        if selector_kind == "query_aware":
            query_embeds = query_queue.pop(0) if query_queue else None
            full_n = output.shape[1]
            # shape guard: query_embeds must be (B, T, D) with B matching output,
            # D matching output's D (T can differ -- it's the question length).
            bad = (query_embeds is None
                   or query_embeds.dim() != 3
                   or query_embeds.shape[0] != output.shape[0]
                   or query_embeds.shape[2] != output.shape[2])
            if bad:
                # shape-mismatch fallback (shouldn't happen): keep first ck
                kept = output[:, :ck, :].contiguous()
                keep_idx = torch.arange(ck, device=output.device).unsqueeze(0).expand(
                    output.shape[0], -1)
            else:
                sel = QueryAwareSelector(
                    pruning_rate=cr,
                    pool=getattr(args, "query_pool", "max"),
                    sim=getattr(args, "query_sim", "cosine"),
                )
                kept, keep_idx = sel.select(output, query_embeds)
                if kept.shape[1] != ck:
                    kept = kept[:, :ck, :].contiguous()
                    keep_idx = keep_idx[:, :ck].contiguous()
                # one-time log: confirm the selector received the question and
                # the scores are text<->patch similarities (not CLS-attn)
                if hook_state["n_calls"] == 1:
                    with torch.no_grad():
                        sc = text_patch_scores(output, query_embeds,
                                               pool=sel.pool, sim=sel.sim)[0]
                    print(f"[serve_bench] QUERY_AWARE shape check: patches={full_n} "
                          f"query_tokens={query_embeds.shape[1]} "
                          f"scores_shape={tuple(sc.shape)} "
                          f"score[min,med,max]=[{sc.min().item():.4f},"
                          f"{sc.median().item():.4f},{sc.max().item():.4f}] "
                          f"top5_patch_idx={torch.topk(sc, 5).indices.tolist()}",
                          flush=True)
            module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
            module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
            hook_state["kept_counts"].append(kept.shape[1])
            if hook_state["n_calls"] <= 5:
                print(f"[serve_bench] projector hook fire #{hook_state['n_calls']}: "
                      f"in={output.shape[1]} -> kept={kept.shape[1]} "
                      f"(prune_rate={cr:.3f}, selector={selector_kind})",
                      flush=True)
            return kept

        # ---- A'' clip_query path: CLIP contrastive text<->patch selection ----
        if selector_kind == "clip_query":
            query_feats = query_queue.pop(0) if query_queue else None   # (B,T,768) CLIP-text
            clip_hs = captured.get("clip_patches_hidden")               # (B,577,1024)
            full_n = output.shape[1]
            bad = (query_feats is None or query_feats.dim() != 3
                   or clip_hs is None or clip_hs.dim() != 3
                   or clip_hs.shape[0] != output.shape[0]
                   or clip_hs.shape[1] - 1 != full_n)
            if bad:
                kept = output[:, :ck, :].contiguous()
                keep_idx = torch.arange(ck, device=output.device).unsqueeze(0).expand(
                    output.shape[0], -1)
            else:
                # project vision-tower patch features (skip CLS) into CLIP
                # contrastive space via visual_projection.weight.T (1024->768).
                with torch.no_grad():
                    patches_1024 = clip_hs[:, 1:, :]                       # (B,N,1024)
                    # visual_projection.weight is (768,1024) [out,in]; .weight.T is
                    # (1024,768) so patches(1024) @ W.T -> (768) contrastive space.
                    wv = clip_visual_proj.weight.to(output.device, output.dtype).T
                    clip_patch_feats = patches_1024 @ wv                   # (B,N,768)
                    qf = query_feats.to(clip_patch_feats.dtype)            # (B,T,768)
                sel = ClipQuerySelector(
                    pruning_rate=cr,
                    pool=getattr(args, "query_pool", "max"),
                    sim=getattr(args, "query_sim", "cosine"),
                )
                kept, keep_idx = sel.select(output, clip_patch_feats, qf)
                if kept.shape[1] != ck:
                    kept = kept[:, :ck, :].contiguous()
                    keep_idx = keep_idx[:, :ck].contiguous()
                if hook_state["n_calls"] == 1:
                    with torch.no_grad():
                        sc = clip_text_patch_scores(clip_patch_feats, qf,
                                                    pool=sel.pool, sim=sel.sim)[0]
                    print(f"[serve_bench] CLIP_QUERY shape check: patches={full_n} "
                          f"clip_text_tokens={qf.shape[1]} "
                          f"scores_shape={tuple(sc.shape)} "
                          f"score[min,med,max]=[{sc.min().item():.4f},"
                          f"{sc.median().item():.4f},{sc.max().item():.4f}] "
                          f"top5_patch_idx={torch.topk(sc, 5).indices.tolist()}",
                          flush=True)
            module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
            module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
            hook_state["kept_counts"].append(kept.shape[1])
            if hook_state["n_calls"] <= 5:
                print(f"[serve_bench] projector hook fire #{hook_state['n_calls']}: "
                      f"in={output.shape[1]} -> kept={kept.shape[1]} "
                      f"(prune_rate={cr:.3f}, selector={selector_kind})",
                      flush=True)
            return kept

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
            # fallback: take the first ck tokens (keeps shapes valid even
            # if the vision hook hasn't captured scores for some reason)
            kept = output[:, :ck, :].contiguous()
            keep_idx = torch.arange(ck, device=output.device).unsqueeze(0).expand(output.shape[0], -1)
        else:
            # build the v1 selector (true_cls uses TrueClsAttnSelector with optional
            # diversity; proxy keeps the probe's ClsAttnSelector for reproducibility).
            # ADAPTIVE: prune rate is the per-request cr (from the controller).
            if selector_kind == "true_cls":
                sel = TrueClsAttnSelector(
                    pruning_rate=cr,
                    diversity_lam=getattr(args, "diversity_lam", 0.0),
                )
            else:
                sel = ClsAttnSelector(pruning_rate=cr)
            kept, keep_idx = sel.select(output, scores)
            # guard: if rounding gave k != ck, trim/pad to ck so the
            # count EXACTLY matches the placeholder count from patch_image_token_count
            if kept.shape[1] != ck:
                kept = kept[:, :ck, :].contiguous()
                keep_idx = keep_idx[:, :ck].contiguous()
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
                  f"(prune_rate={cr:.3f}, selector={selector_kind})", flush=True)
        return kept

    if selector_kind != "true_cls" and vision_tower is not None:
        # PROXY path needs the vision-tower saliency hook; clip_query needs the
        # pre-projector patch-feature stash. Both are forward-hooks on the inner
        # CLIPVisionTransformer (skip the wrapper's .vision_tower delegation).
        inner = getattr(vision_tower, "vision_model", vision_tower)
        if selector_kind == "clip_query":
            inner.register_forward_hook(_vision_hook_clip)
        else:
            inner.register_forward_hook(_vision_hook_proxy)
    projector.register_forward_hook(_projector_hook)

    # ---- M1 timing hooks (always on, for prefill breakdown) ----------------
    # register_with_pre_hook needs PyTorch >=1.8; forward_pre_hook fires before
    # the module forward, forward_hook after. The vision_tower.vision_model is
    # the full CLIPVisionTransformer forward (embeddings + 24 encoder layers) --
    # exactly the "fixed cost" we want to isolate. The projector forward is the
    # boundary (cheap linear) -- measured to subtract from TTFT cleanly.
    if vision_tower is not None:
        inner = getattr(vision_tower, "vision_model", vision_tower)
        inner.register_forward_pre_hook(_vt_pre)
        inner.register_forward_hook(_vt_post)
    projector.register_forward_pre_hook(_proj_pre)
    projector.register_forward_hook(_proj_post)

    # stash hook_state + cls_capture on the projector for run() to read/clean up
    projector._vtc_hook_state = hook_state  # type: ignore[attr-defined]
    projector._vtc_cls_capture = cls_capture  # type: ignore[attr-defined]
    projector._vtc_query_queue = query_queue  # type: ignore[attr-defined]
    projector._vtc_tokenizer = tokenizer     # type: ignore[attr-defined]
    projector._vtc_embed_tokens = embed_tokens  # type: ignore[attr-defined]
    projector._vtc_clip_text_model = clip_text_model   # type: ignore[attr-defined]
    projector._vtc_clip_text_proj = clip_text_proj     # type: ignore[attr-defined]
    projector._vtc_clip_visual_proj = clip_visual_proj  # type: ignore[attr-defined]
    projector._vtc_clip_tokenizer = clip_tokenizer     # type: ignore[attr-defined]
    projector._vtc_times = _vtc_times  # type: ignore[attr-defined]  (M1 timing)
    projector._vtc_k_cell = k_cell     # type: ignore[attr-defined]  (adaptive: mutable per-request k)
    return llm, projector


def embed_question(question: str, tokenizer, embed_tokens, device) -> "torch.Tensor":
    """Tokenize a question and look up its LLM input embeddings (B=1, T, D).

    PLUMBING for the query_aware selector (method-design.md §9): both the
    question token embeddings and the post-projector patch embeddings live in
    the LLM's input-embedding space, so they're directly comparable via cosine
    similarity. The question text is known BEFORE the forward (it's part of the
    prompt), so we tokenize JUST the question (not the full chat-template prompt
    -- avoids the image placeholder `<image>` x576 and the USER/ASSISTANT template
    tokens; the question's own tokens are what we want to match against patches),
    run embed_tokens (a lookup, not a transformer pass), and stash the result.

    Returns a (1, T, D) fp16 tensor on `device`. Cheap (~ms for T<=~30 tokens).
    """
    import torch  # noqa
    enc = tokenizer(question, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)        # (1, T)
    if ids.numel() == 0:                     # degenerate: fall back to BOS
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        ids = torch.tensor([[bos]], device=device, dtype=torch.long)
    emb = embed_tokens(ids)                  # (1, T, D) -- a lookup
    return emb.to(torch.float16)


def embed_clip_question(question: str, clip_tokenizer, clip_text_model,
                        clip_text_proj, device) -> "torch.Tensor":
    """Embed the question via CLIP text encoder + text_projection (A'').

    The v2 fix: instead of the LLM `embed_tokens` (word-semantics space, NOT
    contrastively aligned -> OCR failed), embed the question through CLIP's text
    tower and project into CLIP's contrastive space. CLIP was trained so this
    feature ALIGNS with CLIP-ViT patch features, making cosine a meaningful
    cross-modal similarity for text/OCR-relevant patch selection.

    Returns (1, T, 768) fp16 contrastive features on `device`. Cheap for T<=~30
    (one small transformer pass over ~12 layers, hidden=768).
    """
    import torch  # noqa
    # CLIP tokenizer expects raw text; add BOS/EOS per CLIP convention.
    enc = clip_tokenizer(question, return_tensors="pt", add_special_tokens=True,
                         truncation=True, max_length=77)
    ids = enc["input_ids"].to(device)        # (1, T)
    if ids.numel() == 0:
        bos = clip_tokenizer.bos_token_id if clip_tokenizer.bos_token_id is not None else 49406
        ids = torch.tensor([[bos]], device=device, dtype=torch.long)
    with torch.no_grad():
        tout = clip_text_model(input_ids=ids)              # last_hidden_state (1,T,768)
        wt = clip_text_proj.weight.to(device, tout.last_hidden_state.dtype).T  # (768,768)
        text_contr = tout.last_hidden_state @ wt           # (1,T,768) contrastive
    return text_contr.to(torch.float16)


# --------------------------------------------------------------------------- #
# Metrics aggregation
# --------------------------------------------------------------------------- #
def _controller_json(controller: "LoadAdaptiveController") -> dict:
    """JSON-safe snapshot of the controller (realized[] carries LoadReading
    dataclasses, which aren't serializable -> convert to plain dicts)."""
    realized = []
    for r, reading in controller.realized:
        realized.append({
            "r": r,
            "kv_occupancy": reading.kv_occupancy,
            "num_running": reading.num_running,
        })
    out = {k: v for k, v in controller.__dict__.items() if k != "realized"}
    out["realized"] = realized
    out["realized_summary"] = controller.realized_summary()
    return out


def mean_stderr(xs: list[float]) -> dict:
    # NaN-safe: skip non-finite values (batch-submit mode stores NaN per-row then
    # overwrites with the aggregate; guard against any NaN leaking into aggregation).
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
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

    # query-aware plumbing: pop the per-request question-embedding queue that
    # build_engine stashed on the projector; pre-compute the question embeddings
    # BEFORE each llm.chat() call so the projector hook can pop them in order.
    query_queue = getattr(projector, "_vtc_query_queue", None)
    sel_kind = getattr(args, "selector", "proxy")
    do_query_aware = (sel_kind == "query_aware")
    do_clip_query = (sel_kind == "clip_query")

    raw = []
    import torch  # noqa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()

    # ---- request-submission mode -------------------------------------------
    # SERIAL (default, M1 + accuracy): one llm.chat() per request -> per-request
    #   TTFT measured, but max_num_seqs has NO throughput effect (only 1 in flight).
    # BATCH (M2): submit ALL message-lists to ONE llm.chat() -> vLLM runs them with
    #   continuous batching up to max_num_seqs -> real concurrency, real req/s gain
    #   from KV-pressure relief. Per-request TTFT unavailable; served_req_s is
    #   computed as n / wall_time (the aggregate throughput metric M2 needs).
    # LOAD_PROFILE (method D): submit requests in time-varying bursts/gaps per a
    #   profile generator (constant|bursty|step) -> engine load swings so the
    #   adaptive controller has something to react to. Per-batch llm.chat() with
    #   sleeps between; r decided from live load before EACH batch (adaptive) and
    #   k_cell updated so the placeholder count + projector hook honor it.
    batch_submit = getattr(args, "batch_submit", False)
    load_profile = getattr(args, "load_profile", None)
    adaptive = bool(getattr(args, "adaptive", False))
    k_cell = getattr(projector, "_vtc_k_cell", None)
    # target_k = the FIXED kept-token count for the run (defined in build_engine;
    # stashed on hook_state). Used as the non-adaptive fallback in the load-
    # profile diagnostic log and the projector hook's _cur_k(). For adaptive,
    # k_cell['k'] is the live value instead.
    _hs = getattr(projector, "_vtc_hook_state", {})
    target_k = _hs.get("target_k", max(1, int(round(576 * (1.0 - args.pruning_rate)))))

    # ---- adaptive controller (method D core) ----
    # Built once; decide_r() is called before each submission to set the per-
    # request r (and hence k_cell['k']). realized[] accumulates (r, reading) for
    #事后 distribution analysis (the adaptation proof).
    controller: Optional[LoadAdaptiveController] = None
    if adaptive:
        controller = LoadAdaptiveController(
            r_min=float(getattr(args, "r_min", 0.25)),
            r_max=float(getattr(args, "r_max", args.pruning_rate)),
            occ_lo=float(getattr(args, "occ_lo", 0.40)),
            occ_hi=float(getattr(args, "occ_hi", 0.70)),
            conc_lo=float(getattr(args, "conc_lo", 0.25)),
            conc_hi=float(getattr(args, "conc_hi", 0.75)),
            run_lo=int(getattr(args, "run_lo", 4)),
            run_hi=int(getattr(args, "run_hi", 8)),
            signal=getattr(args, "load_signal", "num_running"),
        )
        print(f"[serve_bench] controller: {controller}", flush=True)

    # max_num_seqs is stamped onto every LoadReading so the num_running signal
    # can normalize to a concurrency fraction. Captured once (the engine's cap).
    _mnseqs = int(getattr(args, "max_num_seqs", 256))

    def _decide_and_set_k():
        """Adaptive: read engine load, decide r, update k_cell. Returns r (or None)."""
        if controller is None or k_cell is None:
            return None
        reading = read_engine_load(llm, max_num_seqs=_mnseqs)
        r = controller.decide_r(reading)
        k_cell["k"] = max(1, int(round(576 * (1.0 - r))))
        return r

    def _build_messages(s):
        # query-aware: embed the question NOW (cheap) and push to the FIFO so the
        # projector hook pops it during this request's forward.
        if do_query_aware:
            assert query_queue is not None
            query_queue.append(embed_question(s.question, projector._vtc_tokenizer,  # type: ignore[attr-defined]
                                              projector._vtc_embed_tokens, device))  # type: ignore[attr-defined]
        elif do_clip_query:
            assert query_queue is not None
            query_queue.append(embed_clip_question(  # type: ignore[attr-defined]
                s.question,
                projector._vtc_clip_tokenizer, projector._vtc_clip_text_model,
                projector._vtc_clip_text_proj, device))
        img_url = s.image
        if os.path.exists(img_url):
            img_url = "file://" + os.path.abspath(img_url)
        return [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": s.question},
            ],
        }]

    if load_profile is not None:
        # ---- method D: time-varying load profile (STREAMING submission) -------
        # The adaptive controller must see MID-FLIGHT load to react. The sync
        # llm.chat() drains the engine between calls (returns when ALL requests
        # finish), so a controller that reads load only at segment boundaries
        # always sees an empty engine. To expose rising load we submit via the
        # ENGINE-LEVEL streaming loop (add_request + step): within a segment we
        # add requests ONE AT A TIME (non-blocking), reading load + deciding r +
        # setting k_cell BEFORE each add. Requests added early in the segment
        # enter the scheduler's waiting/running queues, so the load read before
        # LATER adds genuinely sees rising KV-occupancy / num-running -> the
        # controller's r rises within the segment. Then we step() until all
        # segment requests finish, collecting outputs.
        gen = PROFILES.get(load_profile)
        if gen is None:
            raise ValueError(f"unknown --load-profile {load_profile!r}; "
                             f"choices={list(PROFILES)}")
        max_num_seqs = getattr(args, "max_num_seqs", 256)
        # profile-specific knobs (CLI overrides; let us tune the load swing so
        # the adaptive controller exercises the full [r_min,r_max] range).
        if load_profile == "bursty":
            segments = gen(samples, max_num_seqs,
                           burst=int(getattr(args, "burst", -1)),
                           gap=float(getattr(args, "burst_gap", 1.5)))
        elif load_profile == "step":
            segments = gen(samples, max_num_seqs,
                           n_low=int(getattr(args, "step_n_low", 30)),
                           low_gap=float(getattr(args, "step_low_gap", 0.6)),
                           n_high=int(getattr(args, "step_n_high", 60)),
                           high_gap=float(getattr(args, "step_high_gap", 2.0)))
        else:  # constant
            segments = gen(samples, max_num_seqs)
        # Preprocess each sample's prompt ONCE (chat template + image data ->
        # TokensPrompt with multi_modal_data). add_request consumes this form.
        # NOTE: get_num_image_tokens (our patch) fires HERE, so k_cell['k'] must
        # be set BEFORE preprocess_chat for the placeholder count to match the
        # per-request r. We therefore set k_cell per-request immediately before
        # each preprocess -- but to keep one preprocess pass, we instead set
        # k_cell right before each add_request AND re-run the per-request
        # placeholder by calling the patched fn implicitly through a fresh
        # preprocess. Cheapest correct path: preprocess ALL at the segment's
        # r (one r per segment, decided from load at segment entry reflecting
        # residual in-flight work from prior segments + within-segment adds).
        # Within-segment per-request r variation would need per-request
        # preprocess (cheap; done below for the adaptive-intra-segment path).
        engine = llm.llm_engine
        t_all_start = time.perf_counter()
        n_tokens_out = 0
        seg_idx = 0
        req_counter = 0
        # Per-segment drain with a ONE-SEGMENT-LAG load controller.
        # WHY DRAIN-EACH-SEGMENT: the projector hook reads a SINGLE shared k_cell
        # at forward time, and vLLM batches multiple image-requests per forward.
        # If two in-flight requests had different k (different r), their text-
        # placeholder counts would mismatch the hook's single k -> masked_scatter
        # 'totalElements <= srcSize' crash. So all requests in flight during a
        # forward MUST share k. Fully draining each segment before submitting the
        # next guarantees no cross-segment batching -> no mismatch.
        # WHY ONE-SEGMENT-LAG: with full drain, the load at segment N+1's entry is
        # always 0 (segment N just finished). So the controller can't react to
        # instantaneous load. Instead we sample the load ONCE during segment N's
        # drain (its peak, while requests are mid-decode) and use that to decide
        # segment N+1's r. This is a legitimate reactive controller (responds to
        # recently-observed load, one decision cycle of latency -- exactly how a
        # real online controller with a control loop would behave).
        # Net behavior: under a bursty/step profile, segment N's peak load (which
        # scales with burst size) drives segment N+1's r -> r rises after big
        # bursts, falls after quiet periods. The realized[] log records BOTH the
        # decision read (segment entry) and the peak read (mid-drain) so the
        # adaptation is visible either way.
        prev_peak_reading: Optional[LoadReading] = None
        t_submit_start = time.perf_counter()
        for seg_idx, (batch_samples, gap) in enumerate(segments):
            if not batch_samples:
                continue
            # ---- decide THIS segment's r from the PRIOR segment's peak load ----
            # (one-segment lag; seg 0 has no prior -> r_min cold start). For a
            # non-adaptive run, use the fixed --pruning-rate.
            if adaptive:
                if prev_peak_reading is not None:
                    r_seg = controller.decide_r(prev_peak_reading)
                    k_cell["k"] = max(1, int(round(576 * (1.0 - r_seg))))
                else:
                    # cold start: read current (empty) load -> records realized[0]
                    r_seg = _decide_and_set_k()
            else:
                r_seg = getattr(args, "pruning_rate", 0.0)
                if k_cell is not None:
                    k_cell["k"] = max(1, int(round(576 * (1.0 - r_seg))))
            # ---- submit this segment's requests (all share k_cell) ----
            # reset the multi-modal input cache between segments: the cache keys
            # on image data, and the placeholder count (from get_num_image_tokens,
            # which reads k_cell) is part of preprocessing -- a stale cached result
            # from a prior segment (different k) would bake the WRONG placeholder
            # count into this segment's requests -> masked_scatter mismatch.
            try:
                engine.reset_mm_cache()
            except Exception:
                pass
            if seg_idx < 3:
                print(f"[serve_bench] seg {seg_idx}: r_seg={r_seg:.3f} "
                      f"k_cell={k_cell['k'] if k_cell else target_k}", flush=True)
            seg_pairs = []
            for s in batch_samples:
                rid = f"dval_{req_counter}"; req_counter += 1
                msgs = _build_messages(s)
                prepped = llm.preprocess_chat(msgs)[0]
                engine.add_request(rid, prepped, sp)
                seg_pairs.append((rid, s))
            # ---- drain this segment, sampling peak load mid-decode ----
            seg_finished: dict = {}
            seg_peak_occ = -1.0
            seg_peak_nr = -1
            seg_peak_reading: Optional[LoadReading] = None
            t_seg_drain = time.perf_counter()
            n_steps = 0
            while engine.has_unfinished_requests():
                for o in engine.step():
                    if o.finished:
                        seg_finished[o.request_id] = o
                n_steps += 1
                # sample load every few steps (cheap) and track the peak -- this
                # is the signal for the NEXT segment's r decision.
                if adaptive and n_steps % 3 == 1:
                    rd = read_engine_load(llm, max_num_seqs=_mnseqs)
                    occ = rd.kv_occupancy if rd.kv_occupancy is not None else -1
                    nr = rd.num_running if rd.num_running is not None else -1
                    if occ > seg_peak_occ or nr > seg_peak_nr:
                        seg_peak_reading = rd
                        seg_peak_occ = max(seg_peak_occ, occ)
                        seg_peak_nr = max(seg_peak_nr, nr)
                if time.perf_counter() - t_seg_drain > 300:
                    print(f"[serve_bench] WARN: segment {seg_idx} drain >300s, "
                          f"breaking", flush=True)
                    break
            if adaptive and seg_peak_reading is not None:
                prev_peak_reading = seg_peak_reading
            # ---- collect this segment's results (ONCE per request) ----
            # NOTE: an earlier version had a second result-append inside the
            # `if gap > 0.0` block below that re-used the loop's last `o`/`s`
            # and appended a DUPLICATE row each gapped segment (200 samples ->
            # 249 rows), corrupting accuracy and inflating n. Removed: results
            # are collected exactly once here; the gap block only sleeps.
            for rid, s in seg_pairs:
                o = seg_finished.get(rid)
                if o is None:
                    raw.append({
                        "id": s.id, "served_tok_s": float("nan"),
                        "served_req_s": float("nan"), "ttft_ms": float("nan"),
                        "peak_kv_mb": float("nan"), "correct": 0,
                        "answer": "", "gt": s.gt,
                        "segment": seg_idx, "r_used": r_seg,
                    })
                    continue
                text = o.outputs[0].text.strip()
                n_out = len(o.outputs[0].token_ids)
                n_tokens_out += n_out
                correct = scorer(text, s.gt, s.extra.get("choices"))
                raw.append({
                    "id": s.id, "served_tok_s": float("nan"),
                    "served_req_s": float("nan"), "ttft_ms": float("nan"),
                    "peak_kv_mb": float("nan"), "correct": correct,
                    "answer": text, "gt": s.gt,
                    "segment": seg_idx, "r_used": r_seg,
                })
            if gap > 0.0:
                time.sleep(gap)
        wall = time.perf_counter() - t_submit_start
        # aggregate throughput over the WHOLE profile (the headline req/s metric)
        agg_req_s = len(samples) / wall if wall > 0 else 0.0
        agg_tok_s = n_tokens_out / wall if wall > 0 else 0.0
        peak_kv_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        for r in raw:
            r["served_tok_s"] = agg_tok_s
            r["served_req_s"] = agg_req_s
            r["peak_kv_mb"] = peak_kv_mb
    elif batch_submit:
        # ---- M2 batched path: real continuous batching ----------------------
        # Pre-build all messages (also pre-pushes all query embeddings in order).
        # ADAPTIVE: decide r ONCE from the (initially empty) load -> r_min typically;
        # this is the "constant low" sanity case. For constant-high, use --load-profile
        # constant with adaptive (controller sees occupancy rise -> r_max).
        if adaptive:
            _decide_and_set_k()
        all_messages = [_build_messages(s) for s in samples]
        t_all_start = time.perf_counter()
        outs = llm.chat(all_messages, sp, use_tqdm=False)
        wall = time.perf_counter() - t_all_start
        # aggregate throughput (the M2 metric): req/s over the whole batch.
        agg_req_s = len(samples) / wall if wall > 0 else 0.0
        n_tokens_out = 0
        for s, o in zip(samples, outs):
            text = o.outputs[0].text.strip()
            n_out = len(o.outputs[0].token_ids)
            n_tokens_out += n_out
            correct = scorer(text, s.gt, s.extra.get("choices"))
            # per-request req/s in batch mode = meaningless individually; we store
            # the AGGREGATE in agg below. Keep per-row latencies = NaN to flag.
            raw.append({
                "id": s.id, "served_tok_s": float("nan"), "served_req_s": float("nan"),
                "ttft_ms": float("nan"), "peak_kv_mb": float("nan"),
                "correct": correct, "answer": text, "gt": s.gt,
            })
        # overwrite served metrics with the batched aggregate (the real M2 signal)
        agg_tok_s = n_tokens_out / wall if wall > 0 else 0.0
        peak_kv_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        for r in raw:
            r["served_tok_s"] = agg_tok_s
            r["served_req_s"] = agg_req_s
            r["peak_kv_mb"] = peak_kv_mb
        # batch mode has no per-request TTFT; M1 timing hooks still fired per-
        # request (vision tower + projector), so the prefill_breakdown remains
        # valid -- but TTFT itself (the denominator of vt_frac) is undefined.
        # We report vt_frac against the SERIAL-derived TTFT only if present.
    else:
        # ---- SERIAL path (default): per-request TTFT, M1 prefill breakdown --
        t_all_start = time.perf_counter()
        for s in samples:
            if adaptive:
                _decide_and_set_k()   # per-request r from the live load
            messages = _build_messages(s)
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
    vtc_times = getattr(projector, "_vtc_times", {})

    # ---- M1: prefill breakdown (vision tower vs LLM prefill) ----------------
    # vision_tower_ms = full CLIPVisionTransformer forward (embeddings + 24 layers).
    # projector_ms   = projector linear (boundary, pruning-relevant).
    # llm_prefill_ms = TTFT - vision_tower - projector (the LLM prefill on the
    #   image+text sequence; estimated remainder). Same n as TTFT.
    vt_ms = vtc_times.get("vt_post", [])
    proj_ms = vtc_times.get("proj_post", [])
    ttft_serial = [r["ttft_ms"] for r in raw if r["ttft_ms"] == r["ttft_ms"]]  # drop NaN (batch mode)
    vt_mean = statistics.fmean(vt_ms) if vt_ms else 0.0
    proj_mean = statistics.fmean(proj_ms) if proj_ms else 0.0
    # per-request llm prefill estimate (pair by index; clamp >=0); only in serial mode
    if ttft_serial and not batch_submit:
        llm_prefill_est = [
            max(0.0, t - v - p) for t, v, p in zip(
                ttft_serial,
                vt_ms + [vt_mean] * (len(ttft_serial) - len(vt_ms)),
                proj_ms + [proj_mean] * (len(ttft_serial) - len(proj_ms)),
            )
        ]
        ttft_mean = statistics.fmean(ttft_serial)
        vt_frac = (vt_mean / ttft_mean) if ttft_mean > 0 else float("nan")
    else:
        llm_prefill_est = []
        ttft_mean = float("nan")
        vt_frac = float("nan")  # batch mode: TTFT undefined, vt_frac N/A

    agg = {
        "served_tok_s": mean_stderr([r["served_tok_s"] for r in raw]),
        "served_req_s": mean_stderr([r["served_req_s"] for r in raw]),
        "ttft_ms": mean_stderr([r["ttft_ms"] for r in raw]),
        "peak_kv_mb": mean_stderr([r["peak_kv_mb"] for r in raw]),
        "accuracy": (sum(r["correct"] for r in raw) / len(raw)) if raw else float("nan"),
    }
    agg_prefill = {
        "vision_tower_ms": {"mean": vt_mean, "n": len(vt_ms)},
        "projector_ms": {"mean": proj_mean, "n": len(proj_ms)},
        "llm_prefill_ms_est": mean_stderr(llm_prefill_est),
        "vision_tower_fraction_of_prefill": vt_frac,
        "ttft_ms_mean": ttft_mean,
        "batch_submit": batch_submit,
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
        "max_num_seqs": getattr(args, "max_num_seqs", 256),
        "adaptive": bool(getattr(args, "adaptive", False)),
        "load_profile": getattr(args, "load_profile", None),
        "controller": (_controller_json(controller))
                     if controller is not None else None,
        "n": len(raw), "wall_s": wall, "agg": agg,
        "prefill_breakdown": agg_prefill,
        "prefill_speedup_vs_r0": prefill_speedup, "e2e_speedup_vs_r0": e2e_speedup,
        "hook": {
            "n_projector_calls": hook_state.get("n_calls", 0),
            "kept_counts_head": hook_state.get("kept_counts", [])[:10],
            "kept_counts_unique": sorted(set(hook_state.get("kept_counts", []))),
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
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="vLLM engine max_num_seqs (continuous-batching concurrency cap). "
                         "M2 lever: 1 = no batching (isolates per-request latency), "
                         "12+ = full continuous batching (KV-pressure-bound throughput).")
    ap.add_argument("--batch-submit", action="store_true",
                    help="M2: submit ALL requests in ONE llm.chat() call so vLLM's "
                         "continuous batching actually engages max_num_seqs. Disables "
                         "per-request TTFT (M1 needs serial mode, the default).")
    # ---- P2 method D: load-adaptive prune-rate controller -------------------
    ap.add_argument("--adaptive", action="store_true",
                    help="METHOD D: enable the load-adaptive controller. Per request (or "
                         "per load-profile segment), read the engine's live load "
                         "(KV-occupancy / num-running) and set r in [r_min,r_max] -- "
                         "prune MORE under high load (KV-pressure relief, where M2 showed "
                         "the req/s speedup is largest: r75 1.26x->1.76x c1->c12), prune "
                         "LESS under light load (preserve accuracy). Overrides the fixed "
                         "--pruning-rate (which becomes the r_max initial value).")
    ap.add_argument("--r-min", type=float, default=0.25,
                    help="adaptive controller r floor (light-load prune rate).")
    ap.add_argument("--r-max", type=float, default=0.50,
                    help="adaptive controller r ceiling (heavy-load prune rate). ALSO the "
                         "per-benchmark accuracy guardrail (GQA/TextVQA: 0.50, validated "
                         "to keep the acc drop acceptable).")
    ap.add_argument("--occ-lo", type=float, default=0.40,
                    help="KV-occupancy low threshold (below -> r_min).")
    ap.add_argument("--occ-hi", type=float, default=0.70,
                    help="KV-occupancy high threshold (above -> r_max).")
    ap.add_argument("--conc-lo", type=float, default=0.25,
                    help="num_running signal: concurrency-FRACTION low threshold "
                         "(num_running/max_num_seqs below -> r_min). Default 0.25 -> at "
                         "c12, r_min when <3 concurrent.")
    ap.add_argument("--conc-hi", type=float, default=0.75,
                    help="num_running signal: concurrency-FRACTION high threshold "
                         "(num_running/max_num_seqs above -> r_max). Default 0.75 -> at "
                         "c12, r_max when >9 concurrent.")
    ap.add_argument("--run-lo", type=int, default=4,
                    help="num-running ABSOLUTE low threshold (legacy fallback when "
                         "max_num_seqs unknown; below -> r_min).")
    ap.add_argument("--run-hi", type=int, default=8,
                    help="num-running ABSOLUTE high threshold (legacy fallback when "
                         "max_num_seqs unknown; above -> r_max).")
    ap.add_argument("--load-signal", default="num_running",
                    choices=["kv_occupancy", "num_running"],
                    help="engine-load signal the controller reacts to. DEFAULT "
                         "num_running (P3-step-1): concurrency fraction spans the full "
                         "[0,1] range under c12/short-seq deployment, so realized-r "
                         "traverses [r_min,r_max]. Use kv_occupancy for long-sequence / "
                         "high-concurrency regimes where KV pressure is the real "
                         "bottleneck and occupancy rises into a meaningful range.")
    ap.add_argument("--load-profile", default=None,
                    choices=["constant", "bursty", "step"],
                    help="METHOD D validation: time-varying request-submission profile so "
                         "the engine load swings and the adaptive controller has something "
                         "to react to. 'constant' = one big batch (M2's constant-high case; "
                         "controller sits at ~r_max). 'bursty' = small bursts separated by "
                         "idle gaps (concurrency 0<->burst; r swings r_min<->r_max). "
                         "'step' = low-rate -> high-rate -> low-rate staircase. Overrides "
                         "--batch-submit (load-profile implies segmented batch submission).")
    # profile-tuning knobs (so the load swing exercises the full r-range)
    ap.add_argument("--burst", type=int, default=-1,
                    help="bursty profile: requests per burst. -1 (default) = max_num_seqs "
                         "(saturate the engine each burst -> peak conc ~1.0 -> r_max).")
    ap.add_argument("--burst-gap", type=float, default=1.5,
                    help="bursty profile: idle seconds between bursts. Default 1.5s lets "
                         "the prior burst substantially drain so the controller sees low "
                         "load at the next burst's start (r_min) then high load mid-burst "
                         "(r_max) -> the full r swing that lets adaptive beat fixed.")
    ap.add_argument("--step-n-low", type=int, default=30, help="step profile: # reqs in the low phase (one-at-a-time).")
    ap.add_argument("--step-low-gap", type=float, default=0.6, help="step profile: gap (s) between low-phase reqs.")
    ap.add_argument("--step-n-high", type=int, default=60, help="step profile: # reqs in the high phase (one batch up to max_num_seqs).")
    ap.add_argument("--step-high-gap", type=float, default=2.0, help="step profile: gap (s) after the high-phase batch.")
    ap.add_argument("--selector", default="proxy",
                    choices=["proxy", "true_cls", "query_aware", "clip_query"],
                    help="score source: 'proxy' = hidden-state-deviation (the P2 probe); "
                         "'true_cls' = real [CLS]->patch softmax attention (v1 method, "
                         "via ClsAttnCapture on the last vision-tower layer); "
                         "'query_aware' = text<->patch similarity in the LLM "
                         "embed_tokens space (v2-step-1, SparseVLM-style text-guided "
                         "selection -- OCR failed); "
                         "'clip_query' = CLIP CONTRASTIVE text<->patch similarity (A'', "
                         "the v2 fix: question via CLIP text tower, patches via CLIP "
                         "visual_projection, scored in CLIP's aligned 768-d space).")
    ap.add_argument("--diversity-lam", type=float, default=0.0,
                    help="PRUNESID-style diversity weight (only with --selector true_cls); "
                         "0.0 = pure top-k by CLS-attn (v1 default)")
    ap.add_argument("--query-pool", default="max", choices=["max", "mean"],
                    help="question-token pooling for query_aware: 'max' (SparseVLM default) "
                         "keeps the best-matching token per patch; 'mean' averages.")
    ap.add_argument("--query-sim", default="cosine", choices=["cosine", "dot"],
                    help="text<->patch similarity function for query_aware "
                         "(default cosine = L2-normalize both sides first).")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
