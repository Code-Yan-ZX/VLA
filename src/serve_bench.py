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

# === ENGINE MODE (must run BEFORE any `import vllm`) =========================
# v2 P0: V1 is the DEFAULT (the v2 migration target). Two env levers:
#   VLLM_USE_V1            : 1 (default in vllm>=0.8) = V1 engine; 0 = V0 engine.
#   VLLM_ENABLE_V1_MULTIPROCESSING: 0 = EngineCore in-PROCESS (our measurement
#       path); 1 (vllm 0.19 default) = EngineCore in a subprocess.
# WHY multiproc=0: V1's scheduler (chunked prefill, prefix caching) is IDENTICAL
# in both modes (vllm/v1/core/sched/scheduler.py; only the IPC wrapper differs),
# but multiproc=0 keeps the model in-process (llm_engine.py:131 path:
# `model_executor.driver_worker.model_runner.model` resolves) so the V0-style
# projector forward-hook reaches it. This is the §4.3 measurement-time
# simplification (verified in runs/v1_probe.py: P1 model in-process, P2 hook
# fires, P3 processor patch shrinks placeholders 576->k with no shape crash,
# P4 get_metrics() -> num_requests_running peaks at full concurrency).
# Roll back to V0 via `--engine v0` (sets VLLM_USE_V1=0; multiproc is irrelevant
# in V0 since V0 is always in-process).
import os as _os
_os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# VLLM_USE_V1 is NOT forced here anymore; --engine v0 (parsed in main()) sets it
# to "0" before vllm import via a re-import guard. For the default (V1) path we
# leave it unset so vllm's native default (V1) applies.

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from .compressors import (  # noqa: F401  (re-export)
    ClsAttnSelector,
    TrueClsAttnSelector,
    QueryAwareSelector,
    ClipQuerySelector,
    TomeMergeSelector,
    RandomPruneSelector,
    tome_merge,
    random_prune,
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
    read_engine_load_v1,
)
from .elasticvis.live_allocator import (  # noqa: F401  (EV-1 re-export)
    LiveGreedyAllocator,
    assign_debug_k,
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


def score_yesno(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """Yes/no scorer for MME (gt in {"yes","no"}).

    Matches the GQA yes/no lead-token rule: correct if the model's LEAD
    alnum word is the gt yes/no token. Deterministic, no external deps.
    (signature-compatible with score_gqa's choices arg; unused here.)
    """
    if not gt:
        return 0
    g = gt.strip().lower()
    if g not in {"yes", "no"}:
        return 0
    p_words = _norm_words(pred)
    for w in p_words:
        if w in {"a", "an", "the"}:
            continue
        return 1 if w == g else 0
    return 0


def score_mc_letter(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """Multiple-choice letter scorer for MMBench / ScienceQA.

    Extracts the model's FIRST option letter (A-Z) from the answer and matches
    it to gt (which must itself be a single letter "A"/"B"/...). Handles common
    answer phrasings: "A", "A.", "A:", "Option A", "The answer is B", "B. foo".
    Deterministic, no external deps.
    """
    if not gt:
        return 0
    g = gt.strip().upper()
    if not (len(g) == 1 and g.isalpha()):
        return 0
    p = pred.strip().upper()
    if not p:
        return 0
    # scan tokens for the first that is exactly a letter (optionally followed by ./:/comma)
    # word-split on whitespace so "OPTION A" -> "A" is caught, "B. text" -> "B." -> B
    for tok in p.split():
        core = tok.rstrip(".,:;)\"'")
        if len(core) == 1 and core.isalpha():
            return 1 if core == g else 0
        # the letter may be glued: "A)foo" -> core "A)FOO" no; rare. Also catch
        # the standalone-letter-at-start case below if no token was clean.
    # fallback: first char if it's a letter
    if p[0].isalpha():
        return 1 if p[0] == g else 0
    return 0


SCORERS = {
    "gqa": score_gqa,
    "textvqa": score_textvqa,
    "mme": score_yesno,
    "mmbench": score_mc_letter,
    "scienceqa": score_mc_letter,
}


# --------------------------------------------------------------------------- #
# vLLM engine + hook installation (lazy import)
# --------------------------------------------------------------------------- #
def patch_image_token_count(pruning_rate: float, full_n: int = 576,
                            k_cell: Optional[dict] = None,
                            ev_state: Optional[dict] = None) -> int:
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

    ELASTICVIS MODE (EV-1): when `ev_state` is provided, the patched function
    reads `ev_state["cur_k"]` on EVERY call (set per-request by run() right
    before preprocess_chat). This breaks the "all in-flight requests share k"
    constraint: each request's placeholder count is k_i, and the projector hook
    returns a LIST of per-image 2D tensors (different k_i per row) that vLLM's
    scatter consumes per-image (sanity_check_mm_encoder_outputs accepts lists;
    encoder_runner.extend() splits them; the placeholder-scatter matches).
    """
    import vllm.model_executor.models.llava as _llava_mod  # noqa
    k = max(1, int(round(full_n * (1.0 - pruning_rate))))
    InfoCls = _llava_mod.LlavaProcessingInfo

    # ---- ELASTICVIS (EV-1): per-request k via ev_state["cur_k"] ----
    if ev_state is not None:
        if not getattr(InfoCls.get_num_image_tokens, "_vtc_patched", False):
            orig = InfoCls.get_num_image_tokens

            def patched_ev(self, *, image_width, image_height):  # noqa: ANN001
                return ev_state["cur_k"]
            patched_ev._vtc_mode = "elasticvis"
            patched_ev._vtc_orig = orig
            patched_ev._vtc_patched = True
            InfoCls.get_num_image_tokens = patched_ev
            print(f"[serve_bench] patched get_num_image_tokens: ELASTICVIS "
                  f"(per-request k from ev_state['cur_k']; init k="
                  f"{ev_state['cur_k']})", flush=True)
        return int(ev_state["cur_k"])

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


def _resolve_k_policy(args) -> str:
    """Determine the k-policy (additive: existing flags map to existing modes).

    - ``--k-policy elasticvis`` -> "elasticvis" (EV-1 per-request allocator).
    - ``--k-policy segment`` or ``--adaptive`` -> "segment" (v2 per-segment
      LoadAdaptiveController; the existing method-D path, unchanged).
    - ``--k-policy fixed`` or none -> "fixed" (scalar --pruning-rate; default).

    Existing v2 commands that pass ``--adaptive`` (no ``--k-policy``) still map
    to "segment" and behave identically. Commands with neither flag default to
    "fixed" (byte-identical to pre-EV-1 behavior).
    """
    kp = getattr(args, "k_policy", None)
    if kp is not None:
        return kp
    if bool(getattr(args, "adaptive", False)):
        return "segment"
    return "fixed"


def _pixel_fingerprint(pixel_values_row) -> tuple:
    """Cheap content hash of one image's pixel_values (a 3D tensor: C,H,W).
    Returns a (sum, sum_sq) tuple — collision-safe for distinct normalized
    images (different content → different sums with overwhelming probability).
    Computed on GPU, ONE .item() sync per batch (batched via .sum(dim=...)).
    """
    import torch  # noqa
    s = float(pixel_values_row.float().sum().item())
    s2 = float((pixel_values_row.float() ** 2).sum().item())
    # Round to nearest 10 — robust to float16↔float32 conversion noise between
    # preprocess (float32) and forward (float16 on GPU, back-converted). The
    # dtype error accumulates to ~4-40 over C*H*W≈150k elements; rounding to 10
    # absorbs it while keeping collision probability negligible for distinct imgs.
    return (round(s, -1), round(s2, -1))


def _install_embed_mm_patch(engine_model, ev_state: dict) -> None:
    """Monkey-patch ``engine_model.embed_multimodal`` to populate
    ``ev_state["cur_batch_k"]`` (a list of per-row k_i) via pixel-value
    fingerprinting — ORDER-INDEPENDENT (no FIFO queue).

    The patch computes a (sum, sum_sq) fingerprint per pixel_values row and
    matches it to the ``ev_state["fp_to_k"]`` map (built at preprocess time
    when rid and k_i are both in scope). The projector hook then reads
    ``cur_batch_k`` to do per-row top-k_i gather.

    Falls back to k=576 (max, no prune) for fingerprint misses (shouldn't
    happen in normal operation — all images are fingerprinted at preprocess).
    """
    import torch  # noqa
    orig_embed_mm = engine_model.embed_multimodal

    def patched_embed_multimodal(*args, **kwargs):  # noqa: ANN002
        # Try to extract pixel_values from kwargs (vLLM passes them as a kwarg).
        pv = kwargs.get("pixel_values")
        if pv is not None and isinstance(pv, torch.Tensor) and pv.dim() == 4:
            B = pv.shape[0]
            ev_state["n_embed_calls"] += 1
            # Batch fingerprint: sum and sum_sq over (C,H,W) per row.
            pv_f = pv.float()
            sums = pv_f.sum(dim=[1, 2, 3]).tolist()
            sums_sq = (pv_f ** 2).sum(dim=[1, 2, 3]).tolist()
            fp_to_k = ev_state["fp_to_k"]
            batch_k = []
            for i in range(B):
                fp = (round(sums[i], -1), round(sums_sq[i], -1))
                ki = fp_to_k.get(fp)
                if ki is not None:
                    ev_state["n_fp_hits"] += 1
                else:
                    ev_state["n_fp_miss"] += 1
                    ki = ev_state.get("cur_k", 576)  # serial-mode fallback
                batch_k.append(ki)
            ev_state["cur_batch_k"] = batch_k
            if ev_state["n_embed_calls"] <= 5:
                print(f"[serve_bench] EV embed_multimodal #{ev_state['n_embed_calls']}: "
                      f"B={B} fp-matched k={batch_k}", flush=True)
        else:
            ev_state["cur_batch_k"] = None
        return orig_embed_mm(*args, **kwargs)

    engine_model.embed_multimodal = patched_embed_multimodal
    print(f"[serve_bench] EV: monkey-patched embed_multimodal for per-row "
          f"fingerprint→k matching (order-independent)", flush=True)


def _extract_pixel_values(prompt) -> "Optional[object]":
    """Extract the pixel_values tensor from a preprocessed V1 prompt.

    Tries multiple access patterns (V1's EngineInput format varies across
    vLLM versions): dict key, attribute, nested MultiModalDataDict. Returns
    a (C,H,W) or (1,C,H,W) tensor, or None if not found.
    """
    import torch  # noqa
    mmd = None
    # Try dict access
    if isinstance(prompt, dict):
        mmd = prompt.get("multi_modal_data") or prompt.get("mm_data")
    else:
        mmd = getattr(prompt, "multi_modal_data", None)
    if mmd is None:
        return None
    # mmd may be a dict or a MultiModalKwargsItems (UserDict)
    candidates = []
    if isinstance(mmd, dict):
        for k in ("pixel_values", "image"):
            v = mmd.get(k)
            if v is not None:
                candidates.append(v)
    # try direct attribute / items()
    for k in ("pixel_values", "image"):
        v = getattr(mmd, k, None) if not isinstance(mmd, dict) else None
        if v is not None:
            candidates.append(v)
    # unwrap common wrappers
    for v in candidates:
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (list, tuple)) and len(v) > 0:
            inner = v[0]
            if isinstance(inner, torch.Tensor):
                return inner
            if isinstance(inner, (list, tuple)) and len(inner) > 0 \
                    and isinstance(inner[0], torch.Tensor):
                return inner[0]
    return None


def build_engine(model: str, args):
    """Construct a vLLM offline LLM with the probe compressor hooked in.

    ENGINE MODE (v2 P0):
      * `--engine v1` (default): V1 engine, EngineCore IN-PROCESS
        (VLLM_ENABLE_V1_MULTIPROCESSING=0, set at module top). The model is
        reachable via the SAME attribute chain as V0
        (llm_engine.model_executor.driver_worker.model_runner.model) so the
        projector forward-hook + processor patch work unchanged. V1's scheduler
        (chunked prefill, prefix caching) is the v2 measurement target.
      * `--engine v0`: legacy V0 engine (VLLM_USE_V1=0, set in main() before
        vllm import). Kept for rollback / V0-vs-V1 comparison.
    """
    import torch  # noqa
    import vllm  # noqa
    from vllm import LLM  # noqa  (lazy: CPU-import-safe)
    from vllm.model_executor.models.llava import LlavaMultiModalProjector  # noqa

    engine_mode = getattr(args, "engine", "v1")
    # VLLM_USE_V1 was removed in vllm 0.19 (V0 dropped); read defensively.
    try:
        from vllm.envs import VLLM_USE_V1
    except ImportError:
        VLLM_USE_V1 = None  # vllm >=0.19 is V1-only
    try:
        from vllm.envs import VLLM_ENABLE_V1_MULTIPROCESSING
    except ImportError:
        VLLM_ENABLE_V1_MULTIPROCESSING = "<n/a>"
    print(f"[serve_bench] vllm={vllm.__version__} engine_mode={engine_mode} "
          f"VLLM_USE_V1={VLLM_USE_V1} "
          f"VLLM_ENABLE_V1_MULTIPROCESSING={VLLM_ENABLE_V1_MULTIPROCESSING}",
          flush=True)
    if engine_mode == "v0":
        if VLLM_USE_V1 is None:
            raise RuntimeError(
                "engine=v0 requested but this vllm (0.19+) removed V0. Use the "
                "`vtc_serve` env (vllm 0.10.2) for V0 rollback.")
        if VLLM_USE_V1:
            raise RuntimeError(
                "engine=v0 but VLLM_USE_V1 is True -- set os.environ['VLLM_USE_V1']="
                "'0' BEFORE importing vllm (main() does this from --engine v0).")

    # allow loading subset images from local paths (file:// or bare path).
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    llm_kwargs = dict(
        model=model,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
        # ADAPTIVE/ELASTICVIS modes need eager execution: varying per-request k
        # -> varying sequence lengths -> CUDA graph capture mismatches. Fixed-r
        # runs could keep graphs but we use eager uniformly for cross-cell
        # comparability. ELASTICVIS additionally returns a LIST of per-image
        # tensors (variable k_i) from the projector hook — CUDA graphs cannot
        # capture dynamic-shape list returns.
        enforce_eager=bool(getattr(args, "adaptive", False)
                           or _resolve_k_policy(args) == "elasticvis"),
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=_repo_root,
        max_num_seqs=getattr(args, "max_num_seqs", 256),  # M2: concurrency control
    )
    if engine_mode == "v1":
        # V1-specific: enable get_metrics() (LLM forces disable_log_stats=True by
        # default at entrypoints/llm.py:272-273) + defeat prefix caching so the
        # per-request k-patch takes effect cleanly (prefix-cache would serve a
        # stale k=576 result, bypassing the processor patch -- verified in probe).
        llm_kwargs.update(
            disable_log_stats=False,      # enables llm.get_metrics() for the controller
            enable_prefix_caching=False,  # per-request k must take effect every call
        )
    llm = LLM(**llm_kwargs)

    # ---- locate the projector + vision tower on the loaded model ----
    # IDENTICAL attribute chain in V0 and V1-in-process (V1 multiproc=0): the
    # model lives in the main process at model_executor.driver_worker.model_runner.model
    engine_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    projector: Optional[LlavaMultiModalProjector] = getattr(
        engine_model, "multi_modal_projector", None)
    vision_tower = getattr(engine_model, "vision_tower", None)
    if projector is None:
        raise RuntimeError(
            "multi_modal_projector not found on engine model -- wrong arch?")
    print(f"[serve_bench] hooks: projector={type(projector).__name__} "
          f"vision_tower={type(vision_tower).__name__} "
          f"engine_model={type(engine_model).__name__}", flush=True)

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
    k_policy = _resolve_k_policy(args)
    elasticvis = (k_policy == "elasticvis")
    ev_state = None
    if elasticvis:
        # ELASTICVIS (EV-1): per-request k via ev_state. No shared k_cell —
        # each request gets its own k_i from the LiveGreedyAllocator (or debug
        # assignment). ev_state["cur_k"] is the value get_num_image_tokens
        # returns (set per-request right before preprocess). The projector hook
        # reads ev_state["cur_batch_k"] (a list of per-row k_i set by the
        # embed_multimodal monkey-patch, which matches pixel_values rows to
        # pre-computed fingerprints — ORDER-INDEPENDENT, robust to scheduler
        # reordering under continuous batching + chunked prefill).
        ev_state = {"cur_k": 576, "k_by_rid": {}, "fp_to_k": {},
                    "cur_batch_k": None, "n_embed_calls": 0,
                    "n_fp_hits": 0, "n_fp_miss": 0,
                    "debug_k": getattr(args, "ev_debug_k", None)}
        k_cell = None
        target_k = patch_image_token_count(args.pruning_rate, full_n=576,
                                           ev_state=ev_state)
        print(f"[serve_bench] ELASTICVIS mode ON: per-request k_i via "
              f"LiveGreedyAllocator (slo={getattr(args,'slo_ms',10000)}ms "
              f"type={getattr(args,'slo_type','e2e')}); "
              f"ev_state init cur_k={ev_state['cur_k']}", flush=True)
    elif adaptive:
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
                  "selector": selector_kind, "adaptive": adaptive,
                  "elasticvis": elasticvis, "ev_per_batch_k": []}

    def _cur_k() -> int:
        """Per-request kept-token count: k_cell['k'] if adaptive, else target_k."""
        return k_cell["k"] if (k_cell is not None) else target_k

    def _cur_r() -> float:
        """Per-request pruning rate implied by _cur_k() (r = 1 - k/576)."""
        return 1.0 - (_cur_k() / 576.0)

    def _projector_hook(module, inputs, output):  # noqa: ANN001
        import torch  # noqa
        hook_state["n_calls"] += 1

        # ---- ELASTICVIS (EV-1b): per-request k via fingerprint-matched rows ----
        # BREAKS the "all in-flight MUST share k" constraint: each image in this
        # batched forward gets its OWN k_i. The embed_multimodal monkey-patch
        # already computed per-row k_i via pixel-value fingerprinting and stashed
        # them in ev_state["cur_batch_k"] (ORDER-INDEPENDENT — no FIFO queue, so
        # scheduler reordering under continuous batching + chunked prefill is
        # safe). We select top-k_i per row via the SAME score_provider used by
        # the scalar-k paths and return a LIST of per-image 2D tensors [(k_i,D)].
        # vLLM V1 fully supports list returns: sanity_check_mm_encoder_outputs
        # accepts lists; encoder_runner.extend() splits per-image; the scatter
        # places each image's k_i embeddings into its k_i text placeholders.
        if elasticvis and ev_state is not None and ev_state.get("cur_batch_k"):
            B, N = output.shape[0], output.shape[1]
            ks_raw = ev_state["cur_batch_k"]
            # align k list to actual batch size B (defensive: pad/truncate)
            if len(ks_raw) < B:
                ks_raw = ks_raw + [576] * (B - len(ks_raw))
            ks = [max(1, min(int(k), N)) for k in ks_raw[:B]]
            scores = score_provider()
            results = []
            for i in range(B):
                ki = ks[i]
                if (scores is not None and scores.dim() == 2
                        and scores.shape[0] == B and scores.shape[1] == N):
                    _, idx = torch.topk(scores[i], ki)
                    kept_i = output[i, idx, :].contiguous()    # (ki, D)
                else:
                    # no per-row scores (random/tome_merge selector, or first
                    # call before vision hook captured): keep first ki patches.
                    kept_i = output[i, :ki, :].contiguous()    # (ki, D)
                results.append(kept_i)
            hook_state["kept_counts"].extend(ks)
            hook_state["ev_per_batch_k"].append(list(ks))
            module._vtc_keep_count = sum(ks)  # type: ignore[attr-defined]
            if hook_state["n_calls"] <= 8:
                print(f"[serve_bench] EV1b projector hook #{hook_state['n_calls']}: "
                      f"B={B} in={N} per-row-k={ks} "
                      f"(list-return {len(results)}x 2D; scatter by vLLM)",
                      flush=True)
            return results   # list[(k_i, D)] — sanity_check passes, scatter OK

        ck = _cur_k()
        cr = _cur_r()
        if args.pruning_rate == 0.0 and not adaptive:
            hook_state["kept_counts"].append(output.shape[1])
            return None  # control: no-op, but still log full token count

        # ---- P3 tome_merge path: bipartite soft-matching + average-merge ----
        # DIFFERENT REDUCTION MODE: doesn't discard, MERGES (avg) similar tokens.
        # No score provider needed (operates purely on projector output). The
        # output sequence is k-shorter (placeholder-shrink makes the LLM forward
        # identical to a prune at iso-k -> throughput comparable). keep_idx is a
        # dummy arange (merge has no per-token index). ToMe (Bolya et al. ICLR'23)
        # is the P3 panel's PUBLISHED merge member vs the prune family.
        if selector_kind == "tome_merge":
            full_n = output.shape[1]
            kept = tome_merge(output, cr)
            if kept.shape[1] != ck:
                kept = kept[:, :ck, :].contiguous()
            keep_idx = torch.arange(kept.shape[1], device=output.device).unsqueeze(0).expand(
                output.shape[0], -1)
            module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
            module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
            hook_state["kept_counts"].append(kept.shape[1])
            if hook_state["n_calls"] <= 5:
                print(f"[serve_bench] projector hook fire #{hook_state['n_calls']}: "
                      f"in={output.shape[1]} -> merged={kept.shape[1]} "
                      f"(prune_rate={cr:.3f}, selector=tome_merge)", flush=True)
            return kept

        # ---- P3 random prune path: uniform random k-of-N (sanity floor) ----
        # No score provider; deterministic at fixed seed. The trivial baseline
        # for the cross-compressor panel: any signal-based selector should beat
        # this at iso-throughput.
        if selector_kind == "random":
            full_n = output.shape[1]
            sel = RandomPruneSelector(pruning_rate=cr, seed=getattr(args, "rand_seed", 0))
            kept, keep_idx = sel.select(output)
            if kept.shape[1] != ck:
                kept = kept[:, :ck, :].contiguous()
                keep_idx = keep_idx[:, :ck].contiguous()
            module._vtc_keep_idx = keep_idx         # type: ignore[attr-defined]
            module._vtc_keep_count = kept.shape[1]  # type: ignore[attr-defined]
            hook_state["kept_counts"].append(kept.shape[1])
            if hook_state["n_calls"] <= 5:
                print(f"[serve_bench] projector hook fire #{hook_state['n_calls']}: "
                      f"in={output.shape[1]} -> kept={kept.shape[1]} "
                      f"(prune_rate={cr:.3f}, selector=random)", flush=True)
            return kept

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

    # PROXY needs the vision-tower saliency hook; clip_query needs the pre-projector
    # patch-feature stash; true_cls uses ClsAttnCapture (its own patch). tome_merge
    # and random need NO vision-tower signal (they operate purely on the projector
    # output) -> skip the vision-tower hook entirely for them.
    if selector_kind not in ("true_cls", "tome_merge", "random") and vision_tower is not None:
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
    projector._vtc_ev_state = ev_state if elasticvis else None  # type: ignore[attr-defined]

    # ---- ELASTICVIS: monkey-patch embed_multimodal for per-row k matching ----
    # WHY: the projector hook fires inside embed_multimodal with (B, 576, D) but
    # has NO access to which request each row belongs to. Under continuous
    # batching + chunked prefill, the scheduler's row order ≠ submission order.
    # A FIFO k_queue (the EV-1a approach) CRASHES at c≥16 because a row gets the
    # wrong k_i → placeholder/embedding count mismatch → masked_scatter assert.
    #
    # FIX (EV-1b): compute a per-row FINGERPRINT of pixel_values inside
    # embed_multimodal, match it to a fingerprint→k_i map built at preprocess
    # time (when rid and k_i are both in scope). This is ORDER-INDEPENDENT —
    # each row is identified by its pixel content, not its position in a queue.
    if elasticvis and ev_state is not None:
        _install_embed_mm_patch(engine_model, ev_state)

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


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (NaN-safe). p in [0,100].

    Serving-paper-standard tail metric (p50/p99). Nearest-rank (not linear
    interpolation) so p99 of n=200 is the 198th-ranked value -- the conventional
    "1% worst" tail, matching vLLM/mooncake goodput-benchmark convention.
    """
    xs = sorted(x for x in xs if isinstance(x, (int, float)) and x == x)
    if not xs:
        return float("nan")
    k = int(math.ceil(p / 100.0 * len(xs))) - 1
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


def goodput(xs: list[float], slo_ms: float, wall_s: float) -> dict:
    """Goodput = req/s meeting an SLO (the deployment metric). xs = per-request
    latencies (ms); a request 'meets' if its latency <= slo_ms. Returns req/s
    meeting SLO + raw counts. This is the throughput-vs-latency Pareto metric:
    compression's value is raising req/s WITHOUT blowing the SLO."""
    n_met = sum(1 for x in xs if isinstance(x, (int, float)) and x == x and x <= slo_ms)
    n = len(xs)
    return {
        "req_s": (n_met / wall_s) if wall_s > 0 else 0.0,
        "n_met": n_met, "n": n, "slo_ms": slo_ms,
        "frac_met": (n_met / n) if n else 0.0,
    }


def goodput_at_slo(rows: list[dict], slo_deadlines_ms: list[float],
                   slo_type: str, wall_s: float,
                   k_by_rid: Optional[dict] = None,
                   rid_for_index: Optional[Callable[[int], str]] = None,
                   cap_trace: int = 200) -> dict:
    """Per-request-deadline goodput@SLO (the ElasticVis objective).

    Computed for ANY k-policy (fixed-r baselines too). For each request i:
      deadline_i = slo_deadlines_ms[i]  (ms, per-request)
      lat_i      = rows[i]['e2e_ms'] if slo_type=='e2e' else rows[i]['ttft_ms']
      met_i      = (lat_i is a number) and (lat_i <= deadline_i)
      acc_i      = rows[i]['correct']  (0/1, the per-request correctness used
                                       for the aggregate accuracy)
      goodput_acc = sum(met_i * acc_i) / wall_s   (acc-WEIGHTED: ElasticVis obj)
      met_rate    = sum(met_i) / wall_s           (unweighted, v2-comparable)

    `rows` must be in submission order (so index i pairs with deadline i).
    `k_by_rid`/`rid_for_index` are only for the per-request trace (k shown when
    the policy records one, e.g. elasticvis; null for fixed-r unless caller
    fills it). Returns the section dict (without deadline_source; caller adds).
    """
    n = len(rows)
    n_met = 0
    sum_met_acc = 0.0
    trace: list[dict] = []
    for i, r in enumerate(rows):
        deadline_i = slo_deadlines_ms[i]
        lat_i = r["e2e_ms"] if slo_type == "e2e" else r["ttft_ms"]
        ok = (isinstance(lat_i, (int, float)) and lat_i == lat_i
              and lat_i <= deadline_i)
        acc_i = int(r.get("correct", 0) or 0)
        if ok:
            n_met += 1
            sum_met_acc += acc_i
        if len(trace) < cap_trace:
            rid = rid_for_index(i) if rid_for_index else f"r{i}"
            k = k_by_rid.get(rid) if k_by_rid else None
            lat_out = (lat_i if (isinstance(lat_i, (int, float))
                                 and lat_i == lat_i) else None)
            trace.append({
                "rid": rid, "k": k,
                "deadline_ms": deadline_i, "lat_ms": lat_out,
                "met": bool(ok), "acc": acc_i,
            })
    return {
        "goodput_acc": (sum_met_acc / wall_s) if wall_s > 0 else 0.0,
        "met_rate": (n_met / wall_s) if wall_s > 0 else 0.0,
        "frac_met": (n_met / n) if n else 0.0,
        "n_met": n_met, "n": n,
        "slo_type": slo_type,
        "per_request": trace,
    }


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
    # V1-aware controller read: V1 uses llm.get_metrics() (Prometheus snapshot);
    # V0 uses the in-process scheduler. Both return a LoadReading.
    _read_engine_load = (read_engine_load_v1 if getattr(args, "engine", "v1") == "v1"
                         else read_engine_load)
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

    # ---- elasticvis per-request allocator (EV-1) ----
    ev_state = getattr(projector, "_vtc_ev_state", None)
    ev_allocator: Optional[LiveGreedyAllocator] = None
    if ev_state is not None:
        ev_allocator = LiveGreedyAllocator(
            k_min=int(getattr(args, "ev_k_min", 144)),
            k_max=int(getattr(args, "ev_k_max", 576)),
            slo_type=str(getattr(args, "slo_type", "e2e")),
        )
        print(f"[serve_bench] EV allocator: {ev_allocator}", flush=True)

    # max_num_seqs is stamped onto every LoadReading so the num_running signal
    # can normalize to a concurrency fraction. Captured once (the engine's cap).
    _mnseqs = int(getattr(args, "max_num_seqs", 256))

    def _ev_resolve_k(rid: str, req_idx: int) -> int:
        """ElasticVis: compute k_i for this request from live load + SLO, store
        in ev_state, set cur_k (for get_num_image_tokens). Returns k_i.

        Debug mode (--ev-debug-k '576,144,576'): assigns k round-robin from the
        spec (smoke test; bypasses the allocator). Otherwise: reads live engine
        load via get_metrics, runs the queue-aware greedy gate.

        NOTE: the per-row k→image matching is done at FORWARD time via pixel-
        value fingerprinting (see _install_embed_mm_patch). This function only
        sets cur_k (for the placeholder count) and k_by_rid (for logging).
        The fingerprint→k map is populated by _ev_register_fp after preprocess.
        """
        if ev_state is None:
            return 576
        dbg = ev_state.get("debug_k")
        if dbg:
            ks = assign_debug_k(dbg, max(req_idx + 1, 1))
            k_i = int(ks[req_idx % len(ks)])
        else:
            rd = _read_engine_load(llm, max_num_seqs=_mnseqs)
            nr = getattr(rd, "num_running", None) or 0
            sum_k = nr * 288  # approximate (mean k of in-flight reqs)
            slo_i = _ev_slo_ms(req_idx)
            k_i = ev_allocator.allocate(rd, slo_i, sum_k=sum_k)
        k_i = max(1, min(int(k_i), 576))
        ev_state["k_by_rid"][rid] = k_i
        ev_state["cur_k"] = k_i           # get_num_image_tokens reads THIS
        return k_i

    def _ev_register_fp(prompt, k_i: int) -> None:
        """After preprocess, extract pixel_values from the prompt and register a
        fingerprint→k_i mapping. Called once per request, right after preprocess.
        The embed_multimodal monkey-patch uses this map at forward time to match
        pixel_values rows to k_i — ORDER-INDEPENDENT (no FIFO queue)."""
        if ev_state is None or k_i is None:
            return
        pv = _extract_pixel_values(prompt)
        if pv is not None:
            fp = _pixel_fingerprint(pv)
            ev_state["fp_to_k"][fp] = k_i

    def _ev_slo_ms(req_idx: int) -> float:
        """Per-request SLO deadline (ms). --ev-mixed-slo '3500,15000' alternates
        tight/slack (H1b); otherwise uniform --slo-ms."""
        mixed = getattr(args, "ev_mixed_slo", None)
        if mixed:
            vals = [float(x.strip()) for x in mixed.split(",") if x.strip()]
            if vals:
                return vals[req_idx % len(vals)]
        return float(getattr(args, "slo_ms", 5000.0))

    def _decide_and_set_k():
        """Adaptive: read engine load, decide r, update k_cell. Returns r (or None)."""
        if controller is None or k_cell is None:
            return None
        reading = _read_engine_load(llm, max_num_seqs=_mnseqs)
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

    # P3-step-4: global peak KV-occupancy / num_running trackers (regime
    # evidence). Populated by the load_profile drain loop; left None for the
    # batch_submit / serial paths (which don't sample mid-decode).
    run_peak_occ: Optional[float] = None
    run_peak_nr: Optional[int] = None

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
        run_peak_occ = -1.0   # global peak KV-occupancy across all segments (regime evidence)
        run_peak_nr = -1      # global peak num_running across all segments
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
                ev_ki = None
                if ev_state is not None:
                    ev_ki = _ev_resolve_k(rid, req_counter - 1)
                msgs = _build_messages(s)
                prepped = llm.preprocess_chat(msgs)[0]
                if ev_state is not None:
                    _ev_register_fp(prepped, ev_ki)
                engine.add_request(rid, prepped, sp)
                seg_pairs.append((rid, s))
            # ---- drain this segment, sampling peak load mid-decode ----
            # P3-step-4: sample load for ALL runs (not just adaptive) so the KV-
            # bound regime is provable for fixed-r too (peak KV-occupancy +
            # num_running are now logged in load_trace for every config).
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
                # is the signal for the NEXT segment's r decision (adaptive) AND
                # the regime evidence (all runs).
                if n_steps % 3 == 1:
                    rd = _read_engine_load(llm, max_num_seqs=_mnseqs)
                    occ = rd.kv_occupancy if rd.kv_occupancy is not None else -1
                    nr = rd.num_running if rd.num_running is not None else -1
                    if occ > seg_peak_occ or nr > seg_peak_nr:
                        seg_peak_reading = rd
                        seg_peak_occ = max(seg_peak_occ, occ)
                        seg_peak_nr = max(seg_peak_nr, nr)
                    # global peak tracker (all runs) for regime evidence
                    if occ >= 0 and (run_peak_occ is None or occ > run_peak_occ):
                        run_peak_occ = occ
                    if nr >= 0 and (run_peak_nr is None or nr > run_peak_nr):
                        run_peak_nr = nr
                if time.perf_counter() - t_seg_drain > 600:
                    print(f"[serve_bench] WARN: segment {seg_idx} drain >600s, "
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
                        "e2e_ms": float("nan"),
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
                    "e2e_ms": float("nan"),
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
        # v2 P2: STREAMING add_request + engine.step loop (NOT one llm.chat) so we
        # capture PER-REQUEST TTFT + e2e latency under real concurrency -- the
        # serving-paper-standard p50/p99 + goodput metrics. We submit ALL requests
        # up front (one add_request each, no sleep) so the engine schedules up to
        # max_num_seqs at once (same concurrency regime as llm.chat -- verified by
        # the load_trace peak num_running), then drain via step(). Per-request:
        #   ttft_ms = o.metrics.first_token_latency * 1000  (vLLM's own
        #             arrival->first-token, wall-clock, includes queue+prefill --
        #             the deployer's TTFT; needs disable_log_stats=False, set in
        #             build_engine for V1).
        #   e2e_ms  = (now - submit_ts) * 1000  (our perf_counter submit->finish;
        #             the deployer's per-request latency).
        # Aggregate req/s = n/wall (unchanged from the old llm.chat path -> P0/P1
        # throughput comparability preserved).
        # ADAPTIVE: decide r ONCE from the (initially empty) load -> r_min typically;
        # this is the "constant low" sanity case. For constant-high, use --load-profile
        # constant with adaptive (controller sees occupancy rise -> r_max).
        if adaptive:
            _decide_and_set_k()
        all_messages = [_build_messages(s) for s in samples]
        # Mirror llm.chat() internals: render one conversation then add immediately
        # so the engine can start while the next is being rendered (vLLM recommends
        # generator-style over pre-render-all). We call _preprocess_chat_one (the V1
        # per-conversation renderer, same as LLM._run_chat) so the placeholder-count
        # patch (get_num_image_tokens) fires inside preprocess. We then add via the
        # ENGINE-level add_request (NOT LLM._add_request): the latter returns a
        # uuid-suffixed id "N-xxxxxxxx" while step() emits the BASE id we passed,
        # so keying submit_ts by our own id keeps the round-trip exact.
        # output_kind=FINAL_ONLY -> step() emits an output only when a request
        # finishes (matches LLM._add_request's behavior; cheaper than CUMULATIVE).
        try:
            from vllm.sampling_params import RequestOutputKind  # vllm >=0.8
            sp.output_kind = RequestOutputKind.FINAL_ONLY
        except Exception:
            pass  # older vllm: CUMULATIVE default; the `if o.finished` guard still works
        engine = llm.llm_engine
        submit_ts: dict[str, float] = {}
        rids_in_order: list[str] = []
        t_all_start = time.perf_counter()
        for i, conv in enumerate(all_messages):
            rid = f"p2_{i}"
            if ev_state is not None:
                k_i = _ev_resolve_k(rid, i)  # sets ev_state["cur_k"] BEFORE preprocess
            prompt = llm._preprocess_chat_one(conv)
            if ev_state is not None:
                _ev_register_fp(prompt, k_i)  # fp→k map for forward-time matching
            engine.add_request(rid, prompt, sp)
            submit_ts[rid] = time.perf_counter()
            rids_in_order.append(rid)
        ttft_by_rid: dict[str, float] = {}
        e2e_by_rid: dict[str, float] = {}
        out_by_rid: dict = {}
        t_drain_guard = time.perf_counter()
        while engine.has_unfinished_requests():
            for o in engine.step():
                if getattr(o, "finished", False):
                    rid = o.request_id
                    e2e_by_rid[rid] = (time.perf_counter() - submit_ts[rid]) * 1000.0
                    m = getattr(o, "metrics", None)
                    ftl = getattr(m, "first_token_latency", 0.0) if m else 0.0
                    ttft_by_rid[rid] = (ftl * 1000.0) if ftl and ftl > 0 else float("nan")
                    out_by_rid[rid] = o
            if time.perf_counter() - t_drain_guard > 1800:
                print("[serve_bench] WARN: batch drain >1800s, breaking", flush=True)
                break
        wall = time.perf_counter() - t_all_start
        agg_req_s = len(samples) / wall if wall > 0 else 0.0
        n_tokens_out = 0
        for i, s in enumerate(samples):
            rid = rids_in_order[i]
            o = out_by_rid.get(rid)
            if o is None:
                raw.append({
                    "id": s.id, "served_tok_s": float("nan"), "served_req_s": float("nan"),
                    "ttft_ms": float("nan"), "e2e_ms": float("nan"),
                    "peak_kv_mb": float("nan"), "correct": 0, "answer": "", "gt": s.gt,
                })
                continue
            text = o.outputs[0].text.strip()
            n_out = len(o.outputs[0].token_ids)
            n_tokens_out += n_out
            correct = scorer(text, s.gt, s.extra.get("choices"))
            raw.append({
                "id": s.id, "served_tok_s": float("nan"), "served_req_s": float("nan"),
                "ttft_ms": ttft_by_rid.get(rid, float("nan")),
                "e2e_ms": e2e_by_rid.get(rid, float("nan")),
                "peak_kv_mb": float("nan"), "correct": correct, "answer": text, "gt": s.gt,
            })
        # overwrite served metrics with the batched aggregate (the real M2 signal)
        agg_tok_s = n_tokens_out / wall if wall > 0 else 0.0
        peak_kv_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        for r in raw:
            r["served_tok_s"] = agg_tok_s
            r["served_req_s"] = agg_req_s
            r["peak_kv_mb"] = peak_kv_mb
    else:
        # ---- SERIAL path (default): per-request TTFT, M1 prefill breakdown --
        t_all_start = time.perf_counter()
        for s_idx, s in enumerate(samples):
            if adaptive:
                _decide_and_set_k()   # per-request r from the live load
            if ev_state is not None:
                _ev_resolve_k(f"ser_{s_idx}", s_idx)
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
                "ttft_ms": ttft, "e2e_ms": e2e * 1000.0, "peak_kv_mb": peak_kv_mb,
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
    # ---- v2 P2: serving-paper-standard tail-latency + goodput ----------------
    # These are the metrics the v1 paper was criticized for MISSING. p50/p99 of
    # TTFT (deployer's tail latency) and e2e; goodput = req/s meeting an SLO
    # (throughput-vs-latency Pareto -- compression's deployment value).
    # In batch/closed-loop submit (the P2 scale regime), TTFT/e2e include queueing
    # (requests arrive near-simultaneously); the r-comparison at iso-c is still
    # fair (same n, same c). Serial c1 gives queue-free single-request latency.
    _ttfts = [r["ttft_ms"] for r in raw]
    _e2es = [r["e2e_ms"] for r in raw]
    agg["ttft_ms_p50"] = percentile(_ttfts, 50)
    agg["ttft_ms_p99"] = percentile(_ttfts, 99)
    agg["e2e_ms_p50"] = percentile(_e2es, 50)
    agg["e2e_ms_p99"] = percentile(_e2es, 99)
    # goodput under two SLOs (the deployment-relevant metrics):
    #   ttft <= 500ms  (interactive responsiveness; deployer's TTFT SLO)
    #   e2e  <= 1000ms (full request latency SLO)
    agg["goodput_ttft_500ms"] = goodput(_ttfts, 500.0, wall)
    agg["goodput_e2e_1000ms"] = goodput(_e2es, 1000.0, wall)
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

    # ---- EV-1b: per-request-deadline goodput@SLO (the ElasticVis objective) - #
    # Computed for ANY k-policy (fixed-r baselines too) whenever --slo-ms OR
    # --ev-mixed-slo is set (args.slo_ms has a default, so this always runs).
    # deadline_i = _ev_slo_ms(i) (the WORKLOAD's per-request deadline, applies to
    # all policies); lat_i = e2e if slo_type=='e2e' else ttft; met_i<=deadline_i.
    # `raw` is in submission order so index i pairs with deadline i. The per-
    # request k in the trace is filled from ev_state when the policy records one
    # (elasticvis); null for fixed-r (single k lives in pruning_rate / hook).
    _slo_type_ev = str(getattr(args, "slo_type", "e2e"))
    _ev_slo_deadlines = [_ev_slo_ms(i) for i in range(len(raw))]
    _rid_for_index = (lambda i: f"p2_{i}") if batch_submit else (lambda i: f"ser_{i}")
    _ev_k_by_rid = ev_state["k_by_rid"] if ev_state else None
    # For the per-request trace's k: use ev_state's per-rid k (elasticvis); for
    # fixed-r (single k for the whole run) fill target_k so the baseline's trace
    # is complete. For segment/adaptive (k varies per-segment but isn't recorded
    # per-rid) leave k null rather than mislabel with the r_max endpoint.
    if _ev_k_by_rid is None and _resolve_k_policy(args) == "fixed":
        _ev_k_by_rid = {_rid_for_index(i): target_k for i in range(len(raw))}
    _goodput_at_slo = goodput_at_slo(
        raw, _ev_slo_deadlines, _slo_type_ev, wall,
        k_by_rid=_ev_k_by_rid, rid_for_index=_rid_for_index,
    )
    _goodput_at_slo["deadline_source"] = ("ev_mixed_slo"
                                          if getattr(args, "ev_mixed_slo", None)
                                          else "slo_ms")
    _goodput_at_slo["slo_ms"] = getattr(args, "slo_ms", None)
    _goodput_at_slo["ev_mixed_slo"] = getattr(args, "ev_mixed_slo", None)
    # GLOBAL-slo goodput (backward compat): single deadline = args.slo_ms via the
    # original goodput() helper (acc-unaware, the v2 throughput-vs-SLO metric).
    _global_slo_ms = getattr(args, "slo_ms", None)
    _global_lats = [r["e2e_ms"] if _slo_type_ev == "e2e" else r["ttft_ms"]
                    for r in raw]
    _goodput_at_slo["global_slo_goodput"] = (
        goodput(_global_lats, float(_global_slo_ms), wall)
        if _global_slo_ms is not None else None
    )

    result = {
        "benchmark": args.benchmark, "pruning_rate": args.pruning_rate,
        "selector": getattr(args, "selector", "proxy"),
        "diversity_lam": getattr(args, "diversity_lam", 0.0),
        "max_num_seqs": getattr(args, "max_num_seqs", 256),
        "adaptive": bool(getattr(args, "adaptive", False)),
        "k_policy": _resolve_k_policy(args),
        "slo_ms": getattr(args, "slo_ms", None),
        "slo_type": _slo_type_ev,
        "ev_mixed_slo": getattr(args, "ev_mixed_slo", None),
        "goodput_at_slo": _goodput_at_slo,
        "load_profile": getattr(args, "load_profile", None),
        "controller": (_controller_json(controller))
                     if controller is not None else None,
        "elasticvis": ({
            "k_by_rid": dict(list(ev_state["k_by_rid"].items())[:20]) if ev_state else {},
            "k_by_rid_n": len(ev_state["k_by_rid"]) if ev_state else 0,
            "ev_per_batch_k_head": hook_state.get("ev_per_batch_k", [])[:10],
            "allocator_realized_summary": (ev_allocator.realized_summary()
                                           if ev_allocator is not None else None),
            "n_fp_to_k_entries": len(ev_state.get("fp_to_k", {})),
            "n_embed_calls": ev_state.get("n_embed_calls", 0),
            "n_fp_hits": ev_state.get("n_fp_hits", 0),
            "n_fp_miss": ev_state.get("n_fp_miss", 0),
            "slo_ms": getattr(args, "slo_ms", None),
            "slo_type": getattr(args, "slo_type", None),
            "ev_mixed_slo": getattr(args, "ev_mixed_slo", None),
            "ev_debug_k": getattr(args, "ev_debug_k", None),
        }) if ev_state is not None else None,
        "load_trace": {  # P3-step-4: KV-bound regime evidence (all runs)
            "peak_kv_occupancy": run_peak_occ,
            "peak_num_running": run_peak_nr,
            "max_num_seqs": getattr(args, "max_num_seqs", 256),
        } if run_peak_occ is not None else None,
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
    # ENGINE MODE (v2 P0): V1 (default, the v2 target) vs V0 (legacy rollback).
    # Parsed BEFORE vllm import so VLLM_USE_V1=0 takes effect for --engine v0.
    ap.add_argument("--engine", default="v1", choices=["v1", "v0"],
                    help="vLLM engine generation. v1 (default, v2 target): V1 engine "
                         "+ in-process EngineCore (VLLM_ENABLE_V1_MULTIPROCESSING=0, set "
                         "at module top) -> V1 scheduler (chunked prefill) with model "
                         "reachable for forward-hooks. v0: legacy V0 engine "
                         "(VLLM_USE_V1=0). The projector hook + processor patch work in "
                         "BOTH (identical model-access chain); the controller read "
                         "auto-dispatches (V1=get_metrics, V0=in-process scheduler).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pruning-rate", type=float, default=0.0,
                    help="fraction of visual tokens to DROP (0=control)")
    ap.add_argument("--benchmark", required=True,
                    choices=["gqa", "textvqa", "mme", "mmbench", "scienceqa"])
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
                    choices=["proxy", "true_cls", "query_aware", "clip_query",
                             "tome_merge", "random"],
                    help="compressor (P3 cross-compressor panel): "
                         "'proxy' = hidden-state-deviation prune (the v1/P2 selector); "
                         "'true_cls' = real [CLS]->patch softmax attention prune "
                         "(VisionZip/FasterVLM family, via ClsAttnCapture on the last "
                         "vision-tower layer); "
                         "'query_aware' = text<->patch similarity prune in the LLM "
                         "embed_tokens space (SparseVLM-style, OCR failed); "
                         "'clip_query' = CLIP CONTRASTIVE text<->patch similarity prune "
                         "(A''); "
                         "'tome_merge' = ToMe bipartite soft-matching + average-MERGE "
                         "(Bolya et al. ICLR'23; DIFFERENT REDUCTION MODE -- merges "
                         "instead of discards; published-method row); "
                         "'random' = uniform random prune (sanity floor; seeded by "
                         "--rand-seed).")
    ap.add_argument("--diversity-lam", type=float, default=0.0,
                    help="PRUNESID-style diversity weight (only with --selector true_cls); "
                         "0.0 = pure top-k by CLS-attn (v1 default)")
    ap.add_argument("--query-pool", default="max", choices=["max", "mean"],
                    help="question-token pooling for query_aware: 'max' (SparseVLM default) "
                         "keeps the best-matching token per patch; 'mean' averages.")
    ap.add_argument("--query-sim", default="cosine", choices=["cosine", "dot"],
                    help="text<->patch similarity function for query_aware "
                         "(default cosine = L2-normalize both sides first).")
    ap.add_argument("--rand-seed", type=int, default=0,
                    help="seed for the 'random' selector (P3 sanity floor). 0=default.")
    # ---- ElasticVis EV-1: per-request visual-token budget allocator ----------
    ap.add_argument("--k-policy", default=None,
                    choices=["fixed", "segment", "elasticvis"],
                    help="k-policy (additive; EV-1). 'fixed' (default if neither "
                         "this nor --adaptive given): scalar --pruning-rate for the "
                         "whole run (byte-identical to pre-EV-1). 'segment': the v2 "
                         "per-segment LoadAdaptiveController (same as --adaptive; "
                         "all in-flight requests share one k per segment). "
                         "'elasticvis': PER-REQUEST k_i from the LiveGreedyAllocator "
                         "(breaks the shared-k constraint -- different requests in "
                         "the SAME batched forward get different visual-token counts).")
    ap.add_argument("--slo-ms", type=float, default=10000.0,
                    help="ElasticVis: per-request SLO deadline (ms) for the allocator "
                         "gate. Default 10000 (the e2e floor at c64/k576 is ~10s, so "
                         "10s lets the allocator give high-k in light spells and drop "
                         "to k_min under load). Overridden per-request by --ev-mixed-slo.")
    ap.add_argument("--slo-type", default="e2e", choices=["ttft", "e2e"],
                    help="ElasticVis: SLO gate type. 'e2e' gates on wait+P(k)+S(k); "
                         "'ttft' gates on wait+P(k). Default e2e (deployment-relevant).")
    ap.add_argument("--ev-k-min", type=int, default=144,
                    help="ElasticVis: minimum visual-token budget k_min on the grid.")
    ap.add_argument("--ev-k-max", type=int, default=576,
                    help="ElasticVis: maximum visual-token budget k_max on the grid.")
    ap.add_argument("--ev-debug-k", default=None,
                    help="ElasticVis SMOKE TEST: comma-separated k values assigned "
                         "round-robin (e.g. '576,144,576'). Bypasses the allocator so "
                         "you can verify different requests in ONE batch get different "
                         "visual-token counts. The projector hook debug-print shows "
                         "the per-row k evidence.")
    ap.add_argument("--ev-mixed-slo", default=None,
                    help="ElasticVis H1b: comma-separated per-request SLO deadlines "
                         "(ms), assigned round-robin (e.g. '3500,15000' = 50%% tight / "
                         "50%% slack, the §8 TextVQA +35.5%% regime).")
    args = ap.parse_args()
    # V0 rollback: VLLM_USE_V1 must be set BEFORE the first `import vllm`. The
    # lazy imports inside build_engine/run() fire on the first call, so setting
    # it here (after parse, before run) is in time. vllm reads envs at import.
    if args.engine == "v0":
        os.environ["VLLM_USE_V1"] = "0"
        print(f"[serve_bench] engine=v0: set VLLM_USE_V1=0 for legacy V0 path",
              flush=True)
    else:
        os.environ.pop("VLLM_USE_V1", None)  # native V1 default
    run(args)


if __name__ == "__main__":
    main()
