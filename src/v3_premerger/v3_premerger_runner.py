"""V3 go/no-go runner: PRE-merger vs POST-merger pruning on Qwen3-VL-8B-Instruct.

Isolates the STAGE effect (prune BEFORE vs AFTER the native 2x2 merger) at
iso-final-token-budget, using the SAME text-agnostic L2-norm selector at each
respective stage.

Modes:
  --mode none   (A) no pruning (baseline)
  --mode post   (B) POST-merger: hook model._process_image_input, prune each
                      per-image embed (post-split, full multiscale row) to
                      k_i = round(full_i*(1-r)) by L2-norm. (== v2_p1 baseline.)
  --mode pre    (C) PRE-merger: monkey-patch visual.merger.forward (AND each
                      visual.deepstack_merger_list[*].forward for qwen3vl) to
                      slice the merger input. A forward_pre_hook canNOT be used:
                      Qwen2.5-VL's merger class is decorated with vLLM
                      @support_torch_compile, whose __call__ calls self.forward
                      directly and bypasses nn.Module forward_pre_hooks; wrapping
                      .forward is hook-bypass-proof. All mergers consume the same
                      block-major hidden_states (groups of 4 consecutive tokens =
                      1 merge-unit). ONE keep-mask over merge-units, computed once
                      from the first merger's input (deepstack[0], layer-8
                      features) and cached, is applied to all 4 -> the deepstack
                      cat (qwen3_vl.py L654) never sees a seq mismatch.
                      _process_image_input is replaced to split by the PRUNED
                      per-image counts (k_units). Processor placeholder patch is
                      IDENTICAL to post-merger (scales by (1-r)).

enforce_eager=True (variable per-request pruning breaks encoder CUDA graphs).
One cell per fresh process. M-RoPE handled by vLLM's recompute_mrope_positions
(same as v2_p1; output shape is identical between post and pre).

--mask-ranking {stage,swap} (M3 causal control, drafts/v3_merger_aware_design.md
§2): the pre vs post difference is ONLY which ranking picks the 2x2 merge-units
(a kept unit's merged token is identical at either stage). swap crosses the
ranking against the forward path: mode=post+swap runs the full post path but
selects units with the PRE ranking (deepstack[0]-input unit scores, computed
identically to pre mode, in setup_post_merger_swap); mode=pre+swap runs the
sliced pre path but selects with the POST ranking (PreMergerPruner(mask_ranking=
"swap") runs all mergers on the full input once to derive it). By unit
equivalence, post+swap must reproduce pre-standard accuracy exactly and
pre+swap must reproduce post-standard -- isolating RANKING as the only source
of the pre/post gap.

--mode hybrid (merger-aware selection, design §4c): post forward path
(everything merged) + a per-image HYBRID unit mask = agreement(top-k PRE AND
top-k POST) UNION contested budget routed to text: --hybrid-text-frac t of the
contested slots go to the PRE ranking among high-Sobel-edge (text) units, the
rest to the POST ranking among low-edge units. Keeps EXACTLY k units (iso-token
with pre/post). --save-unit-scores stashes a per-image disagreement summary
(Jaccard@k, Spearman(pre,post), mean edge) into per_sample.
"""
from __future__ import annotations
import os, sys, time, json, argparse

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import functools
import torch
import vllm
import dataclasses as _dc
import vllm.model_executor.models.qwen3_vl as _q3vl_mod
import vllm.model_executor.models.qwen2_5_vl as _q2vl_mod

# Model-family registry. qwen2vl = Qwen2.5-VL (single native 2x2 merger, NO
# deepstack); qwen3vl = Qwen3-VL (merger + 3 deepstack mergers). The two
# mergers are structurally identical (forward(x).view(-1, hidden_size) where
# hidden_size = ctx * spatial_merge_size**2; consecutive-4 input tokens form
# one merge-unit in BOTH), so the pre/post SELECTOR logic is shared -- only the
# hook TARGET SET differs: qwen2vl hooks visual.merger ONLY.
MODELS = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}
MODEL = MODELS["qwen3vl"]   # default; overridden in main() per --model-family/--model

# --------------------------------------------------------------------------- #
# Data + scoring (inlined from src/serve_bench.py, identical to v2_p1_runner).
# --------------------------------------------------------------------------- #
from dataclasses import dataclass
from typing import Optional


@dataclass
class Sample:
    id: str
    image: str
    question: str
    gt: str
    extra: dict


def load_subset(path: str) -> list[Sample]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(Sample(
                id=str(o["id"]), image=o["image"], question=o["question"],
                gt=str(o["gt"]),
                extra={k: v for k, v in o.items()
                       if k not in {"id", "image", "question", "gt"}}))
    return out


def _norm_words(s: str) -> list[str]:
    out = []
    for tok in "".join(c if (c.isalnum() or c.isspace()) else " "
                       for c in s.strip().lower()).split():
        out.append(tok)
    return out


def _singular(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 2 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 1 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def score_gqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    if not gt:
        return 0
    p_words = _norm_words(pred)
    g_norm = "".join(c for c in gt.strip().lower() if c.isalnum() or c.isspace()).strip()
    g_words = g_norm.split()
    if not g_words:
        return 0
    if g_norm in {"yes", "no"}:
        lead = None
        for w in p_words:
            if w not in {"a", "an", "the"}:
                lead = w
                break
        if lead in {"yes", "no"} and lead == g_norm:
            return 1
        return 0
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
    for s in syns:
        s_words = s.split()
        if len(s_words) == 1:
            sg = _singular(s)
            if any(w == s or _singular(w) == sg for w in p_words):
                return 1
        else:
            if s in p_text:
                return 1
    return 0


def score_textvqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
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


# DocVQA: same VQA/exact-match convention as TextVQA (gt is a ';'-joined list
# of short text spans; task spec says "DocVQA uses VQA-accuracy / exact-match
# like TextVQA"). Reuses score_textvqa unchanged.
score_docvqa = score_textvqa


def score_yesno(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """Yes/no scorer for MME (gt in {"yes","no"}). Lead alnum token must == gt.
    (Copied verbatim from src/serve_bench.py:score_yesno so the runner is
    self-contained.)"""
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
    """Multiple-choice letter scorer for MMBench / ScienceQA. Extracts the
    model's first option letter (A-Z) and matches gt (single letter).
    (Copied verbatim from src/serve_bench.py:score_mc_letter.)"""
    if not gt:
        return 0
    g = gt.strip().upper()
    if not (len(g) == 1 and g.isalpha()):
        return 0
    p = pred.strip().upper()
    if not p:
        return 0
    for tok in p.split():
        core = tok.rstrip(".,:;)\"'")
        if len(core) == 1 and core.isalpha():
            return 1 if core == g else 0
    if p[0].isalpha():
        return 1 if p[0] == g else 0
    return 0


def score_chartqa(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """ChartQA relaxed accuracy (lmms-eval convention): numeric answers within
    +/-5% relative tolerance (trailing-% normalized), else normalized exact
    match. Ported from lmms_eval/tasks/chartqa/utils.py:relaxed_correctness.
    (Copied verbatim from src/serve_bench.py:score_chartqa.)"""
    if not gt:
        return 0
    p = pred.strip()
    g = gt.strip()

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None

    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf:
        return int(abs(pf - gf) / abs(gf) <= 0.05)
    return int(" ".join(_norm_words(p)) == " ".join(_norm_words(g)))


def score_ocrbench(pred: str, gt: str, choices: Optional[list[str]] = None) -> int:
    """OCRBench accuracy (lmms-eval convention): correct if ANY ';'-joined GT
    answer is contained in the normalized (lowercase/strip/newline->space)
    output. HME100k rows carry choices=["__nospace__"] -> space-insensitive
    containment (LaTeX math), matching lmms_eval/tasks/ocrbench/utils.py.
    (Copied verbatim from src/serve_bench.py:score_ocrbench.)"""
    if not gt:
        return 0
    nospace = bool(choices) and "__nospace__" in choices
    p = pred.lower().strip().replace("\n", " ")
    for g in gt.split(";"):
        g = g.lower().strip().replace("\n", " ")
        if not g:
            continue
        if nospace:
            if g.replace(" ", "") in p.replace(" ", ""):
                return 1
        elif g in p:
            return 1
    return 0


SCORERS = {
    "gqa": score_gqa,
    "textvqa": score_textvqa,
    "docvqa": score_docvqa,
    "mme": score_yesno,
    "mmbench": score_mc_letter,
    "scienceqa": score_mc_letter,
    "chartqa": score_chartqa,
    "ocrbench": score_ocrbench,
}


# --------------------------------------------------------------------------- #
# Merge-unit / token scoring functions -- the SELECTOR plug-in point.
#   l2   : L2-norm of feature vectors (the ORIGINAL text-agnostic selector; the
#          established v3 baseline). Behaviour identical to pre-change code.
#   attn : global-centroid-distance saliency proxy. Each unit's (or token's)
#          mean feature's L2 distance from the GLOBAL mean feature across the
#          whole image -> measures distinctiveness/informativeness rather than
#          magnitude. This is a DIFFERENT selector from L2, used ONLY to test
#          stage-effect robustness (does pre>post hold with a different score?).
#          NOTE on "attn" naming: true ViT self-attention is unavailable under
#          vLLM 0.19's flash-attn eager path (weights are never materialised;
#          forcing output_attentions on a 65k-token document is infeasible at
#          ~17GB/head in fp16). We therefore substitute a cheap attention PROXY
#          computed from the same merger-prehook hidden states L2 uses. CLS-
#          attention is deliberately avoided (project history: CLS under-attends
#          text/OCR). Mean-attention-received semantics are approximated by the
#          centroid-distance "stands out from the crowd" score.
# --------------------------------------------------------------------------- #
def _score_tokens(hs, selector: str):
    """hs: [n_tok, ctx] -> importance score [n_tok].  (post-merger path.)"""
    if selector == "l2":
        return hs.float().norm(dim=-1)
    f = hs.float()                                     # [n_tok, ctx]
    return (f - f.mean(dim=0, keepdim=True)).norm(dim=-1)


def _score_units(feats, selector: str):
    """feats: [num_units, unit, ctx] -> importance score [num_units].  (pre-)"""
    if selector == "l2":
        return feats.float().norm(dim=-1).mean(dim=-1)
    uf = feats.float().mean(dim=1)                     # [num_units, ctx]
    return (uf - uf.mean(dim=0, keepdim=True)).norm(dim=-1)


# --------------------------------------------------------------------------- #
# J5 query-aware (QA) pre-merger saliency helpers (notes/j5_qa_gate_design.md).
#   qsim_i = max_{t in question tokens} cos( merger(unit_feat_i), embed(q_t) )
#     - merger(unit_feat_i): the NATIVE merger MLP run on the unit's input rows
#       -> one LLM-space row per unit (the merger's main forward, reusing the
#       already-registered PreMergerPruner merger handle; no extra hook).
#     - embed(q_t): question token q_t through the LLM word-embedding layer
#       (shared space: the merger output is exactly what enters the LLM at
#       layer 0, alongside word embeddings). Read in-process off the vLLM
#       model (enforce_eager=True keeps weights addressable).
#   Combination (per image): s_i = (1-λ)·minmax(sel_i) + λ·minmax(qsim_i),
#   each path min-max normalized to [0,1] BEFORE weighting so λ is a genuine
#   trade-off knob (top-k is rank-based, so the absolute scale is irrelevant).
# All helpers are NO-OPs unless qa_lambda>0 -- the plain pre-merger path never
# touches them (bit-identical at λ=0).
# --------------------------------------------------------------------------- #
def _minmax(x):
    """Per-vector min-max to [0,1]; constant vectors -> zeros (safe div)."""
    lo, hi = x.min(), x.max()
    if hi > lo:
        return (x - lo) / (hi - lo)
    return torch.zeros_like(x)


def _find_embed_tokens(model):
    """Locate the LLM word-embedding nn.Embedding on the in-process vLLM model.
    Qwen2.5-VL and Qwen3-VL both expose it at ``language_model.model.
    embed_tokens`` (verified against vLLM 0.19: qwen2_5_vl.py L1143/L1472,
    qwen3_vl.py Qwen3LLMForCausalLM.model -> Qwen3LLMModel.embed_tokens). We try
    that + a few common variants, then fall back to a named_modules scan for the
    LARGEST nn.Embedding (the text vocab >> any vision position embedding).
    Returns (embed_module, path_str) or (None, note)."""
    import torch.nn as nn
    candidates = [
        "model.language_model.model.embed_tokens",    # qwen3vl top-level wrapper
        "language_model.model.embed_tokens",          # qwen2.5vl
        "model.language_model.embed_tokens",
        "language_model.embed_tokens",
        "model.model.embed_tokens",
        "model.embed_tokens",
        "embed_tokens",
    ]

    def _is_word_embed(m):
        # vLLM serves the LLM word embedding as VocabParallelEmbedding (NOT an
        # nn.Embedding subclass); accept any 2-D weight with a vocab-scale row
        # count (Qwen text vocab ~152k >> ViT pos_embed 2304).
        w = getattr(m, "weight", None)
        return (w is not None and getattr(w, "ndim", 0) == 2
                and w.shape[0] >= 100000)

    for path in candidates:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and _is_word_embed(obj):
            return obj, path
    # name-suffix scan: '*.embed_tokens' with the largest vocab row count
    best, best_path = None, None
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "embed_tokens" and _is_word_embed(mod):
            if best is None or mod.weight.shape[0] > best.weight.shape[0]:
                best, best_path = mod, name
    if best is not None:
        return best, f"named_modules:{best_path}"
    # final fallback: largest nn.Embedding (HF-style weights)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and mod.num_embeddings >= 100000 and (
                best is None or mod.num_embeddings > best.weight.shape[0]):
            best, best_path = mod, name
    if best is not None:
        return best, f"named_modules:{best_path}"
    return None, "NOT_FOUND"


def _qa_tokenize_question(tokenizer, question: str) -> list:
    """Tokenize the RAW question text (no chat template, no special tokens) into
    token ids. The question is what carries the query intent; the baked
    instruction suffix ('Answer the question using a single word...') is shared
    boilerplate and is harmless to include, so we embed the full string as-is."""
    try:
        return list(tokenizer(question, add_special_tokens=False)["input_ids"])
    except Exception:
        return list(tokenizer.encode(question, add_special_tokens=False))


def _qa_offline_unit_counts(samples, max_pixels: int, model_id: str):
    """Per-sample full merge-unit counts, recomputed offline with the SAME HF
    image processor vLLM uses (no model weights, no GPU -- mirrors the guard in
    attach_hybrid_unit_scores). Needed to key question embeddings by unit count
    so the blend survives vLLM's batch reordering + warmup encoder-cache replay
    (the warmup request re-encodes sample 0). Returns list[int] or None on
    failure (caller falls back to ordered FIFO)."""
    try:
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
        counts = []
        for smp in samples:
            kw = {}
            if max_pixels and max_pixels > 0:
                kw["max_pixels"] = max_pixels
            g = proc.image_processor(Image.open(smp.image),
                                     return_tensors="pt", **kw)["image_grid_thw"]
            counts.append(int(g[0].prod()) // 4)   # spatial_merge_size**2 == 4
        return counts
    except Exception as e:
        print(f"[qa] offline unit-count recompute failed "
              f"({type(e).__name__}: {str(e)[:160]}); using ordered FIFO fallback",
              file=sys.stderr, flush=True)
        return None


def _qa_precompute(model, tokenizer, samples, embed_cache: bool,
                   max_pixels: int, model_id: str):
    """Build the J5 QA state BEFORE generation (CPU/embed-read only, no forward
    through the LLM). Computes each question's normalized word-embedding tokens
    (optionally cached by question string) and keys them by the sample's offline
    unit count for robust pop-during-generation. Returns a state dict consumed
    by PreMergerPruner._qa_blend_scores. ``main_merger_orig`` is filled in by
    setup_pre_merger (it owns the merger handle)."""
    import torch.nn.functional as F
    embed, embed_path = _find_embed_tokens(model)
    diag = {"embed_found": embed is not None, "embed_path": embed_path,
            "n_samples": len(samples), "cache_hits": 0,
            "grid_recompute_ok": False, "blends": 0, "blend_fallback": 0,
            "pops": 0, "pop_fallback": 0, "pop_miss": 0,
            "n_empty_question": 0}
    state = {"embed": embed, "main_merger_orig": None,
             "count_queues": {}, "ordered": [], "qsim_log": [],
             "sample_counts": None, "diag": diag}
    if embed is None:
        print("[qa] word-embedding layer NOT found -- qsim disabled, pre-merger "
              "falls back to the plain selector (λ effectively 0).",
              file=sys.stderr, flush=True)
        return state

    cache = {}

    def _q_embed(question):
        if embed_cache and question in cache:
            diag["cache_hits"] += 1
            return cache[question]
        ids = _qa_tokenize_question(tokenizer, question)
        emb = None
        nv = int(embed.num_embeddings)
        if not diag.get("nv_printed"):
            print(f"[qa] embed path={embed_path} num_embeddings={nv} "
                  f"first_q_ids={len(ids)} max_id={max(ids) if ids else -1}",
                  file=sys.stderr, flush=True)
            diag["nv_printed"] = True
        if ids:
            oob = [i for i in ids if not (0 <= i < nv)]
            if oob:
                diag["oob_ids"] = diag.get("oob_ids", 0) + len(oob)
                ids = [i for i in ids if 0 <= i < nv]
        if ids:
            try:
                with torch.no_grad():
                    ids_t = torch.as_tensor(ids, dtype=torch.long,
                                            device=embed.weight.device)
                    e = embed(ids_t).float()             # [nqt, hidden_llm]
                    emb = F.normalize(e, dim=-1).contiguous()
            except Exception as ex:
                diag["embed_errors"] = diag.get("embed_errors", 0) + 1
                if diag["embed_errors"] <= 3:
                    print(f"[qa] embed() failed ({type(ex).__name__}: "
                          f"{str(ex)[:120]}) -- qsim disabled for this question",
                          file=sys.stderr, flush=True)
                emb = None
        else:
            diag["n_empty_question"] += 1
        if embed_cache:
            cache[question] = emb
        return emb

    qembeds = [_q_embed(s.question) for s in samples]
    counts = _qa_offline_unit_counts(samples, max_pixels, model_id)
    state["sample_counts"] = counts
    if counts is not None:
        diag["grid_recompute_ok"] = True
        for c, qe in zip(counts, qembeds):
            state["count_queues"].setdefault(c, []).append(qe)
        # extra copy of sample-0's embedding for the WARMUP request (it encodes
        # sample 0 once before the timed batch, popping one entry).
        if qembeds and counts:
            state["count_queues"][counts[0]] = \
                [qembeds[0]] + state["count_queues"][counts[0]]
    # ordered FIFO fallback (used only if a runtime count has no queued match,
    # e.g. offline recompute failed or a processor/count mismatch).
    state["ordered"] = ([qembeds[0]] if qembeds else []) + list(qembeds)
    return state


def attach_qa_per_sample(qa_state, samples, per_sample, diag):
    """Best-effort attach of the per-image mean qsim (recorded during generation
    in qa_state['qsim_log'] as (unit_count, mean_qsim)) to per_sample. qsim_log[0]
    is the warmup request (dropped). When offline unit counts are available we
    greedily match each sample to the next unconsumed log entry with the same
    unit count (order-preserving; robust to batch reordering and the warmup
    replay, same guard idea as attach_hybrid_unit_scores); otherwise we zip in
    order. Attachment is ANALYSIS-ONLY (the gate reads answer/gt, not qsim).
    Sets diag['qa_attached']/['qa_total']."""
    log = qa_state.get("qsim_log", [])
    entries = list(log[1:]) if log else []              # drop warmup entry
    total = len(entries)
    counts = qa_state.get("sample_counts")
    attached = 0
    if counts is not None and len(counts) == len(per_sample):
        for i in range(len(per_sample)):
            c = counts[i]
            for j in range(len(entries)):               # first matching count
                if entries[j][0] == c:
                    per_sample[i]["qa_mean_qsim"] = entries[j][1]
                    del entries[j]
                    attached += 1
                    break
    else:
        for i in range(len(per_sample)):
            if i < len(entries):
                per_sample[i]["qa_mean_qsim"] = entries[i][1]
                attached += 1
    diag["qa_attached"] = attached
    diag["qa_total"] = total


def parse_args():
    ap = argparse.ArgumentParser()
    # --mode/--benchmark/--subset are required for a real run but NOT for
    # --dry-check (which validates hook setup on dummy modules without a GPU).
    # main() enforces their presence when not dry-checking.
    ap.add_argument("--mode", required=False, default=None,
                    choices=["none", "post", "pre", "hybrid"])
    ap.add_argument("--r", type=float, default=0.0,
                    help="prune ratio; k_i = round(full_i*(1-r)). "
                         "{0.5,0.75,0.875} -> keep {50,25,12.5}% of merge-units.")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization. Lower on a shared GPU "
                         "(e.g. 0.55 when another user holds ~18GB on A40-46G).")
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="vLLM max_model_len. Raise for huge-image benchmarks "
                         "(DocVQA documents); baseline was hardcoded 8192.")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="vLLM max_num_batched_tokens. In vLLM 0.19 V1 this ALSO "
                         "gates the multimodal encoder cache size (scheduler.py: "
                         "encoder_cache_size = max_num_batched_tokens). DocVQA huge "
                         "images produce image-item embed_length up to ~16k tokens; "
                         "vLLM default 8192 -> ValueError 'exceeds pre-allocated "
                         "encoder cache size 8192' + cascading OOM. Raise to >=32768 "
                         "for DocVQA post-merger cells. None = vLLM default.")
    ap.add_argument("--benchmark", required=False, default=None,
                    choices=["gqa", "textvqa", "docvqa", "mme", "mmbench",
                             "scienceqa", "chartqa", "ocrbench"])
    ap.add_argument("--subset", required=False, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--selector", default="l2", choices=["l2", "attn"],
                    help="l2 = L2-norm selector (default, original behavior); "
                         "attn = global-centroid-distance saliency proxy -- a "
                         "DIFFERENT selector for stage-effect robustness.")
    ap.add_argument("--visionzip-style", action="store_true",
                    help="Dominant + context token paradigm (VisionZip proxy). "
                         "Instead of pruning all non-dominant units, merge them "
                         "into context tokens via grouped averaging. K_dom + K_ctx "
                         "= K (iso-token with vanilla pre-merger). Faithful to "
                         "VisionZip's dominant/context split but uses L2 scores "
                         "as attention proxy (no FlashAttention disable needed).")
    ap.add_argument("--visionzip-dom-ratio", type=float, default=0.7,
                    help="Fraction of K allocated to dominant tokens under "
                         "--visionzip-style (default 0.7 -> 70% dominant, "
                         "30% context).")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="if >0, pass max_pixels to the image processor to cap "
                         "image resolution (pre-merger tokens ~= max_pixels/256). "
                         "Fixes the DocVQA encoder-cache crash on huge documents "
                         "(keeps pre-merger token count <= ~6000 for ~1.5M px).")
    ap.add_argument("--mask-ranking", default="stage",
                    choices=["stage", "swap"],
                    help="Causal control for the ranking-vs-forward-path question "
                         "(M3 in drafts/v3_merger_aware_design.md). stage (default): "
                         "each mode selects units with its OWN stage's ranking "
                         "(post -> post merged-token scores, pre -> pre block-8 "
                         "unit scores; original behavior). swap: cross the ranking "
                         "against the forward path -- mode=post + swap runs the FULL "
                         "post-merger forward (everything merged) but SELECTS units "
                         "with the PRE ranking (deepstack[0]-input unit scores, "
                         "computed EXACTLY as pre mode); mode=pre + swap runs the "
                         "sliced pre forward but selects with the POST ranking "
                         "(computed by running all mergers on the full input once). "
                         "Because a kept unit's merged token is identical at either "
                         "stage (2x2 merge-unit equivalence), post+swap must "
                         "reproduce pre-standard accuracy and pre+swap must "
                         "reproduce post-standard -- isolating RANKING as the only "
                         "source of the pre/post gap.")
    ap.add_argument("--hybrid-text-frac", type=float, default=0.5,
                    help="--mode hybrid ONLY (merger-aware selection, design §4c): "
                         "fraction t in [0,1] of the CONTESTED budget (k - |agreement|) "
                         "routed to the PRE ranking among high-edge (text) units; the "
                         "rest goes to the POST ranking among low-edge units. t=1 -> "
                         "all contested budget to pre/text (OCR-protective); t=0 -> all "
                         "to post (object-friendly). Agreement units are always kept. "
                         "Keeps exactly k units (iso-token with pre/post).")
    ap.add_argument("--save-unit-scores", action="store_true",
                    help="--mode hybrid ONLY: stash a per-image disagreement summary "
                         "(n_units, k, agreement size, Jaccard@k, Spearman(pre,post), "
                         "mean Sobel edge, routing branch) into per_sample[i]['unit_scores']. "
                         "Recomputed grid counts guard the FIFO attachment against "
                         "vLLM encoder-cache replays.")
    ap.add_argument("--qa-lambda", type=float, default=0.0,
                    help="J5 query-aware pre-merger saliency (QA-pre), --mode pre "
                         "ONLY (stage ranking, standard dominant-only path). 0.0 "
                         "(default) = qsim OFF: the qsim code path is never entered "
                         "and behavior is BIT-IDENTICAL to plain pre-merger (zero "
                         "overhead). >0 blends a query-similarity signal into the "
                         "per-image pre ranking: s_i = (1-λ)·minmax(sel_i) + "
                         "λ·minmax(qsim_i), where qsim_i = max_t cos(merger(unit_i), "
                         "embed(q_t)) -- each unit's native-merger LLM-space embedding "
                         "vs the question's word-embedding tokens (cos-max). λ is "
                         "chosen on a held-out dev slice (scripts/j5_qa_dev_select.py).")
    ap.add_argument("--qa-embed-cache", action="store_true",
                    help="--qa-lambda>0 ONLY: cache each question's normalized token "
                         "embeddings keyed by the raw question string, so repeated "
                         "questions in a subset (n=200 sets reuse prompts) are embedded "
                         "once. Pure compute saving; no effect on the selection math.")
    ap.add_argument("--seed", type=int, default=0,
                    help="vLLM/torch RNG seed (0 = default). For repeat runs; at "
                         "temp=0 variance comes from GPU-kernel non-determinism.")
    ap.add_argument("--model-family", default="qwen3vl",
                    choices=["qwen3vl", "qwen2vl"],
                    help="Architecture family. qwen3vl (DEFAULT) hooks "
                         "visual.merger + visual.deepstack_merger_list (current "
                         "behavior, bit-identical). qwen2vl hooks visual.merger "
                         "ONLY -- Qwen2.5-VL has no deepstack mergers.")
    ap.add_argument("--model", default=None,
                    help="HF model id override. If given, family is auto-detected "
                         "from the id (Qwen2* -> qwen2vl, else qwen3vl), overriding "
                         "--model-family. Default = MODELS[family].")
    ap.add_argument("--dry-check", action="store_true",
                    help="No-GPU path: import-check + construct hook setup on a "
                         "dummy model object for the chosen family, then exit. "
                         "Catches syntax/logic errors without loading vLLM/touching "
                         "a GPU. Use while the model is still downloading.")
    ap.add_argument("--out", required=False, default=None,
                    help="Output JSON path (required unless --dry-check).")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Processor placeholder patch (IDENTICAL for post and pre, IDENTICAL across
# families): scale each image's placeholder list by (1-r). Both families define
# _get_prompt_updates on their own processor class returning PromptUpdate
# objects with .modality and a .replacement(item_idx)->list callable, so the
# wrapper is generic -- only the ProcCls differs.
#   qwen3vl: Qwen3VLMultiModalProcessor._get_prompt_updates
#   qwen2vl: Qwen2_5_VLMultiModalProcessor._get_prompt_updates
# --------------------------------------------------------------------------- #
def detect_family(model_id: str) -> str:
    """Auto-detect family from HF model id. Qwen2/Qwen2.5-VL -> qwen2vl;
    everything else (incl. Qwen3-VL) -> qwen3vl."""
    mid = model_id.lower()
    if "qwen2" in mid:          # qwen2-vl / qwen2.5-vl / qwen2_5_vl
        return "qwen2vl"
    return "qwen3vl"


def _proc_class(family: str):
    """The multimodal processor class carrying _get_prompt_updates for a family."""
    return (_q2vl_mod.Qwen2_5_VLMultiModalProcessor if family == "qwen2vl"
            else _q3vl_mod.Qwen3VLMultiModalProcessor)


def patch_processor(r: float, family: str = "qwen3vl"):
    ProcCls = _proc_class(family)
    if getattr(ProcCls._get_prompt_updates, "_vtc_patched", False):
        ProcCls._get_prompt_updates = ProcCls._get_prompt_updates._vtc_orig
    _orig = ProcCls._get_prompt_updates
    log = {"counts": []}

    def _patched(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        prompts = _orig(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        if r == 0.0:
            return prompts
        out = []
        for p in prompts:
            if getattr(p, "modality", None) == "image":
                orig_repl = p.replacement

                def _scaled(item_idx, _or=orig_repl, _r=r):
                    t = _or(item_idx)
                    n = len(t)
                    k = max(1, int(round(n * (1.0 - _r))))
                    log["counts"].append((n, k))
                    return list(t[:k])
                p = _dc.replace(p, replacement=_scaled)
            out.append(p)
        return out
    _patched._vtc_orig = _orig
    _patched._vtc_patched = True
    ProcCls._get_prompt_updates = _patched
    return log


# --------------------------------------------------------------------------- #
# Qwen2.5-VL-ONLY M-RoPE fix (root cause of the pre/post acc~0.004 collapse).
#
# vLLM's Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions advances
# the position cursor by the FULL image grid (llm_grid_t*llm_grid_h*llm_grid_w)
# per image, but patch_processor scales the image placeholder to
# k = round(full*(1-r)) tokens.  The returned position array (text + FULL grid)
# is therefore LONGER than the scaled prompt; gpu_model_runner._calc_mrope_-
# positions silently truncates it to num_prompt_tokens, so every TEXT token that
# follows the image inherits a 2-D grid position (constant t-axis, small h/w)
# instead of a diagonal text position.  Qwen2.5-VL's BLOCK mrope ([16,24,24])
# concentrates that error into whole dim-blocks -> incoherent/garbage answers
# (acc~0.004, n_answered~31/500).  Qwen3-VL's INTERLEAVED mrope ([24,20,20])
# spreads the identical overshoot across dims and tolerates it (acc~0.38), which
# is why only the qwen2vl branch collapsed.  Verified empirically via DBG_MROPE:
# both families overshoot identically; qwen2vl trailing-text tokens get e.g.
# [15,20,25] (t frozen) and emit '' / 'addCriterion...', qwen3vl gets [4,9,4]
# and still emits real words.
#
# Fix: recompute positions with the ACTUAL placeholder count k per image (count
# the image_token_id run at each offset) so the cursor advances by k.  Trailing
# text then gets correct diagonal positions.  Image tokens keep the first k grid
# positions -- bitwise identical to what truncation already assigned (the kept
# tokens are index-sorted, so this mapping is order-preserving) -- so ONLY the
# catastrophic trailing-text corruption is removed.  For r=0 (k==full) this is
# byte-identical to stock vLLM; the qwen3vl branch is never touched.
# --------------------------------------------------------------------------- #
def _capped_image_path(path: str, max_pixels: int) -> str:
    """Enforce a pixel budget by PIL pre-resize, because vLLM 0.19 V1 ignores
    BOTH engine-level and per-request mm_processor_kwargs={'max_pixels':...}
    for Qwen3-VL (verified: DocVQA ptid identical with/without either), and
    transformers 4.57's processor ignores the kwarg too. Edges round to
    multiples of 32 (= patch 16 x merge 2); results cached under
    runs/data/_capped_cache (jpeg q=95). No-op when max_pixels<=0 or the
    image is already within budget."""
    if not max_pixels or max_pixels <= 0:
        return path
    try:
        from PIL import Image
        import math, hashlib
        with Image.open(path) as im:
            w, h = im.size
            if w * h <= max_pixels:
                return path
            scale = math.sqrt(max_pixels / float(w * h))
            nw = max(32, round(w * scale / 32) * 32)
            nh = max(32, round(h * scale / 32) * 32)
            key = hashlib.md5(f"{path}|{nw}x{nh}".encode()).hexdigest()
            ext = os.path.splitext(path)[1].lower() or ".jpg"
            cache = os.path.join("runs", "data", "_capped_cache", key + ext)
            if not os.path.exists(cache):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                im2 = im.resize((nw, nh))
                if ext in (".jpg", ".jpeg"):
                    im2.convert("RGB").save(cache, quality=95)
                else:
                    im2.save(cache)
            return os.path.abspath(cache)
    except Exception as e:
        print(f"[cap] pre-resize failed for {path} ({type(e).__name__}: "
              f"{str(e)[:120]}); using native image", file=sys.stderr,
              flush=True)
        return path


def setup_qwen2vl_mrope_fix(model):
    import numpy as _np
    config = model.config
    image_token_id = config.image_token_id
    spatial_merge_size = config.vision_config.spatial_merge_size
    _orig = model.get_mrope_input_positions
    diag = {"calls": 0, "pruned_images": 0}

    def _fixed(input_tokens, mm_features):
        # Any non-image media (runner is image-only) -> defer to stock path.
        if any(getattr(f, "modality", "image") != "image" for f in mm_features):
            return _orig(input_tokens, mm_features)
        toks = list(input_tokens)
        n = len(toks)
        llm_pos_ids_list = []
        st = 0
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            full = llm_grid_t * llm_grid_h * llm_grid_w
            # actual placeholder count = run of image_token_id starting @ offset
            k = 0
            i = offset
            while i < n and toks[i] == image_token_id:
                k += 1
                i += 1
            if k <= 0 or k >= full:
                k = full                       # r=0 -> identical to stock vLLM
            else:
                diag["pruned_images"] += 1
            text_len = offset - st
            st_idx = (llm_pos_ids_list[-1].max() + 1
                      if len(llm_pos_ids_list) > 0 else 0)
            llm_pos_ids_list.append(
                _np.broadcast_to(_np.arange(text_len), (3, text_len)) + st_idx)
            grid_indices = _np.indices(
                (llm_grid_t, llm_grid_h, llm_grid_w)).reshape(3, -1)
            llm_pos_ids_list.append(
                grid_indices[:, :k] + text_len + st_idx)
            st = offset + k                    # advance by KEPT count, not full
        if st < n:
            st_idx = (llm_pos_ids_list[-1].max() + 1
                      if len(llm_pos_ids_list) > 0 else 0)
            text_len = n - st
            llm_pos_ids_list.append(
                _np.broadcast_to(_np.arange(text_len), (3, text_len)) + st_idx)
        llm_positions = _np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        delta = (llm_positions.max() + 1 - n).item()
        diag["calls"] += 1
        return torch.from_numpy(llm_positions), delta

    # Instance attribute: the worker calls model.get_mrope_input_positions(...)
    # on this same in-process object, so the bare function (no implicit self)
    # is invoked with (input_tokens, mm_features).
    model.get_mrope_input_positions = _fixed
    return diag


# --------------------------------------------------------------------------- #
# (B) POST-merger: wrap _process_image_input, prune post-split. (== v2_p1.)
# --------------------------------------------------------------------------- #
def setup_post_merger(model, r: float, selector: str = "l2"):
    _orig = model._process_image_input
    diag = {"fires": 0, "nk": [], "selector": selector}

    def _patched(image_input):
        splits = _orig(image_input)
        diag["fires"] += 1
        if r == 0.0:
            return splits
        out = []
        for s in splits:
            n = int(s.shape[0])
            k = max(1, int(round(n * (1.0 - r))))
            score = _score_tokens(s, selector)
            idx = torch.topk(score, k).indices.sort().values
            out.append(s.index_select(0, idx).contiguous())
        if len(diag["nk"]) < 8:
            diag["nk"].append((int(splits[0].shape[0]), int(out[0].shape[0])))
        return tuple(out)
    model._process_image_input = _patched
    return diag


# --------------------------------------------------------------------------- #
# (B-swap) POST forward path + PRE ranking selection (M3 causal control).
#   The full post-merger forward runs UNCHANGED (all mergers see full input);
#   we only OBSERVE the first-called merger's input -- deepstack_merger_list[0]
#   for qwen3vl (called inside the block loop at layer 8; verified
#   mask_computed_at='deepstack_0' in pre-mode diag), visual.merger for qwen2vl
#   -- compute per-unit scores with _score_units EXACTLY as pre mode does, and
#   cache them per image (split by grid_thw). The _process_image_input wrapper
#   then selects each image's top-k merged tokens by those PRE unit scores
#   (merged token i <-> unit i is 1:1: mergers map 4 consecutive block-major
#   tokens -> 1 output token in row-major unit order). By the unit-equivalence
#   argument, the selected merged tokens are bitwise the same tensors pre mode
#   would have produced, so post+swap must reproduce pre-standard accuracy.
# --------------------------------------------------------------------------- #
def setup_post_merger_swap(model, r: float, selector: str = "l2",
                           family: str = "qwen3vl"):
    visual = model.visual
    unit = visual.spatial_merge_size ** 2
    state = {"grid_thw": None, "queue": []}   # queue: per-image PRE unit scores
    diag = {"fires": 0, "nk": [], "selector": selector,
            "mask_ranking": "swap:post-path+pre-ranking",
            "score_passes": 0, "consumed": 0, "fallback_stage": 0,
            "first_merger": "merger" if family == "qwen2vl" else "deepstack_0"}

    # (1) visual.forward pre_hook: capture grid_thw -> per-image unit counts.
    def _visual_prehook(module, args, kwargs):
        g = kwargs.get("grid_thw")
        if g is None and len(args) >= 2:
            g = args[1]
        if g is not None:
            state["grid_thw"] = (g.detach() if torch.is_tensor(g)
                                 else torch.as_tensor(g))
    visual.register_forward_pre_hook(_visual_prehook, with_kwargs=True)

    # (2) observe the FIRST merger pre-mode masks from; compute PRE unit scores
    #     without touching the input (post path merges everything).
    first_merger = (visual.merger if family == "qwen2vl"
                    else visual.deepstack_merger_list[0])
    orig_fwd = first_merger.forward

    def _wrapped(*args, _orig=orig_fwd, **kwargs):
        hs = args[0]                                   # [seq, 1, ctx]
        if r != 0.0 and state["grid_thw"] is not None:
            seq = hs.shape[0]
            ctx = hs.shape[-1]
            total_units = seq // unit
            feats = hs.reshape(total_units, unit, ctx)
            scores = _score_units(feats, selector)     # [total_units]
            full_units = (state["grid_thw"].prod(-1) // unit).tolist()
            if sum(full_units) == total_units:
                off = 0
                for f in full_units:
                    state["queue"].append(scores[off:off + f])
                    off += f
                diag["score_passes"] += 1
            else:                                      # defensive: grid mismatch
                diag["fallback_stage"] += len(full_units)
        return _orig(*args, **kwargs)
    first_merger.forward = _wrapped
    if hasattr(first_merger, "do_not_compile"):        # see _wrap_merger_forward
        first_merger.do_not_compile = True

    # (3) _process_image_input: same split contract as stock post mode, but the
    #     top-k index set comes from the cached PRE ranking (per image, FIFO).
    _orig_pii = model._process_image_input

    def _patched(image_input):
        splits = _orig_pii(image_input)
        diag["fires"] += 1
        if r == 0.0:
            return splits
        out = []
        for s in splits:
            n = int(s.shape[0])
            k = max(1, int(round(n * (1.0 - r))))
            pre_i = state["queue"].pop(0) if state["queue"] else None
            if pre_i is not None and int(pre_i.shape[0]) == n:
                diag["consumed"] += 1
                idx = torch.topk(pre_i.to(device=s.device), k).indices.sort().values
            else:
                # encoder-cache replay or grid mismatch -> fall back to the
                # stage (post) ranking rather than crash; diag exposes it.
                diag["fallback_stage"] += 1
                idx = torch.topk(_score_tokens(s, selector), k).indices.sort().values
            out.append(s.index_select(0, idx).contiguous())
        if len(diag["nk"]) < 8:
            diag["nk"].append((int(splits[0].shape[0]), int(out[0].shape[0])))
        return tuple(out)
    model._process_image_input = _patched
    return diag, state          # state exposed so main() can report queue balance


# --------------------------------------------------------------------------- #
# (B-hybrid) MERGER-AWARE HYBRID SELECTION (Task 5 headline method;
#   drafts/v3_merger_aware_design.md §4c). Forward path = POST (everything
#   merged, numerically untouched -- same as mode=post/swap). Selection = a
#   per-image HYBRID mask over the k kept 2x2 merge-units built from BOTH
#   rankings at once:
#     PRE  = deepstack[0]-input unit scores (_score_units, exactly as pre mode)
#     POST = merged-token scores (_score_tokens on each split)
#     EDGE = per-unit Sobel energy reconstructed from the visual encoder's OWN
#            input pixels (text-stroke proxy; identical pooling convention as
#            scripts/mechanism_token_survival.py:unit_edge_from_image -- 32px
#            units, block-major -- but computed from the exact pixels the model
#            sees, so it needs no image-file I/O and is immune to vLLM's
#            encoder-cache replay pairing issues).
#   Agreement set A = units in top-k under BOTH rankings (stage-robust; keep
#   all). Contested budget (k - |A|) is ROUTED TO TEXT per --hybrid-text-frac
#   t in [0,1]: round((k-|A|)*t) slots go to the PRE ranking among high-edge
#   (above per-image contested-median edge; text) units, the rest to the POST
#   ranking among low-edge units (overflow fills from the other pool). Keeps
#   EXACTLY k units => iso-token with pre/post. t=1 => all contested budget to
#   pre/text; t=0 => all to post. By unit equivalence (M3: swap==pre), running
#   selection at the post stage with any unit mask is equivalent to pre-stage
#   selection, so hybrid runs as "post forward + filter merged tokens".
# --------------------------------------------------------------------------- #
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)   # OPENAI_CLIP_STD, the Qwen-VL
_LUMA = (0.299, 0.587, 0.114)                      # processor's normalization


def _unit_edge_from_pixels(pixels, t: int, h: int, w: int):
    """Per-32px-merge-unit Sobel edge energy from raw visual-encoder input
    pixels. pixels: torch.Tensor [t*h*w, dim] (unit-major rows: groups of 4
    consecutive rows = one 2x2 unit, inner order (ph,pw) raster; dim =
    channels*temporal_patch*patch*patch, the HF Qwen-VL processor flatten
    order). Returns float64 numpy [num_units] in block-major unit order.
    Normalization is per-channel affine, so Sobel of the denormalized luma
    equals Sobel of sum_c(luma_c*std_c*x_c) up to an irrelevant additive
    constant -> we apply the (luma*std) weighting directly to normalized px."""
    import numpy as np
    from scipy.ndimage import sobel as _sobel
    dim = int(pixels.shape[1])
    p = int(round((dim / 6.0) ** 0.5))
    if 6 * p * p != dim:
        raise ValueError(f"pixel dim {dim} != 6*p^2 (patch layout unknown)")
    m = 2                                            # spatial_merge_size
    uh, uw = h // m, w // m
    x = np.asarray(pixels.detach().float().cpu()).reshape(
        t, uh, uw, m, m, 3, 2, p, p)
    wgt = (np.array(_LUMA) * np.array(_CLIP_STD)).reshape(1, 1, 1, 1, 1, 3, 1, 1, 1)
    g = (x * wgt).sum(axis=5).mean(axis=(0, 5))      # [uh,uw,m,m,p,p] (mean grid_t, temporal)
    img = g.transpose(0, 2, 4, 1, 3, 5).reshape(uh * m * p, uw * m * p)
    ex, ey = _sobel(img, axis=1), _sobel(img, axis=0)
    edge = np.hypot(ex, ey)
    unit_edge = edge.reshape(uh, m * p, uw, m * p).mean(axis=(1, 3))
    return unit_edge.reshape(-1).astype(np.float64)  # unit-major


def _spearman_np(a, b) -> float:
    """Spearman rho between two 1-D numpy arrays (ties: average ranks)."""
    import numpy as np
    from scipy.stats import rankdata
    if len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _hybrid_select(pre, post, edge, k: int, text_frac: float):
    """Per-image hybrid unit selection. pre/post/edge: 1-D tensors [num_units]
    on the same device (edge may be float). Returns (idx [k] sorted long,
    stats dict). Keeps EXACTLY k = min(k, num_units) units."""
    f = int(pre.shape[0])
    k = max(1, min(k, f))
    dev = pre.device
    pre_ord = torch.argsort(pre, descending=True)
    post_ord = torch.argsort(post, descending=True)
    in_pre = torch.zeros(f, dtype=torch.bool, device=dev)
    in_post = torch.zeros(f, dtype=torch.bool, device=dev)
    in_pre[pre_ord[:k]] = True
    in_post[post_ord[:k]] = True
    agree = in_pre & in_post
    n_agree = int(agree.sum())
    if n_agree >= k:
        # agreement set already covers the budget: keep the k best-consensus
        # units (smallest summed rank across BOTH rankings).
        pre_rank = torch.empty(f, device=dev)
        post_rank = torch.empty(f, device=dev)
        arange_f = torch.arange(f, device=dev, dtype=pre_rank.dtype)
        pre_rank[pre_ord] = arange_f
        post_rank[post_ord] = arange_f
        cand = agree.nonzero().squeeze(-1)
        order = cand[torch.argsort((pre_rank + post_rank)[cand])][:k]
        return order.sort().values, {"branch": "agree_ge_k", "agree_n": n_agree,
                                     "pre_taken": -1, "post_taken": -1}
    # ---- contested budget routed to text ----
    picked = agree.clone()
    contested = (in_pre ^ in_post).nonzero().squeeze(-1)
    b = k - n_agree
    n_pre = int(round(b * float(text_frac)))
    if contested.numel():
        c_edge = edge.index_select(0, contested)
        med = c_edge.median()
        high = contested[c_edge > med]       # text units
        low = contested[c_edge <= med]       # non-text units
    else:
        high = low = contested

    def _take(pool, scores, want):
        if want <= 0 or pool.numel() == 0:
            return 0
        avail = pool[~picked.index_select(0, pool)]
        if avail.numel() == 0:
            return 0
        m = min(want, int(avail.numel()))
        sel = avail[torch.topk(scores.index_select(0, avail), m).indices]
        picked[sel] = True
        return int(sel.numel())

    got_pre = _take(high, pre, n_pre)        # pre ranking among text units
    if got_pre < n_pre:                      # text pool exhausted -> pre on low-edge
        got_pre += _take(low, pre, n_pre - got_pre)
    rem = k - int(picked.sum())
    got_post = _take(low, post, rem)         # post ranking among non-text units
    if got_post < rem:                       # low-edge pool exhausted -> post on high
        got_post += _take(high, post, rem - got_post)
    assert int(picked.sum()) == k, \
        f"hybrid mask size {int(picked.sum())} != k={k}"
    idx = picked.nonzero().squeeze(-1)       # ascending
    return idx, {"branch": "routed", "agree_n": n_agree,
                 "pre_taken": got_pre, "post_taken": k - n_agree - got_pre}


def setup_hybrid(model, r: float, selector: str = "l2", family: str = "qwen3vl",
                 text_frac: float = 0.5, save_scores: bool = False):
    """POST forward path (everything merged) + hybrid pre/post/edge mask
    selection. Returns (diag, state); state["stats_log"] (when save_scores)
    holds one summary dict per popped image (FIFO; first entry = warmup)."""
    visual = model.visual
    unit = visual.spatial_merge_size ** 2
    state = {"grid_thw": None, "pre_queue": [], "edge_queue": [],
             "stats_log": []}
    diag = {"fires": 0, "nk": [], "selector": selector,
            "mask": "hybrid", "hybrid_text_frac": text_frac,
            "save_unit_scores": save_scores,
            "score_passes": 0, "consumed": 0, "fallback_stage": 0,
            "edge_fallback": 0, "n_agree_ge_k": 0, "agree_fracs": [],
            "first_merger": "merger" if family == "qwen2vl" else "deepstack_0"}

    # (1) visual.forward pre_hook: capture grid_thw AND compute per-unit Sobel
    #     edge from the encoder's OWN input pixels (per image, FIFO).
    def _visual_prehook(module, args, kwargs):
        g = kwargs.get("grid_thw")
        px = kwargs.get("hidden_states")
        if (g is None or px is None) and len(args) >= 1:
            px = args[0] if px is None else px
            if g is None and len(args) >= 2:
                g = args[1]
        if g is None:
            return
        if not torch.is_tensor(g):
            g = torch.as_tensor(g)
        state["grid_thw"] = g.detach() if torch.is_tensor(g) else g
        if r == 0.0:
            return
        try:
            off = 0
            for row in g.tolist():
                t_i, h_i, w_i = int(row[0]), int(row[1]), int(row[2])
                n_patch = t_i * h_i * w_i
                edge = _unit_edge_from_pixels(px[off:off + n_patch], t_i, h_i, w_i)
                off += n_patch
                state["edge_queue"].append(edge)
        except Exception as e:                     # defensive: never break the
            diag["edge_fallback"] += 1             # engine on edge computation
            import numpy as np
            if torch.is_tensor(px):
                for row in g.tolist():
                    state["edge_queue"].append(
                        np.zeros(int(row[0] * row[1] * row[2]) // unit,
                                 dtype=np.float64))
            print(f"[hybrid] edge fallback ({type(e).__name__}: {str(e)[:120]})",
                  file=sys.stderr, flush=True)
    visual.register_forward_pre_hook(_visual_prehook, with_kwargs=True)

    # (2) observe the first merger pre-mode ranks from; compute PRE unit scores
    #     without touching the input (post path merges everything) -- identical
    #     to setup_post_merger_swap.
    first_merger = (visual.merger if family == "qwen2vl"
                    else visual.deepstack_merger_list[0])
    orig_fwd = first_merger.forward

    def _wrapped(*args, _orig=orig_fwd, **kwargs):
        hs = args[0]                                   # [seq, 1, ctx]
        if r != 0.0 and state["grid_thw"] is not None:
            seq = hs.shape[0]
            ctx = hs.shape[-1]
            total_units = seq // unit
            feats = hs.reshape(total_units, unit, ctx)
            scores = _score_units(feats, selector)     # [total_units]
            full_units = (state["grid_thw"].prod(-1) // unit).tolist()
            if sum(full_units) == total_units:
                off = 0
                for f in full_units:
                    state["pre_queue"].append(scores[off:off + f])
                    off += f
                diag["score_passes"] += 1
            else:                                      # defensive: grid mismatch
                diag["fallback_stage"] += len(full_units)
        return _orig(*args, **kwargs)
    first_merger.forward = _wrapped
    if hasattr(first_merger, "do_not_compile"):
        first_merger.do_not_compile = True

    # (3) _process_image_input: per split, pop (PRE scores, edge) FIFO, compute
    #     POST scores from the merged tokens, build the hybrid mask, filter.
    _orig_pii = model._process_image_input

    def _patched(image_input):
        splits = _orig_pii(image_input)
        diag["fires"] += 1
        if r == 0.0:
            return splits
        out = []
        for s in splits:
            n = int(s.shape[0])
            k = max(1, int(round(n * (1.0 - r))))
            pre_i = state["pre_queue"].pop(0) if state["pre_queue"] else None
            edge_i = state["edge_queue"].pop(0) if state["edge_queue"] else None
            if (pre_i is not None and edge_i is not None
                    and int(pre_i.shape[0]) == n and len(edge_i) == n):
                diag["consumed"] += 1
                post_i = _score_tokens(s, selector)
                pre_d = pre_i.to(device=s.device)
                edge_t = torch.as_tensor(edge_i, device=s.device,
                                         dtype=torch.float32)
                idx, st = _hybrid_select(pre_d, post_i, edge_t, k, text_frac)
                if st["branch"] == "agree_ge_k":
                    diag["n_agree_ge_k"] += 1
                if len(diag["agree_fracs"]) < 16:
                    diag["agree_fracs"].append(
                        round(st["agree_n"] / max(1, k), 3))
                if save_scores:
                    import numpy as np
                    pre_np = pre_d.float().cpu().numpy()
                    post_np = post_i.float().cpu().numpy()
                    state["stats_log"].append({
                        "n_units": n, "k": k, "agree_n": st["agree_n"],
                        "jaccard_topk": round(st["agree_n"] / max(1, k), 4),
                        "spearman_pre_post": round(_spearman_np(pre_np, post_np), 4),
                        "mean_edge": round(float(np.mean(edge_i)), 5),
                        "branch": st["branch"], "pre_taken": st["pre_taken"],
                        "post_taken": st["post_taken"]})
            else:
                # encoder-cache replay or grid mismatch -> fall back to the
                # post (stage) ranking rather than crash; diag exposes it.
                diag["fallback_stage"] += 1
                if save_scores:
                    state["stats_log"].append(
                        {"n_units": n, "k": k, "fallback": True})
                idx = torch.topk(_score_tokens(s, selector), k).indices.sort().values
            out.append(s.index_select(0, idx).contiguous())
        if len(diag["nk"]) < 8:
            diag["nk"].append((int(splits[0].shape[0]), int(out[0].shape[0])))
        return tuple(out)
    model._process_image_input = _patched
    return diag, state


def attach_hybrid_unit_scores(state, samples, per_sample, model_id: str,
                              max_pixels: int, diag):
    """Best-effort attachment of state["stats_log"] entries to per_sample by
    FIFO order, guarded by per-image num_units. vLLM V1's encoder-cache replay
    serves some requests without firing visual()/merger/pii (both hooks skipped
    together -> the stats queue stays balanced but is missing exactly those
    requests), so a bare positional zip can misalign after a replay: we recompute
    each sample's full unit count offline with the SAME HF processor vLLM uses
    and only attach when it matches the queue head. stats_log[0] = the warmup
    request (dropped). Sets diag["stats_attached"]/["stats_total"]."""
    stats = state["stats_log"][1:]                 # drop warmup entry
    import numpy as np
    full_units = [None] * len(samples)
    try:
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
        for i, smp in enumerate(samples):
            kw = {}
            if max_pixels and max_pixels > 0:
                kw["max_pixels"] = max_pixels
            g = proc.image_processor(Image.open(smp.image),
                                     return_tensors="pt", **kw)["image_grid_thw"]
            full_units[i] = int(g[0].prod()) // 4
    except Exception as e:
        print(f"[hybrid] offline grid recompute failed "
              f"({type(e).__name__}: {str(e)[:160]}); unit scores NOT attached",
              file=sys.stderr, flush=True)
        diag["stats_attached"] = 0
        diag["stats_total"] = len(stats)
        return
    ptr = 0
    attached = 0
    for i, smp in enumerate(samples):
        if i >= len(per_sample):
            break
        if (ptr < len(stats) and full_units[i] is not None
                and stats[ptr].get("n_units") == full_units[i]):
            per_sample[i]["unit_scores"] = stats[ptr]
            ptr += 1
            attached += 1
    diag["stats_attached"] = attached
    diag["stats_total"] = len(stats)
    diag["stats_attach_note"] = (
        "FIFO matched on per-image num_units; unmatched samples were served "
        "from vLLM's encoder-cache replay (no vision forward fired).")


# --------------------------------------------------------------------------- #
# (C) PRE-merger: forward_pre_hooks on the 4 mergers + visual (to capture
# grid_thw) + replace _process_image_input to split by pruned counts.
# --------------------------------------------------------------------------- #
class PreMergerPruner:
    def __init__(self, r: float, spatial_merge_size: int, selector: str = "l2",
                 visionzip_style: bool = False, visionzip_dom_ratio: float = 0.7,
                 mask_ranking: str = "stage",
                 qa_lambda: float = 0.0, qa_state=None):
        self.r = r
        self.sm = spatial_merge_size
        self.unit = spatial_merge_size ** 2          # 4
        self.selector = selector
        self.visionzip_style = visionzip_style
        self.visionzip_dom_ratio = visionzip_dom_ratio
        self.mask_ranking = mask_ranking              # "stage" | "swap" (M3)
        self.qa_lambda = float(qa_lambda)             # J5: 0 = qsim OFF (plain RBM)
        self.qa_state = qa_state                      # J5: from _qa_precompute
        self.full_units = None                        # list[int] per image
        self.k_units = None                           # list[int] per image
        self._mask = None                             # cached token mask
        self.merger_origs = {}                        # tag -> orig forward (swap)
        self.diag = {"visual_calls": 0, "merger_calls": 0,
                     "mask_computed_at": None, "mask_compute_count": 0,
                     "per_tag_calls": {}, "nk": [], "selector": selector,
                     "mask_ranking": mask_ranking,
                     "vz_dom": [], "vz_ctx": []}

    def begin_pass(self, grid_thw):
        """visual.forward pre_hook: capture grid_thw, plan per-image counts."""
        if not torch.is_tensor(grid_thw):
            grid_thw = torch.as_tensor(grid_thw)
        self.full_units = (grid_thw.prod(-1) // self.unit).tolist()
        self.k_units = [max(1, int(round(f * (1.0 - self.r))))
                        for f in self.full_units]
        self._mask = None
        self.diag["visual_calls"] += 1
        if len(self.diag["nk"]) < 8:
            self.diag["nk"].append((self.full_units[0], self.k_units[0]))

    def slice_input(self, hs, module):
        """Return the kept (pruned) merger input hidden_states for one call."""
        self.diag["merger_calls"] += 1
        tag = getattr(module, "_premerger_tag", "?")
        self.diag["per_tag_calls"][tag] = \
            self.diag["per_tag_calls"].get(tag, 0) + 1
        if self.r == 0.0:
            return hs
        seq = hs.shape[0]
        ctx = hs.shape[-1]
        num_units = seq // self.unit
        if self._mask is None:
            # ---- compute per-unit scores + top-k mask ----
            if self.mask_ranking == "swap":
                # M3 control: pre forward path + POST ranking. Run the ORIGINAL
                # (unwrapped) forward of EVERY merger on the FULL input (all
                # mergers consume the same block-major hs), cat the per-unit
                # outputs exactly as qwen3_vl.py visual.forward does, and score
                # the merged tokens with _score_tokens -- i.e. the post ranking.
                assert self.merger_origs, "swap requires merger_origs registered"
                post_feat = torch.cat(
                    [orig(hs) for orig in self.merger_origs.values()], dim=1)
                scores = _score_tokens(post_feat, self.selector)   # [num_units]
                where = f"swap:post-ranking@{tag}"
            else:
                feats = hs.reshape(num_units, self.unit, ctx)
                scores = _score_units(feats, self.selector)        # [num_units]
                where = tag
                # ---- J5 QA-pre: blend query-similarity into the pre ranking ----
                # Only entered when qa_lambda>0 (λ=0 => bit-identical plain RBM).
                # Stage ranking only; swap/visionzip paths are guarded out in main.
                if self.qa_lambda > 0.0 and self.qa_state is not None \
                        and self.qa_state.get("embed") is not None:
                    scores = self._qa_blend_scores(hs, scores, num_units)
            keep = torch.zeros(num_units, dtype=torch.bool,
                               device=hs.device)
            off = 0
            for f, k in zip(self.full_units, self.k_units):
                s_i = scores[off:off + f]
                idx = torch.topk(s_i, k).indices
                keep[off + idx] = True
                off += f
            self._mask = keep.unsqueeze(-1).expand(-1, self.unit).reshape(-1)
            self._vz_scores = scores                 # cached for VisionZip-style
            self.diag["mask_computed_at"] = where
            self.diag["mask_compute_count"] += 1
            self.diag["vz_dom"].clear()
            self.diag["vz_ctx"].clear()
        # ---- VisionZip-style: dominant + context split ----
        # Runs on EVERY merger call (not just first): context tokens are
        # recomputed from the current merger's input features, using the
        # cached per-unit scores.  All mergers produce the same output size
        # (sum k_i per image) so torch.cat across mergers works.
        if self.visionzip_style:
                dom_out = []  # dominant tokens (top-k_dom units per image)
                ctx_out = []  # context tokens per image
                off_img = 0
                feats_vz = hs.reshape(num_units, self.unit, ctx)
                for img_i, (f, k) in enumerate(
                        zip(self.full_units, self.k_units)):
                    k_dom = max(1, int(round(k * self.visionzip_dom_ratio)))
                    k_ctx = k - k_dom
                    s_img = self._vz_scores[off_img:off_img + f]
                    # dominant: top-k_dom units by score (per-image)
                    dom_idx = torch.topk(s_img, k_dom).indices.sort().values
                    _dom_mask = torch.zeros(f, dtype=torch.bool, device=hs.device)
                    _dom_mask[dom_idx] = True
                    _dom_tokens = _dom_mask.unsqueeze(-1).expand(-1, self.unit).reshape(-1)
                    dom_out.append(hs[off_img * self.unit: (off_img + f) * self.unit][_dom_tokens]
                                   .reshape(k_dom, self.unit, ctx))
                    # context: merge remaining (f - k_dom) units into k_ctx groups
                    if k_ctx > 0 and f - k_dom > 0:
                        _nondom = feats_vz[off_img:off_img + f][~_dom_mask]
                        n_nd = _nondom.shape[0]
                        ctx_units_list = []
                        for ci in range(k_ctx):
                            lo = int(ci * n_nd / k_ctx)
                            hi = int((ci + 1) * n_nd / k_ctx)
                            ctx_units_list.append(
                                _nondom[lo:hi].mean(dim=0, keepdim=True))
                        ctx_out.append(torch.cat(ctx_units_list, dim=0))
                    else:
                        ctx_out.append(torch.empty(0, self.unit, ctx,
                                                   device=hs.device, dtype=hs.dtype))
                    if self.diag["merger_calls"] <= len(self.full_units):
                        self.diag["vz_dom"].append(k_dom)
                        self.diag["vz_ctx"].append(k_ctx)
                    off_img += f
                dom_all = torch.cat(dom_out, dim=0) if dom_out else torch.empty(0, self.unit, ctx, device=hs.device, dtype=hs.dtype)
                parts = [dom_all] + ctx_out
                parts = [p for p in parts if p.shape[0] > 0]
                combined = torch.cat(parts, dim=0)
                combined_out = combined.reshape(-1, 1, ctx)
                return combined_out
        # ---- standard pre-merger (dominant-only) ----
        return hs[self._mask]                           # [num_kept, 1, ctx]

    # ---- J5 QA-pre helpers (only active when qa_lambda>0) ------------------ #
    def _qa_pop_embedding(self, f: int):
        """Pop the next normalized question-embedding tensor for an image of `f`
        merge-units. Count-keyed queue first (robust to vLLM batch reordering and
        the warmup encoder-cache replay of sample 0); ordered FIFO fallback.
        Returns [nqt, hidden_llm] (already L2-normalized) or None."""
        qs = self.qa_state
        cq = qs.get("count_queues", {})
        lst = cq.get(f)
        if lst:
            qs["diag"]["pops"] += 1
            return lst.pop(0)
        ordered = qs.get("ordered", [])
        if ordered:
            qs["diag"]["pop_fallback"] += 1
            return ordered.pop(0)
        qs["diag"]["pop_miss"] += 1
        return None

    def _qa_blend_scores(self, hs, base_scores, num_units):
        """Combine the base pre-ranking with query similarity, per image:
            s_i = (1-λ)·minmax(base_i) + λ·minmax(qsim_i)
        qsim_i = max_t cos( merger(unit_i), embed(q_t) ), where merger is the
        NATIVE main merger run once on the full input (reuses the registered
        merger handle; no extra hook) and embed(q_t) are the question's
        normalized word-embedding tokens. Both paths are min-maxed to [0,1] per
        image before weighting, so λ is a real trade-off knob (top-k is
        rank-based). On ANY failure returns base_scores unchanged (safe: the run
        degrades to plain pre-merger rather than crashing the engine)."""
        import torch.nn.functional as F
        qs = self.qa_state
        try:
            merger_orig = qs.get("main_merger_orig")
            if merger_orig is None:
                return base_scores
            merger_out = merger_orig(hs)                # [num_units, hidden_llm]
            m_norm = F.normalize(merger_out.float(), dim=-1)
            combined = base_scores.float().clone()
            off = 0
            for f in self.full_units:
                base_i = base_scores[off:off + f].float()
                qe = self._qa_pop_embedding(int(f))
                if qe is None:
                    # no question embedding for this image (empty queue / empty
                    # question) -> keep the plain base ranking for it.
                    off += f
                    continue
                qe = qe.to(device=hs.device, non_blocking=True)
                sim = m_norm[off:off + f] @ qe.t()      # [f, nqt]
                qsim_i = sim.max(dim=-1).values         # [f]
                combined[off:off + f] = \
                    (1.0 - self.qa_lambda) * _minmax(base_i) \
                    + self.qa_lambda * _minmax(qsim_i)
                qs["qsim_log"].append(
                    (int(f), round(float(qsim_i.mean()), 5)))
                off += f
            qs["diag"]["blends"] += 1
            return combined
        except Exception as e:
            qs["diag"]["blend_fallback"] += 1
            print(f"[qa] blend fallback ({type(e).__name__}: {str(e)[:120]}); "
                  f"using plain selector ranking", file=sys.stderr, flush=True)
            return base_scores


def _wrap_merger_forward(merger, pruner: "PreMergerPruner"):
    """Monkey-patch ``merger.forward`` to slice its input via ``pruner`` BEFORE
    the native 2x2 merge. Used INSTEAD of ``register_forward_pre_hook``:
    ``Qwen2_5_VisionPatchMerger`` is decorated with ``@support_torch_compile``,
    whose custom ``__call__`` (a) bypasses ``nn.Module`` forward_pre_hooks and
    (b) when ``compile_mm_encoder`` is on, dispatches to the captured aot graph
    and never reaches ``self.forward``. Two measures make the wrap actually run:
      1. ``merger.forward = _wrapped`` -- the eager branch of the decorated
         ``__call__`` is ``return self.forward(*args, **kwargs)``, which hits
         our wrap.
      2. ``merger.do_not_compile = True`` -- forces the decorated ``__call__``
         to ALWAYS take that eager branch (never the compiled-graph branch), so
         pruning runs on every call. Only set for decorated mergers (Qwen3-VL's
         merger is a plain nn.Module with no such attr -> guarded out).
    Numerically identical for both families. Returns the original forward for
    potential restoration."""
    orig_forward = merger.forward

    def _wrapped(*args, _orig=orig_forward, _m=merger, **kwargs):
        import traceback
        hs = args[0]                                   # [seq, 1, ctx]
        kept = pruner.slice_input(hs, _m)
        out = _orig(kept, *args[1:], **kwargs)
        print(f"[DIAG_wrap] tag={getattr(_m, '_premerger_tag', '?')} "
              f"hs={tuple(hs.shape)} kept={tuple(kept.shape)} "
              f"out={tuple(out.shape)} n={pruner.diag['merger_calls']} "
              f"do_not_compile={getattr(_m, 'do_not_compile', 'NA')} "
              f"id_merger={id(_m)} id_self_forward={id(_m.forward)}",
              file=sys.stderr, flush=True)
        if not getattr(_m, "_vtc_stk_done", False):
            for fr in traceback.format_stack()[-6:-1]:
                print("    " + fr.strip().replace("\n", " "), file=sys.stderr, flush=True)
            _m._vtc_stk_done = True
        return out

    merger.forward = _wrapped
    if hasattr(merger, "do_not_compile"):
        merger.do_not_compile = True
    return orig_forward


# --------------------------------------------------------------------------- #
# Qwen2.5-VL pre-merger visual.forward patch: skip reverse_indices after merger.
# Qwen2.5-VL's visual.forward applies window-attention permutation before the
# merger and restores spatial order via reverse_indices after the merger.
# In pre-merger mode the merger input is pruned (by _wrap_merger_forward) so
# the output has k_units < full_units tokens.  reverse_indices maps k_units
# back to full_units — a shape mismatch that crashes (split_with_sizes).
# This patch is a LINE-BY-LINE copy of vLLM 0.19's
# Qwen2_5_VLVisionTransformer.forward (qwen2_5_vl.py:777–877) with ONE
# change: the reverse_indices restoration is SKIPPED, so the output stays
# compressed.
# --------------------------------------------------------------------------- #
def _install_qwen2vl_pre_visual_forward(visual, pruner):
    """Replace Qwen2.5-VL visual.forward to skip reverse_indices after merger.

    IMPORTANT: we assign ``visual.forward = patched_forward`` as an INSTANCE
    attribute.  nn.Module._call_impl accesses ``self.forward`` which for
    instance attributes returns the bare function WITHOUT Python descriptor
    binding — so ``self`` is NOT passed implicitly.  We capture ``visual``
    in the closure instead of relying on ``self``.
    """
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.models.utils import cast_overflow_tensors

    _visual = visual

    def patched_forward(hidden_states, grid_thw):
        # ── vLLM 0.19 qwen2_5_vl.py Qwen2_5_VLVisionTransformer.forward ──
        seq_len, _ = hidden_states.size()
        rotary_pos_emb_cos: list = []
        rotary_pos_emb_sin: list = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = hidden_states.to(device=_visual.device, dtype=_visual.dtype)
        hidden_states = _visual.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            (cos_thw, sin_thw, window_index_thw,
             cu_seqlens_window_thw, cu_seqlens_thw,
             ) = _visual.get_rope_by_thw(t, h, w)
            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * (h // _visual.spatial_merge_size) * (
                w // _visual.spatial_merge_size)
            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)
            rotary_pos_emb_cos.append(cos_thw)
            rotary_pos_emb_sin.append(sin_thw)
            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb_cos = torch.cat(rotary_pos_emb_cos)
        rotary_pos_emb_sin = torch.cat(rotary_pos_emb_sin)
        window_index = torch.cat(window_index)
        reverse_indices = _visual.invert_permutation(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
        max_seqlen_full = _visual.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window = _visual.compute_attn_mask_seqlen(cu_window_seqlens)
        cu_seqlens = cu_seqlens.to(device=_visual.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=_visual.device,
                                                  non_blocking=True)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(device=_visual.device,
                                                   non_blocking=True)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(device=_visual.device,
                                                   non_blocking=True)
        window_index = window_index.to(device=hidden_states.device,
                                       non_blocking=True)
        reverse_indices = reverse_indices.to(device=hidden_states.device,
                                             non_blocking=True)

        hidden_states = hidden_states.reshape(
            seq_len // _visual.spatial_merge_unit, _visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(_visual.blocks):
            if layer_num in _visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens_now,
                                rotary_pos_emb_cos=rotary_pos_emb_cos,
                                rotary_pos_emb_sin=rotary_pos_emb_sin,
                                max_seqlen=max_seqlen_now)

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # merger → wrapped forward prunes input → compressed output
        hidden_states = _visual.merger(hidden_states)
        # ── PATCH: SKIP reverse_indices restoration ──
        # Original (line 877): hidden_states = hidden_states[reverse_indices, :]
        # This would restore full_units from compressed k_units → shape
        # mismatch crash.  Without it, output stays compressed.
        return hidden_states

    patched_forward._premerger_original = visual.forward
    visual.forward = patched_forward
    return patched_forward._premerger_original


def setup_pre_merger(model, r: float, selector: str = "l2", family: str = "qwen3vl",
                      visionzip_style: bool = False, visionzip_dom_ratio: float = 0.7,
                      mask_ranking: str = "stage",
                      qa_lambda: float = 0.0, qa_state=None):
    if mask_ranking == "swap" and visionzip_style:
        raise SystemExit("--mask-ranking swap is not supported with "
                         "--visionzip-style (dominant-only standard path only).")
    visual = model.visual
    sm = visual.spatial_merge_size
    pruner = PreMergerPruner(r, sm, selector, visionzip_style, visionzip_dom_ratio,
                             mask_ranking=mask_ranking,
                             qa_lambda=qa_lambda, qa_state=qa_state)

    # (1) visual.forward pre_hook: capture grid_thw -> plan k_units.
    def _visual_prehook(module, args, kwargs):
        grid_thw = kwargs.get("grid_thw")
        if grid_thw is None and len(args) >= 2:
            grid_thw = args[1]
        if grid_thw is not None:
            pruner.begin_pass(grid_thw)
        return None
    handle_v = visual.register_forward_pre_hook(_visual_prehook, with_kwargs=True)

    # (2) merger [+ deepstack mergers for qwen3vl] pre_hooks.
    #   qwen3vl: hook visual.merger AND each visual.deepstack_merger_list[*] --
    #     all 4 consume the same block-major hidden_states, so ONE cached mask
    #     (computed at the first merger's input) applies to all -> the deepstack
    #     cat never sees a seq mismatch.
    #   qwen2vl: hook visual.merger ONLY -- Qwen2.5-VL has no deepstack
    #     (confirmed: Qwen2_5_VisionTransformer defines no deepstack_merger_list).
    handles = [handle_v]
    visual.merger._premerger_tag = "main"
    targets = [visual.merger]
    if family == "qwen3vl":
        for i, m in enumerate(visual.deepstack_merger_list):
            m._premerger_tag = f"deepstack_{i}"
            targets.append(m)
    for m in targets:
        orig = _wrap_merger_forward(m, pruner)
        pruner.merger_origs[m._premerger_tag] = orig   # for mask_ranking=swap
        handles.append(orig)

    # J5 QA-pre: the qsim signal runs the NATIVE main merger on the unit inputs
    # to reach LLM space. Reuse the already-registered original (unwrapped)
    # main-merger forward -- no extra hook. ("main" tag is set on visual.merger
    # for BOTH families above.) Calling it inside slice_input is side-effect-free
    # and never re-enters the wrap (it is the captured orig forward).
    if qa_state is not None and "main" in pruner.merger_origs:
        qa_state["main_merger_orig"] = pruner.merger_origs["main"]

    # ---- TEMP DIAG: wrap visual.forward + dump compiler/merger facts --------
    _mg = visual.merger
    try:
        _cc = visual.vllm_config.compilation_config
        print(f"[DIAG_cc] compile_mm_encoder={_cc.compile_mm_encoder} "
              f"cudagraph_mm_encoder={_cc.cudagraph_mm_encoder} mode={_cc.mode}",
              file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[DIAG_cc] unavailable ({_e})", file=sys.stderr, flush=True)
    print(f"[DIAG_merger] type={type(_mg).__name__} "
          f"bases={[b.__name__ for b in type(_mg).__bases__]} "
          f"do_not_compile={getattr(_mg, 'do_not_compile', 'NA')} "
          f"forward_name={getattr(_mg.forward, '__name__', '?')} "
          f"call_owner={type(_mg).__call__.__qualname__}",
          file=sys.stderr, flush=True)
    _orig_vf = visual.forward

    def _diag_visual(*args, _ovf=_orig_vf, **kwargs):
        print(f"[DIAG_visual_in] merger_calls={pruner.diag['merger_calls']}",
              file=sys.stderr, flush=True)
        out = _ovf(*args, **kwargs)
        print(f"[DIAG_visual_out] out={tuple(out.shape)} "
              f"merger_calls={pruner.diag['merger_calls']}",
              file=sys.stderr, flush=True)
        return out
    visual.forward = _diag_visual
    # ---- /TEMP DIAG --------------------------------------------------------

    # (2b) Qwen2.5-VL: replace diagnostic wrapper with reverse_indices-skip
    # patch.  Without this, merger output gets inflated from k_units back to
    # full_units by reverse_indices → split_with_sizes crash.
    if family == "qwen2vl":
        _install_qwen2vl_pre_visual_forward(visual, pruner)

    # (3) replace _process_image_input: split by PRUNED counts (k_units).
    _orig_pii = model._process_image_input

    def _patched_pii(image_input):
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(visual.dtype)
            image_embeds = visual(pixel_values, grid_thw=grid_thw)
        if (r == 0.0) or (pruner.k_units is None):
            sizes = (grid_thw.prod(-1) // pruner.unit).tolist()
        else:
            sizes = pruner.k_units
        import sys
        print(f"[DIAG_pii] type={image_input.get('type')} embeds_dim0={image_embeds.shape[0]} "
              f"sizes={sizes} k_units={pruner.k_units} merger_calls={pruner.diag.get('merger_calls')} "
              f"mask_comp={pruner.diag.get('mask_compute_count')}", file=sys.stderr, flush=True)
        return image_embeds.split(sizes)
    model._process_image_input = _patched_pii

    return pruner, handles


# --------------------------------------------------------------------------- #
# No-GPU dry check: validates the family-aware code paths (imports, argparse,
# processor class selection, hook TARGET SET, cached-mask logic, post prune)
# on dummy nn.Module objects + synthetic tensors. Does NOT load vLLM or touch a
# GPU. GPU-dependent and thus UNTESTED here: real vLLM LLM load, the actual
# merger forward under the hook with real weights, end-to-end chat, and runtime
# mrope-position correctness (the runner relies on vLLM's placeholder-count-
# based mrope, identical contract for both families).
# --------------------------------------------------------------------------- #
def run_dry_check(family: str):
    import torch.nn as nn

    class _DummyMerger(nn.Module):
        def forward(self, x):
            # real mergers: [seq, 1, ctx] -> [num_units, hidden]; mimic the
            # 4-to-1 contraction so swap-mode cat/scoring shapes match.
            return x.reshape(x.shape[0] // 4, -1)

    class _DummyVisual(nn.Module):
        def __init__(self, fam):
            super().__init__()
            self.spatial_merge_size = 2
            self.merger = _DummyMerger()
            if fam == "qwen3vl":
                self.deepstack_merger_list = nn.ModuleList(
                    [_DummyMerger() for _ in range(3)])

        def forward(self, x, grid_thw=None):
            return x

    class _DummyModel:
        def __init__(self, fam):
            self.visual = _DummyVisual(fam)
            self._process_image_input = lambda ii: tuple()

    print(f"[dry-check] family={family}  model_id={MODELS[family]}")
    model = _DummyModel(family)

    # (a) processor patch installs on the RIGHT class for this family
    ProcCls = _proc_class(family)
    want = (_q2vl_mod.Qwen2_5_VLMultiModalProcessor if family == "qwen2vl"
            else _q3vl_mod.Qwen3VLMultiModalProcessor)
    assert ProcCls is want, f"proc class mismatch: {ProcCls} vs {want}"
    proc_log = patch_processor(0.75, family)
    assert getattr(ProcCls._get_prompt_updates, "_vtc_patched", False), "patch not installed"
    # restore so repeated dry-checks across families stay clean
    ProcCls._get_prompt_updates = ProcCls._get_prompt_updates._vtc_orig
    print(f"[dry-check]   OK processor patch on {ProcCls.__name__}")

    # (b) pre-merger hook TARGET SET matches family (deepstack included/omitted)
    pruner, handles = setup_pre_merger(model, 0.75, "l2", family)
    n_merger_hooks = len(handles) - 1                 # -1 for the visual.fwd hook
    expected = 4 if family == "qwen3vl" else 1        # merger + 3 deepstack | merger only
    assert n_merger_hooks == expected, \
        f"merger hook count {n_merger_hooks} != expected {expected}"
    print(f"[dry-check]   OK pre-merger targets: {n_merger_hooks} merger hooks "
          f"(expected {expected}); deepstack "
          f"{'INCLUDED' if family == 'qwen3vl' else 'OMITTED (correct for qwen2vl)'}")

    # (c) cached-once-per-pass mask + per-image topk on a synthetic 2-image batch
    unit = model.visual.spatial_merge_size ** 2        # 4
    full_units = [8, 4]
    grid_thw = torch.tensor([[1, 4, 8], [1, 4, 4]])   # prod(-1)//4 = [8, 4]
    pruner.begin_pass(grid_thw)
    assert pruner.full_units == full_units, pruner.full_units
    assert pruner.k_units == [2, 1], pruner.k_units   # round([8,4]*0.25) = [2,1]
    seq = sum(full_units) * unit                       # 48 pre-merger tokens
    hs = torch.randn(seq, 1, 768)
    out = pruner.slice_input(hs, model.visual.merger)
    kept = out.shape[0]
    assert kept == sum(pruner.k_units) * unit == 12, kept
    # fire once more -> mask is cached, count stable, no recompute
    _ = pruner.slice_input(hs, model.visual.merger)
    assert pruner.diag["mask_compute_count"] == 1, pruner.diag["mask_compute_count"]
    print(f"[dry-check]   OK mask logic: {seq} pre-merger toks -> {kept} kept "
          f"(units {full_units}->{pruner.k_units}); mask computed once/cached")

    # (d) post-merger prune (family-agnostic): wraps _process_image_input
    model2 = _DummyModel(family)
    called = {"n": 0}
    def _fake_orig(ii):
        called["n"] += 1
        return (torch.randn(40, 16), torch.randn(20, 16))   # 2 image splits
    model2._process_image_input = _fake_orig
    diag = setup_post_merger(model2, 0.75, "l2")
    out = model2._process_image_input(None)
    assert called["n"] == 1 and diag["fires"] == 1
    assert [s.shape[0] for s in out] == [10, 5], \
        [s.shape[0] for s in out]    # round([40,20]*0.25) = [10,5]
    print(f"[dry-check]   OK post-merger prune: splits [40,20] -> "
          f"{[s.shape[0] for s in out]} (r=0.75)")

    # (e) M3 post+swap: POST forward path (nothing sliced) + PRE-ranking select.
    #     grid (1,4,8) -> 8 units -> k=round(8*0.25)=2.
    model3 = _DummyModel(family)
    grid = torch.tensor([[1, 4, 8]])
    full_hs = torch.randn(8 * unit, 1, 16)
    def _fake_pii3(ii):
        return (torch.randn(8, 32),)                # fully merged, 1 image
    model3._process_image_input = _fake_pii3        # BEFORE setup: gets wrapped
    diag3, _state3 = setup_post_merger_swap(model3, 0.75, "l2", family)
    model3.visual(full_hs, grid_thw=grid)           # fires the grid pre_hook
    first = (model3.visual.merger if family == "qwen2vl"
             else model3.visual.deepstack_merger_list[0])
    out_m = first(full_hs)                          # wrapped: queues PRE scores
    assert out_m.shape[0] == 8, out_m.shape         # merger input untouched
    assert diag3["score_passes"] == 1, diag3
    out3 = model3._process_image_input(None)        # selects by PRE ranking
    assert [s.shape[0] for s in out3] == [2], [s.shape for s in out3]
    assert diag3["consumed"] == 1 and diag3["fallback_stage"] == 0, diag3
    print(f"[dry-check]   OK post+swap control: merger untouched (8 units), "
          f"split 8 -> {[s.shape[0] for s in out3]} by PRE ranking "
          f"(consumed={diag3['consumed']}, fallback={diag3['fallback_stage']})")

    # (f) M3 pre+swap: PRE (sliced) forward path + POST-ranking select.
    model4 = _DummyModel(family)
    pruner4, handles4 = setup_pre_merger(model4, 0.75, "l2", family,
                                          mask_ranking="swap")
    pruner4.begin_pass(grid)
    out4 = pruner4.slice_input(full_hs, model4.visual.merger)
    assert out4.shape[0] == pruner4.k_units[0] * unit == 8, out4.shape
    assert str(pruner4.diag["mask_computed_at"]).startswith("swap:"), \
        pruner4.diag["mask_computed_at"]
    print(f"[dry-check]   OK pre+swap mask: selected by POST ranking "
          f"(mask_computed_at={pruner4.diag['mask_computed_at']}); "
          f"kept {out4.shape[0]} tokens")

    # (g) HYBRID: post forward path + hybrid pre/post/edge mask. grid (1,4,8)
    #     -> 8 units -> k=2. Pixels [32, 24] = 6*p^2 with p=2. Checks: edge
    #     queue + PRE queue populate from the hooks, the mask keeps EXACTLY k
    #     merged tokens, diag counters balance, save_scores logs one entry.
    model5 = _DummyModel(family)
    grid5 = torch.tensor([[1, 4, 8]])
    p_pix = 2
    pixels5 = torch.randn(8 * unit, 6 * p_pix * p_pix)

    def _fake_pii5(ii):
        return (torch.randn(8, 32),)                # fully merged, 1 image
    model5._process_image_input = _fake_pii5
    diag5, state5 = setup_hybrid(model5, 0.75, "l2", family,
                                 text_frac=1.0, save_scores=True)
    model5.visual(pixels5, grid_thw=grid5)          # prehook: grid + edge
    assert len(state5["edge_queue"]) == 1 and state5["edge_queue"][0].shape[0] == 8, \
        [a.shape for a in state5["edge_queue"]]
    first5 = (model5.visual.merger if family == "qwen2vl"
              else model5.visual.deepstack_merger_list[0])
    out5m = first5(torch.randn(8 * unit, 1, 16))    # wrapped: queues PRE scores
    assert out5m.shape[0] == 8 and diag5["score_passes"] == 1
    out5 = model5._process_image_input(None)        # hybrid selection
    assert [s.shape[0] for s in out5] == [2], [s.shape for s in out5]
    assert diag5["consumed"] == 1 and diag5["fallback_stage"] == 0, diag5
    assert diag5["edge_fallback"] == 0, diag5
    assert len(state5["stats_log"]) == 1 and \
        state5["stats_log"][0]["n_units"] == 8 and state5["stats_log"][0]["k"] == 2
    # exactly-k also at text_frac=0.0 and on a 2nd image (queue continues)
    model5.visual(pixels5, grid_thw=grid5)
    _ = first5(torch.randn(8 * unit, 1, 16))
    diag5b, _ = diag5, None
    out5b = model5._process_image_input(None)
    assert [s.shape[0] for s in out5b] == [2] and diag5["consumed"] == 2
    print(f"[dry-check]   OK hybrid selection: splits 8 -> {[s.shape[0] for s in out5]} "
          f"by agreement+text-routed mask (consumed={diag5['consumed']}, "
          f"fallback={diag5['fallback_stage']}, edge_fallback={diag5['edge_fallback']}, "
          f"stats={state5['stats_log'][0]['jaccard_topk']})")

    # (h) _hybrid_select unit test: agreement/routing/overflow branches, all
    #     keep exactly k on random inputs (text_frac in {0, 0.5, 1}).
    torch.manual_seed(0)
    for f_u, k_u in [(40, 10), (41, 7), (9, 9), (16, 1)]:
        for tf in (0.0, 0.5, 1.0):
            pre_r = torch.randn(f_u)
            post_r = pre_r + torch.randn(f_u) * 2.0
            edge_r = torch.rand(f_u)
            idx_r, st_r = _hybrid_select(pre_r, post_r, edge_r, k_u, tf)
            assert idx_r.numel() == k_u and idx_r.unique().numel() == k_u, \
                (f_u, k_u, tf, idx_r.numel())
    print(f"[dry-check]   OK _hybrid_select: exactly-k masks across shapes/"
          f"text_frac (branch example: {_hybrid_select(torch.randn(40), torch.randn(40), torch.rand(40), 10, 1.0)[1]['branch']})")

    print(f"[dry-check] ALL PASS for family={family}")


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # Resolve family + model id. --model (if given) triggers auto-detection and
    # overrides --model-family; otherwise use --model-family's standard model.
    if args.model:
        family = detect_family(args.model)
        model_id = args.model
    else:
        family = args.model_family
        model_id = MODELS[family]

    # No-GPU dry check: validate imports + hook setup on dummy modules for the
    # chosen family. Exits before any vLLM/GPU use.
    if args.dry_check:
        run_dry_check(family)
        return

    if args.out is None:
        raise SystemExit("--out is required when not using --dry-check")
    # --dry-check skips these (it never loads data/runs a benchmark); a real
    # run requires all three.
    missing = [n for n, v in (("--mode", args.mode),
                              ("--benchmark", args.benchmark),
                              ("--subset", args.subset)) if not v]
    if missing:
        raise SystemExit("required (unless --dry-check): " + ", ".join(missing))

    r = args.r
    if args.mode == "none":
        r = 0.0
        if args.mask_ranking == "swap":
            raise SystemExit("--mask-ranking swap requires --mode post or pre")
    if args.mode == "hybrid":
        if args.mask_ranking == "swap":
            raise SystemExit("--mask-ranking swap is not applicable to "
                             "--mode hybrid (hybrid IS a crossed-ranking mask)")
        if args.visionzip_style:
            raise SystemExit("--visionzip-style is not supported with "
                             "--mode hybrid (dominant-only standard path only)")
        if not 0.0 <= args.hybrid_text_frac <= 1.0:
            raise SystemExit("--hybrid-text-frac must be in [0,1]")
    # J5 QA-pre is defined for the STANDARD pre path only (stage ranking,
    # dominant-only). Guard the other paths so λ>0 can never silently touch
    # post/hybrid/swap/visionzip behavior. λ=0 is always allowed (no-op).
    if args.qa_lambda > 0.0:
        if not (0.0 <= args.qa_lambda <= 1.0):
            raise SystemExit("--qa-lambda must be in [0,1]")
        if args.mode != "pre":
            raise SystemExit("--qa-lambda>0 requires --mode pre (QA-pre is a "
                             "pre-merger saliency signal)")
        if args.mask_ranking == "swap":
            raise SystemExit("--qa-lambda>0 is not supported with "
                             "--mask-ranking swap (stage ranking only)")
        if args.visionzip_style:
            raise SystemExit("--qa-lambda>0 is not supported with "
                             "--visionzip-style (dominant-only standard path)")
    proc_log = patch_processor(r, family)

    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    llm_kwargs = dict(
        model=model_id, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False, enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=False, enable_prefix_caching=False,
        seed=args.seed,
    )
    # Cap image resolution if requested (DocVQA encoder-cache crash fix).
    # pre-merger tokens ~= max_pixels / (patch=16)^2 ; 1.5M px -> ~5859 tokens.
    if args.max_pixels and args.max_pixels > 0:
        llm_kwargs["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}
    # max_num_batched_tokens also sets the V1 multimodal encoder cache budget
    # (scheduler.py: encoder_cache_size = max_num_batched_tokens). Required for
    # DocVQA post-merger: a single huge document image-item embed_length (~16k)
    # exceeds vLLM's default 8192 -> ValueError + cascading OOM/skip-all.
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - t0
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Loaded early (before mode setup) so the J5 QA-pre path can embed the
    # questions before generation; reused unchanged by the chat loop below.
    samples = load_subset(args.subset)[:args.n]
    if args.max_pixels and args.max_pixels > 0:
        n_cap = 0
        for s in samples:
            new = _capped_image_path(s.image, args.max_pixels)
            if new != s.image:
                n_cap += 1
            s.image = new
        print(f"[cap] {n_cap}/{len(samples)} images pre-resized to <= "
              f"{args.max_pixels} px (vLLM ignores max_pixels kwargs; "
              f"cache runs/data/_capped_cache)", flush=True)

    diag = None
    swap_state = None
    hybrid_state = None
    qa_state = None
    if args.mode == "post":
        if args.mask_ranking == "swap":
            # M3: post forward path + PRE ranking selection.
            diag, swap_state = setup_post_merger_swap(model, r, args.selector,
                                                      family)
        else:
            diag = setup_post_merger(model, r, args.selector)
    elif args.mode == "hybrid":
        # Merger-aware selection: post forward path + hybrid pre/post/edge mask.
        diag, hybrid_state = setup_hybrid(model, r, args.selector, family,
                                          args.hybrid_text_frac,
                                          args.save_unit_scores)
    elif args.mode == "pre":
        # J5 QA-pre: build the query-similarity state (embed questions, key by
        # offline unit count) ONLY when λ>0. λ=0 -> qa_state stays None and
        # setup_pre_merger behaves bit-identically to plain pre-merger.
        if args.qa_lambda > 0.0:
            try:
                tokenizer = llm.get_tokenizer()
            except Exception:
                tokenizer = llm.llm_engine.tokenizer.tokenizer
            qa_state = _qa_precompute(model, tokenizer, samples,
                                      args.qa_embed_cache, args.max_pixels,
                                      model_id)
        pruner, _handles = setup_pre_merger(model, r, args.selector, family,
                                             visionzip_style=args.visionzip_style,
                                             visionzip_dom_ratio=args.visionzip_dom_ratio,
                                             mask_ranking=args.mask_ranking,
                                             qa_lambda=args.qa_lambda,
                                             qa_state=qa_state)
        diag = pruner.diag

    # Qwen2.5-VL ONLY: fix the mrope-position overshoot that collapses pruned
    # acc to ~0.004 (see setup_qwen2vl_mrope_fix docstring).  No-op for r=0 and
    # never installed for qwen3vl -> baseline + qwen3vl behavior untouched.
    mrope_fix_diag = None
    if family == "qwen2vl" and r > 0.0:
        mrope_fix_diag = setup_qwen2vl_mrope_fix(model)

    scorer = SCORERS[args.benchmark]

    def make_msgs(s: Sample):
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://" + s.image}},
            {"type": "text", "text": s.question},
        ]}]

    msgs_all = [make_msgs(s) for s in samples]
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    # vLLM 0.19 V1 IGNORES engine-level mm_processor_kwargs (verified: DocVQA
    # ptid identical with/without the engine kwarg) -> pass it per request.
    chat_kw = {}
    if args.max_pixels and args.max_pixels > 0:
        chat_kw["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}

    # warmup 1 fwd (not timed) so eager kernels are primed
    llm.chat([msgs_all[0]], sampling_params=sp, **chat_kw)

    t0 = time.perf_counter()
    n_skip = 0
    try:
        outs = llm.chat(msgs_all, sampling_params=sp, **chat_kw)
        wall = time.perf_counter() - t0
    except Exception as e:
        # Batched call aborts if ANY single prompt exceeds max_model_len (or
        # another per-sample failure). Re-run per-sample, skip the failures so
        # one huge document cannot abort the whole DocVQA run.
        print(f"[v3] batched chat failed ({type(e).__name__}: {str(e)[:200]}); "
              f"falling back to per-sample (skip-on-error).", flush=True)
        outs = [None] * len(msgs_all)
        t0 = time.perf_counter()
        for i, m in enumerate(msgs_all):
            try:
                outs[i] = llm.chat([m], sampling_params=sp, **chat_kw)[0]
            except Exception:
                outs[i] = None
                n_skip += 1
        wall = time.perf_counter() - t0

    n_ok = 0
    correct = 0
    kept_counts = []
    per_sample = []
    for s, o in zip(samples, outs):
        if o is None:
            per_sample.append({"id": s.id, "correct": 0, "skipped": True,
                               "answer": "", "gt": s.gt, "question": s.question})
            continue
        ans = o.outputs[0].text.strip()
        if ans:
            n_ok += 1
        c = scorer(ans, s.gt, s.extra.get("choices"))
        correct += c
        ptid_len = len(o.prompt_token_ids)
        kept_counts.append(ptid_len)
        per_sample.append({"id": s.id, "correct": int(c), "skipped": False,
                           "answer": ans, "gt": s.gt, "question": s.question,
                           "prompt_token_ids": ptid_len})
    n_scored = len(samples) - n_skip
    req_s = n_scored / wall if wall > 0 else 0.0
    acc = correct / n_scored if n_scored else 0.0

    if swap_state is not None and diag is not None:
        # M3 bookkeeping: every queued PRE-ranking entry must be consumed by
        # exactly one _process_image_input split (0 leftover, 0 fallback in a
        # clean run); nonzero => visual()/pii pairing broke for some images.
        diag["swap_queue_leftover"] = len(swap_state["queue"])
    if hybrid_state is not None and diag is not None:
        # Same FIFO-balance bookkeeping for the hybrid queues.
        diag["hybrid_queue_leftover"] = (len(hybrid_state["pre_queue"])
                                         + len(hybrid_state["edge_queue"]))
        if args.save_unit_scores:
            attach_hybrid_unit_scores(hybrid_state, samples, per_sample,
                                      model_id, args.max_pixels, diag)
    if qa_state is not None and args.qa_lambda > 0.0:
        # J5 QA-pre: stamp qa_lambda on every sample (analysis) + best-effort
        # per-image mean qsim; expose the qsim bookkeeping counters.
        for p in per_sample:
            p["qa_lambda"] = args.qa_lambda
        attach_qa_per_sample(qa_state, samples, per_sample, diag)

    result = {
        "model": model_id, "model_family": family,
        "mode": args.mode, "mask_ranking": args.mask_ranking,
        "benchmark": args.benchmark, "r": r,
        "max_num_seqs": args.max_num_seqs, "n": len(samples),
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "visionzip_style": args.visionzip_style,
        "visionzip_dom_ratio": args.visionzip_dom_ratio,
        "selector": args.selector, "max_pixels": args.max_pixels,
        "seed": args.seed,
        "wall_s": round(wall, 3), "req_per_s": round(req_s, 4),
        "acc": round(acc, 4), "n_answered": n_ok, "n_skipped": n_skip,
        "mean_ptid_len": round(sum(kept_counts) / len(kept_counts), 1) if kept_counts else 0,
        "load_s": round(load_s, 1),
        "proc_placeholder_counts": proc_log["counts"][:3],
        "diag": diag,
        "vllm": vllm.__version__,
        "per_sample": per_sample,
    }
    if args.mode == "hybrid":
        result["hybrid_text_frac"] = args.hybrid_text_frac
        result["save_unit_scores"] = args.save_unit_scores
    if qa_state is not None and args.qa_lambda > 0.0:
        result["qa_lambda"] = args.qa_lambda
        result["qa_embed_cache"] = args.qa_embed_cache
        result["qa"] = qa_state["diag"]                 # embed path + qsim counters
    if mrope_fix_diag is not None:
        result["mrope_fix"] = mrope_fix_diag
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[v3] mode={args.mode} r={r} {args.benchmark}: "
          f"req/s={req_s:.3f} acc={acc:.3f} wall={wall:.1f}s "
          f"mean_ptid={result['mean_ptid_len']:.0f}", flush=True)


if __name__ == "__main__":
    main()
