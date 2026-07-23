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


def parse_args():
    ap = argparse.ArgumentParser()
    # --mode/--benchmark/--subset are required for a real run but NOT for
    # --dry-check (which validates hook setup on dummy modules without a GPU).
    # main() enforces their presence when not dry-checking.
    ap.add_argument("--mode", required=False, default=None,
                    choices=["none", "post", "pre"])
    ap.add_argument("--r", type=float, default=0.0,
                    help="prune ratio; k_i = round(full_i*(1-r)). "
                         "{0.5,0.75,0.875} -> keep {50,25,12.5}% of merge-units.")
    ap.add_argument("--max-num-seqs", type=int, default=16)
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
# (C) PRE-merger: forward_pre_hooks on the 4 mergers + visual (to capture
# grid_thw) + replace _process_image_input to split by pruned counts.
# --------------------------------------------------------------------------- #
class PreMergerPruner:
    def __init__(self, r: float, spatial_merge_size: int, selector: str = "l2",
                 visionzip_style: bool = False, visionzip_dom_ratio: float = 0.7,
                 mask_ranking: str = "stage"):
        self.r = r
        self.sm = spatial_merge_size
        self.unit = spatial_merge_size ** 2          # 4
        self.selector = selector
        self.visionzip_style = visionzip_style
        self.visionzip_dom_ratio = visionzip_dom_ratio
        self.mask_ranking = mask_ranking              # "stage" | "swap" (M3)
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
                      mask_ranking: str = "stage"):
    if mask_ranking == "swap" and visionzip_style:
        raise SystemExit("--mask-ranking swap is not supported with "
                         "--visionzip-style (dominant-only standard path only).")
    visual = model.visual
    sm = visual.spatial_merge_size
    pruner = PreMergerPruner(r, sm, selector, visionzip_style, visionzip_dom_ratio,
                             mask_ranking=mask_ranking)

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
    proc_log = patch_processor(r, family)

    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    llm_kwargs = dict(
        model=model_id, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.90, max_model_len=args.max_model_len,
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

    diag = None
    swap_state = None
    if args.mode == "post":
        if args.mask_ranking == "swap":
            # M3: post forward path + PRE ranking selection.
            diag, swap_state = setup_post_merger_swap(model, r, args.selector,
                                                      family)
        else:
            diag = setup_post_merger(model, r, args.selector)
    elif args.mode == "pre":
        pruner, _handles = setup_pre_merger(model, r, args.selector, family,
                                             visionzip_style=args.visionzip_style,
                                             visionzip_dom_ratio=args.visionzip_dom_ratio,
                                             mask_ranking=args.mask_ranking)
        diag = pruner.diag

    samples = load_subset(args.subset)[:args.n]
    scorer = SCORERS[args.benchmark]

    def make_msgs(s: Sample):
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://" + s.image}},
            {"type": "text", "text": s.question},
        ]}]

    msgs_all = [make_msgs(s) for s in samples]
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # warmup 1 fwd (not timed) so eager kernels are primed
    llm.chat([msgs_all[0]], sampling_params=sp)

    t0 = time.perf_counter()
    n_skip = 0
    try:
        outs = llm.chat(msgs_all, sampling_params=sp)
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
                outs[i] = llm.chat([m], sampling_params=sp)[0]
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
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[v3] mode={args.mode} r={r} {args.benchmark}: "
          f"req/s={req_s:.3f} acc={acc:.3f} wall={wall:.1f}s "
          f"mean_ptid={result['mean_ptid_len']:.0f}", flush=True)


if __name__ == "__main__":
    main()
