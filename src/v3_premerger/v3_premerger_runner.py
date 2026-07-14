"""V3 go/no-go runner: PRE-merger vs POST-merger pruning on Qwen3-VL-8B-Instruct.

Isolates the STAGE effect (prune BEFORE vs AFTER the native 2x2 merger) at
iso-final-token-budget, using the SAME text-agnostic L2-norm selector at each
respective stage.

Modes:
  --mode none   (A) no pruning (baseline)
  --mode post   (B) POST-merger: hook model._process_image_input, prune each
                      per-image embed (post-split, full multiscale row) to
                      k_i = round(full_i*(1-r)) by L2-norm. (== v2_p1 baseline.)
  --mode pre    (C) PRE-merger: register_forward_pre_hook on visual.merger AND
                      each visual.deepstack_merger_list[*]. All 4 mergers consume
                      the same block-major hidden_states (groups of 4 consecutive
                      tokens = 1 merge-unit). ONE keep-mask over merge-units,
                      computed once from the first merger's input (deepstack[0],
                      layer-8 features) and cached, is applied to all 4 -> the
                      deepstack cat (qwen3_vl.py L654) never sees a seq mismatch.
                      _process_image_input is replaced to split by the PRUNED
                      per-image counts (k_units). Processor placeholder patch is
                      IDENTICAL to post-merger (scales by (1-r)).

enforce_eager=True (variable per-request pruning breaks encoder CUDA graphs).
One cell per fresh process. M-RoPE handled by vLLM's recompute_mrope_positions
(same as v2_p1; output shape is identical between post and pre).
"""
from __future__ import annotations
import os, sys, time, json, argparse

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
import vllm
import dataclasses as _dc
import vllm.model_executor.models.qwen3_vl as _q3vl_mod

MODEL = "Qwen/Qwen3-VL-8B-Instruct"

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


SCORERS = {
    "gqa": score_gqa,
    "textvqa": score_textvqa,
    "docvqa": score_docvqa,
    "mme": score_yesno,
    "mmbench": score_mc_letter,
    "scienceqa": score_mc_letter,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["none", "post", "pre"])
    ap.add_argument("--r", type=float, default=0.0,
                    help="prune ratio; k_i = round(full_i*(1-r)). "
                         "{0.5,0.75,0.875} -> keep {50,25,12.5}% of merge-units.")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="vLLM max_model_len. Raise for huge-image benchmarks "
                         "(DocVQA documents); baseline was hardcoded 8192.")
    ap.add_argument("--benchmark", required=True,
                    choices=["gqa", "textvqa", "docvqa", "mme", "mmbench", "scienceqa"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Processor placeholder patch (IDENTICAL for post and pre): scale each image's
# placeholder list by (1-r). The real placeholder path in vLLM 0.19 is
# Qwen3VLMultiModalProcessor._get_prompt_updates -> get_image_replacement_qwen3vl.
# --------------------------------------------------------------------------- #
def patch_processor(r: float):
    ProcCls = _q3vl_mod.Qwen3VLMultiModalProcessor
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
def setup_post_merger(model, r: float):
    _orig = model._process_image_input
    diag = {"fires": 0, "nk": []}

    def _patched(image_input):
        splits = _orig(image_input)
        diag["fires"] += 1
        if r == 0.0:
            return splits
        out = []
        for s in splits:
            n = int(s.shape[0])
            k = max(1, int(round(n * (1.0 - r))))
            score = s.float().norm(dim=-1)
            idx = torch.topk(score, k).indices.sort().values
            out.append(s.index_select(0, idx).contiguous())
        if len(diag["nk"]) < 8:
            diag["nk"].append((int(splits[0].shape[0]), int(out[0].shape[0])))
        return tuple(out)
    model._process_image_input = _patched
    return diag


# --------------------------------------------------------------------------- #
# (C) PRE-merger: forward_pre_hooks on the 4 mergers + visual (to capture
# grid_thw) + replace _process_image_input to split by pruned counts.
# --------------------------------------------------------------------------- #
class PreMergerPruner:
    def __init__(self, r: float, spatial_merge_size: int):
        self.r = r
        self.sm = spatial_merge_size
        self.unit = spatial_merge_size ** 2          # 4
        self.full_units = None                        # list[int] per image
        self.k_units = None                           # list[int] per image
        self._mask = None                             # cached token mask
        self.diag = {"visual_calls": 0, "merger_calls": 0,
                     "mask_computed_at": None, "mask_compute_count": 0,
                     "per_tag_calls": {}, "nk": []}

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

    def merger_prehook(self, module, args):
        """Slices the merger's input hidden_states to the kept merge-units."""
        self.diag["merger_calls"] += 1
        tag = getattr(module, "_premerger_tag", "?")
        self.diag["per_tag_calls"][tag] = \
            self.diag["per_tag_calls"].get(tag, 0) + 1
        if self.r == 0.0:
            return None
        hs = args[0]                                   # [seq, 1, ctx]
        if self._mask is None:
            seq = hs.shape[0]
            ctx = hs.shape[-1]
            num_units = seq // self.unit
            feats = hs.reshape(num_units, self.unit, ctx)
            scores = feats.float().norm(dim=-1).mean(dim=-1)   # [num_units]
            keep = torch.zeros(num_units, dtype=torch.bool,
                               device=hs.device)
            off = 0
            for f, k in zip(self.full_units, self.k_units):
                s_i = scores[off:off + f]
                idx = torch.topk(s_i, k).indices
                keep[off + idx] = True
                off += f
            # expand unit-mask -> token mask (contiguous 4-token blocks)
            tok_mask = keep.unsqueeze(-1).expand(-1, self.unit).reshape(-1)
            self._mask = tok_mask
            self.diag["mask_computed_at"] = tag
            self.diag["mask_compute_count"] += 1
        kept = hs[self._mask]                          # [num_kept, 1, ctx]
        return (kept,)


def setup_pre_merger(model, r: float):
    visual = model.visual
    sm = visual.spatial_merge_size
    pruner = PreMergerPruner(r, sm)

    # (1) visual.forward pre_hook: capture grid_thw -> plan k_units.
    def _visual_prehook(module, args, kwargs):
        grid_thw = kwargs.get("grid_thw")
        if grid_thw is None and len(args) >= 2:
            grid_thw = args[1]
        if grid_thw is not None:
            pruner.begin_pass(grid_thw)
        return None
    handle_v = visual.register_forward_pre_hook(_visual_prehook, with_kwargs=True)

    # (2) merger + deepstack mergers pre_hooks.
    handles = [handle_v]
    visual.merger._premerger_tag = "main"
    targets = [visual.merger]
    for i, m in enumerate(visual.deepstack_merger_list):
        m._premerger_tag = f"deepstack_{i}"
        targets.append(m)
    for m in targets:
        h = m.register_forward_pre_hook(pruner.merger_prehook)
        handles.append(h)

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
        return image_embeds.split(sizes)
    model._process_image_input = _patched_pii

    return pruner, handles


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    r = args.r
    if args.mode == "none":
        r = 0.0
    proc_log = patch_processor(r)

    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.90, max_model_len=args.max_model_len,
        trust_remote_code=False, enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=False, enable_prefix_caching=False,
    )
    load_s = time.perf_counter() - t0
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    diag = None
    if args.mode == "post":
        diag = setup_post_merger(model, r)
    elif args.mode == "pre":
        pruner, _handles = setup_pre_merger(model, r)
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
    for s, o in zip(samples, outs):
        if o is None:
            continue
        ans = o.outputs[0].text.strip()
        if ans:
            n_ok += 1
        correct += scorer(ans, s.gt, s.extra.get("choices"))
        kept_counts.append(len(o.prompt_token_ids))
    n_scored = len(samples) - n_skip
    req_s = n_scored / wall if wall > 0 else 0.0
    acc = correct / n_scored if n_scored else 0.0

    result = {
        "model": MODEL, "mode": args.mode, "benchmark": args.benchmark, "r": r,
        "max_num_seqs": args.max_num_seqs, "n": len(samples),
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "wall_s": round(wall, 3), "req_per_s": round(req_s, 4),
        "acc": round(acc, 4), "n_answered": n_ok, "n_skipped": n_skip,
        "mean_ptid_len": round(sum(kept_counts) / len(kept_counts), 1) if kept_counts else 0,
        "load_s": round(load_s, 1),
        "proc_placeholder_counts": proc_log["counts"][:3],
        "diag": diag,
        "vllm": vllm.__version__,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[v3] mode={args.mode} r={r} {args.benchmark}: "
          f"req/s={req_s:.3f} acc={acc:.3f} wall={wall:.1f}s "
          f"mean_ptid={result['mean_ptid_len']:.0f}", flush=True)


if __name__ == "__main__":
    main()
