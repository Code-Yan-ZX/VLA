#!/usr/bin/env python
"""Hi-Lo Prune official Qwen3-VL-8B path, driven under the CVM frozen protocol.
Algorithm code = /tmp/Hi-Lo_Prune commit 29265e3 UNCHANGED except:
  [FIX-1] removed 3-line dead block referencing undefined `top_index_1`
          (upstream NameError; dead code: converted keep_image_indices only
           when empty). No algorithmic change.
  [SHIM]  text_config.rope_parameters built from rope_scaling (our
          transformers 4.57.6 predates upstream 4.58 config layout).
Inputs are built with the SAME helpers as baselines_hf.py (identical
prompt/processor/max_pixels), bf16, greedy, max_new_tokens=32, bs=1.
Official selection + PA-Fusion run verbatim from hilo_prune packages.
"""
import argparse, json, os, sys, time
import torch

HILO = "/tmp/Hi-Lo_Prune"
sys.path.insert(0, HILO)
sys.path.insert(0, "src/v3_premerger")
import baselines_hf as B  # for load_subset / build_inputs / _cap_image_pixels
from transformers import AutoConfig, AutoProcessor
from hilo_prune.models.qwen3_vl import modeling_hilo_qwen3_vl as M
from hilo_prune.models.qwen3_vl import Qwen3VLForConditionalGeneration as HiLo
from hilo_prune.patches import image_to_merged_patch_features
from PIL import Image

SEL_LOG = []
_orig_sel = M.hierarchical_token_selection
def _traced_sel(pf, tf, *, keep_count, candidate_count):
    out = _orig_sel(pf, tf, keep_count=keep_count, candidate_count=candidate_count)
    SEL_LOG.append({"n": int(pf.shape[0]), "keep": int(len(out.keep)),
                    "cand": int(candidate_count)})
    return out
M.hierarchical_token_selection = _traced_sel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--benchmark", required=True)
ap.add_argument("--subset", required=True)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--keep-frac", type=float, required=True)
ap.add_argument("--candidate-frac", type=float, default=None,
                help="None -> official default 0.3 with clamp >= keep (recorded)")
ap.add_argument("--prune-layer", type=int, default=2)
ap.add_argument("--no-fusion", action="store_true")
ap.add_argument("--max-pixels", type=int, default=4000000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--out", required=True)
a = ap.parse_args()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.manual_seed(a.seed)
t0 = time.perf_counter()
cand_default = None if a.candidate_frac is None else a.candidate_frac
effective_cand = a.candidate_frac if cand_default is not None else max(0.3, a.keep_frac)

processor = AutoProcessor.from_pretrained(a.model)
cfg = AutoConfig.from_pretrained(a.model)
tc = cfg.text_config
rs = getattr(tc, "rope_scaling", None) or {}
tc.rope_parameters = {"rope_type": rs.get("type", "default"),
                      "rope_theta": tc.rope_theta,
                      "mrope_section": rs.get("mrope_section", [24, 20, 20])}  # SHIM
tc.hilo_keep_ratio = a.keep_frac
tc.hilo_candidate_ratio = effective_cand
tc.hilo_prune_layer = a.prune_layer
tc.hilo_use_fusion = not a.no_fusion
tc.hilo_fusion_strength = 0.3
tc.hilo_top_k_ratio = 0.1
tc.hilo_attention_threshold = 0.05
tc.hilo_fusion_layers = list(range(a.prune_layer))
model = HiLo.from_pretrained(a.model, config=cfg, dtype=torch.bfloat16,
                             attn_implementation=a.attn).to("cuda").eval()
load_s = time.perf_counter() - t0

iid = model.config.image_token_id
device = "cuda"
scorer = B.SCORERS[a.benchmark]
samples = B.load_subset(a.subset)[:a.n]
eos = model.generation_config.eos_token_id
eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
per, n_ok, n_skip, n_bad = [], 0, 0, 0
wall0 = time.perf_counter()
for s in samples:
    rec = {"id": s.id, "question": s.question, "gt": s.gt, "answer": None,
           "error": None, "n_image_full": None, "n_image_kept": None}
    try:
        img = Image.open(s.image).convert("RGB")
        inputs = B.build_inputs(processor, img, s.question, a.max_pixels, device)
        n_img = int((inputs["input_ids"][0] == iid).sum())
        grid = inputs["image_grid_thw"][0].cpu()
        pf = image_to_merged_patch_features(
            img, grid,
            patch_size=processor.image_processor.patch_size,
            merge_size=processor.image_processor.merge_size).to(device)
        assert pf.shape[0] == n_img, f"patch rows {pf.shape[0]} != image tokens {n_img}"
        rec["n_image_full"] = n_img
        SEL_LOG.clear()
        pad_id = (model.generation_config.pad_token_id
                  if model.generation_config.pad_token_id is not None
                  else processor.tokenizer.pad_token_id
                  or (sorted(eos_ids)[0] if eos_ids else None))
        with torch.inference_mode():
            gen = model.generate(**inputs, image_tensor=pf, do_sample=False,
                                 max_new_tokens=a.max_tokens,
                                 eos_token_id=list(eos_ids), pad_token_id=pad_id)
        out = gen[0][inputs["input_ids"].shape[1]:]
        ans = processor.decode(out, skip_special_tokens=True).strip()
        rec["answer"] = ans
        rec["n_image_kept"] = SEL_LOG[0]["keep"] if SEL_LOG else n_img
        rec["correct"] = int(scorer(ans, s.gt) > 0) if scorer else None
        n_ok += 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        n_bad += 1
    per.append(rec)
    print(f"[hilo] {s.id} kept={rec['n_image_kept']}/{rec['n_image_full']} "
          f"err={rec['error']}", flush=True)
wall = time.perf_counter() - wall0
ans_bin = [p["correct"] if p.get("correct") is not None else 0 for p in per]
out = {"model": a.model, "mode": "hilo_official", "engine": "hilo-vendored-29265e3",
       "benchmark": a.benchmark, "n": len(per), "seed": a.seed,
       "max_pixels": a.max_pixels, "max_tokens": a.max_tokens,
       "attn": a.attn, "keep_ratio": a.keep_frac,
       "candidate_ratio": effective_cand,
       "candidate_rule": "official default 0.3 clamped to >= keep" if cand_default is None else "explicit",
       "prune_layer": a.prune_layer, "use_fusion": not a.no_fusion,
       "fusion_params": {"strength": 0.3, "top_k_ratio": 0.1, "attention_threshold": 0.05,
                         "layers": list(range(a.prune_layer))},
       "patches": {"hilo_commit": "29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6",
                   "fix1": "dead-code removal of undefined top_index_1 guard (non-algorithmic)",
                   "shim": "text_config.rope_parameters from rope_scaling (tf 4.57.6)",
                   "transformers": __import__("transformers").__version__,
                   "torch": torch.__version__},
       "load_s": round(load_s, 1), "wall_s": round(wall, 1),
       "req_per_s": round(len(per) / wall, 4) if wall else 0,
       "n_scored": n_ok, "n_skipped": n_skip, "n_error": n_bad,
       "acc": round(sum(ans_bin) / max(len(ans_bin), 1), 4),
       "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "per_sample": per}
json.dump(out, open(a.out, "w"), indent=1)
print(f"[hilo] wrote {a.out} n={len(per)} ok={n_ok} err={n_bad} acc_proxy={out['acc']} "
      f"wall={wall:.0f}s ({wall/len(per):.1f}s/req)")
