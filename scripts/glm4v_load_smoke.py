#!/usr/bin/env python3
"""GLM-4V family load smoke (Phase 1 of the stage-law gate onboarding).

Loads the glm4v snapshot with STOCK transformers 4.57.6 (bf16, GPU), verifies:
  (a) config + model load (transformers-native, no remote code),
  (b) the merger module exists at model.visual.merger,
  (c) sensible generation on one OCR-ish image,
  (d) visual token counts in/out of the merger (expect 4x reduction).

One-shot GPU script; exits non-zero on any failure so the gate script can
branch on it. Usage: python scripts/glm4v_load_smoke.py [model_path]
"""
import sys

import torch

MODEL = (sys.argv[1] if len(sys.argv) > 1 else
         "/data/models/modelscope/hub/models/ZhipuAI/GLM-4___1V-9B-Thinking")

print(f"[smoke] model path: {MODEL}", flush=True)
from transformers import AutoConfig, AutoProcessor  # noqa: E402

cfg = AutoConfig.from_pretrained(MD := MODEL)
print(f"[smoke] (a) config OK: {type(cfg).__name__} | image_token_id="
      f"{cfg.image_token_id} | mrope_section="
      f"{cfg.text_config.rope_scaling.get('mrope_section')} | "
      f"spatial_merge_size={cfg.vision_config.spatial_merge_size}", flush=True)
proc = AutoProcessor.from_pretrained(MD)
assert proc.image_token_id == cfg.image_token_id, \
    f"image_token_id mismatch: {proc.image_token_id} vs {cfg.image_token_id}"

from transformers import Glm4vForConditionalGeneration  # noqa: E402

# no `accelerate` in qwen3vl_clean (deliberately untouched env) -> plain CPU
# load then .to("cuda") instead of device_map.
model = Glm4vForConditionalGeneration.from_pretrained(
    MD, torch_dtype=torch.bfloat16).to("cuda")
model.eval()
print(f"[smoke] model loaded: {type(model).__name__}; GPU mem "
      f"{torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

# (b) merger module (+ the strided downsample conv that precedes it)
merger = model.visual.merger
ds = getattr(model.visual, "downsample", None)
print(f"[smoke] (b) merger OK: model.visual.merger = {type(merger).__name__}"
      f" | downsample = {type(ds).__name__ if ds is not None else 'MISSING'}"
      f" | spatial_merge_size = {model.visual.config.spatial_merge_size}",
      flush=True)

# (c)+(d) 1-image forward + generation, counting visual tokens in/out of the
# merge stage. The 4x compression = the strided 2x2 `downsample` conv BEFORE
# the merger MLP (identical layout in transformers and vLLM), so the ratio is
# measured across the whole visual tower: ViT patch tokens (from
# image_grid_thw) vs merged output tokens; the placeholder run in input_ids
# must equal the merged count.
import json  # noqa: E402
from PIL import Image  # noqa: E402

with open("eval/subsets/textvqa_200.jsonl") as f:
    s = json.loads(f.readline())
img = Image.open(s["image"]).convert("RGB")
q = s["question"]
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img}, {"type": "text", "text": q}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")

counts = {}

def _merger_hook(module, inp, out):
    counts["merger_in"] = tuple(inp[0].shape)
    counts["merger_out"] = tuple(out.shape)

def _visual_hook(module, inp, out):
    counts["visual_out"] = tuple(out.shape)

h1 = merger.register_forward_hook(_merger_hook)
h2 = model.visual.register_forward_hook(_visual_hook)
with torch.inference_mode():
    gen = model.generate(**inputs, max_new_tokens=96, do_sample=False)
h1.remove(); h2.remove()
ans = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True)[0].strip()

grid = inputs["image_grid_thw"][0].tolist()          # [t, h, w] in patches
n_patch = grid[0] * grid[1] * grid[2]                # ViT tokens (pre-merge)
n_merged = counts["visual_out"][0]                   # post-merge tokens
n_ph = int((inputs["input_ids"][0] == cfg.image_token_id).sum())
ratio = n_patch / n_merged if n_merged else float("nan")
print(f"[smoke] (d) grid_thw={grid} -> ViT tokens={n_patch}; merged="
      f"{n_merged}; placeholder run={n_ph}; ratio={ratio:.2f} (expect 4.00)",
      flush=True)
print(f"[smoke] (d) merger module: in={counts.get('merger_in')} "
      f"out={counts.get('merger_out')} (post-downsample units -> MLP)",
      flush=True)
print(f"[smoke] (c) Q: {q[:90]}\n[smoke] (c) A: {ans[:400]}", flush=True)
ok_ratio = abs(ratio - 4.0) < 0.01 and n_ph == n_merged
ok_text = len(ans) > 0
print(f"[smoke] VERDICT: {'PASS' if (ok_ratio and ok_text) else 'FAIL'} "
      f"(ratio4x&placeholder={ok_ratio}, nonempty={ok_text})", flush=True)
sys.exit(0 if (ok_ratio and ok_text) else 1)
