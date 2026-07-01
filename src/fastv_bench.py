"""FastV baseline reproduction (accuracy anchor) — the ECCV'24 official repo.

FastV prunes INSIDE the LLM at decoder layer 2 (attention-rank based), so it
CANNOT run inside vLLM (FlashAttention fuses softmax; per-token scores are
inaccessible -- survey §6.5.3). It runs here via the FastV repo's patched
`transformers` + `llava` library (`fastv` env), as the accuracy-only anchor.

Usage (run inside the `fastv` env, on GPU):
    /home/dell/miniconda3/envs/fastv/bin/python -m src.fastv_bench \
        --model-path /media/disk2/YZX/doct/FastV/llava-v1.5-7b \
        --benchmark gqa --subset eval/subsets/gqa_200.jsonl \
        --keep-tokens 288 --agg-layer 2 \
        --metrics-out runs/fastv_baseline/gqa_keep288_metrics.json

What it does:
  1. Loads LLaVA-1.5-7B via the FastV repo's `llava.model.builder.load_pretrained_model`
     (which uses the patched transformers that recognize the fastv config knobs).
  2. Enables FastV pruning: use_fast_v=True, fastv_image_token_length=576,
     fastv_attention_rank=keep_tokens, fastv_agg_layer=agg_layer, fastv_inplace=True.
  3. Runs the eval subset (same JSONL + scorer as serve_bench.py for direct
     accuracy comparability -- no vLLM, so no throughput metrics here, by design).
  4. Writes accuracy + raw answers to --metrics-out.

The model dir at /media/disk2/YZX/doct/FastV/llava-v1.5-7b is the original
LLaVA-format weights (pytorch_model*.bin + mm_projector.bin) that the `llava`
library expects. Do NOT point this at the HF-format runs/models/llava-1.5-7b-hf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# The fastv env's transformers is the patched one at FastV/src/transformers (editable).
# The llava lib is at FastV/src/LLaVA (editable). Both are already wired into the env.


def load_subset(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append({
                "id": str(o["id"]),
                "image": o["image"],
                "question": o["question"],
                "gt": str(o["gt"]),
                "extra": {k: v for k, v in o.items()
                          if k not in {"id", "image", "question", "gt"}},
            })
    return out


# Reuse the exact same scorers as serve_bench.py (iso-eval).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serve_bench import SCORERS  # noqa: E402


def run(args) -> dict:
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from PIL import Image

    samples = load_subset(args.subset)
    if args.limit > 0:
        samples = samples[: args.limit]
    scorer = SCORERS[args.benchmark]

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        args.load_8bit, args.load_4bit, device=args.device,
    )
    model.eval()

    # ---- enable FastV pruning on the LLaMA decoder ----
    # 576 = LLaVA-1.5 image tokens; agg_layer=2 (FastV default "after layer 2");
    # attention_rank = number of image tokens to KEEP (k = N*(1-r)).
    if args.keep_tokens >= 576:
        cfg = model.config
        cfg.use_fast_v = False
        # CRITICAL: leave output_attentions at default (False) for the control --
        # enabling it forces every decoder layer to return attention weights,
        # which is slower and unnecessary when FastV is off.
        print(f"[fastv_bench] FastV DISABLED (keep_tokens={args.keep_tokens} >= 576, control)",
              flush=True)
    else:
        cfg = model.config
        cfg.use_fast_v = True
        cfg.fast_v_sys_length = args.sys_length
        cfg.fast_v_image_token_length = 576
        cfg.fast_v_attention_rank = args.keep_tokens
        cfg.fast_v_agg_layer = args.agg_layer
        cfg.fast_v_inplace = True
        # CRITICAL FIX: the FastV-patched LlamaModel.forward reads
        # `layer_outputs[1]` (the attention weights) at the agg_layer to rank
        # image tokens. That index only exists if each decoder layer returns its
        # attention output, which requires output_attentions=True. The patched
        # modeling_llama.py:635/815-820 falls back to self.config.output_attentions
        # when the forward arg is None -- so we set it on the config (propagates
        # to every layer). Without this, layer_outputs = (hidden_states, ...) and
        # layer_outputs[1] raises IndexError (the FastV-ON crash).
        cfg.output_attentions = True
        # push into the model's decoder (the LlamaModel holds these on self)
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()
        print(f"[fastv_bench] FastV ON: keep {args.keep_tokens}/576 image tokens, "
              f"agg_layer={args.agg_layer}, sys_length={args.sys_length}, inplace=True, "
              f"output_attentions=True (required for layer_outputs[1])",
              flush=True)

    conv_mode = "llava_v1"  # LLaVA-1.5
    raw = []
    n_correct = 0
    for i, s in enumerate(samples):
        img = Image.open(s["image"]).convert("RGB")
        # LLaVA-1.5 image preprocessing (matches the model's image_processor config)
        image_tensor = process_images([img], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        qs = DEFAULT_IMAGE_TOKEN + "\n" + s["question"]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )
        # reset_fastv between requests if the patched model exposes it (Issue #15 workaround)
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv") and cfg.use_fast_v:
            model.model.reset_fastv()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=args.max_tokens,
                use_cache=False,  # FastV requires use_cache=False (inplace drop breaks KV)
            )

        # strip the prompt tokens
        text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        correct = scorer(text, s["gt"], s.get("choices"))
        n_correct += correct
        raw.append({"id": s["id"], "correct": correct, "answer": text, "gt": s["gt"]})
        if (i + 1) % 20 == 0:
            print(f"[fastv_bench] {args.benchmark} {i+1}/{len(samples)} "
                  f"running_acc={n_correct/(i+1):.3f}", flush=True)

    acc = n_correct / len(raw) if raw else float("nan")
    result = {
        "benchmark": args.benchmark,
        "model": args.model_path,
        "fastv_enabled": args.keep_tokens < 576,
        "keep_tokens": args.keep_tokens,
        "agg_layer": args.agg_layer,
        "n": len(raw),
        "accuracy": acc,
        "raw": raw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fastv_bench] {args.benchmark} keep={args.keep_tokens}/576 "
          f"agg_layer={args.agg_layer} n={len(raw)} acc={acc:.3f} -> {args.metrics_out}",
          flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="FastV baseline accuracy (ECCV'24 official repo)")
    ap.add_argument("--model-path", default="/media/disk2/YZX/doct/FastV/llava-v1.5-7b",
                    help="LLaVA-1.5-7B in original LLaVA format (pytorch_model*.bin)")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--benchmark", required=True, choices=["gqa", "textvqa"])
    ap.add_argument("--subset", required=True, help="JSONL subset path (same as serve_bench)")
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--keep-tokens", type=int, default=288,
                    help="image tokens to KEEP (e.g. 288 = r50; 576 = control/FastV-off)")
    ap.add_argument("--agg-layer", type=int, default=2,
                    help="FastV prune-at-layer (2 = after decoder layer 2, the paper default)")
    ap.add_argument("--sys-length", type=int, default=4,
                    help="system prompt length before image tokens (LLaVA-1.5 default 4)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
