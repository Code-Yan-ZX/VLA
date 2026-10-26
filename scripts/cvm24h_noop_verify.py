#!/usr/bin/env python
"""Real 32-request keep-all gate for the original eager quality protocol.

Checks native generate against RBM keep=1, post keep=1, and FastV drop=0.
Every manual forward retains all three Qwen3 deepstack streams. First-step
logits come from the ACTUAL generation calls via an lm_head hook, rather than
a separate approximate prefill. BF16 differences are reported numerically;
byte identity is asserted only for tensors that actually compare equal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/v3_premerger"))
import baselines_hf as B


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(path, report):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(path)


class FirstLogits:
    def __init__(self, head):
        self.value = None
        self.handle = head.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        if self.value is None:
            self.value = output[0, -1].detach().float().cpu()

    def close(self):
        self.handle.remove()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=Path, default=ROOT / "results/cvm2027_supplement_20261026/plan/smoke32_textvqa.jsonl")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    samples = B.load_subset(str(a.subset))[:a.n]
    assert len(samples) == a.n
    report = {"model": "Qwen/Qwen3-VL-8B-Instruct", "dtype": "bfloat16", "attn": "eager",
              "max_pixels": 4000000, "max_new_tokens": 32, "seed": 0,
              "subset": str(a.subset), "subset_sha256": hashlib.sha256(a.subset.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256((ROOT / "src/v3_premerger/baselines_hf.py").read_bytes()).hexdigest(),
              "torch": torch.__version__, "transformers": __import__("transformers").__version__,
              "n_requested": a.n, "n_complete": 0, "status": "partial", "requests": [],
              "logit_policy": "record exact equality, max/mean absolute and relative L2 errors; require finite logits and identical argmax/full greedy tokens; no numerical differences silently relabeled as byte identity"}
    dump(a.out, report)
    if time.time() >= a.deadline_epoch:
        report["stop_reason"] = "deadline before model load"
        dump(a.out, report)
        return
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(
        report["model"], dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(report["model"])
    assert model.visual.config._attn_implementation == "eager"
    assert model.model.language_model.config._attn_implementation == "eager"
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    cfg = {"r": 0., "fastv_k": 3, "ratios": []}
    iid = model.config.image_token_id
    spatial_unit = int(model.visual.spatial_merge_size) ** 2
    tap = B.MergerTap(model.visual)
    rank_calls = []
    original_rank = B.rank_keep_indices

    def rank(attn_w, image_mask, k_keep):
        kept = original_rank(attn_w, image_mask, k_keep)
        rank_calls.append(kept.detach().cpu().tolist())
        return kept
    B.rank_keep_indices = rank
    wall0 = time.perf_counter()
    for sample in samples:
        if time.time() >= a.deadline_epoch:
            report["stop_reason"] = "deadline before request"
            break
        rec = {"id": str(sample.id), "modes": {}, "status": "partial"}
        report["requests"].append(rec)
        failure_payload = {}
        try:
            inputs = B.build_inputs(processor, Image.open(sample.image).convert("RGB"),
                                    sample.question, 4000000, "cuda")
            mask = inputs["input_ids"][0] == iid
            nfull = int(mask.sum())
            seq_len = inputs["input_ids"].shape[1]
            hook = FirstLogits(model.lm_head)
            try:
                native = model.generate(**inputs, do_sample=False, max_new_tokens=32,
                                        pad_token_id=processor.tokenizer.eos_token_id)
                native_tokens = native[0, seq_len:].tolist()
                native_logits = hook.value
            finally:
                hook.close()
            assert native_logits is not None and torch.isfinite(native_logits).all()
            tap.reset()
            ie0, pos0, ds0 = B.capture_prepared_inputs(model, dict(inputs))
            assert ds0 is not None and len(ds0) == 3
            assert all(d.shape[0] == nfull for d in ds0)
            rec.update({"n_image_full": nfull, "n_text": seq_len - nfull,
                        "grid_thw": inputs["image_grid_thw"].cpu().tolist(),
                        "native_tokens": native_tokens,
                        "native_answer": processor.decode(native_tokens, skip_special_tokens=True).strip(),
                        "input_ids_sha256": tensor_hash(inputs["input_ids"]),
                        "position_ids_sha256": tensor_hash(pos0),
                        "deepstack_shapes": [list(d.shape) for d in ds0],
                        "deepstack_sha256": [tensor_hash(d) for d in ds0]})
            for mode in ("pre", "post", "fastv"):
                if time.time() >= a.deadline_epoch:
                    report["stop_reason"] = "deadline before method"
                    dump(a.out, report)
                    return
                failure_payload = {"input_ids": inputs["input_ids"].cpu(),
                                   "native_position_ids": pos0.cpu(), "native_logits": native_logits,
                                   "native_tokens": native_tokens, "id": str(sample.id), "mode": mode}
                rank_calls.clear()
                if mode == "pre":
                    kept, _ = B.premerger_keep_units(tap.first_hs, inputs["image_grid_thw"], 1., spatial_unit)
                elif mode == "post":
                    kept, _ = B.postmerger_keep_tokens(ie0, mask, 1., inputs["image_grid_thw"], spatial_unit)
                else:
                    kept = torch.arange(nfull, device=mask.device)
                ie, pos, ds, mask2 = B.apply_premerger(ie0, pos0, ds0, mask, kept)
                structure = {"kept_image_ids": kept.cpu().tolist(),
                             "kept_ids_are_original_order": torch.equal(kept, torch.arange(nfull, device=kept.device)),
                             "embeds_equal": torch.equal(ie, ie0), "positions_equal": torch.equal(pos, pos0),
                             "mask_equal": torch.equal(mask2, mask),
                             "deepstack_equal": [torch.equal(x, y) for x, y in zip(ds, ds0)],
                             "text_token_count_equal": int((~mask2).sum()) == seq_len - nfull}
                rec["modes"][mode] = structure
                assert structure["kept_ids_are_original_order"] and structure["embeds_equal"]
                assert structure["positions_equal"] and structure["mask_equal"] and structure["text_token_count_equal"]
                assert len(ds) == 3 and all(structure["deepstack_equal"])
                hook = FirstLogits(model.lm_head)
                try:
                    tokens, diag = B.generate_pruned(model, ie, pos, mask2,
                                                     "fastv" if mode == "fastv" else "pre",
                                                     cfg, 32, eos_ids, deepstack=ds)
                    logits = hook.value
                finally:
                    hook.close()
                assert logits is not None
                delta = logits - native_logits
                structure.update({"n_image_kept": int(diag["n_image_kept"]),
                                  "n_deepstack": int(diag["n_deepstack"]),
                                  "prune_plan": diag["prune_plan"], "prune_fired": diag["fired"],
                                  "rank_kept_sequence_ids": list(rank_calls),
                                  "generated_tokens": tokens,
                                  "answer": processor.decode(tokens, skip_special_tokens=True).strip(),
                                  "greedy_tokens_equal": tokens == native_tokens,
                                  "logits_finite": bool(torch.isfinite(logits).all()),
                                  "logits_exactly_equal": torch.equal(logits, native_logits),
                                  "logits_argmax_equal": int(logits.argmax()) == int(native_logits.argmax()),
                                  "logits_max_abs": float(delta.abs().max()),
                                  "logits_mean_abs": float(delta.abs().mean()),
                                  "logits_relative_l2": float(delta.norm() / native_logits.norm().clamp_min(1e-12))})
                failure_payload = {"input_ids": inputs["input_ids"].cpu(), "position_ids": pos.cpu(),
                                   "native_logits": native_logits, "method_logits": logits,
                                   "native_tokens": native_tokens, "method_tokens": tokens,
                                   "id": str(sample.id), "mode": mode}
                assert structure["n_image_kept"] == nfull and structure["n_deepstack"] == 3
                if mode == "fastv":
                    assert rank_calls == [list(range(seq_len))], "FastV keep-all changed the actual sequence set/order"
                assert structure["logits_finite"] and structure["logits_argmax_equal"]
                assert structure["greedy_tokens_equal"], "Actual greedy generation mismatch"
            rec["status"] = "pass"
            report["n_complete"] += 1
            del inputs, native, ie0, pos0, ds0, ie, pos, ds
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            report["status"] = "failed"
            if failure_payload:
                evidence = a.out.with_suffix(".failure.pt")
                torch.save(failure_payload, evidence)
                rec["failure_evidence"] = str(evidence)
            report["wall_s"] = time.perf_counter() - wall0
            dump(a.out, report)
            raise
        report["wall_s"] = time.perf_counter() - wall0
        dump(a.out, report)
        print(f"[noop] {report['n_complete']}/{a.n} {sample.id} pass", flush=True)
    if report["n_complete"] == a.n:
        report["status"] = "pass"
        report["summary"] = {mode: {
            "n": a.n, "full_greedy_matches": sum(r["modes"][mode]["greedy_tokens_equal"] for r in report["requests"]),
            "logits_max_abs": max(r["modes"][mode]["logits_max_abs"] for r in report["requests"]),
            "all_logits_exactly_equal": all(r["modes"][mode]["logits_exactly_equal"] for r in report["requests"])}
            for mode in ("pre", "post", "fastv")}
    report["wall_s"] = time.perf_counter() - wall0
    dump(a.out, report)


if __name__ == "__main__":
    main()
