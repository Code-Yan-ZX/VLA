#!/usr/bin/env python
"""Corrected P3: fixed 200 requests x 3 repeats; original timing files untouched.

Run harness then hilo, serially. Both use vision SDPA / decoder eager, bf16,
greedy, native positions, 4M pixel cap. The request starts with an already
decoded RGB PIL image. Disk reads/decode, model loading and warmup are excluded.
Processor, transfer, vision, selection/fusion, first token and total are timed
on the SAME request; GPU events supplement synchronized wall measurements.
Full evaluation scores from the older attention protocol must not be paired
with these costs as though they were produced by this run. Answers are saved
so the exact fixed timing workload can be scored independently.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import BaseStreamer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/v3_premerger"))
import baselines_hf as B

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ATTN = {"vision_config": "sdpa", "text_config": "eager"}
PLAN = ROOT / "results/cvm2027_supplement_20261026/plan"
MAXTOK = 32
PIXELS = 4000000


def sync():
    torch.cuda.synchronize()


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def tensor_record(tensor):
    """Full byte hash plus a fixed evenly spaced numerical probe (not a norm)."""
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    flat = value.flatten()
    count = min(256, flat.numel())
    idx = torch.linspace(0, flat.numel() - 1, count).long()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "probe": flat[idx].float()}


def dump(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    z = (len(values) - 1) * q
    lo, hi = int(z), min(int(z) + 1, len(values) - 1)
    return values[lo] * (hi - z) + values[hi] * (z - lo) if lo != hi else values[lo]


def summarize(rows, wall, repeats, keep):
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = ("processor_cpu_s", "transfer_s", "vision_s", "selection_s",
            "fusion_s", "prefill_s", "ttft_s", "decode_s", "total_s",
            "compute_gpu_s", "out_len", "peak_alloc_GB", "peak_reserved_GB")
    stats = {}
    for key in keys:
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        if vals:
            stats[key] = {"p50": statistics.median(vals), "p95": quantile(vals, .95)}
    return {"keep_fraction": keep, "actual_k_verified": all(r.get("actual_k_verified") for r in ok),
            "n_attempted": len(rows), "n_success": len(ok), "n_failed": len(rows) - len(ok),
            "workload_wall_s": wall,
            "throughput_req_per_s": sum(len(r["repeats_raw"]) for r in ok) / wall if wall else 0,
            "peak_alloc_GB": max((p["peak_alloc_GB"] for r in ok for p in r["repeats_raw"]), default=None),
            "peak_reserved_GB": max((p["peak_reserved_GB"] for r in ok for p in r["repeats_raw"]), default=None),
            "summary": stats, "requests": rows}


class FirstToken(BaseStreamer):
    """Recognize and skip the exact prompt, rather than assuming callback #2."""
    def __init__(self, prompt):
        self.prompt = prompt.detach().cpu()
        self.calls = 0
        self.prompt_seen = False
        self.first = None
        self.first_event = None

    def put(self, value):
        self.calls += 1
        if self.calls == 1 and torch.equal(value.detach().cpu(), self.prompt):
            self.prompt_seen = True
            return
        if self.first is None:
            self.first_event = event()
            sync()
            self.first = time.perf_counter()

    def end(self):
        pass


class Trace:
    """Hooks inspect sizes and place CUDA events without altering tensors."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.reset()
        for idx, layer in enumerate(model.model.language_model.layers):
            self.handles.append(layer.register_forward_pre_hook(self.layer_hook(idx), with_kwargs=True))
        self.handles.append(model.visual.register_forward_pre_hook(self.vision_before, with_kwargs=True))
        self.handles.append(model.visual.register_forward_hook(self.vision_after, with_kwargs=True))
        self.handles.append(model.model.language_model.register_forward_pre_hook(self.language_before, with_kwargs=True))

    def reset(self):
        self.layers = []
        self.vision_events = None
        self.selection_events = []
        self.fusion_events = []
        self.selection = []
        self.vision_wall_start = None
        self.vision_wall_end = None
        self.lm_start_wall = None

    def layer_hook(self, idx):
        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is not None and hs.shape[1] > 1:
                self.layers.append({"layer": idx, "sequence_length": int(hs.shape[1])})
        return hook

    def vision_before(self, module, args, kwargs):
        sync()
        self.vision_wall_start = time.perf_counter()
        self.vision_events = [event()]

    def vision_after(self, module, args, kwargs, output):
        self.vision_events.append(event())
        sync()
        self.vision_wall_end = time.perf_counter()

    def language_before(self, module, args, kwargs):
        if self.lm_start_wall is None:
            sync()
            self.lm_start_wall = time.perf_counter()

    def finish(self):
        out = {"layer_trace": self.layers,
               "vision_wall_s": self.vision_wall_end - self.vision_wall_start,
               "vision_gpu_s": self.vision_events[0].elapsed_time(self.vision_events[1]) / 1000,
               "selection_gpu_s": sum(x.elapsed_time(y) for x, y in self.selection_events) / 1000,
               "fusion_gpu_s": sum(x.elapsed_time(y) for x, y in self.fusion_events) / 1000}
        return out


@torch.inference_mode()
def generate_manual(model, ie, pos, mask, mode, ds, keep, eos_ids, trace):
    lm = model.model.language_model
    cfg = {"r": 1 - keep, "fastv_k": 3, "ratios": []}
    hidden, positions, cache, imask, diag = B.prefill_pruned(
        model, ie, pos, mask, mode, cfg, deepstack=ds)
    # Generation consumes only the last prefill logit, as native HF generate's
    # logits_to_keep=1 does. Avoid charging unused vocabulary projections to
    # the harness arms. Full-token parity against the existing helper is gated.
    logits = model.lm_head(hidden[:, -1:, :])
    token = int(logits[0, -1].argmax(-1))
    first_event = event()
    sync()
    first_wall = time.perf_counter()
    out = [token]
    cur_pos = int(positions.max())
    kv_len = B._cache_len(cache)
    embed = model.get_input_embeddings()
    for _ in range(MAXTOK - 1):
        if token in eos_ids:
            break
        emb = embed(torch.tensor([[token]], device=ie.device))
        cur_pos += 1
        pos_new = torch.full((3, 1, 1), cur_pos, device=ie.device, dtype=positions.dtype)
        pe = lm.rotary_emb(emb, pos_new)
        dec_mask = torch.zeros(1, 1, 1, kv_len + 1, device=ie.device, dtype=ie.dtype)
        h = emb
        for layer in lm.layers:
            val = layer(h, attention_mask=dec_mask, position_ids=None, past_key_values=cache,
                        use_cache=True, cache_position=None, position_embeddings=pe)
            h = val[0] if isinstance(val, tuple) else val
        kv_len += 1
        token = int(model.lm_head(lm.norm(h))[0, -1].argmax(-1))
        out.append(token)
    return out, diag, first_wall, first_event


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=("harness", "hilo"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=200, help="Smoke only if less than 200; labeled subset")
    ap.add_argument("--configs", nargs="+", help="Optional explicit config tags")
    ap.add_argument("--deadline-epoch", type=float, default=None)
    ap.add_argument("--noop-reference", type=Path, default=PLAN / "timing_v2_noop_reference.pt")
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    ids_path = PLAN / "p3_timing_ids.json"
    ids = json.loads(ids_path.read_text())
    requests = []
    for bench, split in (("textvqa", "textvqa_val.jsonl"), ("ocrbench", "ocrbench.jsonl")):
        samples = {str(s.id): s for s in B.load_subset(str(ROOT / "eval/full_splits" / split))}
        requests.extend((bench, str(sid), samples[str(sid)]) for sid in ids[f"{bench}_100"])
    requests = requests[:a.limit]
    assert a.repeats >= 1 and len(requests) == min(a.limit, 200)
    metadata = {"schema_version": 2, "engine": a.engine, "model": MODEL, "dtype": "bfloat16",
                "attention": ATTN, "seed": 0, "bs": 1, "max_new_tokens": MAXTOK,
                "max_pixels": PIXELS, "repeats": a.repeats, "n_requests": len(requests),
                "timing_ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "boundary": "decoded RGB image -> processor + host-to-device transfer + full vision/merger + selection/fusion + prefill/first token + decode; excludes disk/decode/loading/queue/warmup",
                "aggregation": "per-request median across repeats, then p50/p95 over request medians; raw repeats retained; failures retained, excluded from latency quantiles, success count explicit",
                "workload_wall_boundary": "entire sequential request/repeat loop including local image decode, Python bookkeeping and failure recovery; excludes model loading and warmup; throughput counts successful request executions, not independent samples",
                "segment_note": "CUDA-event segments accompany synchronized wall boundaries. selection/fusion may overlap the prefill segment and must not be summed twice. total_s and ttft_s are direct timestamps of each request.",
                "protocol_note": "shared vision SDPA / text eager differs from previous quality runs; use saved answers and same fixed IDs for paired quality-cost analysis", "results": {}}
    metadata["prefill_lm_head"] = "last token only, matching native HF generation; generated-token parity checked against original all-token projection helper"
    if a.out.exists():
        old = json.loads(a.out.read_text())
        assert old["schema_version"] == 2 and old["attention"] == ATTN
        assert old["timing_ids_sha256"] == metadata["timing_ids_sha256"]
        assert old["script_sha256"] == metadata["script_sha256"], "Refuse mixing versions"
        metadata = old
    processor = AutoProcessor.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    patch_features = None
    if a.engine == "hilo":
        sys.path.insert(0, "/tmp/Hi-Lo_Prune")
        from hilo_prune.models.qwen3_vl import Qwen3VLForConditionalGeneration as ModelClass
        from hilo_prune.models.qwen3_vl import modeling_hilo_qwen3_vl as hilo_module
        from hilo_prune.patches import image_to_merged_patch_features as patch_features
        tc = config.text_config
        rs = getattr(tc, "rope_scaling", None) or {}
        tc.rope_parameters = {"rope_type": rs.get("type", "default"), "rope_theta": tc.rope_theta,
                              "mrope_section": rs.get("mrope_section", [24, 20, 20])}
        tc.hilo_keep_ratio = .25
        tc.hilo_candidate_ratio = .3
        tc.hilo_prune_layer = 2
        tc.hilo_use_fusion = True
        tc.hilo_fusion_strength = .3
        tc.hilo_top_k_ratio = .1
        tc.hilo_attention_threshold = .05
        tc.hilo_fusion_layers = [0, 1]
        configs = [("hilo_0.25", "hilo", .25), ("hilo_0.5", "hilo", .5), ("hilo_0.75", "hilo", .75)]
        metadata["hilo_commit"] = "29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6"
        metadata["hilo_parameters"] = {"candidate": "max(.3,keep)", "prune_layer": 2,
                                       "fusion": True, "strength": .3, "top_k_ratio": .1,
                                       "attention_threshold": .05, "fusion_layers": [0, 1]}
    else:
        ModelClass = AutoModelForImageTextToText
        configs = [("none", "none", 1.)] + [(f"{m}_{k}", m, k) for m in ("pre", "fastv") for k in (.25, .5, .75)]
    if a.configs:
        configs = [c for c in configs if c[0] in a.configs]
        assert len(configs) == len(a.configs), "Unknown config"
    model = ModelClass.from_pretrained(MODEL, config=config, dtype=torch.bfloat16,
                                        attn_implementation=ATTN).to("cuda").eval()
    assert model.visual.config._attn_implementation == "sdpa"
    assert model.model.language_model.config._attn_implementation == "eager"
    assert all(l.self_attn.config._attn_implementation == "eager" for l in model.model.language_model.layers)
    iid = model.config.image_token_id
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    trace = Trace(model)
    tap = B.MergerTap(model.visual) if a.engine == "harness" else None
    original_rank = B.rank_keep_indices

    def rank(*args, **kwargs):
        start = event()
        out = original_rank(*args, **kwargs)
        trace.selection_events.append((start, event()))
        return out
    B.rank_keep_indices = rank
    if a.engine == "hilo":
        original_selection = hilo_module.hierarchical_token_selection
        def select(pf, tf, *, keep_count, candidate_count):
            start = event()
            out = original_selection(pf, tf, keep_count=keep_count, candidate_count=candidate_count)
            trace.selection_events.append((start, event()))
            trace.selection.append({"n": int(pf.shape[0]), "keep": int(out.keep.numel()),
                                    "candidate": int(candidate_count)})
            return out
        hilo_module.hierarchical_token_selection = select
        for layer in model.model.language_model.layers:
            attn = layer.self_attn
            if hasattr(attn, "_compute_selective_boost"):
                fn = attn._compute_selective_boost
                def fusion(*args, _fn=fn, **kwargs):
                    start = event()
                    out = _fn(*args, **kwargs)
                    trace.fusion_events.append((start, event()))
                    return out
                attn._compute_selective_boost = fusion

    # Native/vendored no-op under exactly the shared attention protocol.
    # image_tensor=None explicitly bypasses hierarchical selection/pruning;
    # disable PA-fusion on the live modules as well (keep=1 alone is insufficient).
    noops = []
    fusion_flags = []
    if a.engine == "hilo":
        for layer in model.model.language_model.layers:
            attn = layer.self_attn
            if hasattr(attn, "use_pre_prune_fusion"):
                fusion_flags.append((attn, attn.use_pre_prune_fusion))
                attn.use_pre_prune_fusion = False
    with torch.inference_mode():
        for bench, sid, sample in requests[:32]:
            image = Image.open(sample.image).convert("RGB")
            inputs = B.build_inputs(processor, image, sample.question, PIXELS, "cuda")
            trace.reset()
            if tap:
                tap.reset()
            ie, pos, ds = B.capture_prepared_inputs(model, dict(inputs))
            assert ds is not None and len(ds) == 3, "No-op requires all three deepstack streams"
            prepared = {"input_ids": tensor_record(inputs["input_ids"]),
                        "position_ids": tensor_record(pos),
                        "inputs_embeds": tensor_record(ie),
                        "deepstack": [tensor_record(d) for d in ds]}
            del ie, pos, ds
            trace.reset()
            out = model.generate(**inputs, do_sample=False, max_new_tokens=MAXTOK,
                                 return_dict_in_generate=True, output_logits=True,
                                 eos_token_id=list(eos_ids), pad_token_id=sorted(eos_ids)[0])
            tokens = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
            noops.append({"id": sid, "token": tokens[0], "tokens": tokens,
                          "prepared": prepared,
                          "logits": out.logits[0].detach().float().cpu()})
            if a.engine == "hilo":
                assert not trace.selection, "No-op must bypass official selector"
            del inputs, out
    for attn, enabled in fusion_flags:
        attn.use_pre_prune_fusion = enabled
    if a.engine == "harness":
        torch.save(noops, a.noop_reference)
        metadata["noop"] = {"reference": str(a.noop_reference), "n": len(noops),
                            "attention": ATTN, "max_new_tokens": MAXTOK,
                            "description": "native HF full greedy tokens + first generated-token logits, prepared input/position/deepstack shapes and byte hashes, fixed feature probes; model.generate"}
    else:
        reference = torch.load(a.noop_reference, map_location="cpu", weights_only=True)
        assert [r["id"] for r in reference] == [r["id"] for r in noops]
        diffs = []
        for r, n in zip(reference, noops):
            prepared_diff = {}
            for key in ("input_ids", "position_ids", "inputs_embeds", "deepstack"):
                rl = r["prepared"][key] if key == "deepstack" else [r["prepared"][key]]
                nl = n["prepared"][key] if key == "deepstack" else [n["prepared"][key]]
                prepared_diff[key] = [{"shape": x["shape"], "same_shape": x["shape"] == y["shape"],
                                       "same_byte_hash": x["sha256"] == y["sha256"],
                                       "probe_max_abs": float((x["probe"] - y["probe"]).abs().max())}
                                      for x, y in zip(rl, nl)]
            diffs.append({"id": r["id"], "same_token": r["token"] == n["token"],
                          "same_greedy_tokens": r["tokens"] == n["tokens"],
                          "native_out_len": len(r["tokens"]), "vendored_out_len": len(n["tokens"]),
                          "logits_max_abs": float((r["logits"] - n["logits"]).abs().max()),
                          "logits_mean_abs": float((r["logits"] - n["logits"]).abs().mean()),
                          "prepared": prepared_diff})
        metadata["noop"] = {"selection_explicitly_bypassed": True, "fusion_explicitly_disabled": True,
                            "attention": ATTN, "dtype": "bfloat16", "n": len(diffs),
                            "max_new_tokens": MAXTOK, "comparisons": diffs}
        dump(a.out, metadata)
        assert all(x["same_greedy_tokens"] for x in diffs), "No-op full greedy-token mismatch: see saved metadata"
        assert all(all(t["same_byte_hash"] for t in x["prepared"][k]) for x in diffs
                   for k in ("input_ids", "position_ids")), "No-op input/position mismatch"
    del noops
    print(f"[v2] no-op gate OK {a.engine}", flush=True)

    @torch.inference_mode()
    def measured(bench, sid, sample, image, mode, keep):
        trace.reset()
        if tap:
            tap.reset()
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        cpu_inputs = B.build_inputs(processor, image, sample.question, PIXELS, "cpu")
        pf = None
        if a.engine == "hilo":
            pf = patch_features(image, cpu_inputs["image_grid_thw"][0],
                                patch_size=processor.image_processor.patch_size,
                                merge_size=processor.image_processor.merge_size)
        t_cpu = time.perf_counter()
        start_gpu = event()
        inputs = {k: v.to("cuda") for k, v in cpu_inputs.items()}
        if pf is not None:
            pf = pf.to("cuda")
        transfer_event = event()
        sync()
        t_transfer = time.perf_counter()
        mask = inputs["input_ids"][0] == iid
        nfull = int(mask.sum())
        expected = nfull if mode == "none" else max(1, int(round(nfull * keep)))
        prefill_start = None
        if a.engine == "harness":
            ie, pos, ds = B.capture_prepared_inputs(model, dict(inputs))
            sync()
            t_vision = time.perf_counter()
            selection_wall = 0.
            if mode == "pre":
                se = event()
                kept, _ = B.premerger_keep_units(tap.first_hs, inputs["image_grid_thw"], keep,
                                                  int(model.visual.spatial_merge_size) ** 2)
                ie, pos, ds, mask = B.apply_premerger(ie, pos, ds, mask, kept)
                trace.selection_events.append((se, event()))
                sync()
                selection_wall = time.perf_counter() - t_vision
            assert ds is not None and len(ds) == 3, "All methods need Qwen3 deepstack"
            prefill_start = time.perf_counter()
            gen, diag, t_first, first_event = generate_manual(model, ie, pos, mask, mode, ds, keep, eos_ids, trace)
            nkeep = int(diag["n_image_kept"])
            if mode == "pre":
                nkeep = int(mask.sum())
            assert nkeep == expected, (sid, mode, nfull, expected, nkeep)
            out_ids = gen
            selection_wall = selection_wall if mode == "pre" else None
        else:
            actual_tc = model.model.language_model.config
            assert actual_tc.hilo_keep_ratio == keep
            assert actual_tc.hilo_candidate_ratio == max(.3, keep)
            streamer = FirstToken(inputs["input_ids"])
            pad_id = model.generation_config.pad_token_id
            if pad_id is None:
                pad_id = processor.tokenizer.pad_token_id or sorted(eos_ids)[0]
            gen = model.generate(**inputs, image_tensor=pf, do_sample=False,
                                 max_new_tokens=MAXTOK, streamer=streamer,
                                 eos_token_id=list(eos_ids), pad_token_id=pad_id)
            out_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
            assert streamer.prompt_seen and streamer.first is not None, "Unverified streamer boundary"
            t_first, first_event = streamer.first, streamer.first_event
            assert len(trace.selection) == 1, trace.selection
            nkeep = trace.selection[0]["keep"]
            assert trace.selection[0]["n"] == nfull and nkeep == expected, (sid, nfull, expected, trace.selection)
            selection_wall = None
        end_gpu = event()
        sync()
        t_end = time.perf_counter()
        details = trace.finish()
        # Every layer input length is recorded once during prefill (decode has length 1).
        ntext = int(inputs["input_ids"].shape[1]) - nfull
        for layer in details["layer_trace"]:
            layer["n_image_active"] = layer["sequence_length"] - ntext
        # FastV's ranking layer uses a direct self-attention call and bypasses its layer hook.
        if mode == "fastv":
            present = {x["layer"] for x in details["layer_trace"]}
            for idx in range(len(model.model.language_model.layers)):
                if idx not in present:
                    details["layer_trace"].append({"layer": idx, "sequence_length": nfull + ntext,
                                                    "n_image_active": nfull,
                                                    "note": "direct attention ranking call"})
            details["layer_trace"].sort(key=lambda x: x["layer"])
        assert len(details["layer_trace"]) == len(model.model.language_model.layers), details["layer_trace"]
        return {"processor_cpu_s": t_cpu - t0, "transfer_s": t_transfer - t_cpu,
                "vision_s": details["vision_wall_s"],
                "vision_prepare_s": (t_vision - t_transfer) if a.engine == "harness" else trace.lm_start_wall - t_transfer,
                "selection_s": selection_wall if selection_wall is not None else details["selection_gpu_s"],
                "fusion_s": details["fusion_gpu_s"],
                "prefill_s": t_first - prefill_start if prefill_start is not None else t_first - trace.lm_start_wall,
                "ttft_s": t_first - t0, "decode_s": t_end - t_first, "total_s": t_end - t0,
                "compute_gpu_s": start_gpu.elapsed_time(end_gpu) / 1000,
                "transfer_gpu_s": start_gpu.elapsed_time(transfer_event) / 1000,
                "ttft_gpu_s": start_gpu.elapsed_time(first_event) / 1000,
                "out_len": len(out_ids), "generated_token_ids": out_ids,
                "answer": processor.decode(out_ids, skip_special_tokens=True).strip(),
                "n_image_full": nfull, "n_image_kept": nkeep,
                "actual_k_verified": True,
                "peak_alloc_GB": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_GB": torch.cuda.max_memory_reserved() / 2**30,
                **details}

    for tag, mode, keep in configs:
        if tag in metadata["results"]:
            print(f"[v2] skip complete {tag}", flush=True)
            continue
        if a.deadline_epoch and time.time() >= a.deadline_epoch:
            metadata["stopped_at_deadline"] = True
            dump(a.out, metadata)
            break
        if a.engine == "hilo":
            # from_pretrained deep-copies its input config: mutate the REAL live config.
            for cfg in (model.config.text_config, model.model.config.text_config,
                        model.model.language_model.config):
                cfg.hilo_keep_ratio = keep
                cfg.hilo_candidate_ratio = max(.3, keep)
        gc.collect()
        torch.cuda.empty_cache()
        warmup = []
        for bench, sid, sample in requests[:3]:
            image = Image.open(sample.image).convert("RGB")
            row = measured(bench, sid, sample, image, mode, keep)
            if a.engine == "harness":
                inputs = B.build_inputs(processor, image, sample.question, PIXELS, "cuda")
                tap.reset()
                ie, pos, ds = B.capture_prepared_inputs(model, dict(inputs))
                mask = inputs["input_ids"][0] == iid
                if mode == "pre":
                    kept, _ = B.premerger_keep_units(tap.first_hs, inputs["image_grid_thw"], keep,
                                                      int(model.visual.spatial_merge_size) ** 2)
                    ie, pos, ds, mask = B.apply_premerger(ie, pos, ds, mask, kept)
                ref, _ = B.generate_pruned(model, ie, pos, mask, mode,
                                           {"r": 1-keep, "fastv_k": 3, "ratios": []},
                                           MAXTOK, eos_ids, deepstack=ds)
                assert ref == row["generated_token_ids"], f"Parity failed: {tag} {sid}"
                if mode == "none":
                    # Native generation independently checks the manual no-pruning path.
                    with torch.inference_mode():
                        native = model.generate(**inputs, do_sample=False, max_new_tokens=MAXTOK,
                                                eos_token_id=list(eos_ids), pad_token_id=sorted(eos_ids)[0])
                    native_ids = native[0, inputs["input_ids"].shape[1]:].tolist()
                    assert native_ids == row["generated_token_ids"], f"Native parity failed: {sid}"
                    del native
                del inputs, ie, pos, ds
            warmup.append({"id": sid, "parity": ("exact token equality to generate_pruned and native model.generate" if mode == "none" else "exact token equality to generate_pruned") if a.engine == "harness" else "official generate; verified prompt callback and actual K"})
        metadata.setdefault("warmup", {})[tag] = warmup
        print(f"[v2] warmup/parity OK {tag}", flush=True)
        rows = []
        wall0 = time.perf_counter()
        for bench, sid, sample in requests:
            if a.deadline_epoch and time.time() >= a.deadline_epoch:
                metadata.setdefault("partial", {})[tag] = {"requests": rows, "reason": "deadline"}
                dump(a.out, metadata)
                return
            # Decode outside the per-request boundary and reuse the same RGB image across repeats.
            image = Image.open(sample.image).convert("RGB")
            raw = []
            error = None
            for repeat in range(a.repeats):
                try:
                    raw.append(measured(bench, sid, sample, image, mode, keep))
                except torch.cuda.OutOfMemoryError as exc:
                    error = str(exc)
                    if tap:
                        tap.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
            if error:
                row = {"id": sid, "benchmark": bench, "status": "oom", "error": error,
                       "repeats_raw": raw}
            else:
                row = {k: statistics.median([r[k] for r in raw]) for k in raw[0]
                       if isinstance(raw[0][k], (int, float)) and not isinstance(raw[0][k], bool)}
                row.update({"id": sid, "benchmark": bench, "status": "ok",
                            "actual_k_verified": all(r["actual_k_verified"] for r in raw),
                            "answer": raw[0]["answer"], "layer_trace": raw[0]["layer_trace"],
                            "answers_consistent": len({tuple(r["generated_token_ids"]) for r in raw}) == 1,
                            "repeats_raw": raw})
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f"[v2] {tag} {len(rows)}/{len(requests)} wall={time.perf_counter()-wall0:.1f}s", flush=True)
        metadata["results"][tag] = summarize(rows, time.perf_counter() - wall0, a.repeats, keep)
        dump(a.out, metadata)
        print(f"[v2] COMPLETE {tag}: {metadata['results'][tag]['n_success']}/{len(requests)}", flush=True)
    dump(a.out, metadata)


if __name__ == "__main__":
    main()
