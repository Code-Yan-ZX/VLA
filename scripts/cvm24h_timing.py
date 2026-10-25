#!/usr/bin/env python
"""P3 complete per-request cost measurement (directives 14-16).
Single GPU, bs=1, HF eager (same as eval protocol), 200 pre-fixed requests
(plan/p3_timing_ids.json), 3 repeats each, warmup excluded from stats.
Segments per request (sync boundaries, one measured pass -- no stitched totals):
  processor_s (CPU PIL+resizer+tokenizer, from decoded image),
  vision_merger_s, selection_s (pre-mode top-k+apply; fastv folded into prefill
  because its ranking happens inside LLM layers -- boundary documented),
  prefill_s (up to first token; TTFT = processor EXCLUDED: compute request
  boundary starts after decoded image => TTFT_gpu = vision+select+prefill),
  decode_s, total_s, out_len.
Engine 'hilo': official vendored path; only wall/TTFT(via streamer)/out_len are
separable (internals not exposed without surgery -- reported as such).
Parity gate: instrumented replica must reproduce generate_pruned token ids
exactly on 3 warmup requests, else ABORT.
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, "src/v3_premerger")
import baselines_hf as B
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="harness", choices=["harness", "hilo"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

IDS = json.load(open("results/cvm2027_supplement_20261026/plan/p3_timing_ids.json"))
SPLITS = {"textvqa": "eval/full_splits/textvqa_val.jsonl",
          "ocrbench": "eval/full_splits/ocrbench.jsonl"}
samples = {}
for bench in ("textvqa", "ocrbench"):
    all_s = {s.id: s for s in B.load_subset(SPLITS[bench])}
    for i in IDS[f"{bench}_100"]:
        samples[i] = all_s[i]
REQS = [(i, samples[i]) for i in IDS["textvqa_100"]] + \
       [(i, samples[i]) for i in IDS["ocrbench_100"]]
assert len(REQS) == 200

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEV = "cuda"
MAXTOK = 32
PX = 4000000
REPEATS = 3

def sync():
    torch.cuda.synchronize()

if a.engine == "harness":
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
    iid = model.config.image_token_id
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    spatial_merge = int(getattr(model.visual, "spatial_merge_size", 2))
    spatial_unit = spatial_merge ** 2
    n_layers = model.config.text_config.num_hidden_layers

    CONFIGS = [("none", None), ("pre", 0.25), ("pre", 0.50), ("pre", 0.75),
               ("fastv", 0.25), ("fastv", 0.50), ("fastv", 0.75)]
    # fastv keep0.25 == r 0.75 k3 (same as main table)

    def instrumented(cfgname, keep, sid, s):
        t = {}
        sync(); tw0 = time.perf_counter()
        img = Image.open(s.image).convert("RGB")
        inputs = B.build_inputs(processor, img, s.question, PX, DEV)
        sync(); tw1 = time.perf_counter()
        t["processor_s"] = tw1 - tw0
        mask = (inputs["input_ids"][0] == iid)
        tap = B.MergerTap(model.visual) if cfgname == "pre" else None
        if tap:
            tap.reset()
        ie, pos, ds = B.capture_prepared_inputs(model, dict(inputs))
        sync(); tw2 = time.perf_counter()
        t["vision_merger_s"] = tw2 - tw1
        if cfgname == "none":
            gen, tw3 = model_generate(ie, pos, mask, "none", ds)
        elif cfgname == "pre":
            kept, dpre = B.premerger_keep_units(tap.first_hs, inputs["image_grid_thw"],
                                                keep, spatial_unit)
            ie2, pos2, ds2, m2 = B.apply_premerger(ie, pos, ds, mask, kept)
            sync(); tws = time.perf_counter()
            t["selection_s"] = tws - tw2
            gen, tw3 = model_generate(ie2, pos2, m2, "pre", ds2)
        else:  # fastv: selection inside prefill (layer-3 attention) -- folded
            t["selection_s"] = None
            cfgd = {"r": 1.0 - keep, "fastv_k": 3, "ratios": []}
            plan = B.build_prune_plan("fastv", n_layers, int(mask.sum()), 1.0 - keep, 3, [])
            gen, tw3 = model_generate(ie, pos, mask, "fastv", None, cfgd)
        t["prefill_s"] = tw3[0] - tw2 if cfgname != "pre" else tw3[0] - tws
        t["decode_s"] = tw3[1] - tw3[0]
        t["total_s"] = tw3[1] - tw1
        t["out_len"] = len(gen)
        return t, gen

    def model_generate(ie, pos, mask, stage, ds, cfgd=None):
        """Parity-checked replica of prefill_pruned+decode loop with first-token
        timestamp."""
        LM = model.model.language_model
        hidden, position_ids, cache, image_mask, diag = B.prefill_pruned(
            model, ie, pos, mask, stage,
            cfgd if cfgd else {"r": 0.0, "fastv_k": 3, "ratios": []},
            deepstack=ds)
        logits = model.lm_head(hidden)
        next_tok = int(logits[0, -1].argmax(-1))
        sync(); tt_first = time.perf_counter()
        gen = [next_tok]
        cur_pos = int(position_ids.max())
        kv_len = B._cache_len(cache)
        embed = model.get_input_embeddings()
        dtype = ie.dtype
        device = ie.device
        for _ in range(MAXTOK - 1):
            if next_tok in eos_ids:
                break
            tok_emb = embed(torch.tensor([[next_tok]], device=device))
            cur_pos += 1
            pos_new = torch.full((3, 1, 1), cur_pos, device=device, dtype=position_ids.dtype)
            pe = LM.rotary_emb(tok_emb, pos_new)
            dec_mask = torch.zeros(1, 1, 1, kv_len + 1, device=device, dtype=dtype)
            h = tok_emb
            for layer in LM.layers:
                o = layer(h, attention_mask=dec_mask, position_ids=None,
                          past_key_values=cache, use_cache=True, cache_position=None,
                          position_embeddings=pe)
                h = o[0] if isinstance(o, tuple) else o
            kv_len += 1
            h = LM.norm(h)
            next_tok = int(model.lm_head(h)[0, -1].argmax(-1))
            gen.append(next_tok)
        sync()
        return gen, (tt_first, time.perf_counter())

    # ---- parity gate (3 requests vs original generate_pruned) ----
    parity = []
    for (sid, s) in REQS[:3]:
        inputs = B.build_inputs(processor, Image.open(s.image).convert("RGB"),
                                s.question, PX, DEV)
        mask = (inputs["input_ids"][0] == iid)
        ie, pos, ds = B.capture_prepared_inputs(model, dict(inputs))
        cfgd = {"r": 0.0, "fastv_k": 3, "ratios": []}
        g_orig, _ = B.generate_pruned(model, ie, pos, mask, "none", cfgd, MAXTOK,
                                      eos_ids, deepstack=ds)
        g_rep, _ = model_generate(ie, pos, mask, "none", ds)
        parity.append(g_orig == g_rep)
    if not all(parity):
        raise SystemExit("PARITY FAIL: instrumented replica != generate_pruned")
    print("[timing] parity OK", flush=True)

    results = {}
    if os.path.exists(a.out):   # idempotent restart: keep finished configs
        try:
            results.update(json.load(open(a.out))["results"])
            print(f"[timing] resume with {list(results)} done", flush=True)
        except Exception:
            pass
    for cfgname, keep in CONFIGS:
        tag = f"{cfgname}_{keep}" if keep else "none"
        if tag in results:
            continue
        torch.cuda.reset_peak_memory_stats()
        # warmup 3 requests (excluded)
        for (sid, s) in REQS[:3]:
            instrumented(cfgname, keep, sid, s)
        rows = []
        t_cfg0 = time.perf_counter()
        for (sid, s) in REQS:
            reps = []
            oom = False
            for r in range(REPEATS):
                try:
                    t, gen = instrumented(cfgname, keep, sid, s)
                    reps.append(t)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    oom = True
                    torch.cuda.empty_cache()
                    break
            if oom or not reps:
                rows.append({"id": sid, "oom": True})   # honest: excluded from
                continue                                # p50/p95, listed as OOM
            med = {k: (sorted(r[k] for r in reps)[REPEATS // 2]
                       if reps[0].get(k) is not None else None) for k in reps[0]}
            med["id"] = sid
            med["repeats_raw"] = reps
            rows.append(med)
        wall_cfg = time.perf_counter() - t_cfg0
        ok_rows = [r for r in rows if not r.get("oom")]
        results[tag] = {"requests": rows,
                        "n_oom": len(rows) - len(ok_rows),
                        "workload_wall_s": round(wall_cfg, 1),
                        "throughput_req_per_s": round(3 * len(ok_rows) / wall_cfg, 3),
                        "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "peak_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        json.dump({"engine": "harness-eager", "bs": 1, "repeats": REPEATS,
                   "note": "TTFT_gpu=vision+select+prefill of the SAME measured request; "
                           "processor from decoded image excluded; fastv ranking folded into "
                           "prefill; OOM requests flagged and excluded from percentiles.",
                   "results": results}, open(a.out, "w"), indent=1)
        print(f"[timing] {tag} done wall={wall_cfg:.0f}s oom={results[tag]['n_oom']}", flush=True)
    out = {"engine": "harness-eager", "bs": 1, "repeats": REPEATS,
           "note": "TTFT_gpu=vision+select+prefill segments of the SAME measured "
                   "request; processor from decoded image excluded; fastv ranking "
                   "inside LLM layer 3 folded into prefill.",
           "results": results}
    json.dump(out, open(a.out, "w"), indent=1)
else:
    # HiLo vendored path: wall + TTFT via streamer first-token + out_len only
    sys.path.insert(0, "/tmp/Hi-Lo_Prune")
    from transformers import AutoConfig, AutoProcessor as AP
    from hilo_prune.models.qwen3_vl import Qwen3VLForConditionalGeneration as HiLo
    from hilo_prune.patches import image_to_merged_patch_features
    try:
        from transformers import BaseStreamer
    except ImportError:
        try:
            from transformers.generation.streamers import BaseStreamer
        except ImportError:
            from transformers.streams import BaseStreamer
    processor = AP.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.text_config
    rs = getattr(tc, "rope_scaling", None) or {}
    tc.rope_parameters = {"rope_type": rs.get("type", "default"),
                          "rope_theta": tc.rope_theta,
                          "mrope_section": rs.get("mrope_section", [24, 20, 20])}
    results = {}
    model_ref = [None]
    if os.path.exists(a.out):
        try:
            results.update(json.load(open(a.out))["results"])
            print(f"[timing] resume hilo with {list(results)} done", flush=True)
        except Exception:
            pass
    for keep in (0.25, 0.50, 0.75):
        if f"hilo_{keep}" in results:
            continue
        tc.hilo_keep_ratio = keep
        tc.hilo_candidate_ratio = max(0.3, keep)
        tc.hilo_prune_layer = 2
        tc.hilo_use_fusion = True
        tc.hilo_fusion_strength = 0.3
        tc.hilo_top_k_ratio = 0.1
        tc.hilo_attention_threshold = 0.05
        tc.hilo_fusion_layers = [0, 1]
        if "model" not in dir() and model_ref[0] is None:
            model_ref[0] = HiLo.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(DEV).eval()
        model = model_ref[0]
        torch.cuda.reset_peak_memory_stats()

        class TTFT(BaseStreamer):
            """HF generate() feeds the PROMPT to streamer.put BEFORE decoding,
            so the 1st callback is not a generated token.  TTFT = time of the
            2nd callback (first generated token).  If the implementation never
            emits a prompt-put, the 2nd callback would be token #2 -- measured
            one decode step late (~20ms); acceptable and documented."""
            def __init__(self):
                self.first = None
                self.n = 0
            def put(self, value):
                self.n += 1
                if self.n >= 2 and self.first is None:
                    sync(); self.first = time.perf_counter()
            def end(self):
                pass

        rows = []
        tw0 = time.perf_counter()
        for (sid, s) in REQS:
            reps = []
            oom = False
            for r in range(REPEATS):
                try:
                    sync(); t0 = time.perf_counter()
                    img = Image.open(s.image).convert("RGB")
                    inputs = B.build_inputs(processor, img, s.question, PX, DEV)
                    grid = inputs["image_grid_thw"][0].cpu()
                    pf = image_to_merged_patch_features(
                        img, grid, patch_size=processor.image_processor.patch_size,
                        merge_size=processor.image_processor.merge_size).to(DEV)
                    st = TTFT()
                    sync(); t1 = time.perf_counter()
                    with torch.inference_mode():
                        gen = model.generate(**inputs, image_tensor=pf, do_sample=False,
                                             max_new_tokens=MAXTOK, streamer=st)
                    sync(); t2 = time.perf_counter()
                    reps.append({"processor_s": t1 - t0, "total_s": t2 - t0,
                                 "ttft_s": (st.first - t0) if st.first else None,
                                 "out_len": int(gen.shape[1] - inputs["input_ids"].shape[1])})
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    oom = True
                    torch.cuda.empty_cache()
                    break
            if oom or not reps:
                rows.append({"id": sid, "oom": True})
                continue
            med = {k: sorted(rr[k] for rr in reps)[1] for k in reps[0] if reps[0][k] is not None}
            med["id"] = sid; med["repeats_raw"] = reps
            rows.append(med)
        ok_rows = [r for r in rows if not r.get("oom")]
        results[f"hilo_{keep}"] = {"requests": rows,
                                   "n_oom": len(rows) - len(ok_rows),
                                   "workload_wall_s": round(time.perf_counter()-tw0,1),
                                   "throughput_req_per_s": round(3*len(ok_rows)/(time.perf_counter()-tw0),3),
                                   "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                   "peak_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                                   "note": "vendored official path; vision/select inside generate; segments not separable without surgery -- reported as wall/TTFT only"}
        json.dump({"engine": "hilo-vendored", "bs": 1, "repeats": REPEATS,
                   "note": "vendored official path (sdpa, same as predictions); vision/select inside "
                           "generate -> wall/TTFT only; OOM flagged & excluded from percentiles.",
                   "results": results}, open(a.out, "w"), indent=1)
        print(f"[timing hilo] keep={keep} done oom={results[f'hilo_{keep}']['n_oom']}", flush=True)
    json.dump({"engine": "hilo-vendored", "bs": 1, "repeats": REPEATS,
               "results": results}, open(a.out, "w"), indent=1)
