#!/usr/bin/env python
"""P0 correctness probe (directive 4a/4b/4c): bitwise no-op invariants,
deepstack stream sync audit, logit deltas, native-vs-loop answer parity.
Single GPU pass, n=8 samples.  Smoke-only: never used to pick params/samples.
Output JSON: results/.../plan/p0_probe.json
"""
import json, sys, time
import torch
sys.path.insert(0, "src/v3_premerger")
import baselines_hf as B
from transformers import AutoModelForImageTextToText, AutoProcessor

DEV = "cuda"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SMOKE = "results/cvm2027_supplement_20261026/plan/smoke32_textvqa.jsonl"
OUT = "results/cvm2027_supplement_20261026/plan/p0_probe.json"

t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
iid = model.config.image_token_id
spatial_merge = int(getattr(model.visual, "spatial_merge_size", 2))
spatial_unit = spatial_merge ** 2
eos = model.generation_config.eos_token_id
eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
n_layers = model.config.text_config.num_hidden_layers
from PIL import Image
samples = B.load_subset(SMOKE)[:N]
print(f"[probe] model loaded in {time.perf_counter()-t0:.0f}s, layers={n_layers}, "
      f"unit={spatial_unit}, n_samples={len(samples)}", flush=True)

def logits_last(ie, pos):
    with torch.no_grad():
        out = model.model.language_model(inputs_embeds=ie, position_ids=pos,
                                         use_cache=False)
        return model.lm_head(out.last_hidden_state[:, -1:, :])[0, 0].float()

def eq(a, b):
    return a.shape == b.shape and torch.equal(a, b)

def diff(a, b):
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "cos": cos}

tap = B.MergerTap(model.visual)
recs = []
for s in samples:
    img = Image.open(s.image).convert("RGB")
    inputs = B.build_inputs(processor, img, s.question, 4000000, DEV)
    mask = (inputs["input_ids"][0] == iid)
    tap.reset()
    ie, pos, ds = B.capture_prepared_inputs(model, {k: v for k, v in inputs.items()})
    n_img = int(mask.sum()); n_txt = int((~mask).sum())
    ar = torch.arange(n_img, device=DEV)
    r = {"id": s.id, "n_image_full": n_img, "n_text": n_txt,
         "dtype": str(ie.dtype), "deepstack_streams": (None if ds is None else len(ds)),
         "deepstack_rows": (None if ds is None else [int(e.shape[0]) for e in ds]),
         "grid_thw": inputs["image_grid_thw"].tolist(),
         "ds_rows_match_full": None if ds is None else all(int(e.shape[0]) == n_img for e in ds)}

    # --- (a) pre no-op @ keep=1.0: bitwise identity --------------------------
    ie2, pos2, ds2, m2 = B.apply_premerger(ie, pos, ds, mask, ar)
    lg_nat = logits_last(ie, pos); lg_pre1 = logits_last(ie2, pos2)
    r["noop_pre"] = {"embeds_equal": eq(ie, ie2), "pos_equal": eq(pos, pos2),
                     "mask_equal": eq(m2, mask),
                     "ds_equal": None if ds is None else all(eq(a, b) for a, b in zip(ds, ds2)),
                     "logits": diff(lg_nat, lg_pre1)}

    # --- (a2) post no-op @ keep=1.0 -------------------------------------------
    kept_p, dpost = B.postmerger_keep_tokens(ie, mask, 1.0,
                                             inputs["image_grid_thw"], spatial_unit)
    r["noop_post"] = {"kept_is_arange": eq(kept_p, ar),
                      "diag_kept": dpost["n_image_kept"]}

    # --- (a3) fastv no-op @ r=0 -----------------------------------------------
    plan0 = B.build_prune_plan("fastv", n_layers, n_img, 0.0, 3, [])
    r["noop_fastv"] = {"plan": {str(k): v for k, v in plan0.items()},
                       "identity_budget": all(v >= n_img for v in plan0.values())}

    # --- (b/c) deepstack sync + K consistency at real budgets -----------------
    if tap.first_hs is None:
        raise RuntimeError("merger tap captured nothing")
    for c in (0.25, 0.50, 0.75):
        kept_c, dpre = B.premerger_keep_units(tap.first_hs, inputs["image_grid_thw"],
                                              c, spatial_unit)
        iec, posc, dsc, mc = B.apply_premerger(ie, pos, ds, mask, kept_c)
        k_exp = sum(max(1, round(f * c)) for f in dpre["full_per_image"])
        ok = {"kept": int(kept_c.numel()), "expected_formula_k": k_exp,
              "mask_sum": int(mc.sum()), "text_preserved": int((~mc).sum()) == n_txt,
              "ds_all_sliced": None if dsc is None else all(
                  int(e.shape[0]) == int(kept_c.numel()) for e in dsc)}
        ok["all_ok"] = (ok["kept"] == ok["expected_formula_k"] == ok["mask_sum"]
                        and ok["text_preserved"] and ok["ds_all_sliced"] is not False)
        r[f"pre_keep{int(c*100)}"] = ok

    # --- manual decode loop == native HF generate at keep=1 -------------------
    cfg = {"r": 0.0, "fastv_k": 3, "ratios": []}
    with torch.no_grad():
        gen_h, _dg = B.generate_pruned(model, ie2, pos2, m2, "pre", cfg, 32,
                                       eos_ids, deepstack=ds2)
        ans_h = processor.decode(gen_h, skip_special_tokens=True).strip()
        gen_r = model.generate(**inputs, do_sample=False, max_new_tokens=32)[0]
        # strip prompt
        plen = inputs["input_ids"].shape[1]
        ans_r = processor.decode(gen_r[plen:], skip_special_tokens=True).strip()
    r["answers"] = {"loop_pre1": ans_h, "raw_native": ans_r, "match": ans_h == ans_r}
    recs.append(r)
    print(f"[probe] {s.id} n_img={n_img} noop_pre={r['noop_pre']['embeds_equal']} "
          f"ansmatch={r['answers']['match']}", flush=True)

out = {"model": MODEL, "n": len(recs), "transformers": B.__dict__.get("_TFV", "") or
       __import__("transformers").__version__, "torch": torch.__version__,
       "attn": "eager", "max_pixels": 4000000, "dtype": "bfloat16",
       "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "peak_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
       "samples": recs}
allok = all(x["noop_pre"]["embeds_equal"] and x["noop_pre"]["pos_equal"]
            and x["answers"]["match"] and x["noop_fastv"]["identity_budget"]
            and x["noop_post"]["kept_is_arange"]
            and all(x[f"pre_keep{c}"]["all_ok"] for c in (25, 50, 75)) for x in recs)
out["GATE_P0_pass"] = allok
json.dump(out, open(OUT, "w"), indent=1)
print(f"[probe] GATE_P0_pass={allok} -> {OUT}  ({time.perf_counter()-t0:.0f}s)")
