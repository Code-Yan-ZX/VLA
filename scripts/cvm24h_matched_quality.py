#!/usr/bin/env python
"""Fresh full-split quality under the corrected P3 attention protocol.

Single A40, one process at a time; no scorer is imported or consulted. Three
warmups and every overlapping fixed P3 request are checked blindly against
its first repeat's tokens, answer and actual budget. Failed and unattempted
IDs remain in per_sample, so a partial file cannot silently lose hard cases.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.utils.hub import cached_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/v3_premerger"))
import baselines_hf as B
spec = importlib.util.spec_from_file_location("frozen_cvm24h_timing_v2", ROOT / "scripts/cvm24h_timing_v2.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# The cache snapshot resolved by the corrected P3 run on 2026-10-26.
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
HILO_COMMIT = "29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6"
PLAN = ROOT / "results/cvm2027_supplement_20261026/plan"
ATTN = {"vision_config": "sdpa", "text_config": "eager"}


class ProtocolMismatch(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_dump(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def provenance():
    names = ("config.json", "preprocessor_config.json", "processor_config.json",
             "tokenizer_config.json", "tokenizer.json", "chat_template.json", "chat_template.jinja")
    processor = {}
    for name in names:
        path = cached_file(MODEL, name, revision=REVISION, local_files_only=True,
                           _raise_exceptions_for_missing_entries=False)
        processor[name] = sha(path) if path else None
    index_path = cached_file(MODEL, "model.safetensors.index.json", revision=REVISION, local_files_only=True)
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    shards = []
    for name in sorted(set(weight_map.values())):
        path = Path(cached_file(MODEL, name, revision=REVISION, local_files_only=True))
        blob = path.resolve().name
        shards.append({"file": name, "bytes": path.stat().st_size,
                       "cache_blob_sha256": blob if re.fullmatch(r"[0-9a-f]{64}", blob) else None})
    return {"runner_sha256": sha(__file__),
            "helper_sha256": sha(ROOT / "src/v3_premerger/baselines_hf.py"),
            "v2_sha256": sha(ROOT / "scripts/cvm24h_timing_v2.py"),
            "processor_config_sha256": processor,
            "weight_index_sha256": sha(index_path), "weight_shards": shards,
            "weight_hash_policy": "immutable pinned HF snapshot; shard SHA256 names from content-addressed cache, not a new multi-GB byte rehash",
            "hilo_model_sha256": sha("/tmp/Hi-Lo_Prune/hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py"),
            "hilo_selection_sha256": sha("/tmp/Hi-Lo_Prune/hilo_prune/selection.py"),
            "hilo_patches_sha256": sha("/tmp/Hi-Lo_Prune/hilo_prune/patches.py")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("none", "pre", "fastv", "hilo"), required=True)
    ap.add_argument("--benchmark", choices=("textvqa", "ocrbench"), required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--keep-frac", type=float, default=.25)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    assert a.n > 0
    assert a.mode == "none" or a.keep_frac == .25, "This frozen main comparison permits keep=.25 only"
    keep = 1. if a.mode == "none" else .25
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sources = provenance()
    protocol = {"model": MODEL, "weight_revision": REVISION, "attention": ATTN,
                "dtype": "bfloat16", "max_pixels": 4000000, "max_new_tokens": 32,
                "seed": 0, "mrope": "native", "do_sample": False, "bs": 1,
                "processor": "AutoProcessor from the same pinned weight snapshot",
                "prefill_lm_head": "last token only, frozen corrected P3 generate_manual / official generate",
                "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    samples = B.load_subset(str(a.subset))[:a.n]
    raw_lines = [json.loads(line) for line in a.subset.read_text().splitlines() if line.strip()]
    original_rows = {str(row["id"]): row for row in raw_lines}
    assert len(samples) == a.n and len({str(s.id) for s in samples}) == a.n
    official_answers = {}
    official_image_ids = {}
    annotations_sha = None
    if a.benchmark == "textvqa":
        annotation_path = ROOT / "tmp/cvm_cpu_rescore_20261025/TextVQA_0.5.1_val.json"
        annotations_sha = sha(annotation_path)
        assert annotations_sha == "4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d"
        official_rows = json.loads(annotation_path.read_text())["data"]
        official_answers = {str(row["question_id"]): row["answers"] for row in official_rows}
        official_image_ids = {str(row["question_id"]): row["image_id"] for row in official_rows}
    records = []
    for sample in samples:
        sid = str(sample.id)
        source = original_rows[sid]
        answers = official_answers.get(sid, source.get("answers"))
        if a.benchmark == "textvqa":
            assert answers is not None and len(answers) == 10
            assert ";".join(x.strip() for x in answers) == sample.gt
            image_id = official_image_ids[sid]
            assert image_id is not None
            image_id_source = "official TextVQA_0.5.1_val.json question_id lookup"
        else:
            image_id = source.get("image_id", sid)
            image_id_source = ("preserved OCRBench row image_id" if source.get("image_id") is not None
                               else "preserved OCRBench row index alias")
        records.append({"id": sid, "question_id": source.get("question_id", sid),
                        "image_id": image_id, "image_id_source": image_id_source, "image": sample.image,
                        "question": sample.question, "gt": sample.gt, "answers": answers,
                        "dataset": source.get("dataset"), "question_type": source.get("question_type"),
                        "answer": None, "error": None, "attempted": False, "status": "not_attempted",
                        "skipped": True, "n_image_full": None, "n_image_kept": None,
                        "input_ids_sha256": None, "n_prompt_full": None, "prompt_token_ids": None,
                        "image_grid_thw": None})
    report = {"schema_version": 1, "mode": a.mode, "benchmark": a.benchmark,
              "engine": "hilo-vendored" if a.mode == "hilo" else "hf-corrected-manual",
              "model": MODEL, "keep_fraction": keep, "keep_ratio": keep,
              "expected_n": a.n, "n": a.n, "n_attempted": 0, "n_pending": a.n,
              "n_scored": 0, "n_error": 0, "n_skipped": a.n, "complete": False,
              "protocol": protocol, "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              "source_hashes": sources, "subset": str(a.subset), "subset_sha256": sha(a.subset),
              "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in records).encode()).hexdigest(),
              "official_textvqa_annotations_sha256": annotations_sha,
              "prediction_reuse": False, "warmup_n": 3,
              "wall_boundary": "quality-process wall includes model loading, warmup, local IO and bookkeeping; use corrected P3 for latency comparisons",
              "configuration": {"fastv_k": 3, "fastv_k_semantics": "zero-based layer index; prune after layer 3",
                                "hilo_keep_ratio": keep if a.mode == "hilo" else None,
                                "hilo_candidate_ratio": .3 if a.mode == "hilo" else None,
                                "hilo_prune_layer": 2, "hilo_use_fusion": True,
                                "hilo_fusion_strength": .3, "hilo_top_k_ratio": .1,
                                "hilo_attention_threshold": .05, "hilo_fusion_layers": [0, 1],
                                "hilo_commit": HILO_COMMIT},
              "per_sample": records}
    if a.out.exists():
        old = json.loads(a.out.read_text())
        assert old["protocol"] == protocol and old["source_hashes"] == sources, "Refuse incompatible resume"
        assert old["ids_sha256"] == report["ids_sha256"] and old["mode"] == a.mode
        assert not old.get("protocol_failure"), "Refuse to resume a failed protocol; investigate its saved evidence first"
        if old["complete"]:
            print(f"[matched] already complete {a.out}", flush=True)
            return
        report = old
        records = report["per_sample"]
        report.pop("stop_reason", None)
    start_wall = time.perf_counter()
    previous_wall = report.get("wall_s", 0.)

    def save():
        report["n_attempted"] = sum(r["attempted"] for r in records)
        report["n_pending"] = sum(not r["attempted"] for r in records)
        report["n_scored"] = sum(r["status"] == "ok" for r in records)
        report["n_error"] = sum(r["attempted"] and r["status"] != "ok" for r in records)
        report["n_skipped"] = sum(r["skipped"] for r in records)
        report["complete"] = report["n_pending"] == 0 and not report.get("protocol_failure")
        report["wall_s"] = previous_wall + time.perf_counter() - start_wall
        atomic_dump(a.out, report)
    save()
    if time.time() >= a.deadline_epoch:
        report["stop_reason"] = "deadline before model load"
        save()
        return
    engine = "hilo" if a.mode == "hilo" else "harness"
    reference_path = PLAN / f"timing_{engine}_corrected.json"
    reference_data = json.loads(reference_path.read_text())
    assert reference_data["schema_version"] == 2 and reference_data["attention"] == ATTN
    assert reference_data["script_sha256"] == sources["v2_sha256"], "P3 reference source changed"
    ref_tag = "none" if a.mode == "none" else f"{a.mode}_0.25"
    reference_block = reference_data["results"][ref_tag]
    reference = {str(r["id"]): r for r in reference_block["requests"]
                 if r["benchmark"] == a.benchmark and r["status"] == "ok"}
    prior_checks = report.get("blind_p3_check", {})
    reference_sha = sha(reference_path)
    if prior_checks:
        assert prior_checks["sha256"] == reference_sha, "P3 reference changed during resume"
    report["blind_p3_check"] = {"source": str(reference_path), "sha256": reference_sha,
                               "tag": ref_tag, "warmup": [], "formal_overlap": [],
                               "expected_formal_overlap": sum(r["id"] in reference for r in records),
                               "rule": "compare saved first repeat tokens/answer/N/K only; never inspect scores"}
    report["blind_p3_check"]["formal_overlap"] = prior_checks.get("formal_overlap", [])
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    assert config._commit_hash == REVISION
    selection_log = []
    if a.mode == "hilo":
        sys.path.insert(0, "/tmp/Hi-Lo_Prune")
        from hilo_prune.models.qwen3_vl import Qwen3VLForConditionalGeneration as ModelClass
        from hilo_prune.models.qwen3_vl import modeling_hilo_qwen3_vl as hilo_module
        from hilo_prune.patches import image_to_merged_patch_features
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
        original_selection = hilo_module.hierarchical_token_selection

        def select(pf, tf, *, keep_count, candidate_count):
            out = original_selection(pf, tf, keep_count=keep_count, candidate_count=candidate_count)
            selection_log.append({"n": int(pf.shape[0]), "keep": int(out.keep.numel()),
                                  "candidate": int(candidate_count), "ids": out.keep.cpu().tolist()})
            return out
        hilo_module.hierarchical_token_selection = select
    else:
        ModelClass = AutoModelForImageTextToText
    model = ModelClass.from_pretrained(MODEL, revision=REVISION, config=config, local_files_only=True,
                                        dtype=torch.bfloat16, attn_implementation=ATTN).to("cuda").eval()
    assert model.visual.config._attn_implementation == "sdpa"
    assert model.model.language_model.config._attn_implementation == "eager"
    if a.mode == "hilo":
        live = model.model.language_model.config
        assert live.hilo_keep_ratio == .25 and live.hilo_candidate_ratio == .3
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    iid = model.config.image_token_id
    tap = B.MergerTap(model.visual) if a.mode == "pre" else None
    rank_log = []
    original_rank = B.rank_keep_indices

    def rank(attn_w, image_mask, k_keep):
        out = original_rank(attn_w, image_mask, k_keep)
        kept_mask = image_mask.index_select(0, out)
        ordinal = image_mask.long().cumsum(0) - 1
        rank_log.append(ordinal.index_select(0, out)[kept_mask].cpu().tolist())
        return out
    B.rank_keep_indices = rank

    @torch.inference_mode()
    def infer(sample, rec):
        if tap:
            tap.reset()
        rank_log.clear()
        selection_log.clear()
        image = Image.open(sample.image).convert("RGB")
        inputs = B.build_inputs(processor, image, sample.question, 4000000, "cuda")
        mask = inputs["input_ids"][0] == iid
        nfull = int(mask.sum())
        expected = nfull if a.mode == "none" else max(1, round(nfull * keep))
        rec.update({"n_image_full": nfull, "input_ids_sha256": tensor_sha(inputs["input_ids"]),
                    "n_prompt_full": int(inputs["input_ids"].shape[1]), "actual_k_expected": expected,
                    "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist()})
        if a.mode == "hilo":
            pf = image_to_merged_patch_features(image, inputs["image_grid_thw"][0].cpu(),
                                                patch_size=processor.image_processor.patch_size,
                                                merge_size=processor.image_processor.merge_size).to("cuda")
            if pf.shape[0] != nfull:
                raise ProtocolMismatch(f"patch rows {pf.shape[0]} != input image count {nfull}")
            pad_id = model.generation_config.pad_token_id
            if pad_id is None:
                pad_id = processor.tokenizer.pad_token_id or sorted(eos_ids)[0]
            output = model.generate(**inputs, image_tensor=pf, do_sample=False, max_new_tokens=32,
                                    eos_token_id=list(eos_ids), pad_token_id=pad_id)
            tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
            if len(selection_log) != 1 or selection_log[0]["n"] != nfull:
                raise ProtocolMismatch(f"invalid official selection trace: {selection_log}")
            nkept = selection_log[0]["keep"]
            kept_ids = selection_log[0]["ids"]
            rec["selection_candidate_count"] = selection_log[0]["candidate"]
        else:
            ie, pos, ds = B.capture_prepared_inputs(model, dict(inputs))
            if ds is None or len(ds) != 3 or any(d.shape[0] != nfull for d in ds):
                raise ProtocolMismatch("missing or misaligned Qwen3 deepstack stream")
            kept_ids = None
            if a.mode == "pre":
                kept, _ = B.premerger_keep_units(tap.first_hs, inputs["image_grid_thw"], keep,
                                                  int(model.visual.spatial_merge_size) ** 2)
                kept_ids = kept.cpu().tolist()
                ie, pos, ds, mask = B.apply_premerger(ie, pos, ds, mask, kept)
            tokens, diag, _, _ = T.generate_manual(model, ie, pos, mask, a.mode, ds, keep, eos_ids, None)
            nkept = int(mask.sum()) if a.mode == "pre" else int(diag["n_image_kept"])
            if diag["n_deepstack"] != 3:
                raise ProtocolMismatch("manual generation dropped a deepstack stream")
            if a.mode == "fastv":
                if len(rank_log) != 1:
                    raise ProtocolMismatch(f"unexpected FastV ranking count {len(rank_log)}")
                kept_ids = rank_log[0]
            rec["deepstack_streams"] = 3
        rec.update({"n_image_kept": nkept, "kept_image_ids": kept_ids,
                    "kept_all": a.mode == "none", "prompt_token_ids": rec["n_prompt_full"] - nfull + nkept,
                    "generated_token_ids": tokens, "out_len": len(tokens),
                    "answer": processor.decode(tokens, skip_special_tokens=True).strip()})
        if nkept != expected or (kept_ids is not None and len(kept_ids) != expected):
            raise ProtocolMismatch(f"actual K {nkept} != frozen expected K {expected}; N={nfull}")
        if kept_ids is not None and (kept_ids != sorted(set(kept_ids)) or min(kept_ids) < 0 or max(kept_ids) >= nfull):
            raise ProtocolMismatch("kept visual units are not unique/sorted native indices")
        rec["actual_k_verified"] = True

    def check_reference(rec):
        sid = rec["id"]
        ref = reference[sid]
        first = ref["repeats_raw"][0]
        checks = {"id": sid, "answer_equal": rec["answer"] == first["answer"],
                  "tokens_equal": rec["generated_token_ids"] == first["generated_token_ids"],
                  "fullN_equal": rec["n_image_full"] == first["n_image_full"],
                  "actualK_equal": rec["n_image_kept"] == first["n_image_kept"]}
        if not all(v for k, v in checks.items() if k != "id"):
            rec["p3_reference_failure"] = {"checks": checks, "reference_first_repeat": first}
            raise ProtocolMismatch(f"blind P3 agreement failed for {sid}: {checks}")
        return checks

    filename = "textvqa_val.jsonl" if a.benchmark == "textvqa" else "ocrbench.jsonl"
    canonical = {str(s.id): s for s in B.load_subset(str(ROOT / "eval/full_splits" / filename))}
    warmup_ids = list(reference)[:3]
    assert len(warmup_ids) == 3
    try:
        for sid in warmup_ids:
            if time.time() >= a.deadline_epoch:
                report["stop_reason"] = "deadline during warmup"
                save()
                return
            rec = {"id": sid}
            try:
                infer(canonical[sid], rec)
                report["blind_p3_check"]["warmup"].append(check_reference(rec))
            except Exception:
                report["warmup_failure"] = rec
                raise
        print(f"[matched] {a.mode}/{a.benchmark} 3 blind P3 warmups PASS", flush=True)
        for sample, rec in zip(samples, records):
            if rec["attempted"]:
                continue
            if time.time() >= a.deadline_epoch:
                report["stop_reason"] = "deadline before next request"
                save()
                return
            rec["attempted"] = True
            try:
                infer(sample, rec)
                if rec["id"] in reference:
                    check = check_reference(rec)
                    report["blind_p3_check"]["formal_overlap"].append(check)
                    rec["p3_first_repeat_equal"] = True
                rec.update({"error": None, "status": "ok", "skipped": False})
            except ProtocolMismatch as exc:
                rec.update({"error": f"ProtocolMismatch: {exc}", "status": "protocol_error", "skipped": True})
                raise
            except Exception as exc:
                rec.update({"answer": None, "error": f"{type(exc).__name__}: {exc}",
                            "status": "failed", "skipped": True})
                if tap:
                    tap.reset()
                torch.cuda.empty_cache()
            attempted = sum(r["attempted"] for r in records)
            if attempted % 100 == 0:
                save()
                print(f"[matched] {a.mode}/{a.benchmark} {attempted}/{a.n} "
                      f"error={report['n_error']} wall={report['wall_s']:.1f}s", flush=True)
        save()
        print(f"[matched] COMPLETE {a.mode}/{a.benchmark}: attempted={report['n_attempted']} "
              f"success={report['n_scored']} error={report['n_error']}", flush=True)
    except Exception as exc:
        report["protocol_failure"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
