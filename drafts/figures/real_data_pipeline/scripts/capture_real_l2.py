#!/usr/bin/env python
"""capture_real_l2.py — REAL-DATA PRE vs POST merger L2 token-survival capture.

Captures, for every input image (deterministic sorted order), the per merge-unit
PRE and POST L2 saliency scores from a live Qwen3-VL-8B-Instruct forward pass
(vLLM, capture-only, max_tokens=1, enforce_eager, temperature=0), WITHOUT
changing model numerics (hook wrappers call the original forward unchanged).

PRE  score = mean over the 4 patch-feature L2 norms inside each 2x2 merge unit
             (deepstack_merger_list[0] input = ViT block-8 hidden states).
POST score = L2 norm of the concatenated [main, ds0, ds1, ds2] post-merger row
             per unit (visual-tower output, scored on the per-image split).

Scoring helpers (_score_units, _score_tokens) and the no-op capture wrappers
(wrap_capture, scores_from_cap) are IMPORTED from scripts/mechanism_token_survival.py
and src/v3_premerger/v3_premerger_runner.py — NOT reimplemented here. The capture
numerics are cross-checked at runtime against a direct re-derivation from the
imported helpers (see _cross_check_scores).

Outputs per sample (under --output-dir):
  <sample_id>.npz   pre_l2/post_l2 float32[N]; pre_keep/post_keep bool[N];
                    pre_rank/post_rank int32[N]; grid_thw int32[3];
                    unit_grid_hw int32[2].  (NO hidden states stored.)
  <sample_id>.json  full provenance sidecar.
Plus capture_manifest.json summarising all samples.

Determinism: seed 0, enforce_eager, temperature 0, sorted file order, and a
recorded top-k tie-break rule (score descending; ties -> ascending unit index).

CLI:
  python drafts/figures/real_data_pipeline/scripts/capture_real_l2.py \
      --input-dir drafts/figures/real_data_pipeline/inputs \
      --output-dir drafts/figures/real_data_pipeline/data \
      --model-family qwen3vl --model-id Qwen/Qwen3-VL-8B-Instruct \
      --keep-ratio 0.25 --max-pixels 1500000 --seed 0
  # validator re-capture of ONE sample:
  ... --only-sample <sample_id> --output-dir <temp_dir>
  # resume: skip samples whose <sample_id>.npz AND .json sidecar already
  # exist in --output-dir; if ALL exist the model is never loaded and
  # capture_manifest.json is left untouched:
  ... --skip-existing
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

# ---- repo root (4 levels up: drafts/figures/real_data_pipeline/scripts) ---- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "v3_premerger"))

# ---- offline + single-process vLLM env (mirror mechanism_token_survival.py) -- #
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

import numpy as np
import torch

# Import the verified, no-numerics-change capture core (NOT reimplemented).
import mechanism_token_survival as mts          # wrap_capture, scores_from_cap
from v3_premerger_runner import _score_tokens, _score_units   # EXACT selectors

PATCH = mts.PATCH          # 16 px / patch  (verified vision_config.patch_size)
MERGE = mts.MERGE          # 2  (2x2 spatial merge, verified spatial_merge_size)
SUPPORTED_EXT = (".png", ".jpg", ".jpeg")
PROMPT = "Describe this image."        # fixed neutral prompt (capture-only)
TIE_BREAK_RULE = ("sort by score descending; ties broken by ascending unit "
                  "index (stable deterministic top-k); rank 1 = largest score")
PRE_SCORE_DEF = "pre: mean over 4 patch-feature L2 norms per unit"
POST_SCORE_DEF = ("post: L2 norm of concatenated [main,ds0,ds1,ds2] post-merger "
                  "row per unit")
PRE_HOOK_NAME = "visual.deepstack_merger_list[0]"
POST_HOOK_DEF = "cat([main,ds0,ds1,ds2],dim=1) scored per image split"


# --------------------------------------------------------------------------- #
# Small deterministic helpers
# --------------------------------------------------------------------------- #
def discover_samples(input_dir: str):
    """Return sorted list of dicts {sample_id, filename, path}. Sorted by
    filename (codepoint order) => deterministic. Unsupported ext skipped."""
    out = []
    for fn in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in SUPPORTED_EXT:
            continue                                   # skip .gitkeep, others
        stem = os.path.splitext(fn)[0]
        out.append(dict(sample_id=stem, filename=fn,
                        path=os.path.abspath(os.path.join(input_dir, fn))))
    ids = [s["sample_id"] for s in out]
    assert len(ids) == len(set(ids)), f"non-unique sample stems: {ids}"
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolved_revision(model_id: str) -> str:
    """Read the resolved snapshot revision from the HF cache refs/main."""
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    ref = os.path.join(hub, "models--" + model_id.replace("/", "--"),
                       "refs", "main")
    with open(ref) as f:
        return f.read().strip()


def snapshot_dir(model_id: str, revision: str) -> str:
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    return os.path.join(hub, "models--" + model_id.replace("/", "--"),
                        "snapshots", revision)


def processor_class_name(model_id: str, revision: str) -> str:
    pc = os.path.join(snapshot_dir(model_id, revision), "preprocessor_config.json")
    try:
        with open(pc) as f:
            return json.load(f).get("processor_class", "unknown")
    except Exception:
        return "unknown"


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def deterministic_topk(scores: np.ndarray, k: int):
    """Top-k by score DESC, ties -> ascending unit index. Returns
    (keep_mask bool[N], rank int32[N], order int[N]). rank 1 = largest score."""
    n = scores.shape[0]
    # lexsort: LAST key is primary. primary = -score (asc => score desc),
    # secondary = unit index (asc) => exact recorded tie-break.
    order = np.lexsort((np.arange(n), -scores.astype(np.float64)))
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(1, n + 1, dtype=np.int32)
    return keep, rank, order


def compute_k(n_units: int, keep_ratio: float) -> int:
    return max(1, round(n_units * keep_ratio))         # runner/spec contract


def _cross_check_scores(cap, pre_ref: np.ndarray, post_ref: np.ndarray,
                        num_units: int):
    """Independently re-derive PRE/POST scores with the IMPORTED helpers
    (_score_units / _score_tokens) and assert they match scores_from_cap.
    Genuinely exercises the imported helpers (no reimplementation)."""
    hs_ds0 = cap["ds_in"][0]                           # [N,1,1152]
    ctx = hs_ds0.shape[-1]
    feats = hs_ds0.reshape(num_units, MERGE ** 2, ctx)
    pre_chk = _score_units(feats, "l2").numpy().astype(np.float64)
    main = cap["out"]["merger"]
    ds = [cap["out"][f"ds{i}"] for i in range(3)]
    post_feat = torch.cat([main] + ds, dim=1)          # [num_units, 16384]
    post_chk = _score_tokens(post_feat, "l2").numpy().astype(np.float64)
    assert np.array_equal(pre_chk, pre_ref), "PRE score cross-check mismatch"
    assert np.array_equal(post_chk, post_ref), "POST score cross-check mismatch"


# --------------------------------------------------------------------------- #
# Model load (ONCE)
# --------------------------------------------------------------------------- #
def load_llm(model_id: str, max_pixels: int, seed: int):
    from vllm import LLM, SamplingParams
    # Copied from scripts/mechanism_token_survival.py L509-518 (verified flags).
    llm = LLM(model=model_id, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.90, max_model_len=32768,
              trust_remote_code=False, enforce_eager=True,
              limit_mm_per_prompt={"image": 1},
              allowed_local_media_path=ROOT,
              max_num_seqs=4, enable_prefix_caching=False, seed=seed,
              max_num_batched_tokens=32768,
              mm_processor_kwargs={"max_pixels": max_pixels})
    sp = SamplingParams(max_tokens=1, temperature=0.0)   # capture-only forward
    return llm, sp


# --------------------------------------------------------------------------- #
# Core (importable): run_capture(cfg-dict) -> manifest dict
# --------------------------------------------------------------------------- #
def run_capture(cfg: dict) -> dict:
    input_dir = os.path.abspath(cfg["input_dir"])
    output_dir = os.path.abspath(cfg["output_dir"])
    model_id = cfg["model_id"]
    model_family = cfg["model_family"]
    keep_ratio = float(cfg["keep_ratio"])
    max_pixels = int(cfg["max_pixels"])
    seed = int(cfg["seed"])
    only_sample = cfg.get("only_sample")
    attempts = int(cfg.get("attempts", 3))
    os.makedirs(output_dir, exist_ok=True)

    all_samples = discover_samples(input_dir)
    samples = ([s for s in all_samples if s["sample_id"] == only_sample]
               if only_sample else all_samples)
    if only_sample and not samples:
        raise SystemExit(f"--only-sample {only_sample!r} not found in {input_dir}")

    # ---- resume: skip samples whose NPZ + JSON sidecar already exist ---- #
    if cfg.get("skip_existing"):
        pending, skipped = [], []
        for s in samples:
            have = (os.path.exists(os.path.join(output_dir, s["sample_id"] + ".npz"))
                    and os.path.exists(os.path.join(output_dir, s["sample_id"] + ".json")))
            (skipped if have else pending).append(s)
        if skipped:
            print(f"[capture] skip-existing: {len(skipped)}/{len(samples)} "
                  f"sample(s) already have NPZ+JSON in {output_dir}", flush=True)
        samples = pending
    if not samples:
        if not cfg.get("skip_existing"):
            raise SystemExit(f"[capture] no samples to process in {input_dir}")
        print("[capture] nothing to capture; model NOT loaded, "
              "capture_manifest.json left untouched.", flush=True)
        man_path = os.path.join(output_dir, "capture_manifest.json")
        if os.path.exists(man_path):
            with open(man_path) as f:
                return json.load(f)
        return {}

    print(f"[capture] {len(samples)} sample(s) to process "
          f"(input_dir={input_dir}, output_dir={output_dir})", flush=True)

    revision = resolved_revision(model_id)
    pclass = processor_class_name(model_id, revision)
    commit = git_commit()
    import vllm, transformers
    versions = dict(vllm=vllm.__version__, transformers=transformers.__version__,
                    torch=torch.__version__, numpy=np.__version__)

    llm, sp = load_llm(model_id, max_pixels, seed)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    model_class = type(model).__name__
    visual_class = type(model.visual).__name__
    device_name = torch.cuda.get_device_name(0)
    cap = mts.wrap_capture(model.visual)               # no numerics change

    cmd = " ".join(sys.argv)
    summary_samples = []

    for s in samples:
        sid = s["sample_id"]
        mts.reset(cap)                                  # cap rebuilt per sample
        pre = post = None
        for attempt in range(1, attempts + 1):
            mts.reset(cap)
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "file://" + s["path"]}},
                {"type": "text", "text": PROMPT},
            ]}]
            try:
                llm.chat([msgs], sampling_params=sp)
                pre, post, h, w = mts.scores_from_cap(cap)   # float64, patch grid
                break
            except Exception as e:
                print(f"[capture] {sid}: attempt {attempt}/{attempts} failed "
                      f"({type(e).__name__}: {str(e)[:120]})", flush=True)
        if pre is None:
            raise SystemExit(f"[capture] {sid}: capture failed after "
                             f"{attempts} attempts (grid_thw never captured)")

        t = int(cap["grid_thw"][0][0])
        assert t == 1, f"expected t==1 (still image), got {t}"
        n_units = (h * w) // (MERGE ** 2)
        _cross_check_scores(cap, pre, post, n_units)    # uses imported helpers

        k = compute_k(n_units, keep_ratio)
        pre_keep, pre_rank, _ = deterministic_topk(pre, k)
        post_keep, post_rank, _ = deterministic_topk(post, k)

        # ---- NPZ (EXACTLY these keys; NO hidden states) ---- #
        npz_path = os.path.join(output_dir, f"{sid}.npz")
        np.savez(npz_path,
                 pre_l2=pre.astype(np.float32),
                 post_l2=post.astype(np.float32),
                 pre_keep=pre_keep.astype(bool),
                 post_keep=post_keep.astype(bool),
                 pre_rank=pre_rank.astype(np.int32),
                 post_rank=post_rank.astype(np.int32),
                 grid_thw=np.asarray([t, h, w], dtype=np.int32),
                 unit_grid_hw=np.asarray([h // MERGE, w // MERGE], dtype=np.int32))

        # ---- sidecar JSON (provenance; no private abs paths / secrets) ---- #
        sidecar = dict(
            sample_id=sid,
            original_filename=s["filename"],
            sha256_original=sha256_file(s["path"]),
            model_family=model_family,
            model_id=model_id,
            resolved_revision=revision,
            processor_class=pclass,
            model_class=model_class,
            visual_class=visual_class,
            package_versions=versions,
            repo_git_commit=commit,
            dtype="bfloat16",
            device=f"cuda:0 ({device_name})",
            seed=seed,
            max_pixels=max_pixels,
            keep_ratio=keep_ratio,
            k=int(k),
            n_units=int(n_units),
            patch_size=PATCH,
            spatial_merge_size=MERGE,
            processor_resized_image_size_px=dict(h_px=int(h * PATCH),
                                                 w_px=int(w * PATCH)),
            grid_thw=[t, int(h), int(w)],
            unit_grid_hw=[int(h // MERGE), int(w // MERGE)],
            pre_hook_module=PRE_HOOK_NAME,
            post_hook_definition=POST_HOOK_DEF,
            score_definitions=dict(pre=PRE_SCORE_DEF, post=POST_SCORE_DEF),
            tie_break_rule=TIE_BREAK_RULE,
            scoring_helpers="_score_units/_score_tokens imported from "
                           "src/v3_premerger/v3_premerger_runner.py; "
                           "wrap_capture/scores_from_cap imported from "
                           "scripts/mechanism_token_survival.py (cross-checked)",
            hooks_no_numeric_change="capture wrappers call the original merger "
                                    "forward (_orig) unchanged and return its "
                                    "output; cap is reset+rebuilt per sample and "
                                    "never persisted (only derived float32 arrays)",
            capture_prompt=PROMPT,
            max_generated_tokens=1,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
                            .isoformat(timespec="seconds"),
            capture_command=cmd,
        )
        with open(os.path.join(output_dir, f"{sid}.json"), "w") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)

        jacc = float(np.logical_and(pre_keep, post_keep).sum()
                     / np.logical_or(pre_keep, post_keep).sum())
        summary_samples.append(dict(
            sample_id=sid, original_filename=s["filename"], n_units=int(n_units),
            k=int(k), grid_thw=[t, int(h), int(w)],
            unit_grid_hw=[int(h // MERGE), int(w // MERGE)],
            pre_post_topk_jaccard=round(jacc, 4)))
        print(f"[capture] {sid}: N={n_units} k={k} grid=({h},{w}) "
              f"jaccard(pre_keep,post_keep)={jacc:.3f}", flush=True)

    manifest = dict(
        pipeline="real_data_l2_capture",
        input_dir=cfg["input_dir"],          # as passed (relative => no leak)
        capture_script=os.path.relpath(os.path.abspath(__file__), ROOT),
        model_family=model_family, model_id=model_id,
        resolved_revision=revision,
        processor_class=pclass, model_class=model_class, visual_class=visual_class,
        package_versions=versions, repo_git_commit=commit,
        dtype="bfloat16", device=f"cuda:0 ({device_name})",
        seed=seed, max_pixels=max_pixels, keep_ratio=keep_ratio,
        patch_size=PATCH, spatial_merge_size=MERGE,
        pre_hook_module=PRE_HOOK_NAME, post_hook_definition=POST_HOOK_DEF,
        score_definitions=dict(pre=PRE_SCORE_DEF, post=POST_SCORE_DEF),
        tie_break_rule=TIE_BREAK_RULE, capture_prompt=PROMPT,
        k_formula="max(1, round(N*keep_ratio))",
        n_samples=len(summary_samples),
        samples=summary_samples,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec="seconds"),
        capture_command=cmd,
    )
    with open(os.path.join(output_dir, "capture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[capture] done -> {output_dir} "
          f"({len(summary_samples)} samples)", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-family", default="qwen3vl")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--keep-ratio", type=float, default=0.25)
    ap.add_argument("--max-pixels", type=int, default=1500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-sample", default=None,
                    help="process only this sample_id (validator re-capture)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: skip samples whose <sample_id>.npz AND .json "
                         "sidecar already exist in --output-dir; if everything "
                         "exists the model is never loaded and the manifest is "
                         "left untouched (plotting-only reruns)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="max forward attempts per image (grid_thw occasionally "
                         "missing on a no-op vision pass; a retry fixes it)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_capture(vars(args))


if __name__ == "__main__":
    main()
