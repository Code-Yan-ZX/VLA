#!/usr/bin/env python
"""J5 — held-out dev λ selection for query-aware pre-merger saliency (QA-pre).

Spec: notes/j5_qa_gate_design.md. Pre-registered single query-aware attempt
(DECISIONS 2026-07-23). This script picks λ on a dev slice DISJOINT from the
n=200 gate subsets, so the gate itself stays held-out.

Steps
-----
1. (CPU, pure) Build a dev slice = eval/full_splits/{textvqa_val,docvqa_val}.jsonl
   rows whose id is NOT in eval/subsets/{textvqa,docvqa}_200.jsonl, 32 each
   (random.Random(0) shuffle, take 32). Writes the dev jsonl (same schema) and
   runs/v3_merger_aware/j5/dev_ids.json (audit).
2. (GPU, shells out to the runner) Run QA-pre at λ∈{0,0.3,0.5,0.7} (mode=pre,
   r=0.75, selector=l2) on textvqa+docvqa dev (n=32 each) = 8 cells.
3. (CPU) Score each cell with the OFFICIAL metric (textvqa VQA-acc, docvqa
   ANLS via official_scorers), compute mean(2 benchmarks) per λ, gain over
   plain-pre (λ=0); pick the λ with the largest mean gain among {0.3,0.5,0.7};
   if all three gains ≤ 0 -> λ=0. Writes runs/v3_merger_aware/j5/
   lambda_selection.json.

Idempotent: existing cell JSONs are skipped; re-running reuses them.

Usage
-----
  /home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/j5_qa_dev_select.py
  .../python scripts/j5_qa_dev_select.py --build-only   # CPU slice build only
  .../python scripts/j5_qa_dev_select.py --score-only   # skip GPU, score existing
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PY = sys.executable                                  # the conda env's python
RUNNER = os.path.join(REPO, "src/v3_premerger/v3_premerger_runner.py")
sys.path.insert(0, os.path.join(REPO, "src/v3_premerger"))
import official_scorers as S                          # noqa: E402

OUT = os.path.join(REPO, "runs/v3_merger_aware/j5")
FULL = os.path.join(REPO, "eval/full_splits")
SUB = os.path.join(REPO, "eval/subsets")

DEV_N = 32
SEED = 0
R = 0.75
LAMBDAS = [0.0, 0.3, 0.5, 0.7]
BENCHES = ["textvqa", "docvqa"]
FULL_SPLITS = {"textvqa": "textvqa_val.jsonl", "docvqa": "docvqa_val.jsonl"}
# per-benchmark vLLM flags (mirror j2/j3 gate scripts: docvqa needs the large
# model-len + batched-token budget to avoid the encoder-cache crash).
GMU = "0.9"
FLAGS = {
    "textvqa": ["--max-num-seqs", "8", "--max-model-len", "8192",
                "--gpu-memory-utilization", GMU],
    "docvqa": ["--max-num-seqs", "4", "--max-model-len", "32768",
               "--max-num-batched-tokens", "32768",
               "--gpu-memory-utilization", GMU],
}


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dev_slice():
    """CPU-only: build disjoint dev slices (32 each) + audit id list."""
    os.makedirs(OUT, exist_ok=True)
    dev_ids = {}
    dev_paths = {}
    for b in BENCHES:
        full = _read_jsonl(os.path.join(FULL, FULL_SPLITS[b]))
        gate_ids = set(str(o["id"]) for o in
                       _read_jsonl(os.path.join(SUB, f"{b}_200.jsonl")))
        disjoint = [o for o in full if str(o["id"]) not in gate_ids]
        rng = random.Random(f"{SEED}-{b}")
        rng.shuffle(disjoint)
        chosen = disjoint[:DEV_N]
        dev_paths[b] = os.path.join(OUT, f"dev_{b}.jsonl")
        with open(dev_paths[b], "w") as f:
            for o in chosen:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        dev_ids[b] = [str(o["id"]) for o in chosen]
        print(f"[j5-dev] {b}: {len(full)} full, {len(gate_ids)} gate-excluded, "
              f"{len(disjoint)} disjoint -> {len(chosen)} dev -> {dev_paths[b]}")
    with open(os.path.join(OUT, "dev_ids.json"), "w") as f:
        json.dump({"seed": SEED, "n_per_bench": DEV_N, "r": R,
                   "lamdas": LAMBDAS, "ids": dev_ids}, f, indent=2,
                  ensure_ascii=False)
    print(f"[j5-dev] wrote {os.path.join(OUT, 'dev_ids.json')} (audit)")
    return dev_paths


def wait_for_gpu(free_mib=30000, tries=360, sleep_s=60):
    """GPU etiquette (same as j2/j3): wait for >= free_mib free, up to tries."""
    print(f"[j5-dev] waiting for >= {free_mib} MiB free GPU ...", flush=True)
    for _ in range(tries):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout
            free = int(out.strip().splitlines()[0].strip())
            if free > free_mib:
                print(f"[j5-dev] GPU free {free} MiB", flush=True)
                return True
        except Exception as e:
            print(f"[j5-dev] nvidia-smi failed ({e}); retrying", flush=True)
        time.sleep(sleep_s)
    print("[j5-dev][ABORT] GPU busy after wait window", flush=True)
    return False


def cell_tag(b, lam):
    return f"dev_{b}_pre_r{R:.3f}_qa{lam:.2f}_n{DEV_N}"


def run_cell(b, lam, dev_jsonl):
    """Run one QA-pre cell on the dev slice (GPU). Idempotent (skip if exists)."""
    tag = cell_tag(b, lam)
    out_json = os.path.join(OUT, tag + ".json")
    if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
        print(f"[j5-dev] skip (exists): {tag}")
        return out_json
    cmd = [PY, RUNNER, "--model-family", "qwen3vl", "--mode", "pre",
           "--r", str(R), "--benchmark", b, "--subset", dev_jsonl,
           "--n", str(DEV_N), "--selector", "l2",
           "--qa-lambda", str(lam), "--qa-embed-cache",
           "--out", out_json] + FLAGS[b]
    print(f"[j5-dev] RUN {tag}\n         {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[j5-dev] {tag} rc={rc} wall={time.perf_counter()-t0:.0f}s",
          flush=True)
    return out_json


def official_metric(b, out_json):
    """Mean official metric for a runner output JSON (None if unusable)."""
    if not (os.path.exists(out_json) and os.path.getsize(out_json) > 0):
        return None
    try:
        d = json.load(open(out_json))
    except Exception as e:
        print(f"[j5-dev] cannot read {out_json}: {e}")
        return None
    ps = d.get("per_sample") or []
    if not ps:
        return None
    preds = [str(p.get("answer", "")) for p in ps]
    gts = [str(p.get("gt", "")) for p in ps]
    n = len(ps)
    if b == "textvqa":
        return sum(S.score_textvqa_vqaacc(a, g) for a, g in zip(preds, gts)) / n
    if b == "docvqa":
        return sum(S.score_docvqa_anls(a, g) for a, g in zip(preds, gts)) / n
    return None


def select_lambda():
    """Score all cells, pick λ by mean official gain over plain-pre (λ=0)."""
    metrics = {}                                        # lam -> {bench: acc}
    for lam in LAMBDAS:
        metrics[lam] = {}
        for b in BENCHES:
            metrics[lam][b] = official_metric(b, os.path.join(
                OUT, cell_tag(b, lam) + ".json"))
    means = {}
    for lam in LAMBDAS:
        vals = [metrics[lam][b] for b in BENCHES if metrics[lam][b] is not None]
        means[lam] = (sum(vals) / len(vals)) if vals else None
    base = means.get(0.0)
    gains = {}
    for lam in [0.3, 0.5, 0.7]:
        if means.get(lam) is not None and base is not None:
            gains[lam] = means[lam] - base
        else:
            gains[lam] = None
    valid = {l: g for l, g in gains.items() if g is not None}
    if valid and max(valid.values()) > 0:
        selected = max(valid, key=valid.get)
    else:
        selected = 0.0                                  # all gains <= 0 (or no data)
    result = {
        "r": R, "dev_n": DEV_N, "benches": BENCHES, "lambdas": LAMBDAS,
        "metrics": {str(l): metrics[l] for l in LAMBDAS},
        "means": {str(l): means[l] for l in LAMBDAS},
        "base_lambda0_mean": base,
        "gains_over_lambda0": {str(l): gains[l] for l in [0.3, 0.5, 0.7]},
        "selected_lambda": selected,
        "rule": ("argmax mean official gain over λ=0 among {0.3,0.5,0.7}; "
                 "all gains <=0 -> 0.0"),
    }
    out = os.path.join(OUT, "lambda_selection.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[j5-dev] means={ {str(l): (round(means[l],4) if means[l] is not None else None) for l in LAMBDAS} }")
    print(f"[j5-dev] gains_over_λ0={ {str(l): (round(g,4) if g is not None else None) for l,g in gains.items()} }")
    print(f"[j5-dev] >>> SELECTED λ = {selected}  ->  {out}")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true",
                    help="CPU: build dev slice + dev_ids.json, then exit.")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip GPU: score existing cell JSONs + select λ.")
    ap.add_argument("--no-gpu-wait", action="store_true",
                    help="Skip the GPU-free wait (use only if GPU is reserved).")
    args = ap.parse_args()

    dev_paths = build_dev_slice()
    if args.build_only:
        return

    if not args.score_only:
        if not args.no_gpu_wait and not wait_for_gpu():
            sys.exit(1)
        for lam in LAMBDAS:
            for b in BENCHES:
                run_cell(b, lam, dev_paths[b])

    select_lambda()


if __name__ == "__main__":
    main()
