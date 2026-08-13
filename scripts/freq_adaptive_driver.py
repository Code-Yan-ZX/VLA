#!/usr/bin/env python3
"""Direction A/B orchestrator: frequency-aware scorer (φ=α·L2+β·var) + adaptive
stage router (workload detector -> PRE/POST per image). Training-free RBM
upgrades on Qwen3-VL (qwen3vl family).

Phases (each resumable; cells skipped when the output JSON already exists):
  grid_freq   : {0.5,1.0,2.0} x {0.5,1.0,2.0} on TextVQA-500 @ r=0.75 (PRE path)
  grid_adapt  : tau_hf x tau_ent on TextVQA-500 @ r=0.75 (mode=adaptive)
  eval_freq   : best (alpha,beta) on 4 benches @ {0.75,0.5} vs pre/post
  eval_adapt  : best (tau_hf,tau_ent) on 4 benches @ {0.75,0.5} vs pre/post
  eval_combined: freq-best scorer + adaptive router on 4 benches + ChartQA
                 @ {0.75,0.5} vs RBM/FastV (+ regime map table)

Config lives in one dict at the bottom (grid sets, subset files, n, r list).
Run:  python scripts/freq_adaptive_driver.py --phase grid_freq [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/media/disk2/YZX/research/vla")
PYQ3 = "/home/dell/miniconda3/envs/qwen3vl_clean/bin/python"
RUNNER = REPO / "src/v3_premerger/v3_premerger_runner.py"
OUT = REPO / "experiments"

# Benchmark -> (input jsonl, n, extra flags)
# FULL-SPLIT jsonl + n=500 slice (first N samples) -- the SAME files + n that
# the j7hf baselines used. CRITICAL: the subset jsonls (eval/subsets/*) DROP
# the "Answer the question using a single word or phrase." instruction suffix,
# which makes Qwen3-VL emit verbose answers -> official exact-match scores 0.
# The full-split jsonls keep the suffix, so answers are terse and comparable.
BENCH = {
    "textvqa":  ("eval/full_splits/textvqa_val.jsonl", 500, ["--max-num-seqs", "8", "--max-model-len", "8192"]),
    "docvqa":   ("eval/full_splits/docvqa_val.jsonl", 500, ["--max-num-seqs", "4", "--max-model-len", "32768",
                                                             "--max-num-batched-tokens", "32768", "--max-pixels", "4000000"]),
    "ocrbench": ("eval/full_splits/ocrbench.jsonl", 500, ["--max-num-seqs", "8", "--max-model-len", "8192",
                                                          "--max-pixels", "4000000"]),
    "gqa":      ("eval/full_splits/gqa_testdev.jsonl", 500, ["--max-num-seqs", "8", "--max-model-len", "8192"]),
    "chartqa":  ("eval/subsets/chartqa_200.jsonl", 200, ["--max-num-seqs", "8", "--max-model-len", "8192"]),
}
STD = ["--model-family", "qwen3vl", "--gpu-memory-utilization", "0.9", "--seed", "0"]
ALPHA_GRID = [0.5, 1.0, 2.0]
BETA_GRID = [0.5, 1.0, 2.0]
TAU_HF_GRID = [0.08, 0.13, 0.20]
TAU_ENT_GRID = [2.0, 2.5, 3.0]
R_LIST = [0.75, 0.5]
BENCHES_ALL = ["textvqa", "docvqa", "ocrbench", "gqa", "chartqa"]
BENCHES_CORE = ["textvqa", "docvqa", "ocrbench", "gqa"]

MAX_LOAD_TRIES = 60          # wait for GPU free (1 min each)
FREE_THRESHOLD_MIB = 41000


def log(msg: str):
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
        line = out.strip().splitlines()[0]
        return int(line.split()[0])
    except Exception:
        return -1


def wait_gpu():
    for _ in range(MAX_LOAD_TRIES):
        free = gpu_free_mib()
        if free >= FREE_THRESHOLD_MIB:
            log(f"GPU free {free} MiB; proceeding")
            return
        log(f"GPU busy ({free} MiB free); sleeping 60 s")
        time.sleep(60)
    raise SystemExit(f"[driver] GPU never freed after {MAX_LOAD_TRIES} min")


def run_cell(mode: str, bench: str, r: float, tag: str,
             selector: str = "l2", alpha: float = 1.0, beta: float = 1.0,
             tau_hf: float = 0.3, tau_ent: float = 2.0,
             hf_var_mode: str = "mean1sd",
             force: bool = False) -> str | None:
    """Run one runner cell; returns path to the output JSON (or None on
    failure). Concurrent-cell safety: a lockfile holds the GPU."""
    subset, n, extra = BENCH[bench]
    out_file = OUT / "freq_aware" / f"{tag}.json"
    if not force and out_file.exists() and out_file.stat().st_size > 100:
        log(f"skip {tag} (exists)")
        return str(out_file)
    os.makedirs(out_file.parent, exist_ok=True)
    cmd = [PYQ3, str(RUNNER), *STD, "--mode", mode, "--benchmark", bench,
           "--subset", str(REPO / subset), "--n", str(n), "--r", str(r),
           "--selector", selector,
           "--alpha", str(alpha), "--beta", str(beta),
           "--tau-hf", str(tau_hf), "--tau-ent", str(tau_ent),
           "--hf-var-mode", hf_var_mode,
           *extra, "--out", str(out_file)]
    wait_gpu()
    log(f"RUN {tag}: {mode} {bench} r={r} sel={selector} "
        f"a={alpha} b={beta} t_hf={tau_hf} t_ent={tau_ent} "
        f"var_mode={hf_var_mode} -> {out_file.name}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    log_fmt = " ".join(cmd) + f"\nrc={proc.returncode} wall={wall/60:.1f}min"
    if proc.returncode != 0:
        # vLLM teardown race: the PREVIOUS cell's process can still hold ~5GiB
        # as it exits (free dips marginal-below GMU0.9's 40GiB requirement).
        # Wait for the GPU to settle + retry (self-healing; 2 extra tries).
        log(f"FAIL {tag} rc={proc.returncode} (stderr tail below); retrying\n"
            f"{proc.stderr[-1200:]}")
        for _try in range(2):
            log(f"retry {tag} #{_try+1}; waiting 90 s for GPU settle")
            time.sleep(90)
            wait_gpu()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            if proc.returncode == 0:
                break
            log(f"FAIL(retry {_try+1}) {tag} rc={proc.returncode}\n"
                f"{proc.stderr[-800:]}")
        if proc.returncode != 0:
            log(f"GIVE UP {tag} after retries")
            return None
    try:
        d = json.load(open(out_file))
        log(f"OK   {tag} acc={d.get('acc')} n={d.get('n')} "
            f"skip={d.get('n_skipped')} ptid={d.get('mean_ptid_len')} "
            f"(wall {wall/60:.1f}min)")
        return str(out_file)
    except Exception as e:
        log(f"WARN {tag} json unreadable after rc=0 ({e})")
        return str(out_file)


def acc_of(path: str) -> float | None:
    """OFFICIAL rescore (paper protocol, mirrors j7_main_table / the runner's
    official_rescore helper): the runner's inline 'acc' uses loose per-bench
    matching, which is NOT the peer-reviewed metric. Selection must be on the
    official metric so the grid winner survives the final rescore."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO / "src/v3_premerger"))
        import official_scorers as _S
        import v3_premerger_runner as _R
        d = json.load(open(path))
        ps = d.get("per_sample") or []
        n = len(ps)
        if not n:
            return None
        bench = d.get("benchmark")
        preds = [str(p.get("answer", "")) for p in ps]
        gts = [str(p.get("gt", "")) for p in ps]
        if bench == "textvqa":
            return sum(_S.score_textvqa_vqaacc(a, g)
                       for a, g in zip(preds, gts)) / n
        if bench == "docvqa":
            return sum(_S.score_docvqa_anls(a, g)
                       for a, g in zip(preds, gts)) / n
        if bench == "gqa":
            return _S.score_gqa_batch(preds, gts)["acc"]
        if bench == "ocrbench":
            meta = {}
            for src in [REPO / "eval/full_splits/ocrbench.jsonl",
                        REPO / "eval/subsets/ocrbench_200.jsonl"]:
                if src.exists():
                    for line in open(src):
                        o = json.loads(line)
                        ex = o.get("extras") or {}
                        meta.setdefault(str(o["id"]),
                                        (ex.get("question_type", ""),
                                         ex.get("category")))
            items = [(a, g) + meta.get(str(p.get("id")), ("", ""))
                     for a, g, p in zip(preds, gts, ps)]
            return _S.score_ocrbench_batch(items)["acc"]
        if bench == "chartqa":
            return sum(_R.score_chartqa(a, g)
                       for a, g in zip(preds, gts)) / n
        return d.get("acc")
    except Exception:
        return None


def phase_grid_freq(dry: bool = False):
    tag_of = lambda a, b: f"q3_freq_a{a:g}_b{b:g}_textvqa_r0.75"
    results = {}
    # plain-RBM reference on the SAME subset (selector=l2 == freq alpha=1,beta=0
    # under z-normalization, run natively so the grid has a real baseline).
    if not dry:
        p = run_cell("pre", "textvqa", 0.75, "q3_freq_ref_l2_textvqa_r0.75",
                     selector="l2")
        if p:
            results["ref:l2"] = {"acc": acc_of(p), "file": p}
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            tag = tag_of(alpha, beta)
            if dry:
                print(f"[dry] would run {tag}")
                continue
            p = run_cell("pre", "textvqa", 0.75, tag,
                         selector="freq", alpha=alpha, beta=beta)
            if p:
                results[f"{alpha:g}x{beta:g}"] = {"acc": acc_of(p), "file": p}
    if not dry:
        with open(OUT / "freq_aware" / "grid.json", "w") as f:
            json.dump({"grid": results, "alpha_grid": ALPHA_GRID,
                       "beta_grid": BETA_GRID}, f, indent=1, ensure_ascii=False)
        log(f"grid_freq done -> {OUT/'freq_aware'/'grid.json'}")


def phase_grid_adapt(dry: bool = False):
    tag_of = lambda t1, t2: f"q3_adapt_thf{t1:g}_tent{t2:g}_textvqa_r0.75"
    results = {}
    for t1 in TAU_HF_GRID:
        for t2 in TAU_ENT_GRID:
            tag = tag_of(t1, t2)
            if dry:
                print(f"[dry] would run {tag}")
                continue
            p = run_cell("adaptive", "textvqa", 0.75, tag,
                         selector="l2", tau_hf=t1, tau_ent=t2)
            if p:
                results[f"{t1:g}x{t2:g}"] = {"acc": acc_of(p), "file": p}
    if not dry:
        with open(OUT / "adaptive_stage" / "grid.json", "w") as f:
            json.dump({"grid": results, "tau_hf_grid": TAU_HF_GRID,
                       "tau_ent_grid": TAU_ENT_GRID}, f, indent=1)
        log(f"grid_adapt done -> {OUT/'adaptive_stage'/'grid.json'}")


def best_from(grid_json: Path):
    """Return (key, alpha, beta, acc) of the best cell, reading alpha/beta from
    the cell FILE (robust to arbitrary grid key names like 'refined_1.0x0.6')."""
    try:
        g = json.load(open(grid_json))["grid"]
        ok = {k: v for k, v in g.items() if v.get("acc") is not None}
        if not ok:
            return (None, None, None, None)
        key = max(ok, key=lambda k: ok[k]["acc"])
        v = ok[key]
        if isinstance(v.get("file"), str) and os.path.exists(v["file"]):
            d = json.load(open(v["file"]))
            # adaptive cells run with default alpha=beta=1.0 AND tau_hf/tau_ent;
            # prefer the TAU params for adaptive-mode cells.
            a = b = None
            if d.get("mode") == "adaptive":
                a, b = d.get("tau_hf"), d.get("tau_ent")
            else:
                a, b = d.get("alpha"), d.get("beta")
            if a is not None and b is not None:
                return (key, a, b, v["acc"])
        # fallback: parse a "axb" key
        import re as _re
        m = _re.search(r"([\d.]+)x([\d.]+)", key)
        if m:
            return (key, float(m.group(1)), float(m.group(2)), v["acc"])
        return (key, 1.0, 1.0, v["acc"])
    except Exception:
        return (None, None, None, None)


def phase_eval_freq(dry: bool = False):
    """Best (alpha,beta) across the 4 core benches @ {0.75,0.5}, plus the
    plain pre (RBM) and post (FastV) reference cells for contrast."""
    (OUT / "freq_aware").mkdir(parents=True, exist_ok=True)
    best, alpha, beta, _acc = best_from(OUT / "freq_aware" / "grid.json")
    if best is None and not dry:
        log("freq grid has no valid cell -> skipping eval_freq")
        return
    alpha = float(alpha) if alpha is not None else 1.0
    beta = float(beta) if beta is not None else 1.0
    log(f"freq best (alpha,beta) = ({alpha}, {beta})")
    summary = {"best": best, "alpha": alpha, "beta": beta, "cells": {}}
    for bench in BENCHES_CORE:
        for r in R_LIST:
            tag = f"q3_freq_a{alpha:g}_b{beta:g}_{bench}_r{r:g}"
            tags = {m: f"q3_{m}_{bench}_r{r:g}" for m in ("pre", "post")}
            for name, t in [("freq", tag), *tags.items()]:
                if dry:
                    continue
                mode = "pre" if name == "pre" else (
                    "post" if name == "post" else "pre")
                sel = "l2" if name in ("pre", "post") else "freq"
                a = 1.0 if name != "freq" else alpha
                b = 1.0 if name != "freq" else beta
                p = run_cell(mode, bench, r, t, selector=sel, alpha=a, beta=b)
                if p:
                    summary["cells"].setdefault(f"{bench}@{r:g}", {})[name] = \
                        {"acc": acc_of(p), "file": p}
    if not dry:
        with open(OUT / "freq_aware" / "results.json", "w") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        log(f"eval_freq done -> {OUT/'freq_aware'/'results.json'}")


def phase_eval_adapt(dry: bool = False):
    (OUT / "adaptive_stage").mkdir(parents=True, exist_ok=True)
    best = best_from(OUT / "adaptive_stage" / "grid.json")
    if best[0] is None and not dry:
        log("adapt grid has no valid cell -> skipping eval_adapt")
        return
    t1, t2 = (0.3, 2.0) if best[0] is None else (best[1], best[2])
    log(f"adapt best (tau_hf, tau_ent) = ({t1}, {t2})")
    summary = {"best": best, "tau_hf": t1, "tau_ent": t2, "cells": {}}
    for bench in BENCHES_CORE:
        for r in R_LIST:
            tag = f"q3_adapt_thf{t1:g}_tent{t2:g}_{bench}_r{r:g}"
            tags = {m: f"q3_{m}_{bench}_r{r:g}" for m in ("pre", "post")}
            for name, t in [("adapt", tag), *tags.items()]:
                if dry:
                    continue
                mode = "adaptive" if name == "adapt" else ("pre" if name == "pre"
                                                           else "post")
                sel = "l2"
                p = run_cell(mode, bench, r, t, selector=sel,
                             tau_hf=t1, tau_ent=t2)
                if p:
                    summary["cells"].setdefault(f"{bench}@{r:g}", {})[name] = \
                        {"acc": acc_of(p), "file": p}
    if not dry:
        with open(OUT / "adaptive_stage" / "results.json", "w") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        log(f"eval_adapt done -> {OUT/'adaptive_stage'/'results.json'}")


def phase_eval_combined(dry: bool = False):
    """Combined: freq-best scorer + adaptive router on 4 benches + ChartQA
    @ {0.75,0.5}, vs RBM(pre) and FastV(post). Writes regime map markdown."""
    (OUT / "combined").mkdir(parents=True, exist_ok=True)
    fb = best_from(OUT / "freq_aware" / "grid.json")
    ab = best_from(OUT / "adaptive_stage" / "grid.json")
    alpha = float(fb[1]) if fb[0] is not None else 1.0
    beta = float(fb[2]) if fb[0] is not None else 1.0
    t1 = float(ab[1]) if ab[0] is not None else 0.3
    t2 = float(ab[2]) if ab[0] is not None else 2.0
    log(f"combined: freq=({alpha},{beta}), adapt=({t1},{t2})")
    summary = {"freq_best": fb, "adapt_best": ab,
               "alpha": alpha, "beta": beta, "tau_hf": t1, "tau_ent": t2,
               "cells": {}}
    for bench in BENCHES_ALL:
        for r in R_LIST:
            ctag = f"q3_combined_a{alpha:g}_b{beta:g}_t{t1:g},{t2:g}_{bench}_r{r:g}"
            tags = {"combined": ctag,
                    "rbm": f"q3_pre_{bench}_r{r:g}",
                    "fastv": f"q3_post_{bench}_r{r:g}"}
            for name, t in tags.items():
                if dry:
                    continue
                if name == "combined":
                    mode, sel, a, b, thf, tent = \
                        "adaptive", "freq", alpha, beta, t1, t2
                elif name == "rbm":
                    mode, sel, a, b, thf, tent = \
                        "pre", "l2", 1.0, 1.0, t1, t2
                else:
                    mode, sel, a, b, thf, tent = \
                        "post", "l2", 1.0, 1.0, t1, t2
                p = run_cell(mode, bench, r, t, selector=sel, alpha=a, beta=b,
                             tau_hf=thf, tau_ent=tent)
                if p:
                    summary["cells"].setdefault(f"{bench}@{r:g}", {})[name] = \
                        {"acc": acc_of(p), "file": p}
    if not dry:
        with open(OUT / "combined" / "results.json", "w") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        log(f"eval_combined done -> {OUT/'combined'/'results.json'}")


PHASES = {
    "grid_freq": phase_grid_freq,
    "grid_adapt": phase_grid_adapt,
    "eval_freq": phase_eval_freq,
    "eval_adapt": phase_eval_adapt,
    "eval_combined": phase_eval_combined,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    PHASES[args.phase](args.dry_run)


if __name__ == "__main__":
    main()