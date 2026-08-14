#!/usr/bin/env python3
"""S6 stitching driver: 2026-PAPER-MODULE x RBM probes (goal = beat the
per-benchmark incumbent = max(RBM pre, FastV post) at iso token budget).

Module stitched now: PRUNESID-style importance + diversity selection
(--diversity nms on the RBM pre path; counts unchanged -> placeholder
contract untouched). Phases (each resumable; cells skipped when the output
JSON already exists):

  ref            : RBM pre + FastV post @ r in {0.25, 0.5} x {textvqa, docvqa,
                  ocrbench, gqa} n=200 dev -- same-slice incumbents.
  grid           : --diversity nms over div_tau x div_gamma (6 cells)
                  @ r=0.25 x 4 benches.
  retest         : best (tau,gamma) also at r=0.5 + chartqa (soft-budget cell).
  verify         : winner at n=500 x {0.25, 0.5} x 4 benches (+ chartqa).

Selection metric = OFFICIAL rescore (mirrors j7_main_table), so grid winners
survive the final rescore. Run: python scripts/sota_stitch_driver.py --phase <p>
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

# Benchmark -> (input jsonl, n, extra flags). Dev = n=200; verify uses n=500.
# Full-split jsonls (keep the terse-answer instruction suffix -- the subset
# jsonls drop it and break official exact-match scoring; see freq_adaptive_driver).
BENCH = {
    "textvqa": ("eval/full_splits/textvqa_val.jsonl", 200,
                ["--max-num-seqs", "8", "--max-model-len", "8192"]),
    "docvqa": ("eval/full_splits/docvqa_val.jsonl", 200,
               ["--max-num-seqs", "4", "--max-model-len", "32768",
                "--max-num-batched-tokens", "32768", "--max-pixels", "4000000"]),
    "ocrbench": ("eval/full_splits/ocrbench.jsonl", 200,
                 ["--max-num-seqs", "8", "--max-model-len", "8192",
                  "--max-pixels", "4000000"]),
    "gqa": ("eval/full_splits/gqa_testdev.jsonl", 200,
            ["--max-num-seqs", "8", "--max-model-len", "8192"]),
    "chartqa": ("eval/subsets/chartqa_200.jsonl", 200,
                ["--max-num-seqs", "8", "--max-model-len", "8192"]),
}
STD = ["--model-family", "qwen3vl", "--gpu-memory-utilization", "0.9",
       "--seed", "0"]
BENCHES_CORE = ["textvqa", "docvqa", "ocrbench", "gqa"]

# Diversity grid: div_tau (NMS cosine suppression; lower = more spread) x
# div_gamma (candidate-pool factor; 1.0 = pure importance, no diversity slack).
GRID_DIV = [(t, g) for t in (0.6, 0.75, 0.9) for g in (1.25, 2.0)]

MAX_LOAD_TRIES = 60
FREE_THRESHOLD_MIB = 41000


def log(msg: str):
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0].split()[0])
    except Exception:
        return -1


def wait_gpu():
    for _ in range(MAX_LOAD_TRIES):
        if gpu_free_mib() >= FREE_THRESHOLD_MIB:
            log(f"GPU free {gpu_free_mib()} MiB; proceeding")
            return
        log(f"GPU busy ({gpu_free_mib()} MiB free); sleeping 60 s")
        time.sleep(60)
    raise SystemExit(f"[driver] GPU never freed after {MAX_LOAD_TRIES} min")


def run_cell(mode: str, bench: str, r: float, tag: str, n: int = 0,
             selector: str = "l2", diversity: str = "none",
             div_tau: float = 0.75, div_gamma: float = 1.5,
             force: bool = False) -> str | None:
    """Run one runner cell; returns the output JSON path (or None on failure).
    mode ∈ {pre, post}; diversity ∈ {none (incumbent RBM), nms (stitch probe)}."""
    subset, _, extra = BENCH[bench]
    n_override = n if n else BENCH[bench][1]
    out_file = OUT / "sota_stitch" / f"{tag}.json"
    if not force and out_file.exists() and out_file.stat().st_size > 100:
        log(f"skip {tag} (exists)")
        return str(out_file)
    os.makedirs(out_file.parent, exist_ok=True)
    cmd = [PYQ3, str(RUNNER), *STD, "--mode", mode, "--benchmark", bench,
           "--subset", str(REPO / subset), "--n", str(n_override), "--r", str(r),
           "--selector", selector,
           "--diversity", diversity]
    if diversity == "nms":
        cmd += ["--div-tau", str(div_tau), "--div-gamma", str(div_gamma)]
    cmd += [*extra, "--out", str(out_file)]
    wait_gpu()
    log(f"RUN {tag}: {mode} {bench} r={r} n={n_override} tau={div_tau} "
        f"gam={div_gamma} -> {out_file.name}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        log(f"FAIL {tag} rc={proc.returncode} (stderr tail below); retrying\n"
            f"{proc.stderr[-1200:]}")
        for _try in range(2):
            time.sleep(90)
            wait_gpu()
            proc = subprocess.run(cmd, capture_output=True, text=True)
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
            f"skip={d.get('n_skipped')} (wall {wall/60:.1f}min)")
        return str(out_file)
    except Exception as e:
        log(f"WARN {tag} json unreadable after rc=0 ({e})")
        return str(out_file)


def acc_of(path: str) -> float | None:
    """OFFICIAL rescore (paper protocol): selection must be on the official
    metric so grid winners survive the final rescore."""
    try:
        sys.path.insert(0, str(REPO / "src/v3_premerger"))
        import official_scorers as S  # noqa: PLC0415
        d = json.load(open(path))
        ps = d.get("per_sample") or []
        n = len(ps)
        if not n:
            return None
        bench = d.get("benchmark")
        preds = [str(p.get("answer", "")) for p in ps]
        gts = [str(p.get("gt", "")) for p in ps]
        if bench == "textvqa":
            return sum(S.score_textvqa_vqaacc(a, g) for a, g in zip(preds, gts)) / n
        if bench == "docvqa":
            return sum(S.score_docvqa_anls(a, g) for a, g in zip(preds, gts)) / n
        if bench == "gqa":
            return S.score_gqa_batch(preds, gts)["acc"]
        if bench == "ocrbench":
            return sum(S.score_ocrbench(a, g) for a, g in zip(preds, gts)) / n
        if bench == "chartqa":
            return sum(S.score_chartqa(a, g) for a, g in zip(preds, gts)) / n
    except Exception as e:
        log(f"WARN acc_of {Path(path).name}: {type(e).__name__} {e}")
    return None


def _tag(mode: str, bench: str, r: float, **kw) -> str:
    base = f"q3_{mode}_{bench}_r{r}"
    if kw.get("div_tau") is not None and kw.get("div_gamma") is not None:
        base += f"_tau{kw['div_tau']}_gam{kw['div_gamma']}"
    return base


def phase_ref():
    for r in (0.25, 0.5):
        for bench in BENCHES_CORE:
            for mode in ("pre", "post"):
                run_cell(mode, bench, r, _tag(mode, bench, r))


def phase_grid():
    for r in (0.25,):
        for tau, gam in GRID_DIV:
            for bench in BENCHES_CORE:
                run_cell("pre", bench, r,
                         _tag("pre_dv", bench, r, div_tau=tau, div_gamma=gam),
                         diversity="nms", div_tau=tau, div_gamma=gam)


def phase_retest(best_tau: float, best_gam: float):
    for r in (0.25, 0.5):
        for bench in BENCHES_CORE + (["chartqa"] if r == 0.5 else []):
            run_cell("pre", bench, r,
                     _tag("pre_dv", bench, r, div_tau=best_tau,
                          div_gamma=best_gam),
                     diversity="nms", div_tau=best_tau, div_gamma=best_gam)


def phase_verify(best_tau: float, best_gam: float):
    for r in (0.25, 0.5):
        for bench in BENCHES_CORE:
            # incumbent arms at same n=500 slice
            for mode in ("pre", "post"):
                run_cell(mode, bench, r, _tag(mode, bench, r), n=500)
            run_cell("pre", bench, r,
                     _tag("pre_dv", bench, r, div_tau=best_tau,
                          div_gamma=best_gam),
                     n=500, diversity="nms", div_tau=best_tau,
                     div_gamma=best_gam)
    for bench in ("chartqa",):
        for mode in ("pre", "post"):
            run_cell(mode, bench, 0.5, _tag(mode, bench, 0.5), n=200)
        run_cell("pre", bench, 0.5,
                 _tag("pre_dv", bench, 0.5, div_tau=best_tau,
                      div_gamma=best_gam),
                 n=200, diversity="nms", div_tau=best_tau, div_gamma=best_gam)


def summary():
    """Print the per-bench x per-r table: RBM(pre), FastV(post), best diversity."""
    def _acc(name: str):
        p = OUT / "sota_stitch" / f"{name}.json"
        return acc_of(str(p)) if p.exists() else None

    print("\n=== per-bench r map: incumbent (pre/post) vs best diversity ===")
    print(f" {'bench':9s} {'r':<5} {'RBM_pre':>8} {'FastV_post':>10} "
          f"{'DV_best':>8} {'incumbent'}")
    for bench in BENCHES_CORE:
        for r in (0.25, 0.5):
            pre = _acc(f"q3_pre_{bench}_r{r}")
            post = _acc(f"q3_post_{bench}_r{r}")
            dv_best = None
            for p in sorted((OUT / "sota_stitch").glob(
                    f"q3_pre_dv_{bench}_r{r}_*.json")):
                v = acc_of(str(p))
                if v is not None and (dv_best is None or v > dv_best):
                    dv_best = v

            def f(x):
                return f"{x:.4f}" if x is not None else "   -   "

            prev = pre or 0.0
            postv = post or 0.0
            dv = dv_best or 0.0
            m = max(prev, postv, dv)
            best = ("RBM" if prev == m else
                    ("POST" if postv == m else "DV"))
            print(f" {bench:9s} r={r:<5} {f(pre)} {f(post)} {f(dv_best)} "
                  f" {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ref", "grid", "retest", "verify",
                                        "summary"], required=True)
    ap.add_argument("--best-tau", type=float, default=0.75)
    ap.add_argument("--best-gam", type=float, default=1.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.phase == "ref":
        phase_ref()
    elif args.phase == "grid":
        phase_grid()
    elif args.phase == "retest":
        phase_retest(args.best_tau, args.best_gam)
    elif args.phase == "verify":
        phase_verify(args.best_tau, args.best_gam)
    elif args.phase == "summary":
        summary()


if __name__ == "__main__":
    main()