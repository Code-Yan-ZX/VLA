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
             selector: str = "l2", alpha: float = 1.0, beta: float = 1.0,
             diversity: str = "none", div_tau: float = 0.75,
             div_gamma: float = 1.5, extra_args: list | None = None,
             force: bool = False) -> str | None:
    """Run one runner cell; returns the output JSON path (or None on failure).
    mode ∈ {pre, post}; diversity ∈ {none (incumbent RBM), nms (stitch probe)}.
    extra_args = additional runner flags (e.g. --budget-file/--budget-calib)."""
    subset, _, benchextra = BENCH[bench]
    n_override = n if n else BENCH[bench][1]
    out_file = OUT / "sota_stitch" / f"{tag}.json"
    if not force and out_file.exists() and out_file.stat().st_size > 100:
        log(f"skip {tag} (exists)")
        return str(out_file)
    os.makedirs(out_file.parent, exist_ok=True)
    cmd = [PYQ3, str(RUNNER), *STD, "--mode", mode, "--benchmark", bench,
           "--subset", str(REPO / subset), "--n", str(n_override), "--r", str(r),
           "--selector", selector, "--alpha", str(alpha), "--beta", str(beta),
           "--diversity", diversity]
    if diversity == "nms":
        cmd += ["--div-tau", str(div_tau), "--div-gamma", str(div_gamma)]
    # extra_args AFTER benchextra so per-cell overrides (--budget-file,
    # --max-num-seqs 1) win argparse's last-value-wins
    cmd += [*benchextra, *(extra_args or []), "--out", str(out_file)]
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
            # OCRBench official "final score" needs per-sample question_type +
            # category meta (mirrors freq_adaptive_driver's protocol).
            meta = {}
            for src in (REPO / "eval/full_splits/ocrbench.jsonl",
                        REPO / "eval/subsets/ocrbench_200.jsonl"):
                if src.exists():
                    for line in open(src):
                        o = json.loads(line)
                        ex = o.get("extras") or {}
                        meta.setdefault(str(o["id"]), (ex.get("question_type", ""),
                                                       ex.get("category")))
            items = [(a, g) + meta.get(str(p.get("id")), ("", ""))
                     for a, g, p in zip(preds, gts, ps)]
            return S.score_ocrbench_batch(items)["acc"]
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


def phase_calib():
    """Phase-2 calib dump: pre r=0.05, --budget-calib per bench (n=200,
    request order == sample order via default max-num-seqs of the runner)."""
    for bench in BENCHES_CORE:
        calib_path = OUT / "sota_stitch" / f"calib_{bench}.json"
        run_cell("pre", bench, 0.05, f"calib_{bench}", extra_args=[
            "--budget-calib", str(calib_path)])


def phase_budget(mode: str = "spectral", tau: float = 0.9, clamp: float = 0.5,
                 div_tau: float | None = None):
    """Phase-2 eval: allocator -> per-image budgets (iso-token to r=0.25),
    then pre + --budget-file runs on the 4 benches (max-num-seqs=1). With
    --div-tau>0, ALSO runs the combined cell (budget + diversity nms)."""
    for is_comb in (False, True) if div_tau else (False,):
        for bench in BENCHES_CORE:
            calib_path = OUT / "sota_stitch" / f"calib_{bench}.json"
            if not calib_path.exists():
                log(f"MISSING {calib_path.name}; run phase calib first")
                continue
            bud_path = OUT / "sota_stitch" / \
                f"budgets_{mode}_t{tau}_r0.25_{bench}.json"
            if not bud_path.exists():
                subprocess.run(
                    [PYQ3, str(REPO / "scripts/stitch_budget_alloc.py"),
                     "--calib", str(calib_path), "--r", "0.25",
                     "--mode", mode, "--tau", str(tau), "--clamp", str(clamp),
                     "--out", str(bud_path)], check=True)
            if is_comb:
                tag = f"pre_bd_{mode}t{tau}_tau{div_tau}_{bench}_r0.25"
                run_cell("pre", bench, 0.25, tag,
                         diversity="nms", div_tau=float(div_tau),
                         div_gamma=1.5,
                         extra_args=["--budget-file", str(bud_path),
                                     "--max-num-seqs", "1"])
            else:
                tag = f"pre_budget_{mode}t{tau}_{bench}_r0.25"
                run_cell("pre", bench, 0.25, tag,
                         extra_args=["--budget-file", str(bud_path),
                                     "--max-num-seqs", "1"])


def phase_probe2(best_tau: float, best_gam: float):
    """Composite stitch probe: freq scorer (Direction B: alpha=1,beta=0.6 --
    textvqa +1.2pp@25) x diversity-NMS (best tau/gam from grid). Same 4 benches
    @ r=0.25 dev. Pure selection-level composition, zero count changes."""
    for bench in BENCHES_CORE:
        run_cell("pre", bench, 0.25,
                 _tag("pre_freqdv", bench, 0.25, div_tau=best_tau,
                      div_gamma=best_gam),
                 selector="freq", alpha=1.0, beta=0.6,
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
    """Official-metric table with incumbents, best diversity (tau/gamma), deltas
    and the r=0.25 gate verdict.

    Gate (r=0.25, dev): GO if diversity wins >=1 cell where POST is incumbent
    (textvqa/gqa up to data) AND no >1pp regression on PRE-incumbent cells.
    SOTA-hold = dv >= max(pre, post) - 1e-3 on ALL four cells at r=0.25."""
    def _acc(*names):
        for name in names:
            p = OUT / "sota_stitch" / f"{name}.json"
            if p.exists():
                v = acc_of(str(p))
                if v is not None:
                    return v
        return None

    print("\n=== per-bench r map: incumbent (pre/post) vs best diversity ===")
    print(f" {'bench':9s} {'r':<5} {'RBM_pre':>8} {'FastV_post':>10} "
          f"{'DV_best':>9} {'(tau,gam)':>12} {'dv-pre':>7} {'dv-post':>7}")
    verdict_rows = []
    for bench in BENCHES_CORE:
        for r in (0.25, 0.5):
            pre = _acc(f"q3_pre_{bench}_r{r}")
            post = _acc(f"q3_post_{bench}_r{r}")
            dv_best, dv_cfg = None, None
            for glob_pat in (f"q3_pre_dv_{bench}_r{r}_*.json",
                             f"q3_pre_budget_*_{bench}_r{r}.json",
                             f"q3_pre_bd_*_{bench}_r{r}.json"):
                for p in sorted((OUT / "sota_stitch").glob(glob_pat)):
                    v = acc_of(str(p))
                    if v is not None and (dv_best is None or v > dv_best):
                        dv_best = v
                        dv_cfg = p.stem.replace(f"q3_pre_dv_{bench}_r{r}_", "") \
                            .replace(f"q3_pre_budget_", "") \
                            .replace(f"q3_pre_bd_", "")

            def f(x):
                return f"{x:.4f}" if x is not None else "   -   "

            inc = max(pre or 0, post or 0)
            dv = dv_best if dv_best is not None else None
            best = ("DV" if dv is not None and dv >= inc - 1e-9 else
                    ("RBM" if (pre or 0) >= (post or 0) else "POST"))
            if dv is None:
                dpp = dpo = "  n/a"
            else:
                dpp = f"{100*(dv-(pre or 0)):+.1f}" if pre is not None else "  -"
                dpo = f"{100*(dv-(post or 0)):+.1f}" if post is not None else "  -"
            print(f" {bench:9s} r={r:<5} {f(pre)} {f(post)} {f(dv_best)} "
                  f"{str(dv_cfg):>12} {dpp:>6}pp {dpo:>6}pp  [{best}]")
            if r == 0.25 and pre is not None and post is not None and dv_best:
                verdict_rows.append((bench, pre, post, dv_best))

    print("\n=== r=0.25 gate ===")
    if len(verdict_rows) == 4:
        losses = [(b, p, d) for b, p, q, d in verdict_rows if d < p - 0.01]
        wins_post = [(b, q, d) for b, p, q, d in verdict_rows
                     if q > p and d > q - 0.001]
        wins_pre = [(b, p, d) for b, p, q, d in verdict_rows
                    if p > q and d > p - 0.001]
        sota_hold = all(d >= max(p, q) - 1e-3 for b, p, q, d in verdict_rows)
        print(f" losses (>1pp vs PRE-incumbent): {losses if losses else 'none'}")
        print(f" DV beats PRE on PRE-leading: {[b for b,_,_ in wins_pre]}")
        print(f" DV beats POST on POST-leading: {[b for b,_,_ in wins_post]}")
        go = (not losses) and wins_post
        print(f" gate verdict: {'GO -> promote to n=500' if go else 'NO-GO'} "
              f"(rule: no >1pp PRE regression AND win >=1 POST-led cell)")
        print(f" SOTA-held-all-four (dv>=max(pre,post)): {sota_hold}")
    else:
        print(" cell set incomplete for the gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ref", "grid", "retest", "verify",
                                        "probe2", "calib", "budget",
                                        "summary"], required=True)
    ap.add_argument("--best-tau", type=float, default=0.75)
    ap.add_argument("--best-gam", type=float, default=1.5)
    ap.add_argument("--budget-mode", choices=["spectral", "erank"],
                    default="spectral")
    ap.add_argument("--budget-tau", type=float, default=0.9)
    ap.add_argument("--combine-tau", type=float, default=None,
                    help="phase budget: >0 also runs budget+diversity-NMS "
                         "combined cells at this NMS threshold")
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
    elif args.phase == "probe2":
        phase_probe2(args.best_tau, args.best_gam)
    elif args.phase == "calib":
        phase_calib()
    elif args.phase == "budget":
        phase_budget(args.budget_mode, args.budget_tau,
                     div_tau=args.combine_tau)
    elif args.phase == "summary":
        summary()


if __name__ == "__main__":
    main()