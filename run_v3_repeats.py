#!/usr/bin/env python3
"""c64 repeat experiments: 6 cells x 5 runs -> mean+-std summary.

Cells = fixed-r in {0.0, 0.5, 0.75} x benchmark in {gqa, textvqa} at c64.
Each run: batch-submit, selector proxy, limit=200, max-num-seqs 64, SLO e2e 5000ms.
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/media/disk2/YZX/research/vla")
OUTDIR = REPO / "runs" / "v3_rep"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = OUTDIR / "orchestrator.log"

PRUNING_RATES = [0.0, 0.5, 0.75]
BENCHMARKS = ["gqa", "textvqa"]
N_RUNS = 5

# fields collected per run
FIELDS = [
    ("req/s",        lambda d: d["agg"]["served_req_s"]["mean"]),
    ("p99-ttft(ms)", lambda d: d["agg"]["ttft_ms_p99"]),
    ("p99-e2e(ms)",  lambda d: d["agg"]["e2e_ms_p99"]),
    ("goodput_acc",  lambda d: d["goodput_at_slo"]["goodput_acc"]),
    ("met_rate",     lambda d: d["goodput_at_slo"]["met_rate"]),
    ("accuracy",     lambda d: d["agg"]["accuracy"]),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_cell(bench, rate, run_i):
    out = OUTDIR / f"{bench}_r{rate}_run{run_i}.json"
    cmd = [
        "python", "-m", "src.serve_bench",
        "--model", "runs/models/llava-1.5-7b-hf",
        "--engine", "v1",
        "--selector", "proxy",
        "--benchmark", bench,
        "--subset", f"eval/subsets/{bench}_200.jsonl",
        "--batch-submit",
        "--max-num-seqs", "64",
        "--max-tokens", "32",
        "--limit", "200",
        "--k-policy", "fixed",
        "--pruning-rate", str(rate),
        "--slo-ms", "5000",
        "--slo-type", "e2e",
        "--metrics-out", str(out),
    ]
    log(f"START {bench} r={rate} run {run_i} -> {out.name}")
    t0 = time.time()
    with open(OUTDIR / f"{bench}_r{rate}_run{run_i}.stdout", "w") as sof:
        proc = subprocess.run(
            cmd, cwd=str(REPO),
            stdout=sof, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    if proc.returncode != 0:
        log(f"  FAIL rc={proc.returncode} ({dt:.0f}s) see {out.stem}.stdout")
        return None
    if not out.exists():
        log(f"  FAIL no JSON ({dt:.0f}s)")
        return None
    log(f"  OK ({dt:.0f}s)")
    return out


def extract(path):
    with open(path) as f:
        d = json.load(f)
    row = {}
    for name, fn in FIELDS:
        try:
            row[name] = float(fn(d))
        except Exception as e:
            row[name] = float("nan")
    return row


def fmt_ms(values):
    """mean+-std, blank if any nan."""
    vals = [v for v in values if v == v]  # drop nan
    if len(vals) < 2:
        if not vals:
            return "n/a"
        return f"{vals[0]:.2f}"
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return f"{m:.2f}+-{s:.2f}"


def main():
    # collect all per-run extractions
    # results[(bench,rate)] = list of dict-rows
    results = {}
    cells = [(b, r) for b in BENCHMARKS for r in PRUNING_RATES]
    for bench, rate in cells:
        rows = []
        for run_i in range(N_RUNS):
            out = run_cell(bench, rate, run_i)
            if out is not None:
                rows.append(extract(out))
        results[(bench, rate)] = rows

    # build summary
    lines = []
    lines.append("# v3 c64 Repeats — mean±std over 5 runs\n")
    lines.append("Config: model=llava-1.5-7b-hf, engine=v1, selector=proxy, "
                 "batch-submit, max-num-seqs=64, limit=200, max-tokens=32, "
                 "k-policy=fixed, SLO e2e 5000ms. Seed=0 (natural scheduling/timing variance).\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("")
    header = "| cell | n | req/s | p99-ttft (ms) | p99-e2e (ms) | goodput_acc | met_rate | accuracy |"
    sep    = "|------|---|-------|---------------|--------------|-------------|----------|----------|"
    lines.append(header)
    lines.append(sep)
    flags = []
    for bench in BENCHMARKS:
        for rate in PRUNING_RATES:
            rows = results[(bench, rate)]
            n = len(rows)
            cell = f"{bench}_c64_r{rate}"
            col_vals = {name: [r[name] for r in rows] for name, _ in FIELDS}
            cells_str = []
            high_var = []
            for name, _ in FIELDS:
                vals = col_vals[name]
                vals_clean = [v for v in vals if v == v]
                if not vals_clean:
                    cells_str.append("n/a")
                    continue
                if len(vals_clean) >= 2:
                    m = statistics.mean(vals_clean)
                    s = statistics.stdev(vals_clean)
                    cells_str.append(f"{m:.3f}±{s:.3f}")
                    if m != 0 and abs(s / m) > 0.10:
                        high_var.append(f"{name} (std/mean={abs(s/m):.2%})")
                else:
                    cells_str.append(f"{vals_clean[0]:.3f}")
            lines.append(f"| {cell} | {n} | " + " | ".join(cells_str) + " |")
            if high_var:
                flags.append((cell, high_var))
    lines.append("")
    lines.append("## High-variance flags (std/mean > 10%)")
    if flags:
        for cell, hv in flags:
            lines.append(f"- **{cell}**: {', '.join(hv)}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Per-run raw values")
    for bench in BENCHMARKS:
        for rate in PRUNING_RATES:
            rows = results[(bench, rate)]
            lines.append(f"\n### {bench}_c64_r{rate}")
            rh = "| run | " + " | ".join(name for name, _ in FIELDS) + " |"
            rs = "|----|" + "|".join(["----"] * len(FIELDS)) + "|"
            lines.append(rh)
            lines.append(rs)
            for i, r in enumerate(rows):
                lines.append(f"| {i} | " + " | ".join(
                    ("n/a" if r[name] != r[name] else f"{r[name]:.4f}")
                    for name, _ in FIELDS) + " |")

    summary = "\n".join(lines) + "\n"
    out_md = REPO / "final_results_v3_repeats.md"
    with open(out_md, "w") as f:
        f.write(summary)
    log(f"WROTE {out_md}")
    print("\n\n========== SUMMARY ==========\n")
    print(summary)


if __name__ == "__main__":
    main()
