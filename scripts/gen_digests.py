#!/usr/bin/env python
"""Generate compact experiments/<name>.md digests for each P2 probe metrics file.
Re-scores accuracy from saved raw answers with the fixed scorers (so GQA r0/r50,
which ran pre-scorer-fix, report correct accuracy). Pure CPU, no vLLM import
needed (scorers are standalone). Run: python scripts/gen_digests.py
"""
import json, glob, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from src.serve_bench import score_gqa, score_textvqa  # fixed scorers, lazy vllm import

SCORER = {"gqa": score_gqa, "textvqa": score_textvqa}
EXP = os.path.join(ROOT, "experiments")
SKIP = {"gqa_r0", "gqa_r50"}  # metric-names of the hand-written gate digests

def fmt_x(x): return f"{x:.2f}x" if isinstance(x, (int, float)) else "—"

for p in sorted(glob.glob(os.path.join(ROOT, "runs", "p2_probe", "*_metrics.json"))):
    d = json.load(open(p)); a = d["agg"]; name = os.path.basename(p).replace("_metrics.json", "")
    if name in SKIP:
        continue
    bm = d["benchmark"]; r = d["pruning_rate"]
    raw = d.get("raw", [])
    sc = SCORER[bm]
    correct = sum(sc(o.get("answer", ""), o.get("gt", ""), None) for o in raw)
    acc = correct / len(raw) if raw else 0.0
    px, ex = d.get("prefill_speedup_vs_r0"), d.get("e2e_speedup_vs_r0")
    lines = [
        f"# {name} — {bm.upper()} @ prune {r} (keep {int(round(576*(1-r)))}/576), n={d.get('n')}", "",
        f"- **e2e req/s**: {a['served_req_s']['mean']:.2f}  ({fmt_x(ex)} vs r0)",
        f"- **prefill TTFT**: {a['ttft_ms']['mean']:.0f} ms  ({fmt_x(px)} vs r0)",
        f"- **served tok/s**: {a['served_tok_s']['mean']:.2f}",
        f"- **accuracy (re-scored, fixed scorer)**: {acc:.3f} ({correct}/{len(raw)})",
        f"- **log**: `runs/{name}.log` · **metrics**: `runs/p2_probe/{name}_metrics.json` (gitignored; raw answers saved)",
        "",
        "Full curve + analysis: `eval/p2_probe_summary.md`. Probe n=200 is a seed subset (seed=0) for the go/no-go gate, not the final benchmark number.",
    ]
    with open(os.path.join(EXP, f"probe_{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote experiments/probe_{name}.md  acc={acc:.3f} e2e={fmt_x(ex)}")
