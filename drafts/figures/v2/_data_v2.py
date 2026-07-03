"""V2 figure data loaders. Read from raw v2 run JSON + notes tables; no hardcoded numbers.

Sources (all paths absolute from repo root /media/disk2/YZX/research/vla):
  - runs/v2_p2/batch_c{C}_r{R}.json  -> LLaVA-1.5 V1 scale matrix (c in 1/4/16/64)
  - runs/v2_p2/qwen3vl_c64_r{0,50}.json -> Qwen3-VL c64 (req/s only)
  - runs/v2_p3/{sel}_c64_r{R}.json   -> cross-compressor panel at c64
  - notes/v2_p1_qwen3vl.md §2 table  -> Qwen3-VL c1/c12 (no JSON stored; parse table)
  - notes/lit-survey.md §2 table      -> 37-method throughput tally (Fig 1)
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_P2 = REPO / "runs" / "v2_p2"
RUNS_P3 = REPO / "runs" / "v2_p3"
NOTES_P1 = REPO / "notes" / "v2_p1_qwen3vl.md"


# ---------- Fig 2 / Fig 3: LLaVA-1.5 V1 scale matrix ----------
def scale_matrix():
    """{(c, r): cell} for c in {1,4,16,64}, r in {0.0,0.50,0.75}.

    cell = {req_s, ttft_p50, ttft_p99, e2e_p99, acc, peak_kv_mb, wall_s, n, raw_path}.
    Reads runs/v2_p2/batch_c{C}_r{R}.json.
    """
    out = {}
    for c in (1, 4, 16, 64):
        for r, tag in ((0.0, "r0"), (0.50, "r50"), (0.75, "r75")):
            p = RUNS_P2 / f"batch_c{c}_{tag}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            agg = d["agg"]
            out[(c, r)] = {
                "req_s": agg["served_req_s"]["mean"],
                "ttft_p50": agg["ttft_ms_p50"],
                "ttft_p99": agg["ttft_ms_p99"],
                "e2e_p99": agg["e2e_ms_p99"],
                "acc": agg["accuracy"],
                "peak_kv_mb": agg["peak_kv_mb"]["mean"],
                "wall_s": d["wall_s"],
                "n": d["n"],
                "raw_path": str(p),
            }
    return out


def goodput_at_slo(path_or_raw, slo_ms, metric="ttft_ms", served_req_s=None):
    """Goodput (req/s meeting SLO) = served_req_s * frac(req[metric] <= slo).

    Accepts either a path to a batch JSON (reads raw list) or a (raw_list, req_s) pair.
    """
    if isinstance(path_or_raw, (str, Path)):
        d = json.loads(Path(path_or_raw).read_text())
        raw = d["raw"]
        served_req_s = d["agg"]["served_req_s"]["mean"]
    else:
        raw, served_req_s = path_or_raw
    n = len(raw)
    if n == 0:
        return 0.0
    n_met = sum(1 for r in raw if r[metric] <= slo_ms)
    return served_req_s * n_met / n


# ---------- Fig 3: goodput SLO sweep at c64 ----------
def goodput_sweep_c64(slos_ms=(3000, 5000, 8000, 10000)):
    """{(r, slo_ms): goodput_req_s} at c64 for r in {0,0.50,0.75}, TTFT-SLO."""
    out = {}
    for r, tag in ((0.0, "r0"), (0.50, "r50"), (0.75, "r75")):
        p = RUNS_P2 / f"batch_c64_{tag}.json"
        if not p.exists():
            continue
        for slo in slos_ms:
            out[(r, slo)] = goodput_at_slo(p, slo, metric="ttft_ms")
    return out


# ---------- Fig 5: cross-compressor panel at c64 ----------
P3_COMPRESSORS = ("proxy", "true_cls", "tome_merge", "random")


def cross_compressor_c64():
    """{(compressor, r): cell} at c64 for r in {0.0,0.50,0.75}.

    cell = {req_s, ttft_p99, acc, goodput_5s, raw_path}.
    """
    out = {}
    for sel in P3_COMPRESSORS:
        for r, tag in ((0.0, "r0"), (0.50, "r50"), (0.75, "r75")):
            p = RUNS_P3 / f"{sel}_c64_{tag}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            agg = d["agg"]
            out[(sel, r)] = {
                "req_s": agg["served_req_s"]["mean"],
                "ttft_p99": agg["ttft_ms_p99"],
                "acc": agg["accuracy"],
                "goodput_5s": goodput_at_slo(p, 5000, metric="ttft_ms"),
                "raw_path": str(p),
            }
    return out


# ---------- Fig 4: architecture-conditional (Qwen3-VL c1/c12 from notes, c64 from JSON) ----------
def qwen3vl_c64():
    """{(r): req_s} for Qwen3-VL-8B c64 from JSON (r in {0.0, 0.50})."""
    out = {}
    for r, tag in ((0.0, "r0"), (0.50, "r50")):
        p = RUNS_P2 / f"qwen3vl_c64_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out[r] = d["req_per_s"]
    return out


def qwen3vl_p1_table():
    """Parse notes/v2_p1_qwen3vl.md §2 served-throughput table.

    Returns {(c, r): req_s} for c in {1,4,12}, r in {0.0,0.50,0.75}.
    Table rows: | c | r0 req/s | r50 req/s | r75 req/s | ... |
    """
    txt = NOTES_P1.read_text()
    # locate the §2 table by its header signature
    m = re.search(
        r"\| c \| r0 req/s \| r50 req/s \| r75 req/s \|.*?\n\|[-| ]+\|\n(.*?)(?=\n\n|\n\*\*)",
        txt, re.S,
    )
    if not m:
        raise RuntimeError("could not locate v2_p1 §2 served-throughput table")
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 4:
            continue
        cm = re.match(r"c?(\d+)", cells[0])
        if not cm:
            continue
        c = int(cm.group(1))
        # strip bold markdown (**..**) before float conversion
        nums = []
        try:
            for ci in cells[1:4]:
                nums.append(float(re.sub(r"\*\*", "", ci).strip()))
        except ValueError:
            continue
        r0, r50, r75 = nums
        out[(c, 0.0)] = r0
        out[(c, 0.50)] = r50
        out[(c, 0.75)] = r75
    return out


def arch_conditional_amplification():
    """r75/r0 served speedup vs concurrency, for LLaVA-1.5 and Qwen3-VL-8B.

    Returns {model: {c: (r75/r0 ratio, r0_req_s, r75_req_s)}}.
    LLaVA c1/c4/c16/c64 from scale_matrix; Qwen3-VL c1/c12 from notes, c64 from JSON.
    Qwen3-VL c64 has no r75 (only r0/r50 measured) -> omit r75 ratio, use r50/r0 at c64.
    """
    scale = scale_matrix()
    llava = {}
    for c in (1, 4, 16, 64):
        if (c, 0.0) in scale and (c, 0.75) in scale:
            r0 = scale[(c, 0.0)]["req_s"]
            r75 = scale[(c, 0.75)]["req_s"]
            llava[c] = (r75 / r0, r0, r75)
    qwen = {}
    p1 = qwen3vl_p1_table()
    for c in (1, 12):
        if (c, 0.0) in p1 and (c, 0.75) in p1:
            r0 = p1[(c, 0.0)]
            r75 = p1[(c, 0.75)]
            qwen[c] = (r75 / r0, r0, r75)
    # Qwen3-VL c64: r50/r0 only (r75 not measured)
    q64 = qwen3vl_c64()
    if 0.0 in q64 and 0.50 in q64:
        # mark as r50-based; caller knows Qwen c64 uses r50/r0
        qwen[64] = (q64[0.50] / q64[0.0], q64[0.0], q64[0.50])
    return {"LLaVA-1.5-7B": llava, "Qwen3-VL-8B": qwen}
