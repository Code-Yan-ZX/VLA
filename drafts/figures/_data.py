"""Shared data loaders for P4 figures. Read from source files; no hardcoded numbers.

Sources:
  - notes/lit-survey.md  §2 table (the 37-method throughput tally)
  - eval/final_results.md (Tables A, C-n500, F)
  - runs/p2_d/*.json     (M2 concurrency x prune matrix; step-profile controller)
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LITSURVEY = REPO / "notes" / "lit-survey.md"
FINAL     = REPO / "eval" / "final_results.md"
RUNS      = REPO / "runs" / "p2_d"


# ---------- Fig 1 / Table: the 37-method throughput tally ----------
def parse_lit_throughput():
    """Parse notes/lit-survey.md §2 comparison table.

    Returns list of dicts: {n, method, year, throughput_raw}.
    throughput_raw is the verbatim 'Real throughput?' cell. We then derive:
      wall_clock  = True if cell startswith 'Y' (covers 'Y' and 'Y (partial)')
      deploy      = True if 'DEPLOY' substring present
    """
    txt = LITSURVEY.read_text()
    # isolate §2 table: from '| # |' header up to '### 2.1'
    m = re.search(r"\n\| # \|(.+?)\n\|[-| ]+\|\n(.*?)\n### 2\.1", txt, re.S)
    if not m:
        raise RuntimeError("could not locate lit-survey §2 table")
    # prepend "#|" so header aligns 1:1 with data rows (both split on '|')
    header = ("#|" + m.group(1)).split("|")
    header = [h.strip() for h in header if h.strip()]
    rows = []
    for line in m.group(2).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 8:
            continue
        # columns: # | Method | Year/Venue | Train-free | Base | Benchmarks | Ratio | Accuracy | Real throughput? | Q2.5 | Code | Notes
        try:
            n = int(cells[0])
        except ValueError:
            continue
        method = re.sub(r"\*\*", "", cells[1])
        year = cells[2]
        # find the "Real throughput?" column by header position
        tp_idx = header.index("Real throughput?")
        tp = cells[tp_idx] if tp_idx < len(cells) else ""
        rows.append({"n": n, "method": method, "year": year, "throughput_raw": tp})
    return rows


def throughput_tally():
    """Return (rows, n_total, n_wallclock, n_deploy).

    n_wallclock counts any real wall-clock-style number (cell starts with 'Y'
    or contains 'partial'/'DEPLOY'). n_deploy counts inside a serving engine.
    """
    rows = parse_lit_throughput()
    n_total = len(rows)
    n_wall = 0
    n_dep = 0
    for r in rows:
        tp = r["throughput_raw"]
        tp_plain = re.sub(r"\*", "", tp).strip()
        # wall-clock = a real number (cell marked "Y" or "Y (partial)").
        # bare "partial" (Eval-Framework meta-eval) is NOT a method wall-clock report.
        is_wall = tp_plain.upper().startswith("Y")
        is_dep = "DEPLOY" in tp_plain.upper()
        r["wall_clock"] = is_wall
        r["deploy"] = is_dep
        if is_wall:
            n_wall += 1
        if is_dep:
            n_dep += 1
    return rows, n_total, n_wall, n_dep


# ---------- Fig 2: M2 concurrency x prune matrix (GQA) ----------
def m2_matrix():
    """Read runs/p2_d/m2_*.json -> {(conc, prune): req/s}.

    conc in {1,4,12}, prune in {0, 0.50, 0.75}. Missing cells omitted (c4/r75).
    """
    out = {}
    for conc in (1, 4, 12):
        for pr, tag in ((0.0, "r0"), (0.50, "r50"), (0.75, "r75")):
            p = RUNS / f"m2_c{conc}_{tag}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            out[(conc, pr)] = d["agg"]["served_req_s"]["mean"]
    return out


# ---------- Fig 3: controller step-profile time-series ----------
def step_profile():
    """Read runs/p2_d/p3s1_gqa_adaptive_step_mt32.json -> realized list.

    Each entry: {decision_index, r, num_running, conc_frac, kv_occupancy}.
    conc_frac = num_running / max_num_seqs.
    """
    p = RUNS / "p3s1_gqa_adaptive_step_mt32.json"
    d = json.loads(p.read_text())
    max_seqs = d["max_num_seqs"]
    rz = d["controller"]["realized"]
    series = []
    for i, e in enumerate(rz):
        series.append({
            "i": i,
            "r": e["r"],
            "num_running": e["num_running"],
            "conc_frac": e["num_running"] / max_seqs,
            "kv_occupancy": e["kv_occupancy"],
            "r_min": d["controller"]["r_min"],
            "r_max": d["controller"]["r_max"],
        })
    meta = {
        "max_num_seqs": max_seqs,
        "n_decisions": len(series),
        "load_profile": d["load_profile"],
        "benchmark": d["benchmark"],
        "conc_lo": d["controller"]["conc_lo"],
        "conc_hi": d["controller"]["conc_hi"],
    }
    return series, meta


# ---------- Fig 4: n=500 Pareto (Table C-n500) ----------
def pareto_n500():
    """Parse eval/final_results.md 'Table C-n500' block.

    Returns {benchmark: {adaptive, fixed r25, fixed r50} each {req_s, acc}}.
    """
    txt = FINAL.read_text()
    block = re.search(
        r"### Table C-n500.*?\n(.*?)\n\*\*★ n=500 GATE", txt, re.S).group(1)
    data = {}
    cur = None
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 4:
            continue
        # benchmark header row like | **GQA** (mt32) | adaptive | 2.383 | 0.556 | ...
        c0 = re.sub(r"\*\*", "", cells[0])
        bench_match = next((b for b in
            ("GQA", "MME", "MMBENCH", "SCIENCEQA", "TEXTVQA")
            if c0.upper().startswith(b)), None)
        if bench_match:
            cur = bench_match
            data[cur] = {}
            cells = cells[1:]  # rest are config/req/acc...
            if len(cells) >= 3 and cells[0] in ("adaptive", "fixed r25", "fixed r50"):
                _add_point(data, cur, cells)
        elif cur is not None and cells[0] in ("adaptive", "fixed r25", "fixed r50"):
            _add_point(data, cur, cells)
    return data


def _add_point(data, bench, cells):
    cfg = cells[0]
    # find req/s (first float >= 1) and acc (float <= 1) in cells[1:4]
    vals = []
    for c in cells[1:5]:
        m = re.search(r"\d+\.\d+", c)
        if m:
            vals.append(float(m.group()))
    if len(vals) >= 2:
        data[bench][cfg] = {"req_s": vals[0], "acc": vals[1]}
