#!/usr/bin/env python3
"""mRoPE/scorer provenance audit -- Gate C official rescore re-execution.

Re-scores the RAW answers of the MALT goal-mode Gate C arms (H0n = native
immediate-RBM, H1 = deferred K1) with the SAME official evaluators the
lifetime sweep used (analyze_deferred_sweep.py):
  TextVQA  -> official VQA accuracy        (score_textvqa_vqaacc)
  DocVQA   -> ANLS, per-item continuous     (score_docvqa_anls)
  OCRBench -> official 5-category scoring   (score_ocrbench_batch)
  GQA      -> normalized exact match        (score_gqa)
No raw `correct` field is read as a metric.

Reproduces H1's already-reported official scores first
(textvqa ~0.7433, docvqa ~0.5959, ocrbench ~0.6354, gqa ~0.5400); if they do
not match, the audit must STOP and investigate (scorer version, raw answer
files, symlink target, manifest, answer normalization).

Output: experiments/mrope_scorer_provenance_audit.md (appended by the audit)
and stdout report.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/media/disk2/YZX/research/vla/src/v3_premerger")
from official_scorers import (score_textvqa_vqaacc, score_docvqa_anls,
                              score_gqa, score_ocrbench_batch)

HERE = "/media/disk2/YZX/research/vla"
MALT = f"{HERE}/runs/malt_goal_mode"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
NBOOT = 5000
rng = np.random.default_rng(0)

# Expected H1 official scores (from experiments/deferred_rbm_n200_data/
# sweep_analysis.json mean_tbl, git ea359d8) -- reproduction anchor.
H1_EXPECTED = {"textvqa": 0.7433333333333333,
               "docvqa": 0.5959,
               "ocrbench": 0.6354,
               "gqa": 0.5400}
# Expected H1 raw `correct` (ad-hoc containment) for the mismatch table.
H1_RAW_EXPECTED = {"textvqa": 0.83, "docvqa": 0.47,
                   "ocrbench": 0.6354, "gqa": 0.565}


def load(path):
    with open(path) as f:
        return json.load(f)


def official_value(bench, answer, gt, qtype=None, category=None):
    if bench == "ocrbench":
        return float(score_ocrbench_batch(
            [(answer, gt, qtype or "", category or "")])["per_item"][0])
    if bench == "textvqa":
        return float(score_textvqa_vqaacc(answer, gt))
    if bench == "docvqa":
        return float(score_docvqa_anls(answer, gt))
    return float(score_gqa(answer, gt))


def official_per_sample(cell, bench):
    out = {}
    for s in cell.get("per_sample", []):
        if s.get("skipped"):
            continue
        v = official_value(bench, s.get("answer", ""), s.get("gt", ""),
                           s.get("question_type"), s.get("category"))
        out[str(s["id"])] = (v, 1 if v >= 0.5 else 0)
    return out


def raw_correct_per_sample(cell):
    return {str(s["id"]): int(s.get("correct", 0)) for s in cell["per_sample"]
            if not s.get("skipped")}


def boot_ci_paired(diffs):
    """Paired bootstrap 95% CI over per-sample diff values (mirror of
    analyze_deferred_sweep.boot_ci)."""
    arr = np.array([diffs[i] for i in diffs])
    means = np.empty(NBOOT)
    for k in range(NBOOT):
        idx = rng.integers(0, len(arr), len(arr))
        means[k] = arr[idx].mean()
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main():
    print("=== GATE C OFFICIAL RESCORE (H0n vs H1, n=200 locked) ===\n")

    # --- load arms ---
    arms = {}
    for tag in ("h0n", "h1"):
        cells = {}
        for b in BENCHES:
            p = f"{MALT}/{tag}_{b}_n200.json"
            cells[b] = load(p)
            if os.path.islink(p):
                print(f"  [symlink] {tag}_{b} -> {os.path.realpath(p)}")
        arms[tag] = cells

    # --- official per-sample + raw per-sample ---
    off = {tag: {b: official_per_sample(arms[tag][b], b) for b in BENCHES}
           for tag in ("h0n", "h1")}
    raw = {tag: {b: raw_correct_per_sample(arms[tag][b]) for b in BENCHES}
           for tag in ("h0n", "h1")}

    # --- 0. reproduce H1 official scores ---
    print("\n[0] REPRO H1 (deferred K1) official scores (anchor)")
    repro_ok = True
    for b in BENCHES:
        vals = [off["h1"][b][i][0] for i in off["h1"][b]]
        m = sum(vals) / len(vals)
        exp = H1_EXPECTED[b]
        ok = abs(m - exp) < 5e-4
        repro_ok = repro_ok and ok
        rawm = sum(raw["h1"][b].values()) / max(1, len(raw["h1"][b]))
        print(f"  {b:9s} official={m:.4f} (expect {exp:.4f}) "
              f"{'OK' if ok else '** MISMATCH **'}  raw_correct={rawm:.4f} "
              f"(expect {H1_RAW_EXPECTED[b]:.4f})  n={len(vals)}")
    if not repro_ok:
        print("\n[!!] H1 official repro FAILED -- STOP. Investigate: scorer "
              "version, raw answer files, symlink target, manifest, correct "
              "overwrite, answer normalization.")
        sys.exit(1)
    print("    H1 official scores REPRODUCED -> proceed.\n")

    # --- 1. official per-dataset H0n vs H1 ---
    print("[1] OFFICIAL accuracy (common ids)")
    macro = {}
    diffs = {}          # per-bench per-sample diff dicts
    for b in BENCHES:
        ids = [i for i in off["h0n"][b] if i in off["h1"][b]]
        hn = sum(off["h0n"][b][i][0] for i in ids) / len(ids)
        h1 = sum(off["h1"][b][i][0] for i in ids) / len(ids)
        macro.setdefault("h0n", []).append(hn)
        macro.setdefault("h1", []).append(h1)
        diffs[b] = {i: off["h0n"][b][i][0] - off["h1"][b][i][0] for i in ids}
        lo, hi = boot_ci_paired(diffs[b])
        # win/tie/loss on binary collapse of official metric
        w = t = l = 0
        for i in ids:
            dh, d1 = off["h0n"][b][i][1], off["h1"][b][i][1]
            w += int(dh == 1 and d1 == 0); l += int(dh == 0 and d1 == 1)
            t += int(dh == d1)
        print(f"  {b:9s} H0n={hn:.4f}  H1={h1:.4f}  diff={hn-h1:+.4f} "
              f"CI95=[{lo:+.4f},{hi:+.4f}]  W/T/L={w}/{t}/{l}  n={len(ids)}")

    m0, m1 = sum(macro["h0n"]) / 4, sum(macro["h1"]) / 4
    print(f"\n  OFFICIAL macro  H0n={m0:.4f}  H1={m1:.4f}  diff={m0-m1:+.4f}")

    # --- 2. macro paired bootstrap (re-sample per-bench paired diffs,
    #        averaging across benches -- same structure as Gate C's macro CI
    #        but on OFFICIAL per-sample metric values) ---
    comb = {i: (0.0, 0.0) for b in BENCHES for i in diffs[b]}
    for b in BENCHES:
        for i in diffs[b]:
            hn, h1 = off["h0n"][b][i], off["h1"][b][i]
            comb[i] = (comb[i][0] + hn[0] / 4, comb[i][1] + h1[0] / 4)
    cids = sorted(comb)
    arr = np.array([comb[i][0] - comb[i][1] for i in cids])
    means = np.empty(NBOOT)
    for k in range(NBOOT):
        idx = rng.integers(0, len(arr), len(arr))
        means[k] = arr[idx].mean()
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    print(f"  paired bootstrap 95% CI of OFFICIAL macro diff (H0n-H1): "
          f"[{lo:+.4f}, {hi:+.4f}]  n_pairs={len(cids)}")

    # --- 3. raw vs official discrepancy (per arm) ---
    print("\n[2] raw `correct` vs OFFICIAL mean (Gate C scorer-mismatch scale)")
    for tag in ("h0n", "h1"):
        for b in BENCHES:
            rv = sum(raw[tag][b].values()) / max(1, len(raw[tag][b]))
            ov = sum(off[tag][b][i][0] for i in off[tag][b]) / len(off[tag][b])
            print(f"  {tag} {b:9s} raw_correct={rv:.4f}  official={ov:.4f}  "
                  f"raw-official={rv-ov:+.4f}")

    # --- 4. deferred-effect decomposition on official metrics ---
    print("\n[3] DEFERRED-EFFECT decomposition (OFFICIAL metrics)")
    print("    K0->K1 sweep gain conflates mRoPE + timing; official H0n vs H1 "
          "isolates timing (both native coords).")
    # K0 official = gate_pre25 (vllm-mimic immediate-RBM)
    k0 = {}
    for b in BENCHES:
        cell = load(f"{HERE}/runs/cascade/gate_pre25_{b}.json")
        per = official_per_sample(cell, b)
        ids = [i for i in per if i in off["h1"][b]]
        k0[b] = (sum(per[i][0] for i in ids) / len(ids),
                 len(ids))
    mk0 = sum(v[0] for v in k0.values()) / 4
    print("    K0 (immediate, vllm-mimic) official:")
    for b in BENCHES:
        print(f"      {b:9s} K0={k0[b][0]:.4f} (n={k0[b][1]})")
    print(f"      macro K0={mk0:.4f}")
    print(f"    sweep K0->K1 official macro gain = {m1-mk0:+.4f} "
          f"(mRoPE + timing, CONFOUNDED)")
    print(f"    official H0n(native imm) -> H1(native def) macro = {m0-m1:+.4f} "
          f"(pure deletion timing)")


if __name__ == "__main__":
    main()
