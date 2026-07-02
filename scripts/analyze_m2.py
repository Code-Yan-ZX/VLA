#!/usr/bin/env python3
"""Analyze the M2 matrix once all 6 runs complete. Computes the req/s speedup of
r50/r75 over r0 SEPARATELY at concurrency=1 vs concurrency=12, and prints the
load-adaptive verdict."""
import json, os
D = "/media/disk2/YZX/research/vla/runs/p2_d"

def load(name):
    p = os.path.join(D, name + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))

rows = {}
for c in [1, 12]:
    for r, tag in [(0.0, "r0"), (0.50, "r50"), (0.75, "r75")]:
        j = load(f"m2_c{c}_{tag}")
        if j is None:
            print(f"MISSING: m2_c{c}_{tag}")
            continue
        rows[(c, r)] = {
            "req_s": j["agg"]["served_req_s"]["mean"],
            "tok_s": j["agg"]["served_tok_s"]["mean"],
            "wall": j["wall_s"],
            "acc": j["agg"]["accuracy"],
        }

print("\n=== M2 matrix: req/s (wall_s) [acc] ===")
print(f"{'':<14} {'r0':<22} {'r50':<22} {'r75':<22}")
for c in [1, 12]:
    cells = []
    for r in [0.0, 0.50, 0.75]:
        d = rows.get((c, r))
        if d:
            cells.append(f"{d['req_s']:.3f} ({d['wall']:.0f}s) [{d['acc']:.3f}]")
        else:
            cells.append("—")
    print(f"max_num_seqs={c:<2} " + " ".join(f"{c:<22}" for c in cells))

print("\n=== Speedup of r50/r75 over r0, by concurrency ===")
for c in [1, 12]:
    base = rows.get((c, 0.0))
    if not base:
        continue
    for r, tag in [(0.50, "r50"), (0.75, "r75")]:
        d = rows.get((c, r))
        if d:
            sp = d["req_s"] / base["req_s"]
            print(f"  c={c:<2} {tag}: {sp:.3f}x req/s over r0")

# load-adaptive verdict
print("\n=== Load-adaptive verdict ===")
ok = True
for r, tag in [(0.50, "r50"), (0.75, "r75")]:
    s1 = rows.get((1, r)); s12 = rows.get((12, r))
    b1 = rows.get((1, 0.0)); b12 = rows.get((12, 0.0))
    if s1 and s12 and b1 and b12:
        sp1 = s1["req_s"] / b1["req_s"]
        sp12 = s12["req_s"] / b12["req_s"]
        grow = sp12 - sp1
        verdict = "GROWS with concurrency (load-adaptive JUSTIFIED)" if sp12 > sp1 else "does NOT grow (per-request latency only)"
        print(f"  {tag}: speedup@c1={sp1:.3f}x  speedup@c12={sp12:.3f}x  delta={grow:+.3f}  -> {verdict}")
