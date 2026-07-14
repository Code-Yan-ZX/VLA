"""Analyze v3 pre-merger vs post-merger go/no-go matrix."""
import json, os, glob

CELLS = "runs/v3_premerger_cells"
# keep-ratio (fraction of merge-units kept) <-> r
R_TO_KEEP = {"0.500": 0.500, "0.750": 0.250, "0.875": 0.125}


def load(name):
    p = os.path.join(CELLS, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def fmt(d, key):
    return f"{d[key]:.3f}" if d else "  -  "


print("=" * 78)
print("V3 GO/NO-GO: PRE-merger (C) vs POST-merger (B) vs baseline (A)")
print("Qwen3-VL-8B-Instruct, n=200, max_tokens=32, max_num_seqs=16, enforce_eager")
print("=" * 78)

for bench in ["gqa", "textvqa"]:
    A = load(f"A_{bench}.json")
    print(f"\n### {bench.upper()}  (mean_ptid = mean prompt-token-id length)")
    print(f"{'variant':<22}{'keep%':>7}{'r':>7}{'acc':>8}{'req/s':>9}{'ptid':>9}")
    print("-" * 62)
    if A:
        print(f"{'A baseline':<22}{'100%':>7}{'0.000':>7}"
              f"{A['acc']:>8.3f}{A['req_per_s']:>9.3f}{A['mean_ptid_len']:>9.0f}")
    for r, keep in [("0.500", 0.5), ("0.750", 0.25), ("0.875", 0.125)]:
        for mode, label in [("B", "B post-merger"), ("C", "C pre-merger")]:
            d = load(f"{mode}_{bench}_r{r}.json")
            if d:
                print(f"{label:<22}{keep*100:>6.1f}%{d['r']:>7.3f}"
                      f"{d['acc']:>8.3f}{d['req_per_s']:>9.3f}"
                      f"{d['mean_ptid_len']:>9.0f}")

print("\n" + "=" * 78)
print("ISO-TOKEN ACCURACY: (C pre) - (B post)  [positive = pre-merger wins]")
print("=" * 78)
print(f"{'benchmark':<10}{'keep%':>8}{'B_acc':>8}{'C_acc':>8}{'C-B (pp)':>11}{'B_ptid':>8}{'C_ptid':>8}")
print("-" * 65)
decisive = []
for bench in ["gqa", "textvqa"]:
    for r, keep in [("0.500", 0.5), ("0.750", 0.25), ("0.875", 0.125)]:
        B = load(f"B_{bench}_r{r}.json")
        C = load(f"C_{bench}_r{r}.json")
        if not B or not C:
            print(f"{bench:<10}{keep*100:>7.1f}%{'?':>8}{'?':>8}{'?':>11}")
            continue
        diff = (C["acc"] - B["acc"]) * 100  # percentage points
        flag = " <-- DEEP" if keep <= 0.25 else ""
        print(f"{bench:<10}{keep*100:>7.1f}%{B['acc']:>8.3f}{C['acc']:>9.3f}"
              f"{diff:>+10.1f}pp{B['mean_ptid_len']:>8.0f}{C['mean_ptid_len']:>8.0f}{flag}")
        if keep <= 0.25:
            decisive.append((bench, keep, B["acc"], C["acc"], diff))

print("\n" + "=" * 78)
print("DECISIVE DEEP-POINT SUMMARY (keep <= 25%)")
print("=" * 78)
if decisive:
    for bench, keep, ba, ca, diff in decisive:
        win = "C (pre) WINS" if diff > 0 else "B (post) wins" if diff < 0 else "tie"
        print(f"  {bench} keep={keep*100:.1f}%: B={ba:.3f} C={ca:.3f}  C-B={diff:+.1f}pp  -> {win}")
    pre_wins = [d for d in decisive if d[4] > 0]
    avg = sum(d[4] for d in decisive) / len(decisive)
    n_pre_gt = sum(1 for d in decisive if d[4] >= 2.0)
    print(f"\n  deep cells where C >= B+2pp: {n_pre_gt}/{len(decisive)}")
    print(f"  mean(C-B) over deep cells: {avg:+.2f}pp")
    print("\n  DECISION RULE: GO if C beats B by >=2-3pp at deep points,")
    print(f"  especially if gap GROWS as compression deepens.")
    # gap-growth check
    for bench in ["gqa", "textvqa"]:
        ds = [d for d in decisive if d[0] == bench]
        if len(ds) >= 2:
            ds.sort(key=lambda x: x[1])  # by keep asc (deeper last)
            diffs = [d[4] for d in ds]
            grows = diffs[-1] > diffs[0]
            print(f"  {bench}: C-B by depth (keep25%->12.5%): "
                  f"{diffs[0]:+.1f} -> {diffs[-1]:+.1}pp  grows={grows}")
else:
    print("  (no deep cells found)")
