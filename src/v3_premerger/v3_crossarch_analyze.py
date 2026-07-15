"""V3 cross-architecture analysis: does the workload-conditional pre-vs-post
stage-effect SIGN reproduce on Qwen2.5-VL-7B-Instruct?

Reads runs/v3_crossarch_cells/{A,B,C}_{bench}_r0.750_qwen2vl.json for TextVQA +
GQA, computes C-B (pre - post) per benchmark, and tests the DECISIVE question:

  TextVQA (text-dense): C-B > 0  (pre wins -- same sign as Qwen3-VL)
  GQA     (object):     C-B < 0  (post wins -- same sign as Qwen3-VL)

Both signs reproduce => the stage effect is architecture-general (not a
Qwen3-VL-specific artifact of its deepstack path). Writes _summary.json.

Qwen3-VL reference (@12.5% keep, deep point): TextVQA C-B=+44pp, GQA C-B=-5.5pp.
Here we run @0.75 (25% keep) for a faster cross-arch check; the SIGN is the
claim, not the magnitude.
"""
import json
from pathlib import Path

CELLS = Path("/media/disk2/YZX/research/vla/runs/v3_crossarch_cells")
JSON_OUT = CELLS / "_summary.json"
R = "0.750"          # prune-ratio string used in filenames
FAM = "qwen2vl"

# (bench, expected C-B sign, Qwen3-VL reference for context)
BENCHES = [
    ("textvqa", "positive",
     "Qwen3-VL: C-B=+44pp @12.5%keep / +24pp @50%keep (pre wins; text-dense)"),
    ("gqa", "negative",
     "Qwen3-VL: C-B=-5.5pp @12.5%keep (post wins; object)"),
]


def load(mode, bench):
    p = CELLS / f"{mode}_{bench}_r{R}_{FAM}.json"
    return json.load(open(p)) if p.exists() else None


rows = []
for bench, expect_sign, ref in BENCHES:
    A = load("A", bench); B = load("B", bench); C = load("C", bench)
    rec = {"bench": bench,
           "expected_sign": expect_sign, "qwen3vl_reference": ref,
           "A_acc": A["acc"] if A else None,
           "B_acc": B["acc"] if B else None,
           "C_acc": C["acc"] if C else None,
           "B_ptid": B["mean_ptid_len"] if B else None,
           "C_ptid": C["mean_ptid_len"] if C else None}
    if B and C:
        diff = (C["acc"] - B["acc"]) * 100.0
        rec["diff_CB_pp"] = diff
        obs = "positive" if diff > 0 else "negative" if diff < 0 else "tie"
        rec["observed_sign"] = obs
        rec["sign_reproduced"] = (obs == expect_sign)
    else:
        rec["diff_CB_pp"] = None
        rec["observed_sign"] = "missing"
        rec["sign_reproduced"] = None
    rows.append(rec)

# ---------------- table ----------------
W = 96
print("=" * W)
print(f"V3 CROSS-ARCH: stage-effect SIGN reproduction on Qwen2.5-VL-7B  "
      f"(r={R} == keep 25%, L2 text-agnostic selector)")
print("DECISIVE: TextVQA C-B>0 (pre wins) AND GQA C-B<0 (post wins) "
      "=> sign reproduces => architecture-general")
print("=" * W)
hdr = (f"{'benchmark':<10}{'A base':>9}{'B post':>9}{'C pre':>9}"
       f"{'C-B[pp]':>10}{'expected':>11}{'observed':>11}{'repro?':>9}")
print(hdr)
print("-" * W)
n_repro = 0
n_decided = 0
for r in rows:
    fa = f"{r['A_acc']:.3f}" if r["A_acc"] is not None else "  -"
    fb = f"{r['B_acc']:.3f}" if r["B_acc"] is not None else "  -"
    fc = f"{r['C_acc']:.3f}" if r["C_acc"] is not None else "  -"
    fd = f"{r['diff_CB_pp']:+.1f}" if r["diff_CB_pp"] is not None else " -"
    fr = ("YES" if r["sign_reproduced"] else "NO") \
        if r["sign_reproduced"] is not None else "?"
    if r["sign_reproduced"] is True:
        n_repro += 1
    if r["sign_reproduced"] is not None:
        n_decided += 1
    print(f"{r['bench']:<10}{fa:>9}{fb:>9}{fc:>9}{fd:>10}"
          f"{r['expected_sign']:>11}{r['observed_sign']:>11}{fr:>9}")
print("-" * W)

if n_decided == 2 and n_repro == 2:
    verdict = "SIGN REPRODUCED ON BOTH -> architecture-general stage effect"
elif n_decided == 2 and n_repro == 1:
    verdict = "PARTIAL reproduction (one sign flipped)"
elif n_decided == 2 and n_repro == 0:
    verdict = "NO reproduction (both signs flipped -- effect NOT architecture-general)"
else:
    verdict = "insufficient data (some cells missing/failed)"
print(f"verdict: {verdict}  ({n_repro}/{n_decided} signs reproduced)")

# ---------------- iso-token sanity (B vs C mean_ptid should be close) --------
print("\niso-token check (B vs C mean_ptid_len; large delta => bug in placeholder patch):")
for r in rows:
    if r["B_ptid"] is not None and r["C_ptid"] is not None:
        print(f"  {r['bench']:<10} B_ptid={r['B_ptid']:.0f}  "
              f"C_ptid={r['C_ptid']:.0f}  delta={r['C_ptid']-r['B_ptid']:+.1f}")

# ---------------- write _summary.json ----------------
summary = {
    "model_family": FAM, "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "r": float(R), "keep_fraction": 1.0 - float(R), "selector": "l2",
    "n_signs_reproduced": n_repro, "n_decided": n_decided,
    "verdict": verdict,
    "rows": rows,
}
JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(summary, indent=2))
print(f"\nwrote {JSON_OUT}")
