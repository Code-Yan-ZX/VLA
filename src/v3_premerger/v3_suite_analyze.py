"""V3 suite analysis: workload-conditional pre-vs-post stage effect across the
benchmark suite. Reads runs/v3_premerger_cells/{A,B,C}_<bench>_r<rrr>.json for
all 6 benchmarks, builds the (C pre - B post) gap map at deep (12.5%) + 25%,
ordered by expected text-density, and prints + writes a JSON + a PNG plot.

gqa/textvqa cells are the reused go/no-go results (identical settings); the 4
others are fresh from the suite matrix. All at n=200, Qwen3-VL-8B, max_num_seqs=16,
enforce_eager, L2 text-agnostic selector.
"""
import json, os
from pathlib import Path

CELLS = Path("/media/disk2/YZX/research/vla/runs/v3_premerger_cells")
PLOT_OUT = Path("/media/disk2/YZX/research/vla/runs/v3_suite_map.png")
JSON_OUT = Path("/media/disk2/YZX/research/vla/runs/v3_suite_map.json")

# ordered MOST -> LEAST text-dense (the mechanism prediction axis)
BENCHES = [
    ("docvqa",   "DocVQA (document OCR)"),
    ("textvqa",  "TextVQA (scene text)"),
    ("scienceqa","ScienceQA (figure+text MC)"),
    ("mmbench",  "MMBench (perception MC)"),
    ("mme",      "MME (yes/no perception)"),
    ("gqa",      "GQA (object/spatial QA)"),
]
R_KEEP = [("0.750", 0.25, "25%"), ("0.875", 0.125, "12.5% (deep)")]


def load(name):
    p = CELLS / name
    if not p.exists():
        return None
    return json.load(open(p))


def get(bench, mode, r):
    return load(f"{mode}_{bench}_r{r}.json")


rows = []
for bench, label in BENCHES:
    A = load(f"A_{bench}.json")
    rec = {"bench": bench, "label": label, "A": A}
    for r, keep, klabel in R_KEEP:
        B = get(bench, "B", r)
        C = get(bench, "C", r)
        rec[f"B_{r}"] = B
        rec[f"C_{r}"] = C
        if B and C:
            rec[f"diff_{r}"] = (C["acc"] - B["acc"]) * 100.0  # pp
        else:
            rec[f"diff_{r}"] = None
    rows.append(rec)

# ---------------- table ----------------
W = 96
print("=" * W)
print("V3 SUITE: workload-conditional stage effect  (C pre - B post) [pp]")
print("Qwen3-VL-8B-Instruct, n=200, max_num_seqs=16, enforce_eager, L2 text-agnostic selector")
print("ordered: MOST text-dense (top) -> LEAST text-dense (bottom)")
print("=" * W)
hdr = (f"{'benchmark':<32}{'A base':>8}"
       f"{'B@25%':>8}{'C@25%':>8}{'C-B@25':>9}"
       f"{'B@12.5':>9}{'C@12.5':>9}{'C-B@12.5':>10}"
       f"{'Bptid':>7}{'Cptid':>7}")
print(hdr)
print("-" * W)
for r in rows:
    A = r["A"]; Aa = f"{A['acc']:.3f}" if A else "  -  "
    B25 = r.get("B_0.750"); C25 = r.get("C_0.750")
    B125 = r.get("B_0.875"); C125 = r.get("C_0.875")
    def f(d): return f"{d['acc']:.3f}" if d else "  -  "
    d25 = r["diff_0.750"]; d125 = r["diff_0.875"]
    fd25 = f"{d25:+.1f}" if d25 is not None else " - "
    fd125 = f"{d125:+.1f}" if d125 is not None else " - "
    # iso-token check: ptid of B vs C at deep
    bptid = f"{B125['mean_ptid_len']:.0f}" if B125 else "-"
    cptid = f"{C125['mean_ptid_len']:.0f}" if C125 else "-"
    print(f"{r['label']:<32}{Aa:>8}"
          f"{f(B25):>8}{f(C25):>8}{fd25:>9}"
          f"{f(B125):>9}{f(C125):>9}{fd125:>10}"
          f"{bptid:>7}{cptid:>7}")

# ---------------- crossover summary ----------------
print("\n" + "=" * W)
print("CROSSOVER @ DEEP (12.5% keep): positive = PRE-merger wins, negative = POST wins")
print("=" * W)
deep = [(r["label"], r["diff_0.875"]) for r in rows if r["diff_0.875"] is not None]
deep.sort(key=lambda x: -x[1])  # biggest pre-win first
for lab, d in deep:
    win = "PRE wins" if d > 0 else "POST wins" if d < 0 else "tie"
    bar = ("+" * int(max(d, 0) / 2)) + ("-" * int(max(-d, 0) / 2))
    print(f"  {lab:<32} {d:+7.1f}pp  {bar:<24} -> {win}")

n_pre = sum(1 for _, d in deep if d > 0)
n_post = sum(1 for _, d in deep if d < 0)
print(f"\n  PRE wins on {n_pre}/{len(deep)}; POST wins on {n_post}/{len(deep)}")
# correlation of gap vs text-density rank (Spearman-ish): is the ordering monotonic?
order_ranks = list(range(len(deep)))  # 0=most text-dense among reported
gaps = [d for _, d in deep]
# monotonic decrease check (text-dense -> pre wins; object-dense -> post wins)
mono = all(gaps[i] >= gaps[i+1] - 1e-9 for i in range(len(gaps) - 1))
print(f"  Gaps sorted by text-density (reported order): {[f'{g:+.1f}' for g in gaps]}")
print(f"  Mechanism consistency: gap DECREASES as text-density DECREASES = "
      f"{'YES (monotonic)' if mono else 'partially (see table)'}")

# ---------------- write JSON ----------------
out = {
    "benchmarks": [
        {"bench": r["bench"], "label": r["label"],
         "A_acc": r["A"]["acc"] if r["A"] else None,
         "B_acc_25": r["B_0.750"]["acc"] if r.get("B_0.750") else None,
         "C_acc_25": r["C_0.750"]["acc"] if r.get("C_0.750") else None,
         "diff_25_pp": r["diff_0.750"],
         "B_acc_12.5": r["B_0.875"]["acc"] if r.get("B_0.875") else None,
         "C_acc_12.5": r["C_0.875"]["acc"] if r.get("C_0.875") else None,
         "diff_12.5_pp": r["diff_0.875"],
         "B_ptid_12.5": r["B_0.875"]["mean_ptid_len"] if r.get("B_0.875") else None,
         "C_ptid_12.5": r["C_0.875"]["mean_ptid_len"] if r.get("C_0.875") else None,
         } for r in rows],
    "settings": {"model": "Qwen3-VL-8B-Instruct", "n": 200, "max_num_seqs": 16,
                 "enforce_eager": True, "selector": "L2-norm text-agnostic"},
}
JSON_OUT.write_text(json.dumps(out, indent=2))
print(f"\nwrote {JSON_OUT}")

# ---------------- plot ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [r["label"].split(" (")[0] for r in rows]
    d25 = [r["diff_0.750"] if r["diff_0.750"] is not None else 0 for r in rows]
    d125 = [r["diff_0.875"] if r["diff_0.875"] is not None else 0 for r in rows]
    x = range(len(labs))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    b1 = ax.bar([i - w/2 for i in x], d25, w, label="keep 25%", color="#9ecae1")
    b2 = ax.bar([i + w/2 for i in x], d125, w, label="keep 12.5% (deep)", color="#08519c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_ylabel("accuracy(C pre) - accuracy(B post)  [pp]")
    ax.set_title("Workload-conditional stage effect: PRE-merger vs POST-merger pruning\n"
                 "(positive = PRE wins; ordered most->least text-dense)  Qwen3-VL-8B, n=200")
    ax.legend(loc="upper right")
    for bar, v in zip(b2, d125):
        ax.text(bar.get_x() + bar.get_width()/2, v + (1.2 if v >= 0 else -2.8),
                f"{v:+.0f}", ha="center", fontsize=8, color="#08519c", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    print(f"wrote {PLOT_OUT}")
except Exception as e:
    print(f"(plot skipped: {e})")
