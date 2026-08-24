#!/usr/bin/env python
"""Lifetime-sweep figures: accuracy-vs-K curve and accuracy-vs-compute Pareto."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/media/disk2/YZX/research/vla"
a = json.load(open(f"{HERE}/experiments/deferred_rbm_n200_data/sweep_analysis.json"))
KS = [0, 1, 3, 5, 8]
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
MARK = {0: "o", 1: "s", 3: "^", 5: "D", 8: "v"}

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, b in zip(axes.ravel(), BENCHES):
    acc = [a["mean_tbl"][f"{b}/K{k}"] for k in KS]
    comp = [a["efficiency"][b][str(k)]["compute_sum_Nl2"] for k in KS]
    # parents
    for tag, m, c in (("RBM(K=0)", "o", "tab:gray"),
                      ("FastV-K3", "^", "tab:orange"),
                      ("Full", "*", "tab:green")):
        if tag == "RBM(K=0)":
            ax.scatter(comp[0], acc[0], marker=m, color=c, s=70, label=tag, zorder=5)
            continue
        ac = a["mean_tbl"][f"{b}/{tag.split('-')[0].lower()}"]
        cc = comp[KS.index(3)] if "FastV" in tag else None
        if tag == "FastV-K3":
            cc = a["efficiency"][b][str(3)]["compute_sum_Nl2"]
            ac = a["mean_tbl"][f"{b}/fastv"]
        if tag == "Full":
            cc = a["efficiency"][b][str(8)]["compute_sum_Nl2"] * (36 / (8 + 1 + 27 * (1 / 4)))
            ac = a["mean_tbl"][f"{b}/full"]
        ax.scatter(cc, ac, marker=m, color=c, s=90, label=tag, zorder=5)
    ax.plot(comp, acc, "-", color="tab:blue", alpha=0.5, zorder=3)
    for k, x, y in zip(KS, comp, acc):
        ax.scatter(x, y, marker=MARK[k], color="tab:blue", s=60, zorder=4)
        ax.annotate(f"K={k}", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.set_title(f"{b}  (n=200)", fontsize=11)
    ax.set_xlabel("attention compute proxy  Σ N_l²")
    ax.set_ylabel("accuracy (official rescore)")
    ax.grid(alpha=0.25)
    if b == "ocrbench":
        ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Deferred-RBM fixed-lifetime K sweep: accuracy vs compute", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(f"{HERE}/experiments/deferred_rbm_n200_data", exist_ok=True)
fig.savefig(f"{HERE}/experiments/deferred_rbm_n200_data/sweep_pareto.png", dpi=130)
print("saved sweep_pareto.png")

# accuracy-vs-K curve
fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
for b in BENCHES:
    acc = [a["mean_tbl"][f"{b}/K{k}"] for k in KS]
    ax2.plot(KS, acc, "-o", label=b, lw=2)
    ax2.annotate(f"FastV {a['mean_tbl'][f'{b}/fastv']:.3f}",
                 (7.3, a["mean_tbl"][f"{b}/fastv"]), fontsize=8, color="gray")
ax2.set_xlabel("deletion layer K")
ax2.set_ylabel("accuracy (official rescore)")
ax2.set_xticks(KS)
ax2.set_title("Deferred-RBM accuracy vs lifetime K (dashed = FastV reference)")
ax2.grid(alpha=0.25)
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{HERE}/experiments/deferred_rbm_n200_data/sweep_accuracy_vs_K.png", dpi=130)
print("saved sweep_accuracy_vs_K.png")
