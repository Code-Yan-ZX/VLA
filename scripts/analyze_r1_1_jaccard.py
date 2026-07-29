#!/usr/bin/env python3
"""R1-1 — pre-vs-swap kept-set Jaccard analysis (+ official metric deltas).

Reads runs/r1_1_swap_jaccard/r1_1_{family}_{mode}_{bench}_n32.json (8 cells),
computes per-sample Jaccard(pre_kept, swap_kept) over samples attached in BOTH
runs (kept_fallback entries excluded), distribution stats + random-chance level
(mean k/f), official scores (TextVQA VQA-acc, DocVQA ANLS) and swap-pre answer
identity. Writes runs/r1_1_swap_jaccard/jaccard_summary.json.

Interpretation: Jaccard ~ 1 (with different answers) -> value-level difference
(merger output depends on batch/subset context); Jaccard ~ k/f (chance) ->
swap's ranking capture is misaligned (ordering artifact).
"""
import json
import sys

sys.path.insert(0, "src/v3_premerger")
import official_scorers as S

OUT = "runs/r1_1_swap_jaccard"


def load(tag):
    try:
        with open(f"{OUT}/{tag}.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [missing] {tag}: {type(e).__name__}: {e}")
        return None


def official(d):
    ps = d.get("per_sample") or []
    b = d["benchmark"]
    preds = [str(p.get("answer", "")) for p in ps]
    gts = [str(p.get("gt", "")) for p in ps]
    if b == "textvqa":
        s = sum(S.score_textvqa_vqaacc(a, g) for a, g in zip(preds, gts))
    elif b == "docvqa":
        s = sum(S.score_docvqa_anls(a, g) for a, g in zip(preds, gts))
    else:
        return None
    return s / max(len(ps), 1)


def kept_map(d):
    m = {}
    for p in d.get("per_sample") or []:
        k = p.get("kept_indices")
        if k is not None and not p.get("kept_fallback"):
            m[p["id"]] = set(k)
    return m


def quant(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


summary = {}
for fam in ["qwen3vl", "qwen2vl"]:
    for b in ["docvqa", "textvqa"]:
        pre = load(f"r1_1_{fam}_pre_{b}_n32")
        swap = load(f"r1_1_{fam}_swap_{b}_n32")
        cell = {}
        if pre and swap:
            kp, ks = kept_map(pre), kept_map(swap)
            common = sorted(set(kp) & set(ks))
            if common:
                jac = [len(kp[i] & ks[i]) / max(len(kp[i] | ks[i]), 1)
                       for i in common]
                fup = {p["id"]: p.get("kept_n_units")
                       for p in pre["per_sample"]}
                fkk = {p["id"]: p.get("kept_k") for p in pre["per_sample"]}
                chance = [(fkk.get(i) or 0) / max(fup.get(i) or 1, 1)
                          for i in common]
                cell.update({
                    "n_common": len(common),
                    "n_attached_pre": len(kp),
                    "n_attached_swap": len(ks),
                    "jaccard_mean": round(sum(jac) / len(jac), 4),
                    "jaccard_min": round(min(jac), 4),
                    "jaccard_p25": round(quant(jac, 0.25), 4),
                    "jaccard_med": round(quant(jac, 0.50), 4),
                    "jaccard_p75": round(quant(jac, 0.75), 4),
                    "jaccard_max": round(max(jac), 4),
                    "frac_eq_1": round(sum(1 for j in jac if j >= 0.9999)
                                       / len(jac), 4),
                    "chance_mean_k_over_f": round(sum(chance) / len(chance),
                                                  4),
                })
            po, so = official(pre), official(swap)
            pa = {p["id"]: str(p.get("answer", ""))
                  for p in pre.get("per_sample", [])}
            sa = {p["id"]: str(p.get("answer", ""))
                  for p in swap.get("per_sample", [])}
            ids = sorted(set(pa) & set(sa))
            cell["acc_pre_official"] = round(po, 4) if po is not None else None
            cell["acc_swap_official"] = round(so, 4) if so is not None else None
            cell["delta_swap_minus_pre"] = (round(so - po, 4)
                                            if (po is not None and so is not None)
                                            else None)
            cell["answer_identity"] = (
                f"{sum(1 for i in ids if pa[i] == sa[i])}/{len(ids)}")
            dp, ds = pre.get("diag", {}), swap.get("diag", {})
            cell["diag_pre"] = {k: dp.get(k) for k in
                                ("kept_attached", "kept_total",
                                 "mask_computed_at", "mask_compute_count")}
            cell["diag_swap"] = {k: ds.get(k) for k in
                                 ("kept_attached", "kept_total",
                                  "swap_queue_leftover", "fallback_stage",
                                  "consumed", "score_passes")}
        summary[f"{fam}/{b}"] = cell
        print(f"--- {fam} / {b} ---")
        print(json.dumps(cell, indent=2))

with open(f"{OUT}/jaccard_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("summary written ->", f"{OUT}/jaccard_summary.json")
