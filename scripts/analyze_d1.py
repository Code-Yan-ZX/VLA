#!/usr/bin/env python3
"""Analyze D1 learned-scorer GPU validation: paired per-sample comparison of
learned vs PRE-L2 on the same held-out slice.

Reads runs/d1_learned/{bench}_r{R}_{learned,l2}_heldout200.json (per_sample
with id/correct). Reports per-benchmark:
  - acc learned vs l2, delta + matched-pairs (only samples both answered)
  - exact same-set agreement
"""
from __future__ import annotations

import glob
import json
import sys

sys.path.insert(0, "src/v3_premerger")
import official_scorers as S  # noqa: E402


def load(path):
    d = json.load(open(path))
    ps = d.get("per_sample") or []
    return {str(p.get("id")): (p.get("answer", ""), p.get("gt", ""),
                               p.get("skipped", False))
            for p in ps}, d.get("acc"), d.get("n_answered")


def _ok(b, pred, gt):
    if b == "textvqa":
        return S.score_textvqa_vqaacc(pred, gt) == 1
    if b == "docvqa":
        return S.score_docvqa_anls(pred, gt) >= 0.5
    if b == "gqa":
        return S.score_gqa(pred, gt) == 1
    raise ValueError(b)


def main():
    rr = sys.argv[1] if len(sys.argv) > 1 else "0.75"
    for b in ["textvqa", "docvqa", "gqa"]:
        lpath = f"runs/d1_learned/{b}_r{rr}_learned_heldout200.json"
        kpath = f"runs/d1_learned/{b}_r{rr}_l2_heldout200.json"
        try:
            l, lacc, ln = load(lpath)
            k, kacc, kn = load(kpath)
        except FileNotFoundError as e:
            print(f"[{b}] MISSING {e.filename}")
            continue
        common = [i for i in l if i in k and not l[i][2] and not k[i][2]]
        l_c = sum(1 for i in common if _ok(b, l[i][0], l[i][1]))
        k_c = sum(1 for i in common if _ok(b, k[i][0], k[i][1]))
        n = len(common)
        acc_l, acc_k = l_c / n, k_c / n
        # matched pairs
        better = sum(1 for i in common
                     if _ok(b, l[i][0], l[i][1])
                     and not _ok(b, k[i][0], k[i][1]))
        worse = sum(1 for i in common
                    if _ok(b, k[i][0], k[i][1])
                    and not _ok(b, l[i][0], l[i][1]))
        print(f"[{b}] r={rr} n_common={n}")
        print(f"   acc learned={acc_l:.4f}  l2={acc_k:.4f}  "
              f"delta={acc_l-acc_k:+.4f}  (learned better {better}, worse "
              f"{worse}, tie {n-better-worse})")


if __name__ == "__main__":
    main()
