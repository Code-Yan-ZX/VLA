#!/usr/bin/env python
"""Protocol-fingerprint checker for prediction reuse (directive 10).
Usage: cvm24h_fpcheck.py [--ids-vs <split.jsonl>] fp.json pred1.json [pred2.json ...]
Prints scalar fingerprint keys of each prediction and pairwise comparison of
the canonical keys; checks sample-id sets against a split file when given.
"""
import json, sys, argparse

CANON = ["model", "mode", "r", "r_pre", "r_post", "fastv_k", "mrope",
         "max_pixels", "seed", "benchmark", "n", "max_tokens", "engine",
         "harness", "git", "transformers", "torch"]

def scalars(d):
    return {k: v for k, v in d.items()
            if not isinstance(v, (list, dict)) and k != "per_sample"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-vs", default=None, help="split jsonl with 'id' per line")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    fps = []
    for p in a.paths:
        d = json.load(open(p))
        s = scalars(d)
        ids = [x.get("id") for x in d.get("per_sample", [])]
        fps.append((p, s, ids))
        print(f"\n== {p.split('/')[-1]}  n_records={len(ids)}")
        for k in sorted(s):
            if k in ("acc", "correct", "score", "wall_s", "req_per_s",
                     "mean_ptid_len", "n_scored", "n_skipped"):
                continue
            print(f"   {k} = {s[k]}")
    if len(fps) > 1:
        base = fps[0]
        for p, s, ids in fps[1:]:
            diffs = {k: (base[1].get(k), s.get(k)) for k in set(base[1]) | set(s)
                     if base[1].get(k) != s.get(k)
                     and (k in CANON or k.startswith(("prompt", "tap", "order")))}
            same_ids = ids == base[2]
            print(f"\n[compare] {base[0].split('/')[-1]} vs {p.split('/')[-1]}: "
                  f"canon_diffs={diffs or 'NONE'} ids_identical_order={same_ids}")
    if a.ids_vs:
        want = [json.loads(l)["id"] for l in open(a.ids_vs)]
        for p, s, ids in fps:
            print(f"[ids] {p.split('/')[-1]}: match_prefix={ids == want[:len(ids)]}, "
                  f"count={len(ids)}/{len(want)}")

main()
