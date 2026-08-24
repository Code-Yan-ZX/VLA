#!/usr/bin/env python3
"""Build the MALT goal-mode Phase-1 EXPLORATION manifest (task 2026-08-25).

Non-overlapping with the locked n=200 benchmark manifest, shared by all
subsequent exploration runs (fixed deterministic procedure, no RNG).

  * pool        = full split rows whose id is NOT in the locked {bench}_200.jsonl
  * ordering    = pool sorted by (id) -- fully deterministic, auditable
  * sample      = 64 rows at even stride across the sorted pool (covers the
                  whole difficulty range; ids are near-random w.r.t. difficulty)
  * output      = eval/subsets/{bench}_explore64.jsonl  (verbatim row format)

Benchmarks / full splits (all verified: every candidate image exists locally):
  textvqa -> eval/full_splits/textvqa_val.jsonl
  docvqa  -> eval/full_splits/docvqa_val.jsonl
  gqa     -> eval/full_splits/gqa_testdev.jsonl
  ocrbench-> eval/full_splits/ocrbench.jsonl
"""
import json
import os

ROOT = "/media/disk2/YZX/research/vla"
BENCHES = {
    "textvqa": "textvqa_val.jsonl",
    "docvqa": "docvqa_val.jsonl",
    "gqa": "gqa_testdev.jsonl",
    "ocrbench": "ocrbench.jsonl",
}
N = 64


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    for bench, full_name in BENCHES.items():
        lock_path = os.path.join(ROOT, "eval", "subsets", f"{bench}_200.jsonl")
        full_path = os.path.join(ROOT, "eval", "full_splits", full_name)
        lock_ids = {r["id"] for r in load_rows(lock_path)}
        pool = [r for r in load_rows(full_path) if r["id"] not in lock_ids]
        assert len(lock_ids) == 200, (bench, len(lock_ids))
        assert len(pool) >= N, (bench, len(pool))
        # deterministic even-stride sample over the id-sorted pool
        pool_sorted = sorted(pool, key=lambda r: r["id"])
        idxs = [round(i * (len(pool_sorted) - 1) / (N - 1)) for i in range(N)]
        assert len(set(idxs)) == N, (bench, idxs)
        pick = [pool_sorted[i] for i in idxs]
        for r in pick:
            assert os.path.exists(r["image"]), (bench, r["id"], r["image"])
        out_path = os.path.join(ROOT, "eval", "subsets", f"{bench}_explore64.jsonl")
        with open(out_path, "w") as f:
            for r in pick:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] {bench}: pool={len(pool)} lock={len(lock_ids)} -> "
              f"{len(pick)} explore ids {pick[0]['id']}..{pick[-1]['id']} -> {out_path}")


if __name__ == "__main__":
    main()
