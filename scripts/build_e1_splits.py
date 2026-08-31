#!/usr/bin/env python
"""Q-RBM E1 — deterministic held-out split builder (pre-registered).

Builds eval/subsets/e1_heldout_{bench}_64.jsonl for the 4 E1 benchmarks:
  textvqa, docvqa, gqa, ocrbench
each 64 samples, STRICTLY DISJOINT from BOTH
  - the official gate subsets  eval/subsets/{bench}_200.jsonl
  - the E1 dev/gate set       runs/merger_repr/dev_{bench}_64.jsonl
so that the held-out set has never been touched by any prior selection,
tuning, or gate.

Also verifies (and, if missing, deterministically rebuilds) the dev_64 files
used as the E1 development set, and writes an audit JSON
  runs/e1/splits_audit.json
recording the exact id sets and all pairwise disjointness checks.

Selection is the j5-pattern: random.Random(f"E1-{bench}").shuffle over the
disjoint pool, take 64. Same schema as the gate subsets (id/image/question/gt),
so it plugs straight into the existing harness loader (baselines_hf.Sample).

Usage:
  /home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/build_e1_splits.py
"""
import json
import os
import random

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SUBSETS = os.path.join(REPO, "eval/subsets")
FULL = os.path.join(REPO, "eval/full_splits")
DEV64 = os.path.join(REPO, "runs/merger_repr")
OUT = os.path.join(REPO, "runs/e1")

BENCHES = ["textvqa", "docvqa", "gqa", "ocrbench"]
N = 64
SEED_TAG = "E1-{bench}"

FULL_SPLITS = {
    "textvqa": "textvqa_val.jsonl",
    "docvqa": "docvqa_val.jsonl",
    "gqa": "gqa_testdev.jsonl",
    "ocrbench": "ocrbench.jsonl",
}


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _ids(rows):
    return set(str(o["id"]) for o in rows)


def build():
    os.makedirs(OUT, exist_ok=True)
    audit = {}
    for b in BENCHES:
        full = _read_jsonl(os.path.join(FULL, FULL_SPLITS[b]))
        gate = _read_jsonl(os.path.join(SUBSETS, f"{b}_200.jsonl"))
        gate_ids = _ids(gate)

        dev_path = os.path.join(DEV64, f"dev_{b}_64.jsonl")
        if os.path.exists(dev_path):
            dev = _read_jsonl(dev_path)
            dev_ids = _ids(dev)
            dev_note = "existing"
        else:
            # deterministic fallback (j5-pattern) — only if the dev_64 is gone
            pool = [o for o in full if str(o["id"]) not in gate_ids]
            rng = random.Random(f"dev-{b}")
            rng.shuffle(pool)
            dev = pool[:N]
            dev_ids = _ids(dev)
            os.makedirs(os.path.dirname(dev_path), exist_ok=True)
            with open(dev_path, "w") as f:
                for o in dev:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            dev_note = "REBUILT (fallback)"

        # held-out: disjoint from gate_200 AND dev_64
        pool = [o for o in full
                if str(o["id"]) not in gate_ids and str(o["id"]) not in dev_ids]
        rng = random.Random(SEED_TAG.format(bench=b))
        rng.shuffle(pool)
        chosen = pool[:N]
        held_path = os.path.join(SUBSETS, f"e1_heldout_{b}_64.jsonl")
        with open(held_path, "w") as f:
            for o in chosen:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        held_ids = _ids(chosen)

        audit[b] = {
            "full": len(full),
            "gate_200": len(gate_ids),
            "dev_64": len(dev_ids),
            "dev_note": dev_note,
            "heldout_64": len(held_ids),
            "pool_after_exclusion": len(pool),
            "overlap_dev_gate200": len(dev_ids & gate_ids),
            "overlap_heldout_gate200": len(held_ids & gate_ids),
            "overlap_heldout_dev64": len(held_ids & dev_ids),
            "heldout_path": held_path,
            "seed_tag": SEED_TAG.format(bench=b),
        }
        print(f"[e1-splits] {b}: full={len(full)} gate200={len(gate_ids)} "
              f"dev64={len(dev_ids)} ({dev_note}) "
              f"heldout={len(held_ids)} (dev∩gate={audit[b]['overlap_dev_gate200']}, "
              f"ho∩gate={audit[b]['overlap_heldout_gate200']}, "
              f"ho∩dev={audit[b]['overlap_heldout_dev64']}) -> {held_path}")

    with open(os.path.join(OUT, "splits_audit.json"), "w") as f:
        json.dump({"seed_tag": SEED_TAG, "n": N, "benches": BENCHES,
                   "detail": audit}, f, indent=2, ensure_ascii=False)
    print(f"[e1-splits] audit -> {os.path.join(OUT, 'splits_audit.json')}")
    bad = [b for b, a in audit.items()
           if a["overlap_heldout_gate200"] or a["overlap_heldout_dev64"]
           or a["overlap_dev_gate200"] or a["heldout_64"] != N]
    if bad:
        print(f"[e1-splits][FAIL] disjointness/size violated for: {bad}")
        return 1
    print("[e1-splits] ALL disjointness/size checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
