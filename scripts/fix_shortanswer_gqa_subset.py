#!/usr/bin/env python3
"""One-time, idempotent fix: append the standard lmms-eval short-answer
instruction to the `question` of the GQA 200-sample subset.

Sibling of scripts/fix_shortanswer_subsets.py (TextVQA/DocVQA, 2026-07-23):
the runner prompt is the RAW question only, so Qwen3-VL-8B emits verbose
sentences; GQA exact-match (runner score_gqa, word-normalized) then collapses.
Baking in the canonical lmms-eval short-answer post-prompt aligns GQA with the
TextVQA/DocVQA/ChartQA/OCRBench protocol before the method gate (Task 5),
which evaluates GQA under short-answer.

INSTRUCTION (standard lmms-eval gqa doc_to_text post-prompt):
    "Answer the question using a single word or phrase."

BEHAVIOR
--------
* Backs up the pristine original to eval/subsets/_backup/gqa_200.jsonl (only on
  first run; later runs reuse the pristine backup as source -> idempotent).
* Rewrites eval/subsets/gqa_200.jsonl in place: question -> question + suffix.
  id / image / gt (and any other fields) preserved exactly; 200 lines kept.
* Verifies: same ids/image/gt/order, 200 lines, question == orig + suffix.

SCORER NOTE: the runner's GQA scorer (score_gqa) is a word-normalized
exact-match with singular handling + yes/no lead-token logic -- the GQA
convention. No separate normalization needed; rescore uses it unchanged.

New file; touches only eval/subsets/gqa_200.jsonl (+ _backup). No GPU.
"""

from __future__ import annotations

import json
import os
import shutil

REPO = "/media/disk2/YZX/research/vla"
SUBSET_DIR = os.path.join(REPO, "eval", "subsets")
BACKUP_DIR = os.path.join(SUBSET_DIR, "_backup")

INSTRUCTION = "Answer the question using a single word or phrase."
SUFFIX = "\n" + INSTRUCTION

BENCH = "gqa"
EXPECTED_N = 200


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dump_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def fix_one() -> dict:
    src = os.path.join(SUBSET_DIR, f"{BENCH}_200.jsonl")
    bak = os.path.join(BACKUP_DIR, f"{BENCH}_200.jsonl")
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Pristine source: first run copies current file to backup; later runs use
    # the pristine backup (idempotent -> never double-append the suffix).
    if not os.path.exists(bak):
        shutil.copy2(src, bak)
        created_backup = True
    else:
        created_backup = False
    pristine = load_jsonl(bak)

    fixed = []
    for o in pristine:
        q = str(o.get("question", ""))
        if q.endswith(SUFFIX):
            q = q[: -len(SUFFIX)]
        new = dict(o)
        new["question"] = q + SUFFIX
        fixed.append(new)

    dump_jsonl(fixed, src)

    # ---- verification ----
    check = load_jsonl(src)
    assert len(check) == EXPECTED_N, f"{BENCH}: expected {EXPECTED_N} lines, got {len(check)}"
    assert len(check) == len(pristine), f"{BENCH}: line count changed"
    for a, b in zip(pristine, check):
        assert a["id"] == b["id"], f"{BENCH}: id changed ({a['id']} -> {b['id']})"
        assert a.get("image") == b.get("image"), f"{BENCH}: image changed for id {a['id']}"
        assert a.get("gt") == b.get("gt"), f"{BENCH}: gt changed for id {a['id']}"
        orig_q = str(a.get("question", ""))
        if orig_q.endswith(SUFFIX):
            orig_q = orig_q[: -len(SUFFIX)]
        assert b["question"] == orig_q + SUFFIX, f"{BENCH}: question not = orig+suffix (id {a['id']})"
        assert b["question"].endswith(SUFFIX)
    keys_changed = set()
    for a, b in zip(pristine, check):
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                keys_changed.add(k)
    assert keys_changed == {"question"}, f"{BENCH}: unexpected fields changed: {keys_changed}"

    return {
        "bench": BENCH,
        "created_backup": created_backup,
        "n": len(check),
        "ids_match": True,
        "gt_match": True,
        "image_match": True,
        "only_question_changed": True,
        "spot": [
            {"id": check[i]["id"], "question": check[i]["question"]}
            for i in (0, 1, 2)
        ],
    }


def main():
    r = fix_one()
    print(f"[ok] {BENCH}: n={r['n']} backup_created={r['created_backup']} "
          f"ids_match={r['ids_match']} gt_match={r['gt_match']} "
          f"image_match={r['image_match']} only_question_changed={r['only_question_changed']}")
    for s in r["spot"]:
        print(f"      id={s['id']!r}  q={s['question']!r}")
    print("\nGQA SUBSET FIX APPLIED + VERIFIED")
    print(f"INSTRUCTION appended: {INSTRUCTION!r}")


if __name__ == "__main__":
    main()
