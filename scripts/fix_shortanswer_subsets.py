#!/usr/bin/env python3
"""One-time, idempotent fix: append the standard lmms-eval short-answer
instruction to the `question` of the TextVQA / DocVQA 200-sample subsets.

WHY
---
The v3 runner (src/v3_premerger/v3_premerger_runner.py:~980) builds the chat
prompt as the RAW question only, with NO short-answer constraint. Qwen3-VL-8B
therefore emits verbose multi-sentence answers (median ~132 chars), which makes
the *official* metrics (TextVQA VQA-accuracy, DocVQA ANLS) collapse to ~0: both
require the normalized prediction to (near-)exactly equal a SHORT gt answer.
The ChartQA/OCRBench subsets already bake a single-word instruction into each
question (scripts/build_chartqa_ocrbench.py:134) and score fine; TextVQA/DocVQA
did not. This aligns them with the canonical lmms-eval prompt.

INSTRUCTION (verified standard lmms-eval doc_to_text post-prompt for BOTH
textvqa and docvqa):
    "Answer the question using a single word or phrase."

BEHAVIOR
--------
* Backs up the pristine original to eval/subsets/_backup/<bench>_200.jsonl
  (only on first run; subsequent runs reuse that pristine copy as the source).
* Rewrites eval/subsets/<bench>_200.jsonl in place: question -> question +
  "\\n" + INSTRUCTION.  id / image / gt (and any other fields) are preserved
  exactly; only `question` changes.  Keeps exactly 200 lines per file.
* Verifies: same ids, same image, same gt, same ordering, 200 lines, and that
  each new question is exactly the original question + the appended suffix.

New file only; touches only the two TextVQA/DocVQA subset files (+_backup).
No GPU, no network.
"""

from __future__ import annotations

import json
import os
import shutil

REPO = "/media/disk2/YZX/research/vla"
SUBSET_DIR = os.path.join(REPO, "eval", "subsets")
BACKUP_DIR = os.path.join(SUBSET_DIR, "_backup")

# Verified standard lmms-eval short-answer post-prompt for textvqa AND docvqa.
INSTRUCTION = "Answer the question using a single word or phrase."
SUFFIX = "\n" + INSTRUCTION

BENCHES = ["textvqa", "docvqa"]
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


def fix_one(bench: str) -> dict:
    src = os.path.join(SUBSET_DIR, f"{bench}_200.jsonl")
    bak = os.path.join(BACKUP_DIR, f"{bench}_200.jsonl")
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Establish the pristine source. First run: copy current file to backup and
    # treat it as canonical. Later runs: backup already holds the pristine
    # original, so use it (idempotent -> never double-append the suffix).
    if not os.path.exists(bak):
        shutil.copy2(src, bak)
        created_backup = True
    else:
        created_backup = False
    pristine = load_jsonl(bak)

    # If a prior run already wrote the fixed file but backup is missing, the
    # backup would be the already-fixed text. Guard: strip a trailing suffix
    # from the pristine question so we always append exactly once.
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
    assert len(check) == EXPECTED_N, f"{bench}: expected {EXPECTED_N} lines, got {len(check)}"
    assert len(check) == len(pristine), f"{bench}: line count changed"
    for a, b in zip(pristine, check):
        assert a["id"] == b["id"], f"{bench}: id changed ({a['id']} -> {b['id']})"
        assert a.get("image") == b.get("image"), f"{bench}: image changed for id {a['id']}"
        assert a.get("gt") == b.get("gt"), f"{bench}: gt changed for id {a['id']}"
        orig_q = str(a.get("question", ""))
        if orig_q.endswith(SUFFIX):
            orig_q = orig_q[: -len(SUFFIX)]
        assert b["question"] == orig_q + SUFFIX, f"{bench}: question not = orig+suffix (id {a['id']})"
        assert b["question"].endswith(SUFFIX)
    # only the question field differs
    keys_changed = set()
    for a, b in zip(pristine, check):
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                keys_changed.add(k)
    assert keys_changed == {"question"}, f"{bench}: unexpected fields changed: {keys_changed}"

    return {
        "bench": bench,
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
    report = []
    for b in BENCHES:
        r = fix_one(b)
        report.append(r)
        print(f"[ok] {b}: n={r['n']} backup_created={r['created_backup']} "
              f"ids_match={r['ids_match']} gt_match={r['gt_match']} "
              f"image_match={r['image_match']} only_question_changed={r['only_question_changed']}")
        for s in r["spot"]:
            print(f"      id={s['id']!r}  q={s['question']!r}")
    print("\nALL SUBSET FIXES APPLIED + VERIFIED")
    print(f"INSTRUCTION appended: {INSTRUCTION!r}")


if __name__ == "__main__":
    main()
