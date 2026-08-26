#!/usr/bin/env python
"""Build fixed-seed subsets for P3-step-2 breadth benchmarks (minimal download).

Outputs (serve_bench.load_subset format):
  eval/subsets/mme_200.jsonl        -- 200 MME yes/no samples
  eval/subsets/mmbench_200.jsonl    -- 200 MMBench multiple-choice samples
  eval/subsets/scienceqa_200.jsonl  -- 200 ScienceQA multiple-choice samples

Each line: {"id","image","question","gt","choices"[,...]}  -- `image` is LOCAL.
  * MME: gt in {"yes","no"}; question as-is (yes/no VQA).
  * MMBench: gt = the CORRECT OPTION LETTER ("A"/"B"/...); question formatted
    as "<q>\nA. <c0>\nB. <c1>\n...\nAnswer with the option letter from the
    given choices directly."; choices = list of option texts.
  * ScienceQA: same MC convention as MMBench (gt = correct letter).

MC convention matches the standard MMBench/ScienceQA LLaVA eval: the prompt
ends with "Answer with the option letter from the given choices directly." and
the scorer extracts the model's first A-D token.

Images -> runs/data/{mme,mmbench,scienceqa}/<id>.jpg  (gitignored).

MINIMAL download (stream/sample, don't pull full dumps), CPU/network only.
Seed = 0 (deterministic). Public lmms-lab/official HF sources, no token.
"""
from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path

import PIL.Image
from datasets import load_dataset

ROOT = Path("/media/disk2/YZX/research/vla")
MME_OUT = ROOT / "eval/subsets/mme_200.jsonl"
MMB_OUT = ROOT / "eval/subsets/mmbench_200.jsonl"
SQA_OUT = ROOT / "eval/subsets/scienceqa_200.jsonl"
MME_IMG = ROOT / "runs/data/mme"
MMB_IMG = ROOT / "runs/data/mmbench"
SQA_IMG = ROOT / "runs/data/scienceqa"
N = 200
SEED = 0
SCAN_CAP = 800  # stream-scan up to this many rows to collect N good samples


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_pil(pil_img, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil_img.convert("RGB").save(dest, "JPEG", quality=92)
    return str(dest)


def _pil_from_field(img_field):
    """Robustly extract a PIL image from a datasets image field (dict bytes or PIL)."""
    if isinstance(img_field, dict) and img_field.get("bytes"):
        return PIL.Image.open(io.BytesIO(img_field["bytes"]))
    if isinstance(img_field, dict) and img_field.get("path"):
        return PIL.Image.open(img_field["path"])
    if img_field is not None and hasattr(img_field, "convert"):
        return img_field  # PIL image object
    return None


LETTERS = "ABCDEFGH"


def _format_mc(question: str, choices: list[str]) -> str:
    """Standard MMBench/ScienceQA MC prompt (matches the official LLaVA eval)."""
    lines = [question.strip()]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer with the option letter from the given choices directly.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# MME  (yes/no perception/cognition; lmms-lab/Manual_Verification_of_MME or the
# common lmms-lab/MME). Use lmms-lab/MME which has {image, question, answer}.
# --------------------------------------------------------------------------- #
def build_mme():
    _ensure(MME_IMG)
    t0 = time.time()
    print("[mme] streaming lmms-lab/MME (stop after enough yes/no rows)...")
    ds = load_dataset("lmms-lab/MME", split="test", streaming=True)
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        q = (row.get("question") or "").strip()
        ans = str(row.get("answer") or "").strip().lower()
        img_field = row.get("image")
        if not q or ans not in {"yes", "no"}:
            continue
        pil = _pil_from_field(img_field)
        if pil is None:
            continue
        pool.append((row, pil))
        if len(pool) >= SCAN_CAP:
            break
    print(f"[mme] scanned {scanned}, pool={len(pool)} good in {time.time()-t0:.1f}s")
    rng.shuffle(pool)
    out = []
    for row, pil in pool:
        if len(out) >= N:
            break
        qid = str(row.get("question_id") or row.get("id") or f"m{len(out)}")
        q = (row.get("question") or "").strip()
        ans = str(row.get("answer") or "").strip().lower()
        dest = MME_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        out.append({"id": qid, "image": img_path, "question": q, "gt": ans})
    print(f"[mme] kept {len(out)} in {time.time()-t0:.1f}s")
    MME_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MME_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[mme] wrote {MME_OUT}")
    return out


# --------------------------------------------------------------------------- #
# MMBench  (multiple-choice; lmms-lab/MMBench dev split). Fields:
#   {index, image, question, A, B, C, D, [E], answer (the LETTER), ...}
# --------------------------------------------------------------------------- #
def build_mmbench():
    _ensure(MMB_IMG)
    t0 = time.time()
    print("[mmbench] streaming lmms-lab/MMBench dev (stop after enough MC rows)...")
    # config 'en' = English MMBench; split 'dev' has the answer key (standard eval).
    ds = load_dataset("lmms-lab/MMBench", "en", split="dev", streaming=True)
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        q = (row.get("question") or "").strip()
        if not q:
            continue
        # collect option fields A..D (E present in a few rows). Drop the
        # placeholder "nan"/"" options the HF dataset emits for missing cells
        # (so the model is never shown "C. nan"); also remap the answer letter
        # index to the CLEANED option list.
        raw_opts = []
        for L in "ABCDE":
            v = row.get(L)
            if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
                raw_opts.append((L, v.strip()))
            else:
                break
        if len(raw_opts) < 2:
            continue
        ans_letter = str(row.get("answer") or "").strip().upper()
        # keep only items up to and including the correct letter (drop trailing
        # placeholder opts after the answer if any). If the answer letter isn't
        # in the cleaned set, skip the row.
        ans_idx = next((i for i, (L, _) in enumerate(raw_opts) if L == ans_letter), -1)
        if ans_idx < 0:
            continue
        opts = [t for _, t in raw_opts]
        img_field = row.get("image")
        pil = _pil_from_field(img_field)
        if pil is None:
            continue
        pool.append((q, opts, ans_letter, pil, row))
        if len(pool) >= SCAN_CAP:
            break
    print(f"[mmbench] scanned {scanned}, pool={len(pool)} good in {time.time()-t0:.1f}s")
    rng.shuffle(pool)
    out = []
    for q, opts, ans_letter, pil, row in pool:
        if len(out) >= N:
            break
        qid = str(row.get("index") or row.get("question_id") or f"b{len(out)}")
        dest = MMB_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        prompt = _format_mc(q, opts)
        out.append({"id": qid, "image": img_path, "question": prompt,
                    "gt": ans_letter, "choices": opts})
    print(f"[mmbench] kept {len(out)} in {time.time()-t0:.1f}s")
    MMB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MMB_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[mmbench] wrote {MMB_OUT}")
    return out


# --------------------------------------------------------------------------- #
# ScienceQA  (multiple-choice; lmms-lab/scienceqa test split, IMAGE subset).
# Fields: {image, question, choices (list), answer (INT index), hint, ...}
# We keep only rows WITH an image (the multimodal subset) and convert the int
# answer to a letter.
# --------------------------------------------------------------------------- #
def build_scienceqa():
    _ensure(SQA_IMG)
    t0 = time.time()
    print("[scienceqa] streaming lmms-lab/scienceqa test (image-only rows)...")
    # ScienceQA-IMG = the multimodal (image) subset; 'test' split for held-out eval.
    ds = load_dataset("lmms-lab/scienceqa", "ScienceQA-IMG", split="test", streaming=True)
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        img_field = row.get("image")
        pil = _pil_from_field(img_field)
        if pil is None:
            continue  # text-only ScienceQA item -- skip (we want multimodal)
        q = (row.get("question") or "").strip()
        choices = row.get("choices") or []
        ans_idx = row.get("answer")
        if not q or not choices or ans_idx is None:
            continue
        try:
            ai = int(ans_idx)
        except (TypeError, ValueError):
            continue
        if ai < 0 or ai >= len(choices) or len(choices) > len(LETTERS):
            continue
        pool.append((q, list(choices), ai, pil, row))
        if len(pool) >= SCAN_CAP:
            break
    print(f"[scienceqa] scanned {scanned}, pool={len(pool)} good in {time.time()-t0:.1f}s")
    rng.shuffle(pool)
    out = []
    for q, choices, ai, pil, row in pool:
        if len(out) >= N:
            break
        qid = str(row.get("id") or row.get("question_id") or f"s{len(out)}")
        dest = SQA_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        ans_letter = LETTERS[ai]
        # include the hint (lecture/context) if present -- standard ScienceQA eval prepends it
        hint = (row.get("hint") or "").strip()
        full_q = (hint + "\n" + q) if hint else q
        prompt = _format_mc(full_q, choices)
        out.append({"id": qid, "image": img_path, "question": prompt,
                    "gt": ans_letter, "choices": choices})
    print(f"[scienceqa] kept {len(out)} in {time.time()-t0:.1f}s")
    SQA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SQA_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[scienceqa] wrote {SQA_OUT}")
    return out


def verify(records, benchmark: str):
    n_ok = 0
    for r in records:
        assert r["question"] and r["gt"], f"empty q/gt: {r['id']}"
        if "choices" in r:
            assert r["choices"], f"empty choices: {r['id']}"
        img = PIL.Image.open(r["image"])
        img.verify()
        img2 = PIL.Image.open(r["image"])  # reopen (verify() invalidates)
        assert img2.size[0] > 0 and img2.size[1] > 0
        n_ok += 1
    print(f"[{benchmark}] verified {n_ok}/{len(records)} images open + q/gt/choices non-empty")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "mme", "mmbench", "scienceqa"])
    args = ap.parse_args()
    do = args.only
    if do in {"all", "mme"}:
        m = build_mme(); verify(m, "mme")
    if do in {"all", "mmbench"}:
        b = build_mmbench(); verify(b, "mmbench")
    if do in {"all", "scienceqa"}:
        s = build_scienceqa(); verify(s, "scienceqa")
    print("DONE")
