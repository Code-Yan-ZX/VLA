#!/usr/bin/env python
"""Build the 200-sample DocVQA subset for the V3 suite (text-dense strengthener).

Mirrors build_subsets.build_textvqa: STREAM lmms-lab/DocVQA validation (avoid
pulling the full dump), scan until enough good rows, deterministic seed=0,
save JPEGs to runs/data/docvqa/, write eval/subsets/docvqa_200.jsonl.

DocVQA gt is a short text span (answers list, usually 1 entry; occasionally a
few). We join with ';' so the existing score_textvqa (semicolon-OR exact-match)
scorer applies unchanged -- the task spec says "DocVQA uses VQA-accuracy /
exact-match like TextVQA". DocVQA's native metric is ANLS; exact-match is a
fair, deterministic proxy here and is what we isolate the stage effect with.

Format per line: {"id","image","question","gt"}  -- image is a LOCAL path.
CPU/network only. Public lmms-lab dataset, seed=0, no token.

NOTE: runs with the conda `base` python (has `datasets`+PIL), NOT qwen3vl_clean.
If huggingface.co is unreachable, set HF_ENDPOINT=https://hf-mirror.com (done
below automatically only if the default endpoint fails -- we just always set
the mirror since the host blocks hf.co).
"""
from __future__ import annotations

import io
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import PIL.Image
from datasets import load_dataset

ROOT = Path("/media/disk2/YZX/research/vla")
DOC_OUT = ROOT / "eval/subsets/docvqa_200.jsonl"
DOC_IMG = ROOT / "runs/data/docvqa"
N = 200
SEED = 0
SCAN_CAP = 700  # validation has ~5k+ rows; scan enough for 200 good


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_pil(pil_img, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil_img.convert("RGB").save(dest, "JPEG", quality=92)
    return str(dest)


def _pil_from_field(img_field):
    if isinstance(img_field, dict) and img_field.get("bytes"):
        return PIL.Image.open(io.BytesIO(img_field["bytes"]))
    if isinstance(img_field, dict) and img_field.get("path"):
        return PIL.Image.open(img_field["path"])
    if img_field is not None and hasattr(img_field, "convert"):
        return img_field
    return None


def build_docvqa():
    _ensure(DOC_IMG)
    t0 = time.time()
    print(f"[docvqa] streaming lmms-lab/DocVQA validation (endpoint={os.environ.get('HF_ENDPOINT')})...")
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        q = (row.get("question") or "").strip()
        answers = row.get("answers")
        if isinstance(answers, list):
            ans = [str(a).strip() for a in answers if str(a).strip()]
        else:
            a = str(answers or "").strip()
            ans = [a] if a else []
        if not q or not ans:
            continue
        pil = _pil_from_field(row.get("image"))
        if pil is None:
            continue
        pool.append((row, pil, q, ans))
        if len(pool) >= SCAN_CAP:
            break
    print(f"[docvqa] scanned {scanned}, pool={len(pool)} good in {time.time()-t0:.1f}s")
    rng.shuffle(pool)
    out = []
    for row, pil, q, ans in pool:
        if len(out) >= N:
            break
        qid = str(row.get("questionId") or row.get("id") or f"d{len(out)}")
        dest = DOC_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        gt = ";".join(ans)
        out.append({"id": qid, "image": img_path, "question": q, "gt": gt})
    print(f"[docvqa] kept {len(out)} in {time.time()-t0:.1f}s")
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[docvqa] wrote {DOC_OUT}")
    return out


def verify(records):
    n_ok = 0
    for r in records:
        assert r["question"] and r["gt"], f"empty q/gt: {r['id']}"
        img = PIL.Image.open(r["image"])
        img.verify()
        img2 = PIL.Image.open(r["image"])
        assert img2.size[0] > 0 and img2.size[1] > 0
        n_ok += 1
    print(f"[docvqa] verified {n_ok}/{len(records)} images open + q/gt non-empty")


if __name__ == "__main__":
    recs = build_docvqa()
    verify(recs)
    # quick text-density sanity: print a few gt strings
    print("\nSAMPLE LINES:")
    for r in recs[:3]:
        print("  ", json.dumps({k: (r[k] if k != "image" else "...") for k in r}))
    total = sum(p.stat().st_size for p in DOC_IMG.glob("*.jpg"))
    print(f"[docvqa] local images: {total/1e6:.1f}MB ({len(recs)} files)")
    print("DONE")
