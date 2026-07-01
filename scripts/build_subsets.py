#!/usr/bin/env python
"""Build small fixed-seed subsets for the P2 go/no-go probe (minimal download).

Outputs (serve_bench.load_subset format):
  eval/subsets/gqa_200.jsonl       -- 200 GQA testdev_balanced samples
  eval/subsets/textvqa_200.jsonl   -- 200 TextVQA val samples
Each line: {"id","image","question","gt",[choices]}  -- `image` is a LOCAL path.

Images -> runs/data/{gqa,textvqa}/<id>.jpg  (gitignored).

MINIMAL download strategy:
  - GQA testdev_balanced: full parquet shards (~68 MB total -- small), non-streaming.
  - TextVQA val: STREAMING, stop after enough rows for 200 good samples
    (val parquet is ~920 MB; streaming reads row-by-row and only as many as
    needed -- ~200 images extracted, far under 920 MB pulled).

CPU/network only. Seed = 0 (deterministic). Public datasets, no token.
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
GQA_OUT = ROOT / "eval/subsets/gqa_200.jsonl"
TVQA_OUT = ROOT / "eval/subsets/textvqa_200.jsonl"
GQA_IMG = ROOT / "runs/data/gqa"
TVQA_IMG = ROOT / "runs/data/textvqa"
N = 200
SEED = 0
# for TextVQA streaming: scan up to this many rows to get N good samples
TVQA_SCAN_CAP = 600


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _pil_to_jpg_bytes(pil_img) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def save_pil(pil_img, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil_img.convert("RGB").save(dest, "JPEG", quality=92)
    return str(dest)


def build_gqa():
    """GQA testdev_balanced: download both small shards (~68MB), join by imageId, sample."""
    _ensure(GQA_IMG)
    t0 = time.time()
    print("[gqa] loading testdev_balanced_instructions (~2MB)...")
    instr = load_dataset("lmms-lab/gqa", "testdev_balanced_instructions", split="testdev")
    print(f"[gqa] instructions: {len(instr)} rows in {time.time()-t0:.1f}s")
    t1 = time.time()
    print("[gqa] loading testdev_balanced_images (~66MB)...")
    imgs = load_dataset("lmms-lab/gqa", "testdev_balanced_images", split="testdev")
    print(f"[gqa] images: {len(imgs)} rows in {time.time()-t1:.1f}s")

    # index images by id (imageId) -> PIL image
    img_by_id = {}
    for row in imgs:
        img_by_id[row["id"]] = row["image"]   # PIL.JpegImagePlugin image

    instr_rows = list(instr)
    rng = random.Random(SEED)
    rng.shuffle(instr_rows)

    out = []
    n_no_img = 0
    for r in instr_rows:
        if len(out) >= N:
            break
        qid = str(r["id"])
        image_id = r["imageId"]
        question = (r.get("question") or "").strip()
        answer = (r.get("answer") or "").strip()
        if not question or not answer:
            continue
        pil = img_by_id.get(image_id)
        if pil is None:
            n_no_img += 1
            continue
        dest = GQA_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        # GQA answers are open-vocab; no fixed choice list. serve_bench.score_gqa
        # does exact-match (case/punct-insensitive) with optional choices=None.
        out.append({"id": qid, "image": img_path, "question": question, "gt": answer})
    print(f"[gqa] kept {len(out)} (no_image={n_no_img}) in {time.time()-t0:.1f}s")
    GQA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GQA_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[gqa] wrote {GQA_OUT}")
    return out


def build_textvqa():
    """TextVQA val: STREAM (avoid 920MB), scan until 200 good samples."""
    _ensure(TVQA_IMG)
    t0 = time.time()
    print("[tvqa] streaming validation (stop after 200 good)...")
    ds = load_dataset("lmms-lab/TextVQA", split="validation", streaming=True)
    # first row to confirm schema
    ds_peek = load_dataset("lmms-lab/TextVQA", split="validation", streaming=True)
    first = next(iter(ds_peek))
    print("[tvqa] keys:", list(first.keys()))
    print("[tvqa] sample:", {k: str(v)[:50] for k, v in first.items()})

    # collect rows into memory with reservoir-free approach: scan up to cap,
    # reservoir-sample N so the subset is deterministic across partial pulls.
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        q = (row.get("question") or "").strip()
        answers = row.get("answers")
        if isinstance(answers, list):
            gt = ";".join(str(a).strip() for a in answers if str(a).strip())
        else:
            gt = str(answers or "").strip()
        if not q or not gt:
            continue
        pool.append(row)
        if len(pool) >= TVQA_SCAN_CAP:
            break
    print(f"[tvqa] scanned {scanned} rows, pool={len(pool)} good in {time.time()-t0:.1f}s")

    rng.shuffle(pool)
    out = []
    n_no_img = 0
    for row in pool:
        if len(out) >= N:
            break
        qid = str(row.get("question_id") or row.get("image_id") or f"q{len(out)}")
        q = (row.get("question") or "").strip()
        answers = row.get("answers")
        gt = ";".join(str(a).strip() for a in answers if str(a).strip()) \
            if isinstance(answers, list) else str(answers or "").strip()
        img_field = row.get("image")
        pil = None
        if isinstance(img_field, dict) and img_field.get("bytes"):
            pil = PIL.Image.open(io.BytesIO(img_field["bytes"]))
        elif img_field is not None and hasattr(img_field, "convert"):
            pil = img_field  # PIL image object
        if pil is None:
            n_no_img += 1
            continue
        dest = TVQA_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        out.append({"id": qid, "image": img_path, "question": q, "gt": gt})
    print(f"[tvqa] kept {len(out)} (no_image={n_no_img}) in {time.time()-t0:.1f}s")
    TVQA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TVQA_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[tvqa] wrote {TVQA_OUT}")
    return out


def verify(records, benchmark: str):
    n_ok = 0
    for r in records:
        assert r["question"] and r["gt"], f"empty q/gt: {r['id']}"
        img = PIL.Image.open(r["image"])
        img.verify()
        img2 = PIL.Image.open(r["image"])  # reopen (verify() invalidates)
        assert img2.size[0] > 0 and img2.size[1] > 0
        n_ok += 1
    print(f"[{benchmark}] verified {n_ok}/{len(records)} images open + q/gt non-empty")


def report_bytes():
    total = 0
    by_d = {}
    for name, d in (("gqa", GQA_IMG), ("textvqa", TVQA_IMG)):
        s = sum(p.stat().st_size for p in d.glob("*.jpg"))
        by_d[name] = s
        total += s
    print(f"[bytes] local images: gqa={by_d['gqa']/1e6:.1f}MB "
          f"textvqa={by_d['textvqa']/1e6:.1f}MB total={total/1e6:.1f}MB")
    # HF cache delta
    hf_cache = Path.home() / ".cache/huggingface/datasets"
    if hf_cache.exists():
        cs = sum(p.stat().st_size for p in hf_cache.rglob("*") if p.is_file())
        print(f"[bytes] HF datasets cache total (shared, not just this run): {cs/1e6:.1f}MB")
    return total


if __name__ == "__main__":
    print("=" * 60, "\nGQA\n", "=" * 60)
    g = build_gqa()
    verify(g, "gqa")
    print("=" * 60, "\nTextVQA\n", "=" * 60)
    t = build_textvqa()
    verify(t, "textvqa")
    print("=" * 60)
    report_bytes()
    print("\nSAMPLE LINES:")
    if g: print("  gqa:    ", json.dumps(g[0]))
    if t: print("  textvqa:", json.dumps(t[0]))
    print("DONE")
