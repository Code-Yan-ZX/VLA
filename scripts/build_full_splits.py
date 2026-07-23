#!/usr/bin/env python
"""Build the OFFICIAL FULL evaluation splits (V3 full-matrix) for 4 benchmarks.

Mirrors the subset builders (build_subsets.py, build_docvqa_200.py,
build_chartqa_ocrbench.py) EXACTLY -- same HF sources, same baked short-answer
instruction, same image-save convention (JPEG q92), same gt rules -- but takes
the ENTIRE official split instead of a 200-sample.

Outputs (serve_bench.load_subset format; `image` is a LOCAL ABSOLUTE path):
  eval/full_splits/textvqa_val.jsonl  -- TextVQA validation, 5000 rows
  eval/full_splits/docvqa_val.jsonl   -- DocVQA validation, 5349 rows
  eval/full_splits/ocrbench.jsonl     -- OCRBench test, 1000 rows (5 skills x200)
  eval/full_splits/gqa_testdev.jsonl  -- GQA testdev_balanced, 12578 rows

Schema (identical to the matching *_200.jsonl subsets):
  TextVQA: {id, image, question, gt}      -- gt = ALL 10 answers ';'-joined
                                            (official VQA-acc = min(#match/3,1))
  DocVQA : {id, image, question, gt}      -- gt = answers ';'-joined (usually 1)
  OCRBench:{id, image, question, gt, question_type, dataset, category
            [, choices=["__nospace__"] for HME100k]}
  GQA    : {id, image, question, gt}      -- gt = single answer (exact-match)

Short-answer instruction BAKED INTO question (verbatim, as in the latest
*_200.jsonl subsets -- textvqa_200 / docvqa_200 / gqa_200 all carry it):
    "<raw question>\nAnswer the question using a single word or phrase."
OCRBench keeps the RAW question (ocrbench_200.jsonl has NO baked suffix).

OCRBench `category` = the official 5-way skill grouping (derived from the
fine-grained `question_type`), verified to give EXACTLY 200 per skill on the
full 1000-row split:
  Text Recognition (TR)              <- {Regular, Irregular, Artistic,
                                        Digit String} Text Recognition   (4 x 50)
  Scene Text-centric VQA (ST-VQA)    <- {Scene Text-centric VQA}          (200)
  Document Text-centric VQA (DT-VQA) <- {Doc-oriented VQA}                (200)
  Handwriting Text Recognition (HTR) <- {Handwriting Recognition (IAM,50),
                                        Handwritten Math. Expression Rec.
                                        (HME100k,100), Non-Semantic Text
                                        Recognition (NonSemanticText,50)}  (200)
  Key Information Extraction (KIE)   <- {Key Information Extraction}      (200)
(Non-Semantic Text Recognition is part of HTR in the official OCRBench -- the
50+100+50 split is the only assignment that yields 200 per skill.)
The fine `question_type` and source `dataset` are ALSO carried (mirror subset).

GQA images come from lmms-lab/gqa `testdev_balanced_images` (~66MB curated set
that already contains every image referenced by the 12578 testdev questions) --
NO 20GB full-GQA-image download is needed. Any question whose imageId is absent
from that config is dropped and reported.

CPU/network only. Public lmms-lab/echo840 datasets, no token. ALL HF cache is
kept under /media/disk2 (set HF_HOME / HF_DATASETS_CACHE before running).
Runs with the conda `base` python (has datasets+PIL+pyarrow); qwen3vl_clean
does NOT have `datasets` (every existing builder uses base for the same reason).
"""
from __future__ import annotations

import ast
import io
import json
import os
import time
from collections import Counter
from pathlib import Path

# keep every byte of HF traffic on /media/disk2 (home partition is ~90% full)
os.environ.setdefault("HF_HOME", "/media/disk2/YZX/research/vla/runs/data/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/media/disk2/YZX/research/vla/runs/data/hf_cache")

import PIL.Image
from datasets import Image as HFImage
from datasets import load_dataset

ROOT = Path("/media/disk2/YZX/research/vla")
OUT = ROOT / "eval/full_splits"
IMG = ROOT / "runs/data"
SUFFIX = "\nAnswer the question using a single word or phrase."

# OCRBench fine question_type -> official 5-way skill category
OCR_CAT = {
    "Regular Text Recognition": "Text Recognition",
    "Irregular Text Recognition": "Text Recognition",
    "Artistic Text Recognition": "Text Recognition",
    "Digit String Recognition": "Text Recognition",
    "Scene Text-centric VQA": "Scene Text-centric VQA",
    "Doc-oriented VQA": "Document Text-centric VQA",
    "Handwriting Recognition": "Handwriting Text Recognition",
    "Handwritten Mathematical Expression Recognition": "Handwriting Text Recognition",
    "Non-Semantic Text Recognition": "Handwriting Text Recognition",  # official HTR
    "Key Information Extraction": "Key Information Extraction",
}


def save_pil(pil_img, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil_img.convert("RGB").save(dest, "JPEG", quality=92)
    return str(dest)


def pil_from_field(img_field):
    if isinstance(img_field, dict) and img_field.get("bytes"):
        return PIL.Image.open(io.BytesIO(img_field["bytes"]))
    if isinstance(img_field, dict) and img_field.get("path"):
        return PIL.Image.open(img_field["path"])
    if img_field is not None and hasattr(img_field, "convert"):
        return img_field
    return None


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {len(records)} rows -> {path}")


# --------------------------------------------------------------------------- #
# TextVQA  (lmms-lab/TextVQA, split validation, 5000 rows). gt = ALL answers
# ';'-joined. decode=False pool -> save only (all rows are kept here).
# --------------------------------------------------------------------------- #
def build_textvqa():
    t0 = time.time()
    img_dir = IMG / "textvqa"
    local = sorted(str(p) for p in (ROOT / "runs/data/staging/textvqa").glob("*.parquet"))
    if local:
        print(f"[textvqa] loading {len(local)} LOCAL validation parquet shards "
              "(decode=False pool)...")
        ds = load_dataset("parquet", data_files=local, split="train")
    else:
        print("[textvqa] loading lmms-lab/TextVQA validation from Hub...")
        ds = load_dataset("lmms-lab/TextVQA", split="validation")
    ds = ds.cast_column("image", HFImage(decode=False))
    print(f"[textvqa] rows={len(ds)} in {time.time()-t0:.1f}s")
    out, n_no_img, n_skip = [], 0, 0
    for row in ds:
        q = (row.get("question") or "").strip()
        answers = row.get("answers")
        if isinstance(answers, list):
            gt = ";".join(str(a).strip() for a in answers if str(a).strip())
        else:
            gt = str(answers or "").strip()
        if not q or not gt:
            n_skip += 1
            continue
        qid = str(row.get("question_id") or row.get("image_id") or f"q{len(out)}")
        dest = img_dir / f"{qid}.jpg"
        if dest.exists():
            img_path = str(dest)
        else:
            pil = pil_from_field(row.get("image"))
            if pil is None:
                n_no_img += 1
                continue
            img_path = save_pil(pil, dest)
        out.append({"id": qid, "image": img_path,
                    "question": q + SUFFIX, "gt": gt})
    print(f"[textvqa] kept {len(out)} (no_image={n_no_img}, skip={n_skip}) "
          f"in {time.time()-t0:.1f}s")
    write_jsonl(out, OUT / "textvqa_val.jsonl")
    return out


# --------------------------------------------------------------------------- #
# DocVQA  (lmms-lab/DocVQA "DocVQA", split validation, 5349 rows).
# gt = answers ';'-joined (usually 1).
# --------------------------------------------------------------------------- #
def build_docvqa():
    t0 = time.time()
    img_dir = IMG / "docvqa"
    local = sorted(str(p) for p in (ROOT / "runs/data/staging/docvqa").glob("*.parquet"))
    if local:
        print(f"[docvqa] loading {len(local)} LOCAL validation parquet shards "
              "(decode=False pool)...")
        ds = load_dataset("parquet", data_files=local, split="train")
    else:
        print("[docvqa] loading lmms-lab/DocVQA validation from Hub...")
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    ds = ds.cast_column("image", HFImage(decode=False))
    print(f"[docvqa] rows={len(ds)} in {time.time()-t0:.1f}s")
    out, n_no_img, n_skip = [], 0, 0
    for row in ds:
        q = (row.get("question") or "").strip()
        answers = row.get("answers")
        if isinstance(answers, list):
            ans = [str(a).strip() for a in answers if str(a).strip()]
        else:
            a = str(answers or "").strip()
            ans = [a] if a else []
        if not q or not ans:
            n_skip += 1
            continue
        qid = str(row.get("questionId") or row.get("id") or f"d{len(out)}")
        dest = img_dir / f"{qid}.jpg"
        if dest.exists():
            img_path = str(dest)
        else:
            pil = pil_from_field(row.get("image"))
            if pil is None:
                n_no_img += 1
                continue
            img_path = save_pil(pil, dest)
        out.append({"id": qid, "image": img_path,
                    "question": q + SUFFIX, "gt": ";".join(ans)})
    print(f"[docvqa] kept {len(out)} (no_image={n_no_img}, skip={n_skip}) "
          f"in {time.time()-t0:.1f}s")
    write_jsonl(out, OUT / "docvqa_val.jsonl")
    return out


# --------------------------------------------------------------------------- #
# OCRBench  (echo840/OCRBench, split test, 1000 rows = 5 skills x 200).
# --------------------------------------------------------------------------- #
def _parse_ocr_answers(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            lit = ast.literal_eval(raw)
            if isinstance(lit, list):
                return [str(a).strip() for a in lit if str(a).strip()]
        except (ValueError, SyntaxError):
            pass
        return [raw.strip()]
    return []


def build_ocrbench():
    t0 = time.time()
    img_dir = IMG / "ocrbench"
    print("[ocrbench] loading echo840/OCRBench test (full, decode=False pool)...")
    ds = load_dataset("echo840/OCRBench", split="test")
    ds = ds.cast_column("image", HFImage(decode=False))
    print(f"[ocrbench] rows={len(ds)} in {time.time()-t0:.1f}s")
    out, n_no_img, n_skip = [], 0, 0
    for idx, row in enumerate(ds):
        q = (row.get("question") or "").strip()
        ans = _parse_ocr_answers(row.get("answer"))
        if not q or not ans:
            n_skip += 1
            continue
        qid = f"ocr{idx:04d}"
        dest = img_dir / f"{qid}.jpg"
        if dest.exists():
            img_path = str(dest)
        else:
            pil = pil_from_field(row.get("image"))
            if pil is None:
                n_no_img += 1
                continue
            img_path = save_pil(pil, dest)
        qtype = str(row.get("question_type") or "")
        rec = {"id": qid, "image": img_path, "question": q,  # RAW question
               "gt": ";".join(ans),
               "question_type": qtype,
               "dataset": str(row.get("dataset") or ""),
               "category": OCR_CAT.get(qtype, qtype)}  # 5-way skill
        if rec["dataset"] == "HME100k":
            rec["choices"] = ["__nospace__"]  # scorer flag (mirror subset)
        out.append(rec)
    print(f"[ocrbench] kept {len(out)} (no_image={n_no_img}, skip={n_skip}) "
          f"in {time.time()-t0:.1f}s")
    print("[ocrbench] fine question_type dist:",
          dict(Counter(r["question_type"] for r in out)))
    print("[ocrbench] 5-way category dist:",
          dict(Counter(r["category"] for r in out)))
    write_jsonl(out, OUT / "ocrbench.jsonl")
    return out


# --------------------------------------------------------------------------- #
# GQA  (lmms-lab/gqa testdev_balanced_instructions + _images, 12578 rows).
# --------------------------------------------------------------------------- #
def build_gqa():
    t0 = time.time()
    img_dir = IMG / "gqa"
    print("[gqa] loading testdev_balanced_instructions...")
    instr = load_dataset("lmms-lab/gqa", "testdev_balanced_instructions", split="testdev")
    print(f"[gqa] instructions={len(instr)} in {time.time()-t0:.1f}s")
    t1 = time.time()
    print("[gqa] loading testdev_balanced_images (curated testdev image set)...")
    imgs = load_dataset("lmms-lab/gqa", "testdev_balanced_images", split="testdev")
    img_by_id = {r["id"]: r["image"] for r in imgs}
    print(f"[gqa] images={len(imgs)} indexed in {time.time()-t1:.1f}s")
    # encode each UNIQUE source image ONCE (only ~398 distinct), then reuse the
    # JPEG bytes for every question that references it (per-question file, same
    # convention as the gqa_200/gqa_500 subsets which save <qid>.jpg).
    bytes_by_id = {}
    for iid, pil in img_by_id.items():
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, "JPEG", quality=92)
        bytes_by_id[iid] = buf.getvalue()
    print(f"[gqa] encoded {len(bytes_by_id)} unique images in {time.time()-t1:.1f}s")
    out, n_no_img, n_skip = [], 0, 0
    for r in instr:
        qid = str(r["id"])
        image_id = r["imageId"]
        question = (r.get("question") or "").strip()
        answer = (r.get("answer") or "").strip()
        if not question or not answer:
            n_skip += 1
            continue
        dest = img_dir / f"{qid}.jpg"
        if dest.exists():
            img_path = str(dest)
        else:
            jb = bytes_by_id.get(image_id)
            if jb is None:
                n_no_img += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(jb)
            img_path = str(dest)
        out.append({"id": qid, "image": img_path,
                    "question": question + SUFFIX, "gt": answer})
    print(f"[gqa] kept {len(out)} (no_image={n_no_img}, skip={n_skip}) "
          f"in {time.time()-t0:.1f}s")
    write_jsonl(out, OUT / "gqa_testdev.jsonl")
    return out


def verify(records, benchmark: str, sample: int = 200):
    import random
    rng = random.Random(0)
    idxs = rng.sample(range(len(records)), min(sample, len(records)))
    n_ok = 0
    for i in idxs:
        r = records[i]
        assert r["question"] and r["gt"], f"empty q/gt: {r['id']}"
        assert Path(r["image"]).exists(), f"missing image: {r['id']}"
        im = PIL.Image.open(r["image"]); im.verify()
        im2 = PIL.Image.open(r["image"])
        assert im2.size[0] > 0 and im2.size[1] > 0
        n_ok += 1
    print(f"[{benchmark}] verified {n_ok}/{len(idxs)} sampled images + q/gt OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "textvqa", "docvqa", "ocrbench", "gqa"])
    a = ap.parse_args()
    builders = {"ocrbench": build_ocrbench, "gqa": build_gqa,
                "textvqa": build_textvqa, "docvqa": build_docvqa}
    order = ["ocrbench", "gqa", "textvqa", "docvqa"]  # small -> large
    for name in order:
        if a.only in {"all", name}:
            recs = builders[name]()
            verify(recs, name)
    print("DONE")
