#!/usr/bin/env python
"""Build the 200-sample ChartQA + OCRBench subsets for the V3 SOTA matrix.

Mirrors build_docvqa_200.py / build_subsets_p3s2.py conventions: STREAM the HF
dataset (don't clone the full dump), scan the split, deterministic seed=0
shuffle of the good-row pool, keep N=200, save JPEGs to runs/data/<bm>/, write
eval/subsets/<bm>_200.jsonl, verify() every row.

Sources (lmms-eval convention, same family as the existing subsets):
  * ChartQA  -- lmms-lab/ChartQA, split "test" (default config; 2500 rows =
    1250 human_test + 1250 augmented_test). lmms-eval task "chartqa":
    doc_to_target "answer", metric = relaxed accuracy.
  * OCRBench -- echo840/OCRBench, split "test" (1000 rows; lmms-lab/OCRBench
    does not exist). lmms-eval task "ocrbench": doc_to_target "answer" (a LIST
    of acceptable answers), metric = exact-match accuracy after normalization.

Scoring (implemented in src/serve_bench.py + src/v3_premerger/v3_premerger_runner.py):
  * chartqa  -> relaxed accuracy (numeric answers within +/-5% relative
    tolerance, percentages normalized; else normalized exact match). Ported
    from lmms_eval/tasks/chartqa/utils.py:relaxed_correctness
    (arXiv:2203.10244 sec 5.1). The lmms-eval prompt suffix
    "\\nAnswer the question with a single word." is BAKED INTO `question`
    (runner sends the question verbatim; same convention as the MC builders
    baking the option-letter instruction).
  * ocrbench -> correct if ANY ';'-joined GT answer is contained in the model
    output after lowercase/strip/'\\n'->' ' normalization; HME100k rows
    (LaTeX math expressions) additionally strip ALL spaces on both sides,
    matching lmms_eval/tasks/ocrbench/utils.py. Such rows carry
    choices=["__nospace__"] (a scorer FLAG, not MC options -- the runner
    passes extra["choices"] to the scorer unchanged).

Format per line: {"id","image","question","gt", ...}  -- `image` is a LOCAL
absolute path. Extra row fields: chartqa {"type": human_test|augmented_test};
ocrbench {"question_type", "dataset" [, "choices"]} so per-type scoring is
possible later (lmms-eval reports per question_type).

Subset selection: SCAN_CAP covers the WHOLE split (both are single-parquet, so
the network cost equals a partial scan), the pool is seed=0-shuffled and the
first N kept -> representative mix (chartqa ~100 human / 100 augmented in
expectation). Images are pooled as RAW BYTES (decode=False) and only the 200
kept rows are decoded/saved (~116MB ChartQA / ~85MB OCRBench streamed).

CPU/network only. Public datasets, no token, seed=0.
NOTE: runs with conda `base` python (has `datasets`+PIL); qwen3vl_clean /
vtc_serve do NOT have `datasets`. huggingface.co is reached directly (host
proxy); if your network blocks it, `export HF_ENDPOINT=https://hf-mirror.com`
before running. Do NOT set HF_HUB_OFFLINE (download needed).
"""
from __future__ import annotations

import ast
import io
import json
import random
import time
from pathlib import Path

import PIL.Image
from datasets import Image as HFImage
from datasets import load_dataset

ROOT = Path("/media/disk2/YZX/research/vla")
CQ_OUT = ROOT / "eval/subsets/chartqa_200.jsonl"
OCR_OUT = ROOT / "eval/subsets/ocrbench_200.jsonl"
CQ_IMG = ROOT / "runs/data/chartqa"
OCR_IMG = ROOT / "runs/data/ocrbench"
N = 200
SEED = 0
# > split sizes (2500 / 1000) so the stream EXHAUSTS naturally (no early break
# -> clean process exit; breaking a streaming iterator mid-scan aborts atexit).
CQ_SCAN_CAP = 2600
OCR_SCAN_CAP = 1100


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


# --------------------------------------------------------------------------- #
# ChartQA  (lmms-lab/ChartQA, split test; relaxed accuracy). Fields:
#   {type: human_test|augmented_test, question, answer (single str), image}
# --------------------------------------------------------------------------- #
def build_chartqa():
    _ensure(CQ_IMG)
    t0 = time.time()
    print("[chartqa] streaming lmms-lab/ChartQA test (raw image bytes; decode "
          "only kept rows)...")
    ds = load_dataset("lmms-lab/ChartQA", split="test", streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))  # keep RAM ~ parquet size
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        q = (row.get("question") or "").strip()
        ans = str(row.get("answer") or "").strip()
        img_field = row.get("image")
        if not q or not ans:
            continue
        if not (isinstance(img_field, dict) and img_field.get("bytes")):
            continue
        pool.append((scanned - 1, row.get("type") or "", q, ans, img_field))
        if len(pool) >= CQ_SCAN_CAP:
            break
    print(f"[chartqa] scanned {scanned}, pool={len(pool)} good in {time.time()-t0:.1f}s")
    rng.shuffle(pool)
    out = []
    for idx, typ, q, ans, img_field in pool:
        if len(out) >= N:
            break
        pil = _pil_from_field(img_field)
        if pil is None:
            continue
        qid = f"cq{idx:05d}"  # split index -- rows carry no question id
        dest = CQ_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        # lmms-eval "chartqa" default post_prompt (runner sends question as-is)
        prompt = q + "\nAnswer the question with a single word."
        out.append({"id": qid, "image": img_path, "question": prompt,
                    "gt": ans, "type": typ})
    print(f"[chartqa] kept {len(out)} in {time.time()-t0:.1f}s")
    CQ_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CQ_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[chartqa] wrote {CQ_OUT}")
    return out


# --------------------------------------------------------------------------- #
# OCRBench  (echo840/OCRBench, split test; normalized exact-match accuracy).
# Fields: {dataset, question, question_type, answer (LIST of acceptable
# answers), image}. answer list joined with ';' (docvqa convention -> the
# scorer ORs over acceptable answers).
# --------------------------------------------------------------------------- #
def _parse_ocr_answers(raw) -> list[str]:
    """OCRBench 'answer' is normally a list; guard against stringified-list
    revisions (e.g. \"['CENTRE']\") by literal_eval fallback."""
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
    _ensure(OCR_IMG)
    t0 = time.time()
    print("[ocrbench] streaming echo840/OCRBench test (raw image bytes; decode "
          "only kept rows)...")
    ds = load_dataset("echo840/OCRBench", split="test", streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))
    rng = random.Random(SEED)
    pool = []
    scanned = 0
    for row in ds:
        scanned += 1
        q = (row.get("question") or "").strip()
        ans = _parse_ocr_answers(row.get("answer"))
        img_field = row.get("image")
        if not q or not ans:
            continue
        if not (isinstance(img_field, dict) and img_field.get("bytes")):
            continue
        pool.append((scanned - 1, row, q, ans, img_field))
        if len(pool) >= OCR_SCAN_CAP:
            break
    print(f"[ocrbench] scanned {scanned}, pool={len(pool)} good in {time.time()-t0:.1f}s")
    rng.shuffle(pool)
    out = []
    for idx, row, q, ans, img_field in pool:
        if len(out) >= N:
            break
        pil = _pil_from_field(img_field)
        if pil is None:
            continue
        qid = f"ocr{idx:04d}"
        dest = OCR_IMG / f"{qid}.jpg"
        img_path = save_pil(pil, dest)
        rec = {"id": qid, "image": img_path, "question": q,
               "gt": ";".join(ans),
               "question_type": str(row.get("question_type") or ""),
               "dataset": str(row.get("dataset") or "")}
        if rec["dataset"] == "HME100k":
            # scorer flag: space-insensitive containment (lmms-eval HME100k
            # branch for LaTeX math expressions); NOT MC options.
            rec["choices"] = ["__nospace__"]
        out.append(rec)
    print(f"[ocrbench] kept {len(out)} in {time.time()-t0:.1f}s")
    OCR_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OCR_OUT, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[ocrbench] wrote {OCR_OUT}")
    return out


def verify(records, benchmark: str):
    n_ok = 0
    for r in records:
        assert r["question"] and r["gt"], f"empty q/gt: {r['id']}"
        assert Path(r["image"]).exists(), f"missing image: {r['id']}"
        img = PIL.Image.open(r["image"])
        img.verify()
        img2 = PIL.Image.open(r["image"])  # reopen (verify() invalidates)
        assert img2.size[0] > 0 and img2.size[1] > 0
        n_ok += 1
    print(f"[{benchmark}] verified {n_ok}/{len(records)} images open + q/gt non-empty")


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "chartqa", "ocrbench"])
    args = ap.parse_args()
    results = {}
    if args.only in {"all", "chartqa"}:
        results["chartqa"] = build_chartqa(); verify(results["chartqa"], "chartqa")
    if args.only in {"all", "ocrbench"}:
        results["ocrbench"] = build_ocrbench(); verify(results["ocrbench"], "ocrbench")
    # smoke preview: 3 example rows (image path truncated) + type distribution
    for bm, recs in results.items():
        print(f"\n{bm} SAMPLE LINES:")
        for r in recs[:3]:
            print("  ", json.dumps({k: ("..." if k == "image" else r[k]) for k in r}))
        field = "type" if bm == "chartqa" else "question_type"
        print(f"{bm} {field} dist:", dict(Counter(r.get(field, "?") for r in recs)))
        imgdir = CQ_IMG if bm == "chartqa" else OCR_IMG
        total = sum(p.stat().st_size for p in imgdir.glob("*.jpg"))
        print(f"{bm} local images: {total/1e6:.1f}MB")
    print("DONE")
