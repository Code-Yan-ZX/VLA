#!/usr/bin/env python
"""Generic official-metric comparator for today's supplement (directive 19):
reuses the PINNED scoring logic of scripts/analyze_cvm_*_official.py unchanged
(expB module primitives; ocrbench AST scorer identical to its inline block;
docvqa load_reference).  Any two prediction files (full or pre-fixed subset)
-> attempted-denominator scores, paired bootstrap 20k (fixed seed),
common-answered sensitivity, failure cross-table, actual-K summary.
Usage:
  cvm24h_score.py --bench textvqa|ocrbench|docvqa A.json B.json \
      [--ids subset_ids.txt] [--seed-int 0] [--label-a X --label-b Y] --out J.json
"""
import argparse, ast, hashlib, json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, "scripts")
import analyze_cvm_expB_official as EX
from analyze_cvm_ocrbench_official import COMMIT, URL as OCR_URL, SHA256 as OCR_SHA

CACHE = Path("tmp/cvm_cpu_rescore_20261025")

def fetch(url, path, sha):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    if sha:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, f"SHA mismatch {path}"
    return path

def rows_of(path, ids=None):
    data, digest = EX.load_json(Path(path))
    rows = {str(r["id"]): r for r in data["per_sample"]}
    keep = [str(x.strip()) for x in open(ids)] if ids else None
    if keep:
        rows = {k: v for k, v in rows.items() if k in set(keep)}
    return rows, digest, data

# ---------------- per-benchmark per-row official score ----------------------
def textvqa_scores(rows):
    ref = fetch(EX.REFERENCE_URL, CACHE / "vqa_eval_metric_reference.py", EX.REFERENCE_SHA256)
    normalize = EX.import_file("pin_ref", ref).EvalAIAnswerProcessor()
    ann_path = CACHE / "TextVQA_0.5.1_val.json"
    assert hashlib.sha256(ann_path.read_bytes()).hexdigest() == \
        "4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d"
    official = {str(r["question_id"]): r["answers"]
                for r in json.loads(ann_path.read_bytes())["data"]}
    out = {}
    for sid, row in rows.items():
        if row.get("skipped"):
            out[sid] = 0.0
            continue
        answers = official[sid]
        assert ";".join(a.strip() for a in answers) == row.get("gt"), f"GT drift {sid}"
        out[sid] = EX.vqa_score(row.get("answer", ""), answers, normalize)
    return out

def ocrbench_scores(rows):
    ref = fetch(OCR_URL, CACHE / "ocrbench_utils_reference.py", OCR_SHA256)
    raw = ref.read_bytes()
    nodes = [n for n in ast.parse(raw).body
             if isinstance(n, ast.FunctionDef)
             and n.name in ("_fold_fullwidth_ascii", "ocrbench_process_results")]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ref), "exec"), ns)
    f = ns["ocrbench_process_results"]
    meta = {str(json.loads(l)["id"]): json.loads(l)
            for l in open("eval/full_splits/ocrbench.jsonl")}
    out = {}
    for sid, row in rows.items():
        if row.get("skipped"):
            out[sid] = 0.0
            continue
        m = meta[sid]
        doc = {"answer": str(row["gt"]).split(";"), "dataset": m["dataset"],
               "question_type": m["question_type"]}
        out[sid] = float(f(doc, [row.get("answer", "")])["ocrbench_accuracy"]["score"])
    return out

def docvqa_scores(rows):
    import analyze_cvm_docvqa_official as DOC
    anls = DOC.load_reference(CACHE)
    out = {}
    for sid, row in rows.items():
        if row.get("skipped"):
            out[sid] = 0.0
            continue
        out[sid] = float(anls(str(row["gt"]).split(";"), [row.get("answer", "")])["anls"])
    return out

def hist_kept(rows):
    vals = sorted(r["n_image_kept"] for r in rows.values() if r.get("n_image_kept") is not None)
    if not vals: return None
    from collections import Counter
    c = Counter(vals)
    return {"n": len(vals), "min": vals[0], "max": vals[-1],
            "distinct_top": c.most_common(6)}

SC = {"textvqa": textvqa_scores, "ocrbench": ocrbench_scores, "docvqa": docvqa_scores}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--label-a", default=None); ap.add_argument("--label-b", default=None)
    ap.add_argument("--seed-int", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scorer = SC[a.bench]
    rowsA, shaA, dA = rows_of(a.a, a.ids)
    rowsB, shaB, dB = rows_of(a.b, a.ids)
    assert set(rowsA) == set(rowsB), f"attempted ID sets differ {len(rowsA)} vs {len(rowsB)}"
    sA, sB = scorer(rowsA), scorer(rowsB)
    seed = 20261200 + a.seed_int
    pa = EX.paired(sA, rowsA, sB, rowsB, a.resamples, seed, include_skips=True)
    pb = EX.paired(sA, rowsA, sB, rowsB, a.resamples, seed + 1, include_skips=False)
    def cell(rows, scores, src):
        answered = [i for i in rows if not rows[i].get("skipped")]
        return {"source": str(src), "n": len(rows),
                "attempted_score": sum(scores.values()) / len(rows),
                "answered_score": sum(scores[i] for i in answered) / max(len(answered), 1),
                "skipped": len(rows) - len(answered),
                "n_image_kept_hist": hist_kept(rows)}
    skipA = {i for i, r in rowsA.items() if r.get("skipped")}
    skipB = {i for i, r in rowsB.items() if r.get("skipped")}
    A_cell=cell(rowsA,sA,a.a); B_cell=cell(rowsB,sB,a.b)
    res = {"bench": a.bench, "ids_subset": a.ids,
           "label_a": a.label_a or a.a, "label_b": a.label_b or a.b,
           "sha_a": shaA[:16], "sha_b": shaB[:16], "resamples": a.resamples,
           "A": A_cell, "B": B_cell,
           "paired_attempted_all": pa, "paired_common_answered_sensitivity": pb,
           "failures": {"a_only_skip": sorted(skipA - skipB)[:50],
                        "b_only_skip": sorted(skipB - skipA)[:50],
                        "both_skip": len(skipA & skipB)}}
    if a.bench == "ocrbench":
        for k, mult in (("attempted_score", 1.0),):
            pass
        res["A"]["score_out_of_1000"] = res["A"]["attempted_score"]
        res["B"]["score_out_of_1000"] = res["B"]["attempted_score"]
        res["paired_attempted_all"]["delta_points_out_of_1000"] = pa["delta_pp"] * 10
        res["paired_attempted_all"]["ci95_points_out_of_1000"] = [v * 10 for v in pa["ci95_pp"]]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["bench", "label_a", "label_b"]}, ensure_ascii=False))
    print("A:", {k:res["A"][k] for k in ["n","attempted_score","answered_score","skipped"]}); print("B:", {k:res["B"][k] for k in ["n","attempted_score","answered_score","skipped"]})
    print("paired(attempted):", pa["delta_pp"], pa["ci95_pp"], "p=", pa["sign_flip_p_two_sided"])

main()
