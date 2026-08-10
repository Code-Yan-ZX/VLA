"""Build the qualitative Fig. 1 data bundle."""
from __future__ import annotations
import json, math, shutil
from pathlib import Path
import numpy as np

ROOT = Path("/media/disk2/YZX/research/vla")
DATA_DIR = ROOT / "drafts/figures/qualitative_fig1/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    {"benchmark": "TextVQA", "id": "34863",
     "source_image": ROOT / "runs/data/textvqa/34863.jpg",
     "question": "does it say happy birthday?\nAnswer the question using a single word or phrase.",
     "ground_truth": ["yes"],
     "pre_source": ROOT / "runs/cascade/gate_pre25_textvqa.json",
     "fst_source": ROOT / "runs/cascade/gate_fst25_textvqa.json",
     "post_source": ROOT / "runs/full_matrix/j7_qwen3vl_post_textvqa_r0.750_full.json"},
    {"benchmark": "DocVQA", "id": "52297",
     "source_image": ROOT / "runs/data/docvqa/52297.jpg",
     "question": "What are the dates for the conference?\nAnswer the question using a single word or phrase.",
     "ground_truth": ["July 17 and 18, 1996"],
     "pre_source": ROOT / "runs/cascade/gate_pre25_docvqa.json",
     "fst_source": ROOT / "runs/cascade/gate_fst25_docvqa.json",
     "post_source": ROOT / "runs/j4_baselines_hf/j4d_docvqa_post_cap600k_n200.json"},
    {"benchmark": "OCRBench", "id": "ocr0804",
     "source_image": ROOT / "runs/data/ocrbench/ocr0804.jpg",
     "question": "what is the value for 'CIRCULATION DATES'? Answer this question using the text in the image directly.",
     "ground_truth": ["OCTOBER 1999"],
     "pre_source": ROOT / "runs/cascade/gate_pre25_ocrbench.json",
     "fst_source": ROOT / "runs/cascade/gate_fst25_ocrbench.json",
     "post_source": ROOT / "runs/cascade/gate_post25_ocrbench.json"},
]

def load_sample(p, sid):
    with p.open() as f:
        d = json.load(f)
    for s in d.get("per_sample", []):
        if str(s.get("id", "")) == sid:
            return s
    raise KeyError(f"{sid} not in {p}")

def build_example(ex):
    bench, sid = ex["benchmark"], ex["id"]
    key = f"{bench.lower()}_{sid}"
    out_dir = DATA_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_img = out_dir / "source.jpg"
    if not dst_img.exists():
        shutil.copy(ex["source_image"], dst_img)
    pre_rec = load_sample(ex["pre_source"], sid)
    fst_rec = load_sample(ex["fst_source"], sid)
    post_rec = load_sample(ex["post_source"], sid)
    pre_block = pre_rec.get("pre", {})
    kept_per_image = pre_block.get("kept_per_image", [])
    n_units_full = pre_block.get("n_units_full") or sum(pre_block.get("full_per_image", []))
    n_units_kept = pre_block.get("n_units_kept") or sum(pre_block.get("k_per_image", []))
    if not kept_per_image:
        raise ValueError(f"No kept_per_image for {key}")
    kept_indices = kept_per_image[0]
    # derive square grid shape from n_units_full
    side = int(math.ceil(math.sqrt(n_units_full)))
    if side * side < n_units_full:
        side += 1
    keep = np.zeros(side * side, dtype=bool)
    for idx in kept_indices:
        if idx < side * side:
            keep[idx] = True
    keep = keep.reshape(side, side)
    np.savez_compressed(out_dir / "rbm_mask.npz", keep=keep,
                        kept_indices=np.array(kept_indices, dtype=np.int32),
                        unit_grid_hw=np.array([side, side], dtype=np.int32),
                        n_units_full=int(n_units_full),
                        n_units_kept=int(n_units_kept))
    methods = [
        {"key": "rbm", "label": "RBM (ours)", "answer": pre_rec.get("answer", ""),
         "correct": bool(pre_rec.get("correct", 0)), "ptid": pre_rec.get("prompt_token_ids", 0),
         "mask": f"{key}/rbm_mask.npz", "mask_key": "keep",
         "capture_source": str(ex["pre_source"]), "stage": "pre-merger L2 ranking"},
        {"key": "fastv", "label": "FastV", "answer": fst_rec.get("answer", ""),
         "correct": bool(fst_rec.get("correct", 0)), "ptid": fst_rec.get("prompt_token_ids", 0),
         "mask": f"{key}/rbm_mask.npz", "mask_key": "keep",
         "capture_source": str(ex["fst_source"]), "stage": "FastV layer-K query-conditioned",
         "mask_note": "retained-token indices for FastV not stored; pre (RBM) mask shown for reference"},
        {"key": "post", "label": "Post-L2", "answer": post_rec.get("answer", ""),
         "correct": bool(post_rec.get("correct", 0)), "ptid": post_rec.get("prompt_token_ids", 0),
         "mask": f"{key}/rbm_mask.npz", "mask_key": "keep",
         "capture_source": str(ex["post_source"]), "stage": "post-merger L2 ranking",
         "mask_note": "retained-token indices for Post-L2 not stored; pre (RBM) mask shown for reference"},
    ]
    ptids = [m["ptid"] for m in methods]
    if len(set(ptids)) != 1:
        raise ValueError(f"ptid mismatch for {key}: {ptids}")
    return {"benchmark": bench, "id": sid, "image": f"{key}/source.jpg",
            "question": ex["question"], "ground_truth": ex["ground_truth"],
            "evidence_bbox_source_px": [0, 0, 0, 0], "processor_size_wh": [0, 0],
            "unit_grid_hw": [side, side], "ptid": ptids[0], "keep_ratio": 0.25,
            "n_units_full": int(n_units_full), "n_units_kept": int(n_units_kept),
            "methods": methods}

def main():
    examples = []
    for ex in EXAMPLES:
        print(f"Building {ex['benchmark']} {ex['id']}...")
        examples.append(build_example(ex))
        e = examples[-1]
        print(f"  ptid={e['ptid']}, n_units_kept={e['n_units_kept']}/{e['n_units_full']} ({e['n_units_kept']/e['n_units_full']:.0%})")
    manifest = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "resolved at run time (see provenance.json)",
        "keep_ratio": 0.25,
        "selection_rule": "RBM correct AND >=2 displayed baselines wrong AND same final ptid across all 3 methods",
        "candidate_pool_counts": {
            "TextVQA_cascade_pre25_fst25_j7post_intersection": 24,
            "DocVQA_cascade_pre25_fst25_j4dpost_intersection": 37,
            "OCRBench_cascade_pre25_fst25_post25_intersection": 9,
        },
        "examples": examples,
    }
    with (DATA_DIR / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {DATA_DIR / 'manifest.json'}")

if __name__ == "__main__":
    main()
