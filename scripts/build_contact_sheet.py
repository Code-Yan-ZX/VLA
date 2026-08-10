"""Step C+D: mine OCRBench candidates and build a 12-panel contact sheet.

Three arms on Qwen3-VL-8B at 25% retention:
  (1) RBM = pre25 (pre-merger L2)
  (2) FastV-k3 = locked_fst3_ocrbench_n200.json
  (3) Post-L2 = gate_post25_ocrbench.json (post-merger L2)

Filter: RBM.correct == 1 AND FastV-k3.correct == 0 AND post25.correct == 0.
Rank: GT length ASC, then RBM exact/prefix match preferred, then numeric GTs.
Output: contact_sheet.{png,pdf,svg} + contact_sheet_manifest.json in
        drafts/figures/camera_ready/.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path("/media/disk2/YZX/research/vla")
PRE25 = REPO / "runs/cascade/gate_pre25_ocrbench.json"
FST3 = REPO / "runs/rankbridge/locked_fst3_ocrbench_n200.json"
POST25 = REPO / "runs/cascade/gate_post25_ocrbench.json"
SUBSET = REPO / "eval/subsets/ocrbench_200.jsonl"
OUTDIR = REPO / "drafts/figures/camera_ready"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_arm(path: Path) -> dict:
    d = json.load(open(path))
    return {s["id"]: s for s in d["per_sample"]}


def load_subset() -> dict:
    out = {}
    with open(SUBSET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[o["id"]] = o
    return out


def score_exact(ans: str, gt: str) -> int:
    a = re.sub(r"[^a-z0-9]+", "", ans.lower())
    g = re.sub(r"[^a-z0-9]+", "", gt.lower())
    if not g:
        return 0
    return int(g in a)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("$", "\\$")


def fmt_pred(ok, label, ans, ptid, kept, full):
    mark = "[OK]" if ok else "[X]"
    color = "#137a3c" if ok else "#a02828"
    flat = _escape(ans.replace("\n", " "))
    a = flat[:24]
    a = a + ("..." if len(flat) > 24 else "")
    return (f"{mark} {label}: {a}\n        [{ptid}t, "
            f"{kept}/{full} img]"), color


def main():
    pre = load_arm(PRE25)
    fst = load_arm(FST3)
    if not POST25.exists():
        print(f"FATAL: {POST25} not found. Run cascade post25 first.", file=sys.stderr)
        sys.exit(1)
    pst = load_arm(POST25)
    subset = load_subset()

    common = sorted(set(pre) & set(fst) & set(pst) & set(subset))
    print(f"Common IDs: {len(common)}", flush=True)
    qual = []
    for sid in common:
        ps, fs, ts = pre[sid], fst[sid], pst[sid]
        if ps["skipped"] or fs["skipped"] or ts["skipped"]:
            continue
        if ps["correct"] == 1 and fs["correct"] == 0 and ts["correct"] == 0:
            qual.append({
                "id": sid,
                "image": subset[sid]["image"],
                "question": ps["question"],
                "gt": ps["gt"],
                "pre_ans": ps["answer"],
                "fst_ans": fs["answer"],
                "pst_ans": ts["answer"],
                "pre_ptid": ps["prompt_token_ids"],
                "fst_ptid": fs["prompt_token_ids"],
                "pst_ptid": ts["prompt_token_ids"],
                "pre_nimg_full": ps["n_image_full"],
                "pre_nimg_kept": ps["n_image_kept"],
                "fst_nimg_full": fs["n_image_full"],
                "fst_nimg_kept": fs["n_image_kept"],
                "pst_nimg_full": ts["n_image_full"],
                "pst_nimg_kept": ts["n_image_kept"],
                "pre_ok": 1, "fst_ok": 0, "pst_ok": 0,
            })
    print(f"Qualifying (RBM=1, FastV=0, post=0): {len(qual)}", flush=True)

    def rbm_clean(rec):
        a = rec["pre_ans"].lower()
        score = -len(rec["pre_ans"])
        if "**" in rec["pre_ans"] or "the " in a[:30]:
            score += 50
        if rec["pre_ans"].strip().lower() == rec["gt"].strip().lower():
            score -= 100
        return score

    qual.sort(key=lambda r: (len(r["gt"]), rbm_clean(r)))
    # Pad with 2-arm-rescue (RBM=1, post=0) if fewer than 12 strict cases
    if len(qual) < 12:
        strict_ids = {r["id"] for r in qual}
        qual2 = []
        for sid in common:
            if sid in strict_ids: continue
            ps, fs, ts = pre[sid], fst[sid], pst[sid]
            if ps["skipped"] or ts["skipped"]: continue
            if ps["correct"] == 1 and ts["correct"] == 0:
                qual2.append({
                    "id": sid, "image": subset[sid]["image"],
                    "question": ps["question"], "gt": ps["gt"],
                    "pre_ans": ps["answer"], "fst_ans": fs["answer"],
                    "pst_ans": ts["answer"],
                    "pre_ptid": ps["prompt_token_ids"],
                    "fst_ptid": fs["prompt_token_ids"],
                    "pst_ptid": ts["prompt_token_ids"],
                    "pre_nimg_full": ps["n_image_full"],
                    "pre_nimg_kept": ps["n_image_kept"],
                    "fst_nimg_full": fs["n_image_full"],
                    "fst_nimg_kept": fs["n_image_kept"],
                    "pst_nimg_full": ts["n_image_full"],
                    "pst_nimg_kept": ts["n_image_kept"],
                    "pre_ok": 1, "fst_ok": fs["correct"], "pst_ok": 0,
                })
        qual2.sort(key=lambda r: (len(r["gt"]), rbm_clean(r)))
        qual.extend(qual2[:12 - len(qual)])
    top12 = qual[:12]
    for r in top12:
        print(f"  {r['id']}: q={r['question'][:40]!r} gt={r['gt']!r} "
              f"pre={r['pre_ans'][:30]!r} fst={r['fst_ans'][:20]!r} "
              f"pst={r['pst_ans'][:20]!r}", flush=True)

    manifest = {
        "n_common": len(common),
        "n_qualifying": len(qual),
        "criteria": "RBM.correct==1 AND FastV-k3.correct==0 AND post25.correct==0",
        "rank_key": "(len(gt), rbm_clean): shortest GT first, exact matches preferred",
        "n_top": len(top12),
        "candidates": top12,
    }
    (OUTDIR / "contact_sheet_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))

    if len(qual) < 3:
        print("FATAL: fewer than 3 qualifying candidates; aborting.", file=sys.stderr)
        sys.exit(2)

    ncols, nrows = 6, 2
    panel_w, panel_h = 2.6, 3.4
    fig_w, fig_h = ncols * panel_w, nrows * panel_h
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             gridspec_kw=dict(hspace=0.30, wspace=0.20))
    for i, rec in enumerate(top12):
        r, c = divmod(i, ncols)
        ax = axes[r][c] if nrows > 1 else axes[c]
        ax.set_axis_off()
        try:
            img = Image.open(rec["image"]).convert("RGB")
            iw, ih = img.size
            scale = min(320 / iw, 220 / ih)
            img = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))))
            ax.imshow(img, aspect="auto", extent=(0, 1, 0.55, 1.0))
        except Exception as e:
            ax.text(0.5, 0.78, f"[img err: {e}]", ha="center", va="center",
                    fontsize=6, transform=ax.transAxes)
        q = rec["question"][:50] + ("..." if len(rec["question"]) > 50 else "")
        g = rec["gt"][:30] + ("..." if len(rec["gt"]) > 30 else "")
        ax.text(0, 0.52, f"Q: {_escape(q)}", ha="left", va="top", fontsize=5.5,
                transform=ax.transAxes)
        ax.text(0, 0.43, f"GT: {_escape(g)}", ha="left", va="top", fontsize=6,
                fontweight="bold", color="#1f5b8c", transform=ax.transAxes)
        rbm_text, rbm_col = fmt_pred(rec["pre_ok"], "RBM  ",
                                     rec["pre_ans"], rec["pre_ptid"],
                                     rec["pre_nimg_kept"], rec["pre_nimg_full"])
        fst_text, fst_col = fmt_pred(rec["fst_ok"], "FastV",
                                     rec["fst_ans"], rec["fst_ptid"],
                                     rec["fst_nimg_kept"], rec["fst_nimg_full"])
        pst_text, pst_col = fmt_pred(rec["pst_ok"], "Post ",
                                     rec["pst_ans"], rec["pst_ptid"],
                                     rec["pst_nimg_kept"], rec["pst_nimg_full"])
        ax.text(0, 0.36, rbm_text, ha="left", va="top", fontsize=5.0,
                color=rbm_col, transform=ax.transAxes)
        ax.text(0, 0.24, fst_text, ha="left", va="top", fontsize=5.0,
                color=fst_col, transform=ax.transAxes)
        ax.text(0, 0.12, pst_text, ha="left", va="top", fontsize=5.0,
                color=pst_col, transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
        ax.text(0.5, 1.02, rec["id"], ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", transform=ax.transAxes)

    fig.suptitle("OCRBench @ 25% retention on Qwen3-VL-8B: RBM-rescue candidates",
                 fontsize=11, fontweight="bold", y=0.995)
    legend = ("Green [OK] = correct; Red [X] = wrong. "
              "[Nt, K/F img] = prompt tokens, image-kept/image-full (post-merge). "
              "RBM = pre-merger L2; FastV = query-conditioned at layer 3; "
              "Post = post-merger L2.")
    fig.text(0.5, 0.005, legend, ha="center", va="bottom", fontsize=7)

    for ext in ("png", "pdf", "svg"):
        out = OUTDIR / f"contact_sheet.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  wrote {out}", flush=True)
    plt.close(fig)

    rec_by_recall = sorted(qual, key=lambda r: (-score_exact(r["pre_ans"], r["gt"]),
                                                len(r["gt"])))
    print("\nTop-2 recall-favoring candidates:", flush=True)
    for r in rec_by_recall[:2]:
        print(f"  {r['id']}: gt={r['gt']!r} pre={r['pre_ans'][:50]!r} "
              f"exact={score_exact(r['pre_ans'], r['gt'])}", flush=True)


if __name__ == "__main__":
    main()
