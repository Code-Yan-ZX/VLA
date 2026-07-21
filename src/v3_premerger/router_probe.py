#!/usr/bin/env python3
"""Task 4 Phase 4a — Adaptive stage router probe.

Determines whether a cheap input signal can per-sample predict whether
pre-merger or post-merger pruning is better, and how close the best cheap
signal gets to the oracle per-sample router upper bound.

Usage (two phases):
  Phase 1 — collect per-sample data (GPU, ~1 GPU·h serial):
    bash src/v3_premerger/router_probe.sh
  Phase 2 — analyse (CPU only):
    python src/v3_premerger/router_probe.py
"""
import json, os, re, sys, itertools
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = Path("runs/v3_router_probe")
BENCHMARKS = ["textvqa", "gqa"]
MODES = ["pre", "post"]

# OCR / text-dense keyword regex (case-insensitive)
OCR_KEYWORDS_BASE = [
    "read", "text", "word", "letter", "sign", "document", "ocr",
    "number", "name", "write", "written", "say", "says", "spell",
    "spelling", "printed", "print", "label", "caption", "title",
    "headline", "font", "character", "symbol", "numeral",
]

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_per_sample(benchmark: str, mode: str) -> dict:
    """Return {sample_id: {correct, question, gt, prompt_token_ids, ...}}."""
    p = OUT_DIR / f"{mode}_{benchmark}_r0.750_l2_n200.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing per-sample cell: {p}")
    d = json.load(open(p))
    if "per_sample" not in d:
        raise ValueError(f"{p} has no per_sample field — runner was not patched")
    out = {}
    for s in d["per_sample"]:
        if s.get("skipped"):
            continue
        out[str(s["id"])] = s
    return out


def align(pre: dict, post: dict):
    """Align by sample id, return list of (id, pre_correct, post_correct, question, ptid_pre, ptid_post)."""
    ids = sorted(set(pre.keys()) & set(post.keys()))
    rows = []
    for sid in ids:
        p, q = pre[sid], post[sid]
        rows.append((
            sid,
            int(p["correct"]),
            int(q["correct"]),
            p.get("question", ""),
            p.get("prompt_token_ids", 0),
            q.get("prompt_token_ids", 0),
        ))
    return rows


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def ocr_score(question: str, keywords: list[str]) -> bool:
    """True if any keyword matches (case-insensitive word boundary)."""
    q = question.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", q):
            return True
    return False


def ptid_threshold(ptid: int, thresh: int) -> bool:
    """True if ptid >= thresh (heavy image → predict pre is better)."""
    return ptid >= thresh


# ---------------------------------------------------------------------------
# Router evaluation
# ---------------------------------------------------------------------------

def eval_router(rows, predict_pre_fn):
    """predict_pre_fn(row) -> bool (True = use pre, False = use post).
    Returns (acc, n_pre, n_post)."""
    correct = 0
    n_pre = n_post = 0
    for row in rows:
        sid, pre_c, post_c, q, ptid_pre, ptid_post = row
        use_pre = predict_pre_fn(row)
        if use_pre:
            correct += pre_c
            n_pre += 1
        else:
            correct += post_c
            n_post += 1
    return correct / len(rows) if rows else 0.0, n_pre, n_post


def oracle_acc(rows):
    return np.mean([max(r[1], r[2]) for r in rows]) if rows else 0.0


def always_acc(rows, mode):
    idx = 1 if mode == "pre" else 2
    return np.mean([r[idx] for r in rows]) if rows else 0.0


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_ocr_keywords(rows, keyword_pool):
    """Try all subsets? Too expensive. Try each keyword alone + cumulative top-K."""
    results = []
    # Single keyword
    for kw in keyword_pool:
        acc, np_, npo = eval_router(rows, lambda r, k=kw: ocr_score(r[3], [k]))
        results.append({"signal": f"ocr_single:{kw}", "acc": acc, "n_pre": np_, "n_post": npo})
    # Cumulative: add keywords one by one (greedy)
    used = []
    remaining = list(keyword_pool)
    best_acc = always_acc(rows, "post")  # baseline = all-post
    for step in range(len(keyword_pool)):
        best_kw, best_a, best_np, best_npo = None, -1, 0, 0
        for kw in remaining:
            trial = used + [kw]
            a, np_, npo = eval_router(rows, lambda r, t=trial: ocr_score(r[3], t))
            if a > best_a:
                best_kw, best_a, best_np, best_npo = kw, a, np_, npo
        if best_kw is None:
            break
        used.append(best_kw)
        remaining.remove(best_kw)
        results.append({
            "signal": f"ocr_greedy_top{step+1}",
            "keywords": list(used),
            "acc": best_a, "n_pre": best_np, "n_post": best_npo,
        })
        if best_a <= best_acc:
            break
        best_acc = best_a
    return results


def sweep_ptid_threshold(rows):
    """Sweep over prompt_token_ids thresholds (use pre-merger ptid as proxy for image size)."""
    ptids = sorted(set(r[4] for r in rows))  # ptid_pre
    if not ptids:
        return []
    # Sample ~20 quantile thresholds
    quantiles = np.linspace(0, 1, 21)
    thresholds = sorted(set(int(np.quantile(ptids, q)) for q in quantiles))
    results = []
    for t in thresholds:
        acc, np_, npo = eval_router(rows, lambda r, th=t: ptid_threshold(r[4], th))
        results.append({"signal": f"ptid>={t}", "threshold": t, "acc": acc,
                        "n_pre": np_, "n_post": npo})
    return results


def sweep_combined(rows, best_keywords, ptid_thresholds):
    """OCR-match → pre, else ptid>=thresh → pre, else post."""
    results = []
    for nkw in [1, 2, 3, 5]:
        kws = best_keywords[:nkw]
        if not kws:
            continue
        for t in ptid_thresholds:
            def fn(r, k=kws, th=t):
                return ocr_score(r[3], k) or ptid_threshold(r[4], th)
            acc, np_, npo = eval_router(rows, fn)
            results.append({
                "signal": f"ocr{nkw}_or_ptid>={t}",
                "keywords": kws, "threshold": t,
                "acc": acc, "n_pre": np_, "n_post": npo,
            })
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(sweep_data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (bench, data) in zip(axes, sweep_data.items()):
        if not data["ocr_sweep"] and not data["ptid_sweep"]:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(bench)
            continue

        # OCR sweep (greedy cumulative)
        ocr_greedy = [s for s in data["ocr_sweep"] if s["signal"].startswith("ocr_greedy")]
        if ocr_greedy:
            xs = list(range(1, len(ocr_greedy) + 1))
            ys = [s["acc"] for s in ocr_greedy]
            ax.plot(xs, ys, "o-", color="tab:blue", label="OCR keyword (cumulative)")

        # ptid sweep
        if data["ptid_sweep"]:
            xs_p = [s["threshold"] for s in data["ptid_sweep"]]
            ys_p = [s["acc"] for s in data["ptid_sweep"]]
            ax2 = ax.twinx()
            ax2.plot(xs_p, ys_p, "s--", color="tab:orange", label="ptid threshold")
            ax2.set_ylabel("Router acc (ptid)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Reference lines
        ax.axhline(data["always_pre"], color="tab:green", ls=":", label=f"always-pre ({data['always_pre']:.3f})")
        ax.axhline(data["always_post"], color="tab:red", ls=":", label=f"always-post ({data['always_post']:.3f})")
        ax.axhline(data["oracle"], color="black", ls="--", lw=2, label=f"oracle ({data['oracle']:.3f})")

        ax.set_title(f"{bench} (n={data['n_aligned']})")
        ax.set_xlabel("Signal parameter")
        ax.set_ylabel("Router accuracy")
        ax.legend(loc="best", fontsize=7)
        ax.set_ylim(0.0, 1.0)

    plt.suptitle("Task 4 Phase 4a — Router probe: cheap signal vs oracle upper bound",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[probe] plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    missing = []
    for bm in BENCHMARKS:
        for mode in MODES:
            p = OUT_DIR / f"{mode}_{bm}_r0.750_l2_n200.json"
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("=" * 70)
        print("MISSING per-sample data. Run the following GPU commands first:")
        print("=" * 70)
        for mode in MODES:
            for bm in BENCHMARKS:
                name = f"{mode}_{bm}_r0.750_l2_n200"
                print(f"""
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean && \\
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 && \\
python src/v3_premerger/v3_premerger_runner.py --mode {mode} --r 0.75 --selector l2 \\
  --benchmark {bm} --subset eval/subsets/{bm}_200.jsonl --n 200 --max-num-seqs 16 \\
  --out runs/v3_router_probe/{name}.json""")
        print("=" * 70)
        print(f"Missing files ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Load & align
    all_rows = []
    per_bench = {}
    for bm in BENCHMARKS:
        pre = load_per_sample(bm, "pre")
        post = load_per_sample(bm, "post")
        rows = align(pre, post)
        per_bench[bm] = rows
        all_rows.extend(rows)
        print(f"[probe] {bm}: pre={len(pre)} post={len(post)} aligned={len(rows)}")

    # Compute metrics per benchmark + pooled
    sweep_data = {}
    summary_rows = []
    for label, rows in {**per_bench, "pooled": all_rows}.items():
        ap = always_acc(rows, "pre")
        apo = always_acc(rows, "post")
        ora = oracle_acc(rows)

        ocr_sw = sweep_ocr_keywords(rows, OCR_KEYWORDS_BASE)
        ptid_sw = sweep_ptid_threshold(rows)

        # Find best greedy keywords for combined sweep
        best_greedy = [s for s in ocr_sw if s["signal"].startswith("ocr_greedy")]
        best_kws = best_greedy[-1]["keywords"] if best_greedy else []
        ptid_thresh = [s["threshold"] for s in ptid_sw if s["acc"] == max((x["acc"] for x in ptid_sw), default=0)]
        combined_sw = sweep_combined(rows, best_kws, ptid_thresh[:3]) if best_kws and ptid_thresh else []

        # Best router overall
        all_sweeps = ocr_sw + ptid_sw + combined_sw
        best_router = max(all_sweeps, key=lambda x: x["acc"]) if all_sweeps else None
        best_acc = best_router["acc"] if best_router else max(ap, apo)
        gap = ora - best_acc

        sweep_data[label] = {
            "always_pre": ap, "always_post": apo, "oracle": ora,
            "n_aligned": len(rows),
            "ocr_sweep": ocr_sw, "ptid_sweep": ptid_sw, "combined_sweep": combined_sw,
            "best_router": best_router, "gap_to_oracle": gap,
        }

        summary_rows.append({
            "benchmark": label, "n": len(rows),
            "always_pre": round(ap, 4), "always_post": round(apo, 4),
            "oracle": round(ora, 4),
            "best_router_acc": round(best_acc, 4),
            "best_router_signal": best_router["signal"] if best_router else "N/A",
            "gap_to_oracle": round(gap, 4),
        })

        print(f"\n{'='*60}")
        print(f"  {label.upper()}  (n={len(rows)})")
        print(f"{'='*60}")
        print(f"  always-pre:  {ap:.4f}")
        print(f"  always-post: {apo:.4f}")
        print(f"  oracle:      {ora:.4f}")
        print(f"  best router: {best_acc:.4f}  ({best_router['signal'] if best_router else 'N/A'})")
        print(f"  gap→oracle:  {gap:.4f}")

    # Save summary
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = OUT_DIR / "router_probe_summary.json"
    json.dump({
        "summary": summary_rows,
        "sweep_data": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("ocr_sweep", "ptid_sweep", "combined_sweep")}
                       for k, v in sweep_data.items()},
    }, open(summary_path, "w"), indent=2)
    print(f"\n[probe] summary saved: {summary_path}")

    # Save detailed sweeps
    detail_path = OUT_DIR / "router_probe_sweeps.json"
    json.dump({k: {"ocr_sweep": v["ocr_sweep"], "ptid_sweep": v["ptid_sweep"],
                    "combined_sweep": v["combined_sweep"]}
               for k, v in sweep_data.items()},
              open(detail_path, "w"), indent=2)
    print(f"[probe] sweeps saved: {detail_path}")

    # Plot
    plot_path = OUT_DIR / "router_probe_curves.png"
    plot_results({k: v for k, v in sweep_data.items() if k != "pooled"}, plot_path)

    # Print final recommendation
    pooled = sweep_data["pooled"]
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    print(f"  Pooled oracle upper bound: {pooled['oracle']:.4f}")
    print(f"  Best cheap router:         {pooled['best_router']['acc']:.4f} "
          f"({pooled['best_router']['signal']})")
    print(f"  Gap:                       {pooled['gap_to_oracle']:.4f}")
    if pooled["gap_to_oracle"] < 0.02:
        print("  → Cheap signal is near-oracle. Proceed to 4b with this router.")
    elif pooled["gap_to_oracle"] < 0.05:
        print("  → Cheap signal is decent. Consider 4b but leave room for heavier signal.")
    else:
        print("  → Cheap signal has significant gap. Need heavier signal (e.g. attn-based) for 4b.")


if __name__ == "__main__":
    main()
