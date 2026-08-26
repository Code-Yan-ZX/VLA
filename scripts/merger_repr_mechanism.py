"""Mechanism pre-checks for the Dual-Granularity Merger Representation.

Validates, on a small mixed sample set (NOT an evaluation), the mechanism
hypotheses for the residual-detail token r_i = M(DeltaX_i) - M(0):

  H1. M(0) is a fixed constant (bias response), and r_i ~ 0 for constant groups
      (DeltaX_i = 0 -> r_i = M(0) - M(0) = 0 by construction).
  H2. residual norm correlates with within-group high-frequency content
      (within-unit variance / Sobel-style edge score / merger-rank demotion).
  H3. No numerical explosion / distribution shift (residual norms in a sane
      range vs base norms; no NaN/Inf; residual cosine with base is bounded).
  H4. Full-retention / native-merger path is NOT modified (r=0 -> byte-identical
      native forward; the recorder only OBSERVES).

Design: passive recorder on visual.merger (no pruning, r=0). On each main-merger
call, capture the pre-merge 4-patch units and the merged output, and compute the
residual via the ORIGINAL (unwrapped) merger on the de-meaned units. Uses
vLLM 0.19 (qwen3vl_clean env) exactly as v3_premerger_runner.py does.

Usage:
  /home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/merger_repr_mechanism.py \
      --subset runs/merger_repr/mech_check_12.jsonl --out runs/merger_repr/mech_stats.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_CACHE", "/data/models/huggingface/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_samples(path, n):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            s = json.loads(line)
            out.append({"id": s["id"], "image": s["image"],
                        "question": s["question"], "gt": s["gt"]})
    return out


def make_msgs(s):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file://" + s["image"]}},
        {"type": "text", "text": s["question"]},
    ]}]


class PassiveRecorder:
    """Records per-main-merger-call pre-merge units + residuals (no pruning)."""

    def __init__(self, merger):
        self.orig = merger.forward
        self.m_zero = None            # M(0) cached constant
        self.calls = []               # per-visual-call dicts
        self._pending = None
        merger.forward = self._wrapped

    def _wrapped(self, *args, **kwargs):
        hs = args[0]                  # [seq, 1, ctx]
        out = self.orig(hs, *args[1:], **kwargs)
        self._record(hs, out)
        return out

    def _record(self, hs, out):
        try:
            ctx = hs.shape[-1]
            seq = hs.shape[0]
            num_units = seq // 4
            if num_units == 0 or seq % 4 != 0:
                # sub-unit / dangling patches: skip (not merge-decomposable)
                return
            hs4 = hs[:num_units * 4]
            units = hs4.reshape(num_units, 4, ctx)          # [U, 4, ctx]
            # M(0) = orig(zeros) -> constant; compute once
            if self.m_zero is None:
                z = torch.zeros(4, 1, ctx, dtype=hs.dtype, device=hs.device)
                m0 = self.orig(z)
                self.m_zero = m0[0].float().detach().cpu()  # [d_model]
            delta = units - units.mean(dim=1, keepdim=True)  # [U, 4, ctx]
            delta_flat = delta.reshape(num_units * 4, 1, ctx)
            r_all = self.orig(delta_flat)                    # [U, d_model]
            r_all = r_all.float().detach().cpu() - self.m_zero  # r = M(dX) - M(0)
            b_all = out.float().detach().cpu()               # [U, d_model]
            un = units.float().detach().cpu()
            # per-unit statistics
            b_norm = b_all.norm(dim=-1)                       # [U]
            r_norm = r_all.norm(dim=-1)
            r_maxabs = r_all.abs().amax(dim=-1)
            within_var = un.var(dim=1, unbiased=False).mean(dim=-1)
            l2_score = un.norm(dim=-1).mean(dim=-1)
            tl, tr, bl, br = un[:, 0], un[:, 1], un[:, 2], un[:, 3]
            gx = ((tr + br) - (tl + bl)) * 0.5
            gy = ((bl + br) - (tl + tr)) * 0.5
            edge = gx.norm(dim=-1) + gy.norm(dim=-1)
            cos_rb = torch.nn.functional.cosine_similarity(
                r_all, b_all, dim=-1)
            self.calls.append({
                "num_units": int(num_units),
                "b_norm": b_norm.tolist(),
                "r_norm": r_norm.tolist(),
                "r_maxabs": r_maxabs.tolist(),
                "within_var": within_var.tolist(),
                "l2_score": l2_score.tolist(),
                "edge": edge.tolist(),
                "cos_rb": cos_rb.tolist(),
            })
        except Exception as e:  # recorder must never crash the engine
            print(f"[rec] error: {type(e).__name__}: {str(e)[:160]}",
                  file=sys.stderr, flush=True)


def summarize(calls, out_path):
    import statistics as st

    all_r = [v for c in calls for v in c["r_norm"]]
    all_b = [v for c in calls for v in c["b_norm"]]
    all_ra = [v for c in calls for v in c["r_maxabs"]]
    all_cos = [v for c in calls for v in c["cos_rb"]]
    n = len(all_r)
    mzero = calls[0].get("m_zero") if calls else None
    mzero_norm = float(torch.tensor(mzero).norm()) if mzero else None
    # pooled Spearman between r_norm and each candidate explanatory variable
    def spearman(xs, ys):
        import math
        if len(xs) < 3:
            return None
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0] * len(v)
            for pos, idx in enumerate(order):
                r[idx] = pos + 1
            # tie-average
            d = {}
            for i, val in enumerate(v):
                d.setdefault(val, []).append(i)
            for grp in d.values():
                if len(grp) > 1:
                    m = sum(r[i] for i in grp) / len(grp)
                    for i in grp:
                        r[i] = m
            return r
        rx, ry = rank(xs), rank(ys)
        n2 = len(rx)
        mx, my = sum(rx) / n2, sum(ry) / n2
        cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n2))
        sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n2)))
        sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n2)))
        return cov / (sx * sy) if sx * sy > 0 else None

    def paired(var_key):
        xs, ys = [], []
        for c in calls:
            xs += c[var_key]
            ys += c["r_norm"]
        return spearman(xs, ys)

    # mean-patch L2 "demotion": base norm ratio vs unit rank by l2 score
    demote = []
    for c in calls:
        order = sorted(range(c["num_units"]),
                       key=lambda i: c["l2_score"][i], reverse=True)
        for rank_i, i in enumerate(order):
            # rank-normalized demotion = r_norm relative to the image's
            # r_norm distribution, by l2-score rank (no per-dataset tuning)
            demote.append((rank_i / max(1, c["num_units"]),
                           c["r_norm"][i]))
    dem_x = [d[0] for d in demote]
    dem_y = [d[1] for d in demote]

    summary = {
        "n_units_total": n,
        "n_visual_calls": len(calls),
        "m_zero_norm": mzero_norm,
        "r_norm": {"min": min(all_r), "max": max(all_r),
                   "mean": st.mean(all_r), "p50": st.median(all_r),
                   "p90": sorted(all_r)[int(0.9 * (n - 1))]},
        "b_norm": {"min": min(all_b), "max": max(all_b),
                   "mean": st.mean(all_b), "p50": st.median(all_b)},
        "r_maxabs": {"max": max(all_ra), "mean": st.mean(all_ra)},
        "cos_rb": {"min": min(all_cos), "max": max(all_cos),
                   "mean": st.mean(all_cos)},
        "nan_inf_count": sum(1 for v in all_r if v != v or v in (float("inf"), float("-inf"))),
        "spearman_r_vs_within_var": paired("within_var"),
        "spearman_r_vs_edge": paired("edge"),
        "spearman_r_vs_l2": paired("l2_score"),
        "spearman_r_vs_l2rank_normalized": spearman(dem_x, dem_y),
        "zero_r_units_frac": sum(1 for v in all_r if v < 1e-6) / n,
        "frac_r_gt_0.1_b_norm": sum(1 for a, b in zip(all_r, all_b)
                                    if a > 0.1 * b) / n,
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "calls": calls}, f, indent=1)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="runs/merger_repr/mech_check_12.jsonl")
    ap.add_argument("--out", default="runs/merger_repr/mech_stats.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=0)
    args = ap.parse_args()

    samples = load_samples(args.subset, args.n)
    print(f"[mech] {len(samples)} samples; loading {MODEL_ID} ...", flush=True)

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=MODEL_ID, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.90, max_model_len=16384,
        enforce_eager=True, limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=os.path.abspath(REPO),
        max_num_seqs=1, disable_log_stats=True, enable_prefix_caching=False,
        seed=0,
    )
    llm = LLM(**llm_kwargs)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    merger = model.visual.merger
    rec = PassiveRecorder(merger)
    # stash m_zero into the recorder's summary accessor
    rec.calls_mzero = None

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=0)
    chat_kw = {}
    if args.max_pixels and args.max_pixels > 0:
        chat_kw["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}
    msgs = [make_msgs(s) for s in samples]
    llm.chat(msgs[:1], sampling_params=sp, **chat_kw)  # warmup
    outs = llm.chat(msgs, sampling_params=sp, **chat_kw)

    # pull m_zero out of the recorder (first computed value)
    mzero = getattr(rec, "m_zero", None)
    if mzero is not None and rec.calls:
        rec.calls[0]["m_zero"] = mzero.tolist()
    print(f"[mech] {len(rec.calls)} visual calls recorded", flush=True)
    for s, o in zip(samples, outs):
        print(f"  {s['id']:>12s} -> {o.outputs[0].text.strip()[:50]!r}", flush=True)
    summarize(rec.calls, args.out)


if __name__ == "__main__":
    main()
