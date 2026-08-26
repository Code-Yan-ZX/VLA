"""Mechanism pre-check v2 — compare two residual constructions + demotion.

Construction A (task-spec): r_a = M(DeltaX) - M(0),  DeltaX = X - mean(X) per group
Construction B (self-referential): r_b = M(X) - M(Xbar),  Xbar = group-mean repeated

Hypotheses tested for BOTH:
  H1 constant group -> ~0
  H2 norm correlates with within-group detail (within_var / edge / demotion)
  H3 sane magnitude regime (not 2-7x base; no NaN)
  H4 independence from base (cos bounded)
Also: merger-rank demotion = l2_score(pre-merge) - ||M(X)|| (post-merge); a unit
whose score DROPS after the merge lost detail -> residual should track it.
Also: among the units RBM keeps at 25% budget, what fraction are high-detail
(bottom-var or top-demotion)? If RBM already keeps the detail units, residual
tokens add nothing.
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


class PassiveRecorder2:
    def __init__(self, merger):
        self.orig = merger.forward
        self.calls = []
        merger.forward = self._wrapped

    def _wrapped(self, *args, **kwargs):
        hs = args[0]
        out = self.orig(hs, *args[1:], **kwargs)
        self._record(hs, out)
        return out

    def _record(self, hs, out):
        try:
            ctx = hs.shape[-1]
            seq = hs.shape[0]
            num_units = seq // 4
            if num_units == 0 or seq % 4 != 0:
                return
            hs4 = hs[:num_units * 4]
            units = hs4.reshape(num_units, 4, ctx).float()     # [U,4,ctx]
            mu = units.mean(dim=1, keepdim=True)               # [U,1,ctx]
            # run the merger on: actual, de-meaned, mean-repeated
            def _m(x):
                return self.orig(x.to(hs.dtype)).float()       # [U, hidden]
            b = _m(units.reshape(-1, 1, ctx)).reshape(num_units, -1)
            delta = (units - mu).reshape(-1, 1, ctx)
            r_a = _m(delta).reshape(num_units, -1) - _m(torch.zeros_like(delta)).reshape(num_units, -1)
            xbar = mu.expand(-1, 4, -1).reshape(-1, 1, ctx)
            r_b = b - _m(xbar).reshape(num_units, -1)
            out_f = out.float()
            b_norm = out_f.norm(dim=-1)
            ra_norm = r_a.norm(dim=-1)
            rb_norm = r_b.norm(dim=-1)
            within_var = units.var(dim=1, unbiased=False).mean(dim=-1)
            l2_score = units.norm(dim=-1).mean(dim=-1)
            tl, tr, bl, br = units[:, 0], units[:, 1], units[:, 2], units[:, 3]
            gx = ((tr + br) - (tl + bl)) * 0.5
            gy = ((bl + br) - (tl + tr)) * 0.5
            edge = gx.norm(dim=-1) + gy.norm(dim=-1)
            demotion = (l2_score - b_norm).tolist()
            cos_ra = torch.nn.functional.cosine_similarity(r_a, out_f, dim=-1)
            cos_rb = torch.nn.functional.cosine_similarity(r_b, out_f, dim=-1)
            self.calls.append({
                "num_units": int(num_units),
                "b_norm": b_norm.tolist(),
                "r_a_norm": ra_norm.tolist(),
                "r_b_norm": rb_norm.tolist(),
                "r_b_maxabs": r_b.abs().amax(dim=-1).tolist(),
                "within_var": within_var.tolist(),
                "l2_score": l2_score.tolist(),
                "edge": edge.tolist(),
                "demotion": demotion,
                "cos_ra_b": cos_ra.tolist(),
                "cos_rb_b": cos_rb.tolist(),
            })
        except Exception as e:
            print(f"[rec] error: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="runs/merger_repr/mech_check_12.jsonl")
    ap.add_argument("--out", default="runs/merger_repr/mech_stats2.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    samples = load_samples(args.subset, args.n)
    print(f"[mech2] {len(samples)} samples; loading ...", flush=True)
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
    rec = PassiveRecorder2(merger)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=0)
    msgs = [make_msgs(s) for s in samples]
    llm.chat(msgs[:1], sampling_params=sp)
    llm.chat(msgs, sampling_params=sp)
    print(f"[mech2] {len(rec.calls)} visual calls recorded", flush=True)

    import math, statistics as st
    rows = []
    for c in rec.calls:
        nu = c["num_units"]
        for i in range(nu):
            rows.append({k: c[k][i] for k in
                         ["b_norm","r_a_norm","r_b_norm","r_b_maxabs","within_var",
                          "l2_score","edge","demotion","cos_ra_b","cos_rb_b"]})
    n = len(rows)

    def spearman(xs, ys):
        if len(xs) < 3:
            return None
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0]*len(v)
            for pos, idx in enumerate(order):
                r[idx] = pos+1
            d = {}
            for i, val in enumerate(v):
                d.setdefault(val, []).append(i)
            for grp in d.values():
                if len(grp) > 1:
                    m = sum(r[i] for i in grp)/len(grp)
                    for i in grp:
                        r[i] = m
            return r
        rx, ry = rank(xs), rank(ys)
        mx = sum(rx)/len(rx); my = sum(ry)/len(ry)
        cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
        sx = math.sqrt(sum((rx[i]-mx)**2 for i in range(len(rx))))
        sy = math.sqrt(sum((ry[i]-my)**2 for i in range(len(ry))))
        return cov/(sx*sy) if sx*sy > 0 else None

    def qbuckets(x, y):
        idx = sorted(range(n), key=lambda i: x[i])
        out = []
        for a, b in zip([0,.25,.5,.75,1], [.25,.5,.75,1,1.0001]):
            grp = [y[i] for i in idx[int(a*n):int(b*n)]]
            out.append(round(st.mean(grp), 3))
        return out

    ra = [r["r_a_norm"] for r in rows]; rb = [r["r_b_norm"] for r in rows]
    bb = [r["b_norm"] for r in rows]; vv = [r["within_var"] for r in rows]
    ee = [r["edge"] for r in rows]; dd = [r["demotion"] for r in rows]
    l2s = [r["l2_score"] for r in rows]
    ra_ratio = sorted([a/b for a,b in zip(ra,bb)])
    rb_ratio = sorted([a/b for a,b in zip(rb,bb)])
    cosb = [r["cos_rb_b"] for r in rows]

    # RBM keep at 25%: top-K by l2 score; what detail does it retain?
    # detail metric = within_var; "demoted" units = high positive demotion
    kept_detail = []
    for c in rec.calls:
        nu = c["num_units"]
        K = max(1, int(round(nu * 0.25)))
        idx = sorted(range(nu), key=lambda i: c["l2_score"][i], reverse=True)[:K]
        kept_var = st.mean([c["within_var"][i] for i in idx])
        all_var = st.mean(c["within_var"])
        kept_dem = st.mean([c["demotion"][i] for i in idx])
        all_dem = st.mean(c["demotion"])
        kept_detail.append({"K": K, "nu": nu,
                            "kept_var_rel": kept_var/max(all_var,1e-9),
                            "kept_dem_rel": kept_dem/max(abs(all_dem),1e-9)})

    summary = {
        "n_units": n,
        "r_a": {"ratio_vs_base_p50": ra_ratio[int(.5*n)],
                "ratio_vs_base_p90": ra_ratio[int(.9*n)],
                "spearman_vs_within_var": spearman(vv, ra),
                "spearman_vs_edge": spearman(ee, ra),
                "spearman_vs_demotion": spearman(dd, ra)},
        "r_b": {"ratio_vs_base_p10": rb_ratio[int(.1*n)],
                "ratio_vs_base_p50": rb_ratio[int(.5*n)],
                "ratio_vs_base_p90": rb_ratio[int(.9*n)],
                "spearman_vs_within_var": spearman(vv, rb),
                "spearman_vs_edge": spearman(ee, rb),
                "spearman_vs_demotion": spearman(dd, rb),
                "spearman_vs_l2": spearman(l2s, rb),
                "zero_frac": sum(1 for v in rb if v < 1e-6)/n,
                "maxabs_mean": st.mean([r["r_b_maxabs"] for r in rows]),
                "cos_b": {"mean": st.mean(cosb), "p10": sorted(cosb)[int(.1*n)]}},
        "r_b_norm_by_var_quartile": qbuckets(vv, rb),
        "r_b_norm_by_demotion_quartile": qbuckets(dd, rb),
        "rbm_keep_detail": kept_detail,
        "nan_inf": sum(1 for v in rb if v != v or v in (float("inf"), float("-inf")))
                   + sum(1 for v in ra if v != v or v in (float("inf"), float("-inf"))),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "calls": rec.calls}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
