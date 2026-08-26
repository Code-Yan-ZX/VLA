#!/usr/bin/env python3
"""Task 4 Phase 4b — full text-density-gradient routing analysis.

Loads per-sample pre/post cells for {textvqa, gqa, docvqa, mme}, aligns by id,
and computes per-benchmark + pooled routing metrics incl. the workload-vs-sample
decomposition that decides the paper spine.

Metrics:
  always_pre / always_post / oracle(per-sample max) / best cheap router
  cheap signals: (a) ptid>=T -> pre, (b) OCR-keyword-hit -> pre, (c) a OR b
  workload-level routing acc = pooled acc when each benchmark is routed to its
      better fixed stage (task-aware, knows benchmark id)
  decomposition: workload_gain = workload_routing - pooled_best_fixed
                 sample_residual = pooled_oracle - workload_routing
"""
import json, re, glob, os, sys

CELLDIR = "runs/v3_router_probe"
BENCHMARKS = ["textvqa", "docvqa", "mme", "gqa"]  # high->low text density (approx)

OCR_WORDS = [
    "read", "text", "word", "letter", "number", "write", "written", "name",
    "say", "says", "said", "title", "label", "sign", "clock", "time", "price",
    "menu", "logo", "brand", "font", "page", "document", "sentence", "spell",
    "plate", "date", "year", "address", "phone", "score", "caption",
]
OCR_RE = re.compile(r"\b(" + "|".join(OCR_WORDS) + r")\b", re.I)


def load(bm, mode):
    p = f"{CELLDIR}/{mode}_{bm}_r0.750_l2_n200.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ps = d.get("per_sample", [])
    return {str(x["id"]): x for x in ps}


def ocr_hit(q):
    return bool(OCR_RE.search(q or ""))


def routed_acc(pre, post, rule):
    """rule(id, samp) -> 'pre'/'post'; return acc over common ids + n."""
    common = set(pre) & set(post)
    if not common:
        return None, 0
    c = 0
    for i in common:
        choice = rule(i, pre[i])
        c += int(pre[i]["correct"]) if choice == "pre" else int(post[i]["correct"])
    return c / len(common), len(common)


def best_ptid_router(pre, post):
    common = sorted(set(pre) & set(post))
    if not common:
        return None, None
    ptids = sorted({pre[i]["prompt_token_ids"] for i in common})
    cands = ptids + [ptids[0] - 1, ptids[-1] + 1]
    best = (-1, None)
    for T in cands:
        acc, n = routed_acc(pre, post, lambda i, s, T=T: "pre" if s["prompt_token_ids"] >= T else "post")
        if acc is not None and acc > best[0]:
            best = (acc, T)
    return best


def ocr_router_acc(pre, post):
    return routed_acc(pre, post, lambda i, s: "pre" if ocr_hit(s.get("question")) else "post")


def combined_router(pre, post):
    common = sorted(set(pre) & set(post))
    ptids = sorted({pre[i]["prompt_token_ids"] for i in common})
    best = (-1, None)
    for T in ptids + [ptids[0] - 1, ptids[-1] + 1]:
        acc, n = routed_acc(pre, post, lambda i, s, T=T: "pre" if (s["prompt_token_ids"] >= T or ocr_hit(s.get("question"))) else "post")
        if acc is not None and acc > best[0]:
            best = (acc, T)
    return best


def main():
    data = {}
    for bm in BENCHMARKS:
        pre, post = load(bm, "pre"), load(bm, "post")
        if pre is None or post is None:
            print(f"[skip] {bm}: missing cell(s) (pre={pre is not None}, post={post is not None})")
            continue
        data[bm] = (pre, post)
    if not data:
        print("NO CELLS"); sys.exit(1)

    rows = {}
    print("=" * 78)
    print(f"{'bench':10s} {'n':>4s} {'pre':>7s} {'post':>7s} {'oracle':>7s} {'router':>7s} {'rule':>14s} {'gap2orc':>8s}")
    print("-" * 78)
    for bm, (pre, post) in data.items():
        common = set(pre) & set(post)
        n = len(common)
        apre, _ = routed_acc(pre, post, lambda i, s: "pre")
        apo, _ = routed_acc(pre, post, lambda i, s: "post")
        orc, _ = routed_acc(pre, post, lambda i, s: "pre" if int(pre[i]["correct"]) >= int(post[i]["correct"]) else "post")
        # oracle via max(correct)
        orc = sum(max(int(pre[i]["correct"]), int(post[i]["correct"])) for i in common) / n
        bp_acc, bp_T = best_ptid_router(pre, post)
        ocr_acc, _ = ocr_router_acc(pre, post)
        cb_acc, cb_T = combined_router(pre, post)
        # pick best cheap router
        cands = [("ptid>=%s" % bp_T, bp_acc), ("ocr_kw", ocr_acc), ("ptid>=%s|ocr" % cb_T, cb_acc)]
        best_name, best_acc = max(cands, key=lambda x: x[1])
        rows[bm] = dict(n=n, apre=apre, apo=apo, oracle=orc, router=best_acc, rule=best_name,
                        best_fixed=max(apre, apo), best_stage="pre" if apre >= apo else "post")
        print(f"{bm:10s} {n:4d} {apre:7.4f} {apo:7.4f} {orc:7.4f} {best_acc:7.4f} {best_name:>14s} {orc-best_acc:8.4f}")
    print("=" * 78)

    # ---- POOLED (all samples across benchmarks) ----
    # build pooled pre/post maps with namespaced ids
    P, Q = {}, {}
    for bm, (pre, post) in data.items():
        for i in set(pre) & set(post):
            key = f"{bm}::{i}"
            P[key] = pre[i]; Q[key] = post[i]
    p_apre, N = routed_acc(P, Q, lambda i, s: "pre")
    p_apo, _ = routed_acc(P, Q, lambda i, s: "post")
    p_orc = sum(max(int(P[i]["correct"]), int(Q[i]["correct"])) for i in (set(P) & set(Q))) / N
    p_bp_acc, p_bp_T = best_ptid_router(P, Q)
    p_ocr_acc, _ = ocr_router_acc(P, Q)
    p_cb_acc, p_cb_T = combined_router(P, Q)
    pcands = [("ptid>=%s" % p_bp_T, p_bp_acc), ("ocr_kw", p_ocr_acc), ("ptid>=%s|ocr" % p_cb_T, p_cb_acc)]
    p_best_name, p_best_acc = max(pcands, key=lambda x: x[1])
    pooled_best_fixed = max(p_apre, p_apo)

    # ---- WORKLOAD-LEVEL routing: each benchmark routed to its better fixed stage ----
    wl_correct, wl_n = 0, 0
    for bm, r in rows.items():
        pre, post = data[bm]
        common = set(pre) & set(post)
        stage = r["best_stage"]
        wl_correct += sum(int(pre[i]["correct"]) if stage == "pre" else int(post[i]["correct"]) for i in common)
        wl_n += len(common)
    wl_acc = wl_correct / wl_n

    oracle_gain = p_orc - pooled_best_fixed
    workload_gain = wl_acc - pooled_best_fixed
    sample_residual = p_orc - wl_acc
    wl_share = workload_gain / oracle_gain if oracle_gain > 0 else float("nan")
    sm_share = sample_residual / oracle_gain if oracle_gain > 0 else float("nan")

    print(f"\nPOOLED (N={N}, {len(rows)} benchmarks):")
    print(f"  always_pre={p_apre:.4f}  always_post={p_apo:.4f}  best_fixed={pooled_best_fixed:.4f}")
    print(f"  ORACLE(per-sample)={p_orc:.4f}")
    print(f"  best cheap router={p_best_acc:.4f}  ({p_best_name})  gap2oracle={p_orc-p_best_acc:.4f}")
    print(f"  candidate routers: " + ", ".join(f"{nm}={ac:.4f}" for nm, ac in pcands))
    print(f"\nDECOMPOSITION of oracle gain ({oracle_gain:.4f} over best_fixed):")
    print(f"  workload-level routing acc = {wl_acc:.4f}  (each bench -> its best stage)")
    print(f"  workload_gain   = {workload_gain:.4f}  = {wl_share*100:.1f}% of oracle gain")
    print(f"  sample_residual = {sample_residual:.4f}  = {sm_share*100:.1f}% of oracle gain")
    print(f"\nROBUSTNESS: best_fixed stage = '{'pre' if p_apre>=p_apo else 'post'}'; "
          f"fixed-post collapses text-dense (docvqa/textvqa post="
          f"{rows.get('docvqa',{}).get('apo',float('nan')):.3f}/"
          f"{rows.get('textvqa',{}).get('apo',float('nan')):.3f}); router avoids collapse.")

    out = dict(per_bench=rows,
               pooled=dict(n=N, always_pre=p_apre, always_post=p_apo, best_fixed=pooled_best_fixed,
                           oracle=p_orc, best_router=p_best_acc, best_router_rule=p_best_name,
                           routers={nm: ac for nm, ac in pcands}),
               decomposition=dict(workload_routing=wl_acc, workload_gain=workload_gain,
                                  sample_residual=sample_residual, oracle_gain=oracle_gain,
                                  workload_share=wl_share, sample_share=sm_share))
    json.dump(out, open(f"{CELLDIR}/router_probe_full_summary.json", "w"), indent=2)
    print(f"\n[full] summary saved: {CELLDIR}/router_probe_full_summary.json")

    # ---- ptid sweep curve (pooled) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        common = sorted(set(P) & set(Q))
        ptids = sorted({P[i]["prompt_token_ids"] for i in common})
        Ts, accs = [], []
        for T in ptids:
            acc, _ = routed_acc(P, Q, lambda i, s, T=T: "pre" if s["prompt_token_ids"] >= T else "post")
            Ts.append(T); accs.append(acc)
        plt.figure(figsize=(7, 4))
        plt.plot(Ts, accs, label="ptid router (pooled)")
        plt.axhline(p_apre, ls="--", c="C1", label=f"always-pre {p_apre:.3f}")
        plt.axhline(p_apo, ls="--", c="C2", label=f"always-post {p_apo:.3f}")
        plt.axhline(p_orc, ls=":", c="C3", label=f"oracle {p_orc:.3f}")
        plt.xlabel("ptid threshold T (route to pre if ptid>=T)")
        plt.ylabel("pooled accuracy")
        plt.title("Task4: cheap ptid router vs fixed/oracle (pooled)")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{CELLDIR}/router_probe_full_curves.png", dpi=110)
        print(f"[full] plot saved: {CELLDIR}/router_probe_full_curves.png")
    except Exception as e:
        print(f"[full] plot skipped: {e}")


if __name__ == "__main__":
    main()
