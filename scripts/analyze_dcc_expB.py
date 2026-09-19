#!/usr/bin/env python
"""DCC pre-submission Experiment B analysis: Qwen3 TextVQA full, HF harness.

Arms (results/dcc_presubmit/expB/):
  full_textvqa_rbm_pre_r25.json : --mode pre --r-pre 0.25 --mrope native
  full_textvqa_fastv_k3.json    : --mode fastv --r 0.75 --fastv-k 3
Checks: completion/failure counts, identical sample IDs, identical final
visual token counts (n_image_kept per sample), official VQA-acc +
paired bootstrap 95% CI (RBM - FastV). Reported regardless of sign.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from v3_premerger.paired_stats import run_pair  # noqa: E402

EXP_B = os.path.join(REPO, "results", "dcc_presubmit", "expB")
ARMS = {"rbm": "full_textvqa_rbm_pre_r25.json",
        "fastv": "full_textvqa_fastv_k3.json"}


def main():
    data = {}
    for tag, fn in ARMS.items():
        path = os.path.join(EXP_B, fn)
        if not os.path.exists(path):
            print(f"[MISS] {tag}: {fn}")
            continue
        d = json.load(open(path))
        ps = d.get("per_sample") or []
        n_skip = sum(1 for s in ps if s.get("skipped"))
        print(f"[{tag}] n={d.get('n')} complete={len(ps) - n_skip} "
              f"skipped={n_skip} acc={d.get('acc')} "
              f"mean_ptid={d.get('mean_ptid_len')}")
        data[tag] = d
    if len(data) < 2:
        print("not enough arms for pairing")
        return 1

    rbm = {s["id"]: s for s in data["rbm"]["per_sample"] if not s.get("skipped")}
    fsv = {s["id"]: s for s in data["fastv"]["per_sample"] if not s.get("skipped")}
    common = sorted(set(rbm) & set(fsv), key=str)
    print(f"[pair] common non-skipped IDs: {len(common)} "
          f"(rbm-only {len(set(rbm) - set(fsv))}, fastv-only {len(set(fsv) - set(rbm))})")

    # final visual token counts must match arm-to-arm (both keep 25%)
    mism = [i for i in common
            if rbm[i].get("n_image_kept") != fsv[i].get("n_image_kept")]
    print(f"[iso-token] n_image_kept mismatches: {len(mism)}"
          + (f" e.g. {[(i, rbm[i].get('n_image_kept'), fsv[i].get('n_image_kept')) for i in mism[:5]]}"
             if mism else ""))
    # per-sample VQA acc (baselines_hf 'correct' field == official scorer)
    sa = {i: float(rbm[i]["correct"]) for i in common}
    sb = {i: float(fsv[i]["correct"]) for i in common}
    pair = run_pair(sa, sb, "textvqa", "RBM pre r0.25 (HF)", "FastV r0.75 k3 (HF)",
                    n_resamples=20000, seed=20260919, n_nominal=5000)
    print(f"[pair] VQA-acc RBM={pair['mean_A']:.4f} FastV={pair['mean_B']:.4f}")
    print(f"[pair] delta(RBM-FastV) = {pair['mean_delta_pp']:+.2f} pp, "
          f"95% CI {pair['ci95_pp']}, perm p={pair['perm_p_two_sided']}")
    if "mcnemar" in pair:
        print(f"[pair] McNemar: RBM-only={pair['mcnemar']['A_only_correct']} "
              f"FastV-only={pair['mcnemar']['B_only_correct']} "
              f"p={pair['mcnemar']['p_exact_two_sided']}")
    with open(os.path.join(EXP_B, "analysis_full.json"), "w") as f:
        json.dump(pair, f, indent=2)
    print("wrote", os.path.join(EXP_B, "analysis_full.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
