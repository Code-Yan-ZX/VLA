#!/usr/bin/env python
"""Smoke-test checker for the Deferred-RBM n=200 gate (see smoke_deferred_n200.sh).

Loads the smoke JSONs (8 samples x 4 benches, both arms) + the historical dev
n=64 / parent pre JSONs, and verifies the six implementation-correctness checks
from the task spec plus runner invariance / parent reproducibility.
"""
import json
import os
import sys

HERE = "/media/disk2/YZX/research/vla"
RB = f"{HERE}/runs/deferred_rbm"
CASCADE = f"{HERE}/runs/cascade"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]

fails = []


def check(name, cond, detail=""):
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append((name, detail))
    return cond


def load(p):
    return json.load(open(p))


def rec_key(r):
    return (r["id"], r.get("answer", ""), r.get("prompt_token_ids"),
            r.get("n_image_full"), r.get("n_image_kept"), r.get("n_text"))


print("=== SMOKE: Deferred-RBM (rankbridge rho=1.0) vs immediate-RBM (pre) ===")
for b in BENCHES:
    print(f"-- {b} --")
    df = load(f"{RB}/smoke_deferred_{b}_n8.json")
    pf = load(f"{RB}/smoke_pre_{b}_n8.json")
    dev = load(f"{RB}/dev_deferred_{b}_n64.json")["per_sample"]
    pre = load(f"{CASCADE}/gate_pre25_{b}.json")["per_sample"]
    d = {str(r["id"]): r for r in df["per_sample"] if not r.get("skipped")}
    p = {str(r["id"]): r for r in pf["per_sample"] if not r.get("skipped")}
    dv = {str(r["id"]): r for r in dev if not r.get("skipped")}
    pv = {str(r["id"]): r for r in pre if not r.get("skipped")}

    ids = [str(r["id"]) for r in df["per_sample"] if not r.get("skipped")]
    ok_inv = ok_par = ok_ident = ok_fired = ok_keep = ok_align = True
    n_ident = 0
    for i in ids:
        r = d[i]
        # (1)(2)(3) fired + counts from the DEFERRED arm itself.
        # fired records TOTAL kept sequence length (text + image) after prune:
        # expected exactly one fire at layer 3 with L_after total.
        fired = r.get("rb", {}).get("fired")
        if not (fired == [[3, r["prompt_token_ids"]]]):
            ok_fired = False
            print(f"    id {i}: fired={fired} (expected [[3, "
                  f"{r['prompt_token_ids']}]], i.e. single fire at layer 3)")
        keep_frac = r["n_image_kept"] / max(1, r["n_image_full"])
        # (6) keep-25% not drop-25%: kept==0.25*full (allow +/-1 rounding)
        exp_keep = max(1, round(r["n_image_full"] * 0.25))
        if r["n_image_kept"] != exp_keep:
            ok_keep = False
            print(f"    id {i}: kept {r['n_image_kept']} vs expected 25%={exp_keep}")
        # (5) text/ptid alignment: L_after == n_text + n_image_kept
        if r["prompt_token_ids"] != r["n_text"] + r["n_image_kept"]:
            ok_align = False
            print(f"    id {i}: L_after {r['prompt_token_ids']} != n_text "
                  f"{r['n_text']} + kept {r['n_image_kept']}")
        # (A) runner invariance vs dev n=64 (diag-only change must not alter output)
        if i in dv and rec_key(r) != rec_key(dv[i]):
            ok_inv = False
            print(f"    id {i}: deferred smoke != dev n=64 "
                  f"({rec_key(r)} vs {rec_key(dv[i])})")
        # (4) identity: deferred kept_per_image == immediate-RBM kept_per_image
        dk = r.get("rb", {}).get("kept_per_image")
        pk = p.get(i, {}).get("pre", {}).get("kept_per_image")
        pk_parent = pv.get(i, {}).get("pre", {}).get("kept_per_image")
        if dk is not None and pk is not None:
            n_ident += 1
            if dk != pk:
                ok_ident = False
                print(f"    id {i}: deferred kept != pre smoke kept")
        if dk is not None and pk_parent is not None and dk != pk_parent:
            ok_ident = False
            print(f"    id {i}: deferred kept != parent pre kept")
        # (B) parent reproducibility on answers/counts
        if i in pv:
            a = p[i]
            if (a["answer"], a["prompt_token_ids"]) != (pv[i]["answer"],
                                                        pv[i]["prompt_token_ids"]):
                ok_par = False
                print(f"    id {i}: pre smoke != parent pre (answer/ptid)")
    check(f"{b}: layers0-2 full + layer-3 delete (fired=[3,kept])", ok_fired)
    check(f"{b}: post-layer-3 count == 25% target", ok_keep)
    check(f"{b}: L_after == n_text + n_image_kept (no misalignment)", ok_align)
    check(f"{b}: deferred == dev-n64 (runner invariance)", ok_inv)
    check(f"{b}: pre smoke == parent pre (reproducible parent)", ok_par)
    check(f"{b}: kept indices == immediate-RBM (n={n_ident})", ok_ident)

print(f"\nSMOKE {'PASS' if not fails else 'FAIL'} "
      f"({len(fails)} failing check(s))")
for name, detail in fails:
    print(f"  - {name}: {detail}")
sys.exit(1 if fails else 0)
