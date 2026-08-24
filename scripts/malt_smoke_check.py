#!/usr/bin/env python
"""Gate A checker for the MALT causal-ablation smoke (run_malt_smoke.sh).

Per bench, per arm, verifies:
  C1  keep-set identity: rb.kept_per_image == H0 (pre) kept_per_image
  C2  kept count == round(25% of full) (floor@1) -- keep-25% not drop-25%
  C3  L_after == n_text + n_image_kept (text/ptid/KV alignment)
  C4  transient lifetime: layer_visual_counts == [full, full, kept, kept...]
      (exactly the 2-block MALT-1 lifetime), H0 = [kept, kept, ...]
  C5  no extra merge: n_image_full matches the H0 arm sample-for-sample
  C6  H1 fresh == first 8 of sweep_deferred_K1_*_n200.json (runner invariance
      on the SAME fastv-k 1 config; the historical smoke_deferred_* ran
      fastv-k 3, so it is NOT the reproduction reference)
  C7  generation config pinned (max_tokens=32, greedy) -- via fixed CLI
"""
import json
import os

HERE = "/media/disk2/YZX/research/vla"
OUT = f"{HERE}/runs/malt_goal_mode"
RB = f"{HERE}/runs/deferred_rbm"
BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]
ARMS = ["h0", "h1", "h2", "h3", "h4", "nb"]
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append((name, detail))
    return cond


def load(p):
    return json.load(open(p))


def pset(rec):
    """kept-per-image set as frozenset of local-unit tuples."""
    k = (rec.get("rb", {}).get("kept_per_image")
         or rec.get("pre", {}).get("kept_per_image"))
    return frozenset(tuple(x) for x in k) if k else None


for b in BENCHES:
    print(f"-- {b} --")
    h0 = load(f"{OUT}/smoke_h0_{b}_n8.json")["per_sample"]
    h0map = {str(r["id"]): r for r in h0}
    hist = {str(r["id"]): r
            for r in load(f"{RB}/sweep_deferred_K1_{b}_n200.json")
            ["per_sample"][:8]}
    for arm in ARMS:
        recs = load(f"{OUT}/smoke_{arm}_{b}_n8.json")["per_sample"]
        ok_c1 = ok_c2 = ok_c3 = ok_c4 = ok_c5 = ok_c6 = True
        n_ids = 0
        for r in recs:
            if r.get("skipped"):
                continue
            n_ids += 1
            i = str(r["id"])
            # C1 keep-set identity vs H0
            pk = pset(r)
            pk0 = pset(h0map.get(i, {}))
            if pk is None or pk != pk0:
                ok_c1 = False
                print(f"    {arm}/{i}: kept_per_image {pk} != H0 {pk0}")
            # C2 kept == 25% of the PRE-MERGER full.  NOTE the runner's
            # n_image_full semantics differ by arm: H0 (pre) reports the count
            # that enters the LLM AFTER pre-merger pruning (== n_image_kept),
            # while the deferred arms report the FULL post-merge count.  The
            # full pre-merger unit count is pre.n_units_full == the deferred
            # arms' n_image_full (verified 4/4 benches), so for H0 use that.
            r0 = h0map.get(i, {})
            if arm == "h0":
                full_ref = r0.get("pre", {}).get("n_units_full") or r.get(
                    "n_image_full")
            else:
                full_ref = r["n_image_full"]
            exp = max(1, round(full_ref * 0.25))
            if r["n_image_kept"] != exp:
                ok_c2 = False
                print(f"    {arm}/{i}: kept {r['n_image_kept']} != 25% {exp} "
                      f"(full_ref={full_ref})")
            # C3 alignment
            if r["prompt_token_ids"] != r["n_text"] + r["n_image_kept"]:
                ok_c3 = False
                print(f"    {arm}/{i}: L_after {r['prompt_token_ids']} != "
                      f"n_text {r['n_text']} + kept {r['n_image_kept']}")
            # C4 lifetime
            lvc = r.get("layer_visual_counts")
            if lvc:
                if arm == "h0":
                    if not all(v == r["n_image_kept"] for v in lvc[:3]):
                        ok_c4 = False
                        print(f"    {arm}/{i}: H0 counts {lvc[:3]} != kept")
                else:
                    if not (lvc[0] == r["n_image_full"]
                            and lvc[1] == r["n_image_full"]
                            and lvc[2] == r["n_image_kept"]):
                        ok_c4 = False
                        print(f"    {arm}/{i}: lifetime {lvc[:4]} != "
                              f"[full,full,kept,..] (full={r['n_image_full']}, "
                              f"kept={r['n_image_kept']})")
            # C5 no extra merge: the deferred arms' full post-merge count must
            # equal H0's pre-merger unit count (the same underlying image).
            # (For H0, n_image_full == the pre-pruned count entering the LLM,
            # already covered by C2 against pre.n_units_full.)
            r0 = h0map.get(i, {})
            full0 = r0.get("pre", {}).get("n_units_full") if r0 else None
            if arm != "h0" and full0 is not None \
                    and r["n_image_full"] != full0:
                ok_c5 = False
                print(f"    {arm}/{i}: n_image_full {r['n_image_full']} != "
                      f"H0 pre-units {full0}")
            # C6 H1 fresh == historical deferred smoke
            if arm == "h1" and i in hist:
                h = hist[i]
                if (h.get("answer") != r.get("answer")
                        or h.get("prompt_token_ids") != r.get("prompt_token_ids")
                        or h.get("n_image_full") != r.get("n_image_full")):
                    ok_c6 = False
                    print(f"    h1/{i}: fresh != historical deferred smoke")
        check(f"{arm} C1 keep-set==H0", ok_c1)
        check(f"{arm} C2 kept==25%", ok_c2)
        check(f"{arm} C3 L_after==n_text+kept", ok_c3)
        check(f"{arm} C4 lifetime (2-block)", ok_c4)
        check(f"{arm} C5 no extra merge", ok_c5)
        if arm == "h1":
            check(f"{arm} C6 fresh==historical", ok_c6)

print("=== RESULT ===")
if fails:
    print(f"FAIL: {len(fails)} failing checks")
    for f in fails:
        print("  ", f)
    raise SystemExit(1)
print("ALL GATE-A CHECKS PASSED")
