"""Quick phase-mix report for a steps.json (Stage-1 calibration helper)."""
import json, sys


def report(path):
    d = json.load(open(path))
    steps = d["steps"]
    n = len(steps)
    if n == 0:
        print(f"{path}: EMPTY")
        return
    n_mixed = 0          # steps with BOTH a prefill and a decode member
    n_any_prefill = 0
    n_any_decode = 0
    nms = []
    for s in steps:
        phases = list(s["phase"].values())
        has_p = "prefill" in phases
        has_d = "decode" in phases
        if has_p and has_d:
            n_mixed += 1
        if has_p:
            n_any_prefill += 1
        if has_d:
            n_any_decode += 1
        nms.append(s["n_members"])
    import statistics as st
    print(f"{path}")
    print(f"  n_steps={n}  mixed(prefill+decode)={n_mixed} ({n_mixed/n:.2%})"
          f"  any_prefill={n_any_prefill}  any_decode={n_any_decode}")
    print(f"  n_members: min={min(nms)} max={max(nms)} "
          f"mean={st.mean(nms):.1f} median={st.median(nms)}")
    sat = sum(1 for x in nms if x >= 32)
    print(f"  n_members==32 (saturated): {sat}/{n} ({sat/n:.2%})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
