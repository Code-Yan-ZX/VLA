import json, statistics as st
cells = {
 "homo_chunk":"runs/ev_var/stage1_homo_chunk.json.steps.json",
 "homo_nochunk":"runs/ev_var/stage1_homo_nochunk.json.steps.json",
 "bimodal_chunk":"runs/ev_var/stage1_bimodal_chunk.json.steps.json",
 "bimodal_nochunk":"runs/ev_var/stage1_bimodal_nochunk.json.steps.json",
}
for tag,p in cells.items():
    d=json.load(open(p)); steps=d["steps"]
    allk=[k for s in steps for k in s["k_i"].values()]
    kk=set(allk)
    mixed=sum(1 for s in steps if "prefill" in s["phase"].values() and "decode" in s["phase"].values())
    nms=[s["n_members"] for s in steps]
    print(f"{tag}: n_steps={len(steps)} k_set={sorted(kk)} k_min/max/mean={min(allk)}/{max(allk)}/{st.mean(allk):.0f} "
          f"var_k_mean={st.mean(s['var_k'] for s in steps):.0f} mixed={mixed}({mixed/len(steps)*100:.0f}%) "
          f"n_members_mean={st.mean(nms):.1f} max={max(nms)}")
