## Figure 5 — Cross-compressor panel at c64 (framework generality)

**Two panels, c=64 on LLaVA-1.5-7B / GQA / V1 (n=200), four compressors:**
proxy (saliency, prune), true_cls (CLS-attention, prune), ToMe (cosine-similarity,
**merge**), random (uniform, prune).

**(a) Throughput vs prune rate.** The four curves **overlap at iso-k** —
throughput is **compressor-invariant** (r75 req/s = 19.7–20.7, only ~5% spread).
Selection is O(N), runs once per request, and is invisible next to the
multi-hundred-ms LLM forward; the only systematic difference is ToMe's
merge-compute overhead (−1.5% at r75). The served-throughput win is a property of
the **framework** (placeholder-shrink → genuine KV/compute relief), measurable for
*any* boundary compressor — defusing the "only your proxy" criticism.

**(b) Accuracy vs prune rate.** **ToMe (merge) is highest at r75** (0.540 vs proxy
0.475) — merge's averaging preserves information that prune discards, recovering
~55% of prune's accuracy loss. **Honest red flag:** at r75, uniform *random*
(0.535) **beats both saliency selectors** (proxy 0.475, true_cls 0.490) — the
known FastV-style failure mode (saliency picks central/object patches; many GQA
questions need scattered/specific patches). This is a **selector-design finding**
(motivating query-aware selection), not a framework failure: every compressor
shows the goodput-Pareto win.

*Source:* `runs/v2_p3/{proxy,true_cls,tome_merge,random}_c64_r{0,50,75}.json`
(paper Tables 4–5 / `notes/v2_p3_crosscompressor.md` Tables 1, 3).
*Generator:* `gen_fig5.py`.
