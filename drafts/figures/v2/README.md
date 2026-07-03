# V2 Paper Figures (`drafts/figures/v2/`)

The 5 figures for `drafts/paper_v2.md`. All numbers are read directly from raw
v2 run JSON (deterministic generators, no hardcoded values) and verified to match
the paper tables exactly. 300 dpi PNG, colorblind-safe palette (dataviz skill's
validated default, shared via the parent `_style.py`). V1 figures are preserved
in `drafts/figures/`; this subdirectory holds only v2 variants.

## Figures

| # | File | Caption | Generator | Source data |
|---|------|---------|-----------|-------------|
| 1 | `fig1_gap.png` | `fig1_caption.md` | `gen_fig1.py` | `notes/lit-survey.md` §2 (via parent `_data.throughput_tally()`) |
| 2 | `fig2_concurrency_ceiling.png` | `fig2_caption.md` | `gen_fig2.py` | `runs/v2_p2/batch_c{1,4,16,64}_r{0,50,75}.json` |
| 3 | `fig3_goodput_pareto.png` | `fig3_caption.md` | `gen_fig3.py` | `runs/v2_p2/batch_c64_r{0,50,75}.json` (goodput from per-request `raw[*].ttft_ms`) |
| 4 | `fig4_arch_conditional.png` | `fig4_caption.md` | `gen_fig4.py` | `runs/v2_p2/` + `notes/v2_p1_qwen3vl.md` §2 + `runs/v2_p2/qwen3vl_c64_r{0,50}.json` |
| 5 | `fig5_crosscompressor.png` | `fig5_caption.md` | `gen_fig5.py` | `runs/v2_p3/{proxy,true_cls,tome_merge,random}_c64_r{0,50,75}.json` |

## Shared modules

- `_data_v2.py` — v2 data loaders (`scale_matrix`, `goodput_sweep_c64`,
  `cross_compressor_c64`, `arch_conditional_amplification`, etc.). Reads raw JSON;
  goodput at any SLO is recomputed from per-request TTFT.
- Parent `_style.py` and `_data.py` (in `drafts/figures/`) are imported via
  `sys.path`; palette + rcParams are shared with v1 for visual consistency.

## Reproduce

```bash
cd drafts/figures/v2
for f in gen_fig1 gen_fig2 gen_fig3 gen_fig4 gen_fig5; do python3 $f.py; done
```

## Verification (generator stdout, matches paper tables)

- Fig 2: r75/r0 = 1.19×(c1) → 2.22×(c64); ceiling-lift r0 +12% vs r75 +26% (c16→c64).
- Fig 3: r75/r0 throughput 2.22×; p99 reduction 2.84×; goodput@5s r75/r0 = 7.4×.
- Fig 4: LLaVA 1.19×→2.22×; Qwen3-VL 1.08×→1.29×→1.34× (c64 is r50/r0).
- Fig 5: r75 req/s 19.73–20.73 (5% spread); acc proxy 0.475 / cls 0.490 / tome 0.540 / random 0.535.
