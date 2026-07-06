# ElasticVis — Paper Outline (working spec; numbers finalized after EV-1c)

Working title: **"When Does Per-Request Visual-Token Allocation Pay Off? Accuracy-Budget Gating in VLM Serving"** (or: "ElasticVis: Admission-Time Per-Request Visual-Token Budgets for goodput@SLO in VLM Serving")

## One-sentence thesis
A serving engine should set the visual-token budget **per request at admission** (not a global rate) — but the goodput benefit is **gated by the steepness of accuracy(k)**; we characterize the gate and show ElasticVis wins on steep-accuracy (text-dense) workloads under SLO-heterogeneous traffic.

## Reframed positioning (vs prior art, from novelty scan §0)
- NOT "0/N compressors do per-request budget" (retired: CARES/DyToS/PARCEL exist).
- Clean cell: **admission-time, system-signal-driven (load+SLO headroom) per-request visual-token allocation, goodput@SLO objective, in a VLM serving engine.**
- CARES/DyToS/PARCEL = content/accuracy-driven (no system signal); AdaServe/SLOs-Serve/JITServe = text-domain. ElasticVis = the scheduler/allocator atop an elastic compressor.

## Claims
- **C1 (characterization, SIM-confirmed):** ElasticVis's goodput@SLO benefit over fixed-r is gated by accuracy(k) steepness. Synthetic crossover: mixed-SLO ≈0.15, uniform-SLO ≈0.40. Validated across 5 benchmarks (MME/MMBench/ScienceQA ~0.01→no win; GQA 0.12→boundary; TextVQA 0.28→win).
- **C2 (method, SIM + EV-1c):** ElasticVis (greedy per-request allocator) beats best fixed-r on TextVQA goodput@SLO by **+35.5% (sim)** [EV-1c confirms GPU magnitude] under mixed-SLO; robust across arrival-rate×deadline grid (fails only at saturation).
- **C3 (substrate):** v2 measurement framework — cross-engine (V0/V1) × arch (LLaVA-1.5/Qwen3-VL) × compressor (proxy/cls/tome/random) served-throughput + goodput@SLO + c64 + p50/p99. (The v2 paper, folded in or companion.)

## Sections
1. **Intro** — visual-token compression is global-r; serving engines have unique leverage (load/SLO) to allocate per-request; the gating insight (when it pays off).
2. **Related** — CARES, DyToS, PARCEL, AdaServe, SLOs-Serve, JITServe, EPDServe, vLLM-Omni, DeepSeek-OCR, VISOR. The reframed cell.
3. **ElasticVis** — formalization (§1: objective goodput@SLO=Σ1{met}·acc, constraint LatPred≤SLO, decision k_i), allocator (greedy/Lagrangian), admission-time integration (per-row gather in batched hook, §3).
4. **The gating characterization** — accuracy(k) steepness as the determinant. Synthetic sweep + 5-benchmark validation. The crossover thresholds. (F1)
5. **Evaluation** — (a) v2 framework results (served-throughput, goodput-Pareto, cross-compressor); (b) ElasticVis on TextVQA: goodput@SLO vs fixed-{r}, mixed-SLO (F2); (c) robustness grid (F3); (d) per-request allocation trace (F4).
6. **Discussion** — enforce_eager limitation (dynamic per-request shapes ↔ CUDA graphs; relative comparison unaffected; absolute = lower bound); H2 content extension (per-query steepness, vs CARES); cross-arch (Qwen3-VL native merger).
7. **Conclusion.**

## Figures (planned)
- **F1 gating**: x=accuracy(k) range, y=ElasticVis/best-fixed goodput ratio; two curves (H1 uniform-SLO, H1b mixed-SLO); real benchmarks plotted as points (MME/MMBench/ScienceQA/GQA/TextVQA). [sim data ready]
- **F2 TextVQA win**: ElasticVis vs fixed-{r0..r75} goodput@SLO (mixed-SLO), bar/line. [EV-1c]
- **F3 robustness**: rate×tight-fraction heatmap of ElasticVis/fixed ratio. [sim data ready]
- **F4 allocation trace**: the [144,576,144,576,...] per-row-k trace (smoke/alloc-test, already captured). [ready]
- v2 figures (existing): served-throughput, goodput-Pareto, cross-compressor, arch-conditional.

## What EV-1c MUST deliver (to finalize C2 + F2)
- TextVQA goodput@SLO for: fixed-{r0,r25,r50,r75} + ElasticVis (greedy), SAME workload + SAME per-request mixed-SLO deadlines.
- The goodput@SLO numbers per policy (acc-weighted + unweighted met_rate) → ElasticVis-vs-best-fixed ratio on the REAL engine.
- Confirm the sim direction (+35.5%) holds in magnitude (or report the real magnitude).
- Open: TextVQA accuracy(k) on the serving setup (reuse P2 true_cls 0.555/0.445/0.275 or re-probe); enforce_eager throughput caveat quantified.

## Open risks
- enforce_eager: if it makes everything miss SLO, goodput@SLO is uninformative → need SLO thresholds tuned so a meaningful fraction meets them under eager. Check in EV-1c.
- FIFO scheduling assumption (k_fifo→batch): if vLLM reorders under load, placeholder mismatch. Monitor in EV-1c (larger batch).
- TextVQA acc(k) reuse: P2 true_cls numbers must match the serving engine's current selector (proxy default). May need a clean re-probe.
