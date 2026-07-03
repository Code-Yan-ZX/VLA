# Related-work throughput-reporting tally (37 VLM token-compression methods)

Evidence for Fig. 1 and the main novelty claim. Source: `notes/lit-survey.md` §2 (arXiv-verified 2026-07-01). All 37 methods report FLOPs/token-count; **13/37** report some offline wall-clock number; **0/37** measure served throughput inside a production serving engine (vLLM/SGLang/lmdeploy/TRT-LLM).

| # | Method | Year / Venue | FLOPs / token-count | Any wall-clock | In a serving engine |
|--:|---|---|:---:|:---:|:---:|
| 1 | FasterVLM / VisPruner | 2024 / arXiv (v2 retitled VisPruner) | Y | Y | **0** |
| 2 | SparseVLM | 2024 / ICML'25 | Y | Y | **0** |
| 3 | VisionZip | 2024 / **CVPR'25** | Y | Y | **0** |
| 4 | ToMe (original ViT) | 2023 / ICLR'23 | Y | Y | **0** |
| 5 | Q-Zoom (query-aware) | 2026 | Y | Y | **0** |
| 6 | SparseVILA (decode query-aware) | 2025 / ICCV'25 | Y | Y | **0** |
| 7 | DyCoke (video) | 2024 / CVPR'25 | Y | Y | **0** |
| 8 | LLaVA-UHD v3 | 2025 / arXiv | Y | Y | **0** |
| 9 | PRUNESID | 2026 / **ICLR'26** | Y | Y | **0** |
| 10 | E-AdaPrune | 2026 | Y | Y | **0** |
| 11 | FocusUI | 2026 (CVPR'26 ext.) | Y | Y | **0** |
| 12 | Fourier-VLM | 2025 / arXiv | Y | Y | **0** |
| 13 | PLPHP | 2025 / arXiv | Y | Y | **0** |
| 14 | *…and 24 FLOPs-only others* (e.g. FastV, PyramidDrop, VTC-CLS, TokenPacker, LLaVA-PruMerge, G-Prune, GlimpsePrune, AgilePruner, VisionTrim, METEOR, AdaReTaKe, PPE, RedundancyLens, …) | 2023-2026 | Y | — | **0** |

**Totals:** 37 surveyed · **13** report any wall-clock (offline CUDA latency, prefill, or decode speedup on the authors' own harness) · **0** measure served throughput (req/s, tok/s, TTFT, KV-MB) inside a production serving engine. The closest, SparseVILA, reports 4.0× prefill / 2.6× end-to-end but on its own AWQ pipeline, not vLLM/SGLang/lmdeploy/TRT-LLM. The only serving-engine artifact, vLLM RFC #45098 (`--image-pruning-rate`), is unfinished infrastructure with no benchmarks.

*Two independent sources corroborate the gap is open: the Westlake survey (arXiv 2507.20198) §6.5.3-6.5.4 names the FlashAttention-score root cause that blocks in-LLM pruning from engine integration and calls TTFT/per-token latency "missing"; the Eval-Framework (arXiv 2510.07143) explicitly demands this evaluation.*