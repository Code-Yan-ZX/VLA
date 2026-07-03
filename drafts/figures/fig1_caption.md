**Figure 1.** The served-throughput gap in the VLM token-compression literature.
Of 37 visual-token compressors we surveyed (2023-2026, all arXiv-verified),
**13 report *some* wall-clock-style number** (raw CUDA latency, offline prefill,
or decode speedup, each measured on the authors' own research harness), but
**0 measure served throughput inside a production serving engine**
(vLLM/SGLang/lmdeploy/TRT-LLM). The closest, SparseVILA [2510.17777], reports
4.0× prefill / 2.6× end-to-end but on its own AWQ pipeline, not a serving
engine; the only serving-engine artifact, vLLM RFC #45098, is unfinished
infrastructure with no benchmarks. This paper is the first to close that gap.
*Source: `notes/lit-survey.md` §2.1-§2.2.*
