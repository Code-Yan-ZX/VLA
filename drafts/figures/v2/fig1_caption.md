## Figure 1 — The served-throughput gap (framework motivation)

**Bars over the 37-method landscape (2023–2026).** Of 37 surveyed visual-token
compressors, **37/37 report FLOPs or token-count**, **13/37 report some wall-clock-
style number** (offline CUDA latency, raw prefill/decode, or self-reported
"faster" on the authors' own harness — e.g., SparseVILA's AWQ pipeline), and
**0/37 measure served throughput inside a production serving engine** (vLLM /
SGLang / lmdeploy / TRT-LLM). The gap is unfilled and is what motivates a
*framework* (one that works for any boundary compressor across engines and
architectures), not a single method's number.

*Source:* `notes/lit-survey.md` §2 (the 13 wall-clock-reporting methods are
SparseVLM, VisionZip, SparseVILA, DyCoke, Q-Zoom, LLaVA-UHD, ToMe,
FasterVLM/VisPruner, PRUNESID, E-AdaPrune, FocusUI, Fourier-VLM, PLPHP).
*Generator:* `gen_fig1.py` · *Data loader:* `_data.throughput_tally()` (v1, reused).
