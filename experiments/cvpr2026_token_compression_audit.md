# CVPR 2026 candidate audit vs locked RBM-OT

**Date:** 2026-08-10 · **Scope:** TransPrune, V2Drop, "Recursive Token Reduction", AOT, MacTok.
Leads file `literature_frontier_20260810.md` was re-checked independently; only directly
fetched or officially indexed statements are used below.

## Per-method verified table

| | TransPrune | V2Drop | Recursive Token Reduction | AOT | MacTok |
|---|---|---|---|---|---|
| Exact title | TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model | Variation-aware Vision Token Dropping for Faster Large Vision-Language Models | No official paper with this exact title found; matches MetaCompress talk title (see note) | Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models | MacTok: Robust Continuous Tokenization for Image Generation |
| Venue/status | CVPR 2026 poster, confirmed (virtual page 36981, session 2026-06-07) | CVPR 2026, openaccess + virtual poster 39016 indexed; official repo fetched | UNVERIFIED as distinct CVPR paper; official CVPR 2026 paper is "Rethinking Token Reduction for LVLMs" (MetaCompress), Wang et al. | CVPR 2026 main track, confirmed (virtual page 37594, full abstract fetched) | CVPR 2026 poster 36668 indexed; openaccess page fetch failed (403) |
| Task | Image VQA (8 benchmarks; LLaVA-suite models) | Image + video understanding | Multi-turn VQA (MT-VQA), learned | Video (short + long-video benchmarks) | Image generation tokenizer; not a VLM inference reducer |
| Models | LLaVA-v1.5, LLaVA-Next, Qwen2.5-VL | LLaVA, Qwen2-VL | LLaVA-1.5-13B, LLaVA-NeXT-7B per press summaries; UNVERIFIED in paper body | LLaVA-OneVision-7B, LLaVA-Video-7B-Qwen2 | UNVERIFIED (generation models; 403) |
| Stage | Layer transitions of LVLM token embeddings; encoder-vs-LLM layer location UNVERIFIED (arXiv HTML 404) | Inside LLM (token-wise variation across adjacent LLM layers) | Learned compression mapping applied to visual tokens before LLM | Post-vision-encoder per-frame tokens, before LLM; not at merger input, no native merger reuse reported | Tokenizer training stage (generation) |
| Training | Training-free (confirmed) | Training-free, plug-and-play inference | Learned (requires training) | Training-free (confirmed) | Learned tokenizer |
| Headline metric | >50% inference TFLOPs reduction (TFLOPs) | 94.0%/98.6% perf retained; LLM generation latency −31.5%/−74.2% | ~90% token reduction, TTFT −57~64% (author claim via news; body UNVERIFIED) | 8.3% of prefilling FLOPs, 90% video tokens pruned, 97.6% perf retained | 64-token HD generation, gFID (UNVERIFIED, 403) |
| Baselines | UNVERIFIED (body inaccessible) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| Code + status | github.com/liaolea/TransPrune — public, MIT, real tree (llava/, scripts/), 11 commits | github.com/xuyang-liu16/V2Drop — public, Apache-2.0, 30 commits, functional tree | github.com/MArSha1147/MetaCompress — public but README-ONLY today ("code being organized"); frontier-file claim HEAD 70a620e NOT reproducible | github.com/TyroneLi/AOT — public, functional tree (llava-next/, visionzip/, scripts/), 19 commits | UNVERIFIED — no repo located |
| What % measures | TFLOPs reduction | Accuracy retention + latency | Token-count reduction + TTFT | FLOPs + token retention + accuracy retention | Generation quality (gFID) |

Note on "Recursive Token Reduction": BAAI talk announcement (hub.baai.ac.cn/view/56409,
2026-07-17) uses this English title with Chinese title 面向多轮视觉问答的学习式Token压缩框架,
identical to MetaCompress's official description (same ZJU Song Mingli × Alibaba Security
team). Treated as MetaCompress working title; a distinct paper under this name was NOT
found on openaccess/arXiv/dblp in accessible sources.

## Three questions

**Q1. Does AOT transport discarded tokens at a VLM vision-merger input, keep rank-selected
anchors fixed, aggregate four patch offsets independently, then run the native merger?**
NO. AOT (arxiv.org/html/2603.01400v1; cvpr.thecvf.com/virtual/2026/poster/37594) operates
on already-extracted per-frame video tokens before the LLM; anchors are attention-guided
(CLS/self-attention global top + local window top), fixed during the OT step, but anchors
are updated as a normalized mixture of the original anchor and transport-weighted pruned
tokens — not hard-assigned barycenter replacement into independent 2×2 patch slots, and no
native nonlinear merger runs afterward.

**Q2. Where do TransPrune/V2Drop score, and what tasks?**
V2Drop scores inside the LLM (variation across adjacent LLM layers; arxiv.org/abs/2509.01552,
github.com/xuyang-liu16/V2Drop) on image AND video tasks (LLaVA, Qwen2-VL). TransPrune
scores token-embedding transition variation (TTV) + instruction-guided attention during LVLM
inference (arxiv.org/abs/2507.20630; cvpr.thecvf.com/virtual/2026/poster/36981) on IMAGE-VLM
tasks only (LLaVA-v1.5/Next, Qwen2.5-VL); exact encoder-vs-LLM layer of TTV: UNVERIFIED
(arXiv HTML v3 returned 404). Neither is a tokenizer/generation method.

**Q3. Is any released method identical to locked RBM-OT?**
NO. AOT shares optimal transport + anchor aggregation but differs on every collision element:
stage (post-encoder video tokens vs pre-merger input), anchor policy (attention local-global
vs plain-RBM L2 ranking), transport operation (soft mixture update vs balanced Sinkhorn
barycenter replacing each anchor's four patch slots independently), placement (tokens fed to
LLM directly vs enriched anchors re-enter the model's native nonlinear merger). TransPrune/
V2Drop are layerwise pruning-only (no transport); MetaCompress is learned multi-turn; MacTok
is a generation tokenizer.

## Collision ruling

**VERDICT: NO-COLLISION.** No verified CVPR 2026 (or accessible arXiv/GitHub) work implements
the locked RBM-OT combination: training-free, single-turn image VLM, keep 25% of pre-merger
spatial units by plain-RBM L2 ranking as fixed anchors, balanced Sinkhorn OT (τ=0.05, 20 iter,
cosine cost, equal anchor capacity) barycentering dropped pre-merger descriptors into each
anchor's four patch slots independently, then executing the model's native merger. The closest
work, AOT, shares only the high-level "OT moves dropped-token information into anchors" idea —
explicitly insufficient per the collision rule — and differs on stage, anchor policy, transport
operation, and native-merger placement. Shared use of optimal transport alone is not a collision.

## Verification log (2026-08-10)

Resolved: arxiv.org/abs/2507.20630 · cvpr.thecvf.com/virtual/2026/poster/36981 ·
github.com/liaolea/TransPrune · arxiv.org/abs/2509.01552 · github.com/xuyang-liu16/V2Drop ·
arxiv.org/html/2603.01400v1 · cvpr.thecvf.com/virtual/2026/poster/37594 ·
github.com/TyroneLi/AOT · github.com/MArSha1147/MetaCompress · hub.baai.ac.cn/view/56409
(via search index). Openaccess URLs indexed (not directly fetched): Chen_Variation-aware…,
Zeng_MacTok…, Wang_Rethinking_Token_Reduction…, Li_Token_Reduction_via_Local_and_Global…
Failed/unresolved: arxiv.org/html/2507.20630v3 (404) · openaccess.thecvf.com Zeng_MacTok
page (403) · openaccess.thecvf.com Wang_Rethinking page (403) · MacTok code repo (not found) ·
TransPrune baselines/table details · dblp/CrossRef not reachable in this session.
