# VisionZip Official vs. Our Proxy — Faithfulness Gap Report

**Audit date:** 2026-07-22
**Purpose:** decide how to represent VisionZip in the paper's SOTA matrix (our method = pre-merger pruning on Qwen3-VL-8B; headline = post-merger SOTA collapses on text-dense deep compression, pre-merger fixes it).
**Verdict (TL;DR):** **(c) not runnable on our setup** → report official as reference with model/stage mismatch; our proxy faithfully implements the dominant+contextual *principle* at **pre-merger**, which is our claim axis anyway.

Sources (all fetched 2026-07-22):
- Repo (official): `https://github.com/JIA-Lab-research/VisionZip` (old org alias `dvlab-research/VisionZip` redirects here)
- Paper: arXiv **2412.04467**, "VisionZip: Longer is Better but Not Necessary in Vision Language Models" (Yang et al., CVPR 2025)
- Files read: `visionzip/main.py`, `visionzip/utils.py`, `visionzip/clip_encoder.py` (LLaVA path); `Qwen2_5_VL/README.md`, `Qwen2_5_VL/qwen2_5vl_visionzip.py` (Qwen path)
- Community port (unofficial): `https://github.com/Hanjiangli/VisionZip-Qwen2VL` (Qwen2-VL, linked from official README)

---

## 1. Official algorithm (evidence-anchored)

**Two implementations exist; both select POST-merger/post-encoder, both training-free at inference.**

### LLaVA path (`visionzip/clip_encoder.py`, L45–83; `visionzip/utils.py`)
1. **Dominant tokens** — from CLIP ViT attention at the **penultimate encoder layer (−2)** (`apply_info`: `model.r = [0]*22 + [1] + [0]` for 24-layer CLIP-ViT-L; `metric = encoder.layers[-2].metric`):
   - `cls_attention = attn_weights[:, :, 0, 1:]` → **CLS→patch attention**, `.sum(dim=1)` (sum over heads), `topk(dominant_num)`; the CLS token itself is always kept (hence `dominant-1` in `apply_info`).
   - Paper (arXiv 2412.04467, method): "tokens most attended to by the CLS token", layer −2; for **non-CLS encoders** the fallback is `attn.mean(dim=1).mean(dim=1)` (attention *received*, averaged over heads and queries).
2. **Contextual tokens** — not k-means. From the remaining tokens:
   - metric = **raw key states of layer −2, averaged over heads** (`raw_key_states.mean(1)`, L89 of utils.py), L2-normalized;
   - pick `contextual_num` **uniformly spaced target indices** (`step = N // contextual_num`);
   - assign every other token to its nearest target by **cosine similarity on keys** (`bmm` + `argmax`);
   - merge by **plain count-averaging of hidden states** (`bmm(one_hot.T, hidden) / counts`); `contextual = target_hidden + aggregated_hidden`.
3. **Placement:** inside the vision tower (after ViT encoding, before the LLaVA MLP projector → LLM). Selection acts on the ViT's 576 patch tokens — i.e. *after* the ViT's native pixel-unshuffle pooling; for LLaVA-NeXT this is after the 2×2 anyres merge.
4. **Default budgets** (README/paper): LLaVA-1.5: 64 = **54 dom + 10 ctx** (dominant share 0.84), 128 = 108+20, 192 = 162+30; LLaVA-NeXT: 160 = 135+25, 320 = 270+50, 640 = 540+100. Function signature default: `visionzip(model, dominant=191, contextual=30)`.
5. **Training:** training-free mode available (also fine-tuning / from-scratch modes); no training needed for the reported inference numbers.

### Qwen2.5-VL path (`Qwen2_5_VL/qwen2_5vl_visionzip.py`, L584–615 + L1914–1968)
1. **Dominant score:** attention **received** at the **last ViT layer** — `logits.mean(dim=0)` (mean over heads) → `.sum(dim=0)` (sum over queries) → averaged within each 2×2 merge group (`view(N//4,-1).mean(-1)`) to align with the merger grid → `topk(attn_logits, dominant_num)`.
2. **Contextual metric:** last-layer **keys**, averaged within each 2×2 merge group; same uniform-target / cosine-argmax / count-average merge as the LLaVA path.
3. **Placement (key):** selection runs **inside `Qwen2_5_VLForConditionalGeneration.forward`**, i.e. **AFTER the PatchMerger** — it operates on `inputs_embeds[:, first:last+1]` at the `image_token_id` positions (L1955), then rewrites `position_ids`/`attention_mask` and feeds the LLM. **Post-merger confirmed for the Qwen variant.**
4. **Default budgets:** retention 70% = **65% dominant + 5% contextual** (dominant share 0.93); 50% = 45% + 5% (share 0.90). Contextual is always ~5% of post-merger tokens.
5. **Constraints:** **batch_size = 1 only** (NaViT variable lengths); requires ViT attention outputs (their file replaces vision attention with eager materialized softmax, `return_logits=True` path); training via LLaMA-Factory, eval via lmms-eval.
6. **Authors' own caveat (Qwen2_5_VL/README.md):** *"Qwen2.5VL already uses PatchMerger for visual token compression. As a result, the performance gain from VisionZip is less striking compared to LLaVA."*

### Official Qwen2.5-VL results (README, ~7B, lmms-eval)

| Retain | MME | MMVet | OCRBench | POPE | RealWorldQA | DocVQA | MathVerse |
|---|---|---|---|---|---|---|---|
| 100% | 2316 | 61.6 | **81.5** | 86.7 | 68.6 | **95.1** | 46.3 |
| 70% (65d+5c) | 2334 | 60.0 | 80.9 | 86.4 | 68.2 | 94.5 | 45.8 |
| 50% (45d+5c) | 2209 | 57.0 | **70.5** | 86.3 | 68.6 | 93.8 | 45.1 |

**They stop at 50% retention. No 25% row.** Note the *shape*: general tasks (POPE, RWQA) hold; **text-dense tasks degrade first** — OCRBench −13% at 50% while DocVQA still −1.3. LLaVA-1.5-7B @ 64 tokens (11%): TextVQA 58.2 → 55.5 (−2.7).

---

## 2. Model / hardware support

| | Supported |
|---|---|
| **Paper (2412.04467) models** | LLaVA-1.5-7B/13B, LLaVA-NeXT-7B/13B, Mini-Gemini, Video-LLaVA |
| **Official repo code** | CLIP-based LLaVA (`visionzip/`) + **Qwen2.5-VL** (`Qwen2_5_VL/`, HF transformers patch, LLaMA-Factory + lmms-eval) |
| **Community port** | Qwen2-VL (Hanjiangli/VisionZip-Qwen2VL, unofficial) |
| **Qwen3-VL** | **NO.** No Qwen3-VL file, no mention in README/news; deepstack-merger architecture not handled anywhere |
| **vLLM** | **NO.** Both implementations monkey-patch HF `transformers` forwards; nothing for vLLM serving |
| **Batching** | Qwen variant is **batch_size=1 only** |
| **Our setup** (Qwen3-VL-8B-Instruct, 1× A40 46GB, vLLM 0.19 V1, n=200 jsonl subsets) | unsupported on every axis |

---

## 3. Direct-run verdict: **(c) NOT RUNNABLE HERE**

Blocking reasons (each independently fatal; none fixable in ≤1 GPU·h):

1. **No Qwen3-VL support.** Official Qwen code is a hand-modified `modeling_qwen2_5_vl.py` (2224 lines). Qwen3-VL differs materially (deepstack mergers `visual.deepstack_merger_list`, different ViT block signatures, window config). Porting = new modeling file + ViT-attention plumbing.
2. **vLLM incompatibility.** vLLM 0.19 V1 runs its own `Qwen3VLForConditionalGeneration` (model executor, continuous batching, CUDA-graph prefill). Official code patches HF `nn.Module.forward`s that vLLM never calls. The batch_size=1 requirement also fights vLLM's batching.
3. **Attention materialization.** Dominant scoring needs ViT attention weights. vLLM's vision tower uses FlashAttention (no weights returned). Forcing eager attention in the ViT on DocVQA images (~16k pre-merger tokens) materializes ~16k×16k×16-head matrices per layer → OOM on 46GB A40, plus large slowdown.
4. **No serving-side hook for post-LLM-embed rewriting.** The Qwen variant rewrites `inputs_embeds` + `position_ids` + `attention_mask` inside the LLM forward — vLLM's scheduler/placeholder accounting (`image_token_id` counts, encoder cache sizing) must be rewritten to match; we have adjacent infrastructure (our runner already rescales placeholder lists) but the official scoring path is entirely new code.

**Effort estimate for a faithful port:** new Qwen3-VL modeling fork + vLLM integration + eager-ViT attention + validation ≈ 1–2 dev-days and >6 GPU·h of validation/debug risk (escalation territory per ORCHESTRATION §6). **Not worth it; (c) stands.**

---

## 4. Proxy-vs-official gap table

Our proxy: `src/v3_premerger/v3_premerger_runner.py`, `--visionzip-style --visionzip-dom-ratio` (what it does, ≤10 lines):

> Per image, the cached per-2×2-unit L2 scores (`_vz_scores`, or `--selector attn` centroid-distance) split the kept budget k into `k_dom = round(k·0.7)` dominant + `k_ctx = k − k_dom` contextual. Dominant = top-scored units, passed **natively through the merger**. Contextual = the remaining units split into `k_ctx` **contiguous equal-sized groups, mean-pooled** to one unit each, then merged. Total output = k units/image → **iso-token with vanilla pre-merger**. All of this executes **before** the native 2×2 merge via the `merger.forward` wrap, on every merger call (incl. Qwen3-VL deepstack mergers). No attention weights, no training.

| Aspect | Our proxy | Official (LLaVA) | Official (Qwen2.5-VL) | Gap direction |
|---|---|---|---|---|
| **Stage** | **pre-merger** (prune 2×2 units → native merge) | inside vision tower, post ViT-pool, pre-projector | **post-merger**, inside LLM forward on `inputs_embeds` | **Structural — this is our claim axis, not a bug** |
| **Dominant score** | L2 norm of unit features (or centroid distance) | CLS→patch attn, ViT layer −2, sum over heads | attention *received*, ViT last layer, merged-group avg | Official saliency-aware; ours saliency-free |
| **Context mechanism** | contiguous equal groups → mean | uniform targets + **key-cosine argmax** assignment → count-mean | same as LLaVA path | Official: similarity-assigned; ours: order-assigned. Both mean-merge; ours is a coarser grouping |
| **Dominant share of budget** | **0.70** | 0.84 (54/64) | 0.90–0.93 (65/70, 45/50) | Official keeps more raw tokens, fewer averaged ones → slightly less blur |
| **Context share** | 0.30 | 0.16 | 0.05–0.10 | We average more tokens per context slot |
| **Training** | none | none (training-free mode) | none (FT optional) | same |
| **Model/engine** | Qwen3-VL/2.5-VL + vLLM | CLIP+LLaVA + HF | Qwen2.5-VL + HF, bs=1 | no overlap |

### Is the proxy a fair/conservative stand-in? Does post-merger stage alone cause the text-dense collapse?

**The proxy is NOT a numerical stand-in for official VisionZip** (stage + scorer + ratios all differ) — and it shouldn't be: it is a faithful implementation of the *dominant+contextual principle transplanted to pre-merger*, which is precisely what we claim. Direction of bias if anything **conservative for us**: official's attention scoring and higher dominant share (0.84–0.93 vs 0.70) likely buy a little extra on general benchmarks, so a same-stage comparison would not obviously *favor* our proxy.

**Would official VisionZip also collapse on DocVQA/TextVQA at 25% keep? Yes — mechanistically and empirically:**

1. **Empirical anchor (authors' own numbers):** on Qwen2.5-VL, official post-merger VisionZip at **50%** retention already loses **OCRBench 81.5 → 70.5 (−13%)** while DocVQA only −1.3. Text-dense tasks degrade *first*; they simply never published 25% (their "retain 10%" headline is on general LLaVA benchmarks, not DocVQA/OCRBench). Our post-merger proxy at 25%: DocVQA 0.39, TextVQA 0.255 — consistent extrapolation of the authors' own 50% trajectory.
2. **Stage mechanism:** post-merger selection can only choose among 2×2-merged units — information destroyed by the merger is unrecoverable regardless of scorer. At 25% keep, 75% of merge units (each covering 4× the spatial area of a patch) are dropped. In document images, text strokes are spatially *uniform*, not salient-object-concentrated → any per-token scorer must drop most glyphs.
3. **Scorer mechanism (attention ≠ OCR relevance):** attention-received / CLS-attention rewards visually salient regions (headers, figures, layout anchors), not legibility of body text. The official scorer buys nothing for dense small text; arguably worse than a uniform spatial spread.
4. **Context-merge mechanism:** at 25% keep with official ratios, ~5% contextual tokens must absorb ~70% of units by **count-averaging** (~14:1 merge ratio) — averaged glyphs become unreadable sludge.

**Conclusion:** the stage (post-merger) is **sufficient** to cause text-dense collapse at deep compression; the attention-based scorer does not rescue it (and official's own Qwen2.5-VL table shows the collapse onset at 50%). Honest caveat for the paper: we cannot produce an official Qwen3-VL number at 25%; the claim rests on (i) official's published 50%-retention trajectory, (ii) our post-merger proxy reproducing collapse at 25%, (iii) the mechanism above.

---

## 5. Recommendation for the paper's SOTA matrix

1. **Do NOT claim head-to-head numbers against official VisionZip on Qwen3-VL** — not runnable (verdict c), and cross-model/cross-engine comparison would be rejected by reviewers.
2. **SOTA matrix column "VisionZip":**
   - In the Qwen3-VL matrix: label as **"VisionZip (principle, pre-merger port — ours)"** with a footnote: *official VisionZip is post-merger and has no Qwen3-VL implementation; we implement its dominant+contextual principle pre-merger (dom ratio 0.7, mean-merge context).*
   - As a **reference row / separate table**: cite official paper numbers — LLaVA-1.5-7B @64 tokens (TextVQA 55.5 vs 58.2 vanilla, Table in arXiv 2412.04467) and Qwen2.5-VL @50% (DocVQA 93.8, OCRBench 70.5 vs 81.5) — **with explicit model/ratio/stage mismatch noted**. The Qwen2.5-VL 50% row is our strongest external anchor: it is the official method, on the same model family, already showing text-dense degradation where general tasks hold.
3. **Framing sentence (draft):** *"VisionZip (Yang et al., CVPR'25) selects dominant tokens by ViT attention and merges the rest into contextual tokens — post-merger. The authors report robustness at 50% retention on Qwen2.5-VL but already lose 13% OCRBench while POPE/RWQA hold, and do not evaluate 25%. Our post-merger proxy reproduces the collapse at 25% (DocVQA 0.39); moving the same dominant+contextual principle pre-merger restores it, isolating stage — not scorer — as the cause."*
4. **Optional ≤1 GPU·h ablation to harden the stage claim** (recommended next step): run our existing proxy in **post-merger mode** with `--visionzip-style` (our runner already has `--mode post`; needs the dom+ctx split wired into the post path if not already there). Same engine, same model, same benchmarks, only stage differs → a clean stage-effect ablation that no official code can give us.
5. Cite: arXiv 2412.04467; repo `JIA-Lab-research/VisionZip` (incl. Qwen2_5_VL/README authors' caveat); optionally Hanjiangli/VisionZip-Qwen2VL as "no Qwen3-VL port exists in the community either".
