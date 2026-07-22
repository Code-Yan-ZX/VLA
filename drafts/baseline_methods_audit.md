# Baseline Methods Audit — VLM Visual-Token Pruning (Reproducibility)

**Date:** 2026-07-22
**Our setup:** Qwen3-VL-8B-Instruct · 1× A40 46 GB (serial) · vLLM 0.19 V1 serving · benchmarks TextVQA/DocVQA/ChartQA/OCRBench/GQA (n=200 jsonl, `serve_bench` format) · our method = **pre-merger** pruning (before Qwen3-VL's 2×2 vision merger) · budget ≈ 2–4 GPU·h/method incl. adaptation.
**Methods audited:** QuietPrune, Hi-Lo Prune, IF-Prune.

> **TL;DR:** Of the three, only **IF-Prune** has usable public code, but it is **trained** (needs a per-model KL-entropy estimator LoRA that is *not* released for any Qwen model), supports only **Qwen2-VL / Qwen2.5-VL** (no Qwen3-VL), runs on **HF transformers only** (no vLLM), and uses **dual small+large models**. **None of the three is reproducible as a fair Qwen3-VL-8B / vLLM baseline within ~4 GPU·h.** QuietPrune and Hi-Lo Prune have **no usable code at all**.

> **Naming corrections (verified):**
> - "IF-Prune" is **Information-Flow** guided, **not** "instruction-aware".
> - The VLM method "Hi-Lo" is **"Hi-Lo Prune: Look at What You'll Lose before Pruning with Hierarchical Token Selection"** (CVPR 2026). It is **NOT** "HiLo-Token" (arXiv 2606.13898), which is a *Diffusion-Transformer image-editing* token-compression paper — irrelevant to VLM understanding.

---

## 1. QuietPrune

**Paper:** *QuietPrune: Query-Guided Early Token Pruning for Vision-Language Models* — Gao et al., **CVPR 2026**.
Links: [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.pdf) · [CVF HTML](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html) · [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/40193)

- **a. Mechanism:** Query-guided *early* (in-ViT / pre-LLM) pruning. A small **adapter converts the text query into a visual-domain [Q-CLS] token**, whose attention over ViT tokens scores their query-relevance; low-score visual tokens are dropped early. Signal = query→visual attention via a learned [Q-CLS]. Stage = **early / pre-attention in the vision tower**. **Not training-free** — the [Q-CLS] adapter must be trained.
- **b. Code:** **No public repository found.** GitHub repo search for "QuietPrune" returns **0 results** (verified 2026-07-22). No code link on the CVF page or CVPR virtual site.
- **c. Model support:** Unknown (no code). Paper targets generic LVLMs; base models unverified from public sources.
- **d. Integration cost on our setup:** **N/A — no code to integrate.** Would require full re-implementation of the [Q-CLS] adapter **and training it** on Qwen3-VL's ViT — a research re-implementation, far beyond adaptation. Estimate: **>40 engineer-hours + significant training GPU·h** if attempted from scratch.
- **e. Same-budget feasibility:** **No.** No code + requires training = not reproducible in budget.

**Verdict: NOT reproducible (no code; needs trained adapter).**

---

## 2. Hi-Lo Prune

**Paper:** *Hi-Lo Prune: Look at What You'll Lose before Pruning with Hierarchical Token Selection* — Sun et al. (first-author surname Sun; full author list unverified), **CVPR 2026**.
Links: [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.pdf) · [CVF HTML](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html) · [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/40023) · [supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Sun_Hi-Lo_Prune_Look_CVPR_2026_supplemental.pdf)

- **a. Mechanism:** "Look at what you'll lose before pruning." Coarse-to-fine **hierarchical token selection** that estimates the information loss of dropping each token, plus a **Prune-Aware Fusion (PAF)** that transfers information from pruned tokens into retained ones. Signal = hierarchical loss-estimate + low-frequency preservation. Stage = visual-token stage (compatible with pre-/post-merger). **Training-free** (per abstract). Per the paper it is evaluated on **Qwen2-VL, Qwen2.5-VL, and Qwen3-VL**.
- **b. Code:** **No usable public code.** The only matching GitHub repo, [`sealost/Hi-Lo_Prune`](https://github.com/sealost/Hi-Lo_Prune) (0 stars, 0 forks, created=pushed 2026-03-19), is an **empty placeholder**: size 0, file tree = a single 13-byte `README.md` containing only `# Hi-Lo-Prune`. The CVPR paper notes code "will be made publicly available" (supplemental) but nothing has been released as of 2026-07-22. **No LICENSE.**
- **c. Model support:** Per the paper, **Qwen2-VL / Qwen2.5-VL / Qwen3-VL** — the best architectural fit to our stack of all three methods (cannot verify in code since none exists).
- **d. Integration cost on our setup:** **N/A from code — would be a from-paper re-implementation.** Because it is training-free and single-model and targets Qwen-VL, a re-implementation would be the *cheapest* of the three in principle: implement hierarchical loss-score + PAF in the Qwen3-VL visual-token path, then serve. Estimate: **~16–32 engineer-hours** to re-implement + debug from the paper/supplemental, then **~0.5–1 GPU·h** inference for n=200×2 on Qwen3-VL-8B. But this is a re-implementation, not a reproduction of their code, and risks diverging from their numbers.
- **e. Same-budget feasibility:** **No (as reproduction — no code).** The *inference* part fits the GPU budget easily, but the re-implementation cost (human hours + validation risk) puts it outside a quick baseline. It is the best **strategic watch** target: if the authors release code, it becomes by far the easiest and best-matched baseline for us.

**Verdict: NOT reproducible now (code is an empty placeholder); best future fit (training-free + Qwen3-VL). Set a watch.**

---

## 3. IF-Prune

**Paper:** *IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models* — Guohao Sun, Yufei Wang, Sizhuo Ma, Yuege Xie, Yuting Cheng, Zhiqiang Tao, Jian Wang (Snap Research + Rochester Institute of Technology), **CVPR 2026**.
Links: [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.pdf) · [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/38052)

- **a. Mechanism:** Information-theoretic pruning via a **variational information bottleneck**. A **small auxiliary VLM** (the base model + a **trained LoRA** + extra trained heads `mean_logvar_lgkld`, shipped as `non_lora_state_dict.bin`) is run in `infer_entropy_only=True` mode to produce a **per-visual-token entropy/importance score** (KL/information-flow). The **large model** then receives `visual_token_importance` + a `large_model_prune_ratio` (e.g. 0.2) and drops the lowest-importance tokens during `generate()`. Signal = learned per-token entropy from a small estimator. Stage = **post-merger** visual tokens inside the LLM (importance maps to the post-2×2-merger grid, `grid_h//2, grid_w//2`). **NOT training-free** — the KL estimator must be fine-tuned (SFT/LoRA; the repo also ships GRPO/DPO variants).
- **b. Code availability:** Repo [`snap-research/EVLM-IF-Prune`](https://github.com/snap-research/EVLM-IF-Prune).
  - **Stars = 2, forks = 0**, created 2026-03-13, **last push 2026-03-17** (dormant ~4 months), 0 open issues, Python.
  - **License: none committed.** README *claims* Apache-2.0 and references a `LICENSE` file, but **no LICENSE file exists in the repo** (GitHub API `license.spdx_id = null`) — a reuse/legality flag for a paper baseline.
  - **Code completeness:** Substantial and runnable for the *supported* models. Three backends, each a fork of the upstream modeling code: `qwen_kl/` (Qwen2-VL **and** Qwen2.5-VL: `modeling_qwen2_vl.py`, `modeling_qwen2_5_vl.py`), `internvl_kl/` (InternVL2/2.5), `llava_kl/` (LLaVA-1.5/NeXT-style). Full training stack (`train_sft/cls/dpo/grpo`, DeepSpeed ZeRO configs) + eval (`textvqa_eval.py`, `infographicsvqa_eval.py`, ChartQA, lmms-eval integration). Real, complete research code — but **research-grade, single-purpose, not maintained.**
- **c. Model support:** **Qwen2-VL, Qwen2.5-VL, InternVL2/2.5, LLaVA-1.5/NeXT.** **No Qwen3-VL** (no `modeling_qwen3_vl.py`). Backend = **HuggingFace transformers + flash-attn + DeepSpeed/PEFT only**; **no vLLM**. The `lmms-eval` integration is for the InternVL path only.
- **d. Integration cost on our setup (Qwen3-VL-8B + vLLM): HIGH.** Three independent blockers:
  1. **No Qwen3-VL modeling code** → must port the KL/pruning logic from `modeling_qwen2_5_vl.py` into Qwen3-VL's (different ViT + merger) modeling. ~8–16 engineer-hours + debugging.
  2. **Trained estimator required, none released for Qwen.** The only released pruning-KL checkpoint on HF is [`ZachSun/internvl2_5_1b_pruning_kl_lora32`](https://huggingface.co/ZachSun/internvl2_5_1b_pruning_kl_lora32) (a merged **InternVL2.5-1B**). The HF repo `ZachSun/visual_pruning_kl` is a **dataset** (~98.6k; `merged.json`, `merged_cot.json`), **not** a Qwen checkpoint. The test script hardcodes a local `./ckpt/qwen2vl-lora-kl-l32-2epo` that is **not distributed**. So for any Qwen model you must **train** the estimator LoRA + heads yourself (their `finetune_lora.sh` is an **8-GPU DeepSpeed ZeRO-3** recipe, 2 epochs over the ~100k image dataset).
  3. **No vLLM path.** Inference is a custom dual-forward HF pipeline (small estimator → importance → large model `generate()`), incompatible with our vLLM `serve_bench` flow; would need a custom offline-inference harness.
  - **Total adaptation: ~20–40 engineer-hours + training run.**
- **e. Same-budget feasibility:** **No for a fair Qwen3-VL comparison.** Training the KL estimator alone on a single A40 dominates: even at Qwen2-VL-2B/3B, 2 epochs over ~100k image samples with ZeRO-3 offload ≈ **20–60 GPU·h** (and far more for 7–8B); loading two 8B models (estimator + target) also stresses the 46 GB budget. **Far beyond ~4 GPU·h.**
  - **Only "doable-today" variant (NOT a like-for-like baseline):** run their **released InternVL2.5-1B** estimator+target on our n=200×2 benchmarks via HF. Inference-only on a 1B model ≈ **0.2–0.5 GPU·h** + ~4–8 engineer-hours to adapt to our jsonl/format. Feasible in GPU budget, **but it is a different model family/size and a different backend**, so it measures *their recipe on their model*, not a baseline comparable to our Qwen3-VL-8B/vLLM method. Useful only as a sanity/literature datapoint.

**Verdict: Real code, but trained + Qwen2/2.5-only (no Qwen3-VL) + HF-only + dormant + no LICENSE → NOT reproducible in budget on our stack.**

---

## Comparison Table

| Criterion | QuietPrune | Hi-Lo Prune | IF-Prune |
|---|---|---|---|
| Paper / venue | CVPR 2026 (Gao et al.) | CVPR 2026 (Sun et al.) | CVPR 2026 (G. Sun et al., Snap Research) |
| Mechanism signal | Query→visual attn via learned [Q-CLS] | Hierarchical loss-estimate + Prune-Aware Fusion | Learned per-token entropy (variational info-bottleneck) |
| Stage | Early / in-ViT (pre-LLM) | Visual-token (pre/post-merger compatible) | **Post-merger**, in-LLM |
| Training-free? | **No** (trained adapter) | **Yes** (per abstract) | **No** (trained KL estimator LoRA + heads) |
| Public code? | **None** (0 GitHub results) | **Placeholder only** (empty 13-byte README repo) | **Yes** (`snap-research/EVLM-IF-Prune`) |
| Repo health | — | — | 2★ / 0 forks, last push 2026-03-17 (dormant), **no LICENSE file** |
| Qwen2.5-VL | unknown | yes (paper) | **yes** (code) |
| Qwen3-VL | unknown | **yes** (paper) | **no** |
| LLaVA-family | unknown | n/a (paper) | yes (code) |
| Backend | — | — | HF transformers only (**no vLLM**) |
| Fits our Qwen3-VL-8B + vLLM | No | (would, if code existed) | **No** (no Qwen3-VL; no vLLM; dual-model; trained) |
| Released weights for Qwen | — | — | **No** (only InternVL2.5-1B estimator) |
| Reproducible in ~4 GPU·h? | **No** | **No** (no code) | **No** (training ≫ budget) |

---

## Ranking by reproducibility-ease (code completeness × Qwen-VL support × training-free × integration cost)

1. **IF-Prune** — the *only* one with usable code, so the only one you could even start today. But it loses hard on training-free (trained estimator), Qwen3-VL support (absent), and integration (dual-model, HF-only, port + retrain). Ranks #1 by default on the code-completeness gate, not because it's easy.
2. **Hi-Lo Prune** — best *fit* to our stack (training-free, single-model, Qwen3-VL in paper), so it would rank #1 **if code existed**. Today it is a placeholder-only repo, so it is a re-implementation project, not a reproduction.
3. **QuietPrune** — no code **and** requires a trained query adapter; lowest reproducibility.

---

## Recommendation

**None of the three is reproducible as a fair, like-for-like Qwen3-VL-8B / vLLM baseline within ~4 GPU·h. This is the honest outcome — do not force one of these into the baseline table.**

- The blocking reasons are structural, not effort-tunable within budget:
  - **QuietPrune & Hi-Lo Prune:** no usable public code (verified 2026-07-22).
  - **IF-Prune:** code exists but requires (i) a trained per-model KL estimator with **no Qwen checkpoint released**, (ii) a Qwen3-VL modeling port, (iii) a non-vLLM dual-model inference harness. The training step alone is ≫ 4 GPU·h on a single A40.

**Concrete next-step recommendations (in priority order):**

1. **Set a watch on Hi-Lo Prune code release.** It is the single best future baseline for us (training-free + Qwen3-VL + single-model). Check `sealost/Hi-Lo_Prune`, the authors' profiles, and PapersWithCode monthly; it could flip from "impossible" to "easiest" with one commit. (Low cost: a GitHub watch + periodic check.)
2. **If a baseline is needed now and the model need not be Qwen3-VL:** the only audited method runnable today is **IF-Prune on its released InternVL2.5-1B estimator** (HF, inference-only). GPU·h ≈ **0.5–1 GPU·h** (0.2–0.5 inference + ~0.5 bring-up) + ~4–8 engineer-hours to adapt our n=200×2 jsonl. **Explicitly report it as "their recipe on their 1B model," not as a Qwen3-VL-8B baseline.** Do not present it as comparable to our method.
3. **If a training-free Qwen3-VL-8B baseline is genuinely required for the paper, it must come from outside this audited trio** (e.g., a training-free attention/CLS-based pruner that already ships Qwen2.5/3-VL code). This is **out of scope of this audit** — flag for a separate candidate search rather than forcing QuietPrune/Hi-Lo/IF-Prune.
4. **Do NOT invest in porting IF-Prune to Qwen3-VL now:** estimated **~20–60 GPU·h** (training-dominated) + ~20–40 engineer-hours, ~5–15× over budget, for a method whose license is not even committed.

### GPU·h summary
| Option | GPU·h | Comparable to our method? | Feasible in budget? |
|---|---|---|---|
| QuietPrune on Qwen3-VL | n/a (no code; +training if reimplemented ≫4) | — | **No** |
| Hi-Lo Prune on Qwen3-VL (re-impl.) | ~0.5–1 inference + ~16–32 eng-hrs | Would be, if reimplemented correctly | **No (no code; reimpl risk)** |
| IF-Prune on Qwen3-VL (port + train) | **~20–60** | Would be | **No** |
| IF-Prune on InternVL2.5-1B (their recipe) | **~0.5–1** | **No (different model)** | GPU-yes, but not a fair baseline |

---

## Verification notes (how each claim was checked, 2026-07-22)
- Papers identified/verified via web + CVF Open Access (all three are **CVPR 2026**).
- IF-Prune repo verified via **authenticated GitHub API**: metadata (2★, pushed 2026-03-17, `license=null`), full recursive file tree (backends `qwen_kl/internvl_kl/llava_kl`, Qwen2-VL **and** Qwen2.5-VL modeling files, training stack), raw `README.md`, `INSTALLATION.md`, `qwen_kl/test_qwen2vl.py` (dual small/large model + `infer_entropy_only` + hardcoded local `./ckpt/...` LoRA), `qwen_kl/scripts/finetune_lora.sh` (8-GPU ZeRO-3 SFT, `mean_logvar_lgkld` excluded-from-LoRA trainable heads).
- "No Qwen checkpoint" verified via HF API: `ZachSun/internvl2_5_1b_pruning_kl_lora32` (InternVL2.5-1B merged weights) is the only pruning-KL model; `ZachSun/visual_pruning_kl` is a **dataset** (`merged.json`), not weights.
- QuietPrune: GitHub repo search "QuietPrune" → **0 results**. Hi-Lo Prune: only `sealost/Hi-Lo_Prune` → **empty placeholder** (tree = single 13-byte `README.md` reading `# Hi-Lo-Prune`).
- Caveat: Hi-Lo Prune full author list and exact base-model matrix are from secondary search snippets (first author surname "Sun" confirmed via CVF URL slug); verify against the PDF before citing. QuietPrune first-author given name not confirmed (cited as "Gao et al.").
