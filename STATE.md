# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md
> 最近更新：2026-07-23 · **机制纠正+评测修正+方法 gate 完成**：corrected mechanism 因果证实（swap≡pre）、官方指标 HOLD 无翻转、GQA 假象纠正（pre==post）、hybrid/router gate FAIL（诚实负结果）。详见 drafts/v3_merger_aware_results.md

## ★ 当前主线：merger-aware token selection 方法论文（pre-merger pruning 升级）
**机制（因果证实, n=64/bench, L2）**：merger 重写 unit saliency/ranking 且 anti-text。M1 pre/post ranking Spearman ρ=0.14(doc)/0.33(text)/0.36(gqa)；M2 rank_shift↔edge +0.44(doc)，pre-kept/post-dropped 单元 edge 0.64 vs 0.12（post 专丢文字）；**M3 ranking-swap 对照 swap≡pre**（doc 0.465=0.465 200/200、text 0.603≈0.598 198/200）→ pre>post gap 100% 来自 ranking（forward path/位置恒定，排除 mrope 混淆）。mask=2×2 unit 粒度，kept unit 的 merged token pre/post 相同 → 只差"用哪个 ranking 选"。
**评测修正（官方指标, HOLD 无翻转）**：旧 containment 指标 + verbose 生成（median 132 字符句子）高估 ~100×。修 = short-answer prompt 烤入 subset（同 ChartQA/OCRBench 惯例）+ 官方 VQA-acc/ANLS 离线重评分。keep=25% n=200：**TextVQA VQA-acc pre .598 vs post .215（+38.3pp ~6σ）/ DocVQA ANLS pre .465 vs post .200（+26.5pp ~6σ）**。selector-invariant：attn pre .553>post .200（+35pp）。OCRBench/ChartQA 沿用移植 scorer（短答已烤入）。
**⚠️ claim 纠正**：GQA "post 胜 object −6pp" 是 verbose+containment 假象 → 正式评测 **GQA pre==post==0.51**（n=100，待 n=200 确证）。stage law 改为 **pre 弱占优（object 平手、text-dense 大胜），无 crossover**。
**方法 gate（n=100, user 预注册判据, FAIL）**：hybrid（agreement+disagreement→text）OCRBench −8pp vs pre、GQA 无 gap 可补、TextVQA 增益噪声级；disagreement router 0.484≤always-pre 0.494（oracle 0.576，+8.2pp 为 query 依赖、图像级信号够不到）。**因 pre≥post 占 84–97% 图（无 post 更优 regime），任何向 post 路由/混合只伤。**
**▶ 存活的"方法" = rank-before-merge（pre-merger selection）**：merger 前特征上算 saliency→选→再 merge。新颖（pre×native-merger cell 空）+ 机制支撑（swap 对照）+ 弱占优 + selector-invariant。adaptive/hybrid 作 bounding negative 如实报（强化机制 claim）。
**▶ 下一步（已升级 user 定方向）**：A=机制主导方法论文（rank-before-merge 为方法 + 诚实负结果，推荐）/ B=投 query-aware pre-merger 够 +8.2pp oracle（险，query-aware 曾 boundary 三连败但 pre 未试）/ C=A 先行 B 作扩展。写作红线：GQA tie 先 n=200 确证；scope=Qwen3-VL 单架构；hybrid/router 负结果如实报；ChartQA budget regime 只报方向。升级：投稿前。

## ★ method search 历史（均 GPU 证伪，作 negative 库）
selector 三连败（CLS/LLM-cosine/CLIP，OCR 失败）｜load-adaptive controller（n=500 null）｜ElasticVis allocator（EV-1e 负）｜EV-VAR variance（负，p=0.84）｜**merger-aware hybrid+disagreement-router（2026-07-23 gate FAIL，pre 是 fixed point）**。pre-merger = 正交新维度（机制证实）。

## ★ v2 测量论文（fallback，drafts/paper_v2.md，ERA 后降级）
salvage = sharpen 到 continuous-batching/goodput/p99/TTFT。留作 fallback/支撑。

## 资产 + 约束
framework `src/serve_bench.py`+`compressors.py`；v3 `src/v3_premerger/`(runner: mode pre/post/hybrid + mask-ranking swap + save-unit-scores；official_scorers.py)；机制 `scripts/mechanism_token_survival.py`；数据 eval/subsets/*_200.jsonl(textvqa/docvqa/gqa 已烤 short-answer)+runs/v3_merger_aware/{rescore_rerun,survival_capture,swap,hybrid_gate,router}/；模型 Qwen3-VL-8B；env `qwen3vl_clean`(vllm0.19 V1)。1× A40 串行共享机（跑前 nvidia-smi，勿 kill 他人）。提交用户名义 **Code-Yan-ZX** 禁 AI 署名。升级：凭据/>6GPU·h/claim 推翻/投稿前。详见 DECISIONS.md / drafts/v3_merger_aware_results.md。
