# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md
> 最近更新：2026-07-21 · 创新型论文主线 pre-merger pruning · **spine 锁定=发现主导；证据整合就绪（`drafts/v3_evidence.md`）；待方向分析（新 agent）**

## ★ 当前主线：创新型论文 — V3 pre-merger pruning ✅ GO（2026-07-14）
方向：Qwen3-VL 原生 2×2 merger **之前**剪 token（prune merger input, preserve merger）—— 正交于全部失败尝试。Gate 全过：① novelty=GO（GitHub 源码核实 VisionZip 实为 POST-merger → pre-merger cell 空）② feasibility=GO（hook merger+3 deepstack, 不 fork）③ GPU go/no-go=**GO 强结果**。
**✅ go/no-go**（n=200 Qwen3-VL-8B, iso-token pre vs post, L2 text-agnostic selector 控变量）：**TextVQA pre 大胜** keep{50,25,12.5}%→ptid{387,203,111} pre{0.75,0.70,0.62} vs post{0.51,0.26,0.18}，Δ{+24,+44,+44}pp（深度保 75% baseline vs post 21%，3.5×）。**GQA(object) post 胜**（Δ−2.5~−6pp）。throughput pre≈post（merger 仅 10% TTFT，如 F2）。**机制**：merger 是 lossy 聚合破坏 text 高频→post 灾难掉点，pre(raw patch)保 text；object 上 merger helpful→post 胜 = **workload-conditional stage 效应**（比 uniform win 更强）。代码 `src/v3_premerger/`。
**📌 SPINE（user 定 A，发现主导）**：headline = lossy-merger 机制 + workload-conditional stage law（pre/post 优劣随 text-density 单调）+ **post-merger SOTA（VisionZip 类）在 text-dense 深压灾难崩溃（DocVQA 0.77→0.39、TextVQA→0.255）、pre-merger 是鲁棒修复**（field-relevant 强发现，戳现有方法软肋）。方法 = pre-merger pruning（已核实空 cell）+ adaptive stage selection（ptid 廉价信号、workload 级）作实用节；router +2pp 如实报不夸大。
**✅ Task4 adaptive router（2026-07-21 完成）**：离线全梯度 4 bench×pre/post per-sample（id 对齐已核、oracle 复现）。pooled(N=774)：always-pre 0.634 / always-post 0.452 / oracle 0.702 / 廉价 router(ptid≥94) 0.655（胜两固定、距 oracle 4.65pp）。**分解**：workload 级仅占 oracle 增益 27%、sample 级 73%（query 依赖，ptid 等图像级廉价信号够不到，需 query-aware 重 router）；OCR 关键词路由更差(0.539)，有效信号=ptid。→ router 当 headline 太弱(+2pp)故 spine 转发现主导。代码 `router_probe_full.py`(commit 3cc3f31)。per-bench：DocVQA pre0.725/post0.390；TextVQA 0.695/0.255；MME(n174) 0.822/0.833≈tie；GQA 0.320/0.380。
**🔧 Task1 DocVQA 修复 ✅(2026-07-16)**：crash 根因 vLLM0.19 `encoder_cache_size = max_num_batched_tokens` 默认 8192→大图崩。Runner +`--max-num-batched-tokens`(默认 None,DocVQA=32768)。clean：A=0.78(ptid4154)；@25% B=0.39/C=0.725=**+33.5pp**；@12.5% B=0.135/C=0.61=**+47.5pp**。越深压 pre 优势越大。
**⚠️ 跨架构(Qwen2.5-VL-7B)搁置**：pre crash 已修但 mrope 错位仍挡 B+C（grid_thw 未同步 prune→位置编码错位）。预估 1-2 GPU·h 不值，搁置。
**▶ 下一步**：① ✅ VisionZip-style 补 DocVQA（=0.390 iso-token ptid1054，与 post dom-only 同→崩）② ✅ 误差棒（TextVQA n500 +46.6pp/16.7σ；DocVQA n200 +33.5pp/7.2σ；docvqa_500 子集不存在，n200 已足）③ ✅ stage-law 图+证据整合→`drafts/v3_evidence.md`+`drafts/figures/stage_law.png`。**④ 方向分析：user 将派新 agent 评估后续努力方向**（候选：起草论文 / 补强 OCR-Bench·ChartQA·机制图·定性例子 / 跨架构）。**交接**：新 agent 读 STATE+DECISIONS+`drafts/v3_evidence.md` 起；本会话已 push。注：本会话子 agent 工具链连续 API 故障(InvalidParameter "Model not exist")，GPU 编排改主窗口后台 bash 直跑（工具不可用之妥协）。

## ★ method search 历史（均 GPU 证伪，作 negative 库）
selector 三连败（CLS/LLM-cosine/CLIP，OCR 失败）｜load-adaptive controller（n=500 null）｜ElasticVis allocator（EV-1e 负）｜EV-VAR variance（Stage 1 负，F=0.18 p=0.84）。**教训**：boundary TF selector 打不过 intra-LLM OCR；scheduling-based 被 total-token bound。pre-merger = 新维度（compression-architecture）。

## ★ v2 测量论文（fallback，drafts/paper_v2.md，ERA 后降级）
v2 = 0/37 served-throughput framework（ERA arXiv:2606.31982 打破头条）。salvage = sharpen 到 continuous-batching/goodput/p99/TTFT。留作 fallback/支撑，非主线。

## 资产 + 约束
framework `src/serve_bench.py`+`compressors.py`+`src/elasticvis/`；v3 `src/v3_premerger/`(runner+router_probe*)；数据 eval/subsets/* + runs/；模型 LLaVA-1.5-7B + Qwen3-VL-8B；env `qwen3vl_clean`(vllm0.19 V1)。算力 1× A40 46GB 串行。提交用户名义禁 AI 署名。升级找人：凭据/>6GPU·h/claim推翻/投稿前。详见 DECISIONS.md。
