# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md
> 最近更新：2026-07-15 · **用户重锚：要创新型论文（新方法），非测量论文；方向 = pre-merger pruning（Qwen3-VL 原生 merger 之前剪枝）· 验证中**

## ★ 当前主线：创新型论文 — V3 pre-merger pruning ✅ GO（2026-07-14）
方向：Qwen3-VL 原生 2×2 merger **之前**剪 token（prune merger input, preserve merger）—— 正交于全部失败尝试。Gate 全过：① novelty=GO（GitHub 源码核实 VisionZip 实为 POST-merger → pre-merger cell 空）② feasibility=GO（hook merger+3 deepstack, 不 fork）③ GPU go/no-go=**GO 强结果**。
**✅ go/no-go**（n=200 Qwen3-VL-8B, iso-token pre vs post, L2 text-agnostic selector 控变量）：**TextVQA pre 大胜** keep{50,25,12.5}%→ptid{387,203,111} pre{0.75,0.70,0.62} vs post{0.51,0.26,0.18}，Δ{+24,+44,+44}pp（深度保 75% baseline vs post 21%，3.5×）。**GQA(object) post 胜**（Δ−2.5~−6pp）。throughput pre≈post（merger 仅 10% TTFT，如 F2）。**机制**：merger 是 lossy 聚合破坏 text 高频→post 灾难掉点，pre(raw patch)保 text；object 上 merger helpful→post 胜 = **workload-conditional stage 效应**（比 uniform win 更强）。代码 `src/v3_premerger/`。
**Spine**：pre-merger 保 text/OCR 深压精度（post 灾难失败）+ workload-conditional 机制 → 暗示 adaptive stage 方法。**✅ suite law（5 benchmark, 单调 in text-density, @12.5% keep C−B）**：TextVQA(场景文字)**+44.0**→MME(yes/no)**+3.5**→MMBench(感知MC)−1.5→ScienceQA(图+文MC)−2.0→GQA(物体)**−5.5**（pre 胜 text-dense，post 胜 object/MC；与 lossy-merger 机制吻合）。**✅ DocVQA（最 text-dense）**：@25% keep iso-token pre 0.725 vs post 0.39 = **+33.5pp CLEAN**（post 从 baseline 0.77 崩到 0.39，pre 持 0.725）；@12.5% post cell 在 16k-巨图上 vLLM 崩溃（非干净，不 claim +61pp）。机制强证。**下一步**：修 @12.5% DocVQA post 崩溃（bound img/post-hook）→ 更强 selector（L2 是 probe-grade）→ 跨架构(Qwen2.5-VL/InternVL3) → n=500 mean±std → vs VisionZip(post-merger SOTA) → adaptive-stage 方法 → 写论文。
**🔧 Task1 DocVQA 修复 ✅(2026-07-16)**：crash 根因 vLLM0.19 `encoder_cache_size = max_num_batched_tokens` 默认 8192→大图崩。Runner +`--max-num-batched-tokens`(默认 None,DocVQA=32768)。全部 cell 同配置(mnbt=32768,max_pixels=1.5M)clean：A=0.78(200/0/ptid=4154)；@25% B=0.39/C=0.725(iso-ptid=1054)=**+33.5pp**；@12.5% B=0.135/C=0.61(iso-ptid=538)=**+47.5pp**（首次干净深度对比）。越深压 pre 优势越大，强证 text-dense→pre 保文字机制。Task 1 ✅。
**⚠️ 跨架构(Qwen2.5-VL-7B)尝试 2026-07-15/16 = 进度：pre crash 已修但 mrope 错位仍挡 B+C**：A baseline 正常(textvqa0.886/gqa0.514 n=500)。✅**pre crash 修了**(runner qwen2vl visual.forward patch 跳 reverse_indices,n=200 不再 split_with_sizes)。❌**B post 和 C pre acc 均≈0**(B 0.004,C 0.005;n_ans 仅 23-31/500 和 2-13/200)。共享根因=placeholders 数正确但 image_grid_thw 未同步 prune→mrope 位置编码错位→视觉 token 位置偏移→LLM 无法理解。需深入 vLLM Qwen2.5-VL 的 get_rope_index 路径修改 grid_thw 或 rope 计算，预估 1-2 GPU·h but 当前 1h 已用。决策=搁置 Task 2 blocker，转向 Task 3(VisionZip)。代码`src/v3_premerger/v3_premerger_runner.py`(已 commit)。

## ★ method search 历史（均 GPU 证伪，作 negative 库）
selector 三连败（CLS/LLM-cosine/CLIP，OCR 失败）｜load-adaptive controller（n=500 null）｜ElasticVis allocator（EV-1e 负）｜EV-VAR variance（Stage 1 负，{var_k,max_k} F=0.18 p=0.84）。**教训**：boundary TF selector 打不过 intra-LLM OCR；scheduling-based 被 total-token bound。pre-merger = 新维度（compression-architecture，非 selector/scheduling）。

## ★ v2 测量论文（fallback，drafts/paper_v2.md，ERA 后降级）
v2 = 0/37 served-throughput framework（**ERA arXiv:2606.31982 打破头条**：在 vLLM 内测了 served req/s）。salvage = sharpen 到 continuous-batching/goodput/p99/TTFT。已做：§5.9 negative 节 + 5 图 + RTP-LLM 修（commit f3449f6）。留作 fallback/支撑，非主线。

## 资产 + 约束
framework `src/serve_bench.py`(V1 c64 goodput+openloop+per-req k+`--log-step-composition`)+`compressors.py`+`src/elasticvis/`；数据 eval/subsets/* + runs/v2_p*/；模型 LLaVA-1.5-7B + Qwen3-VL-8B；env `qwen3vl_clean`(vllm0.19 V1)。算力 1× A40 46GB 串行。提交用户名义禁 AI 署名。升级找人：凭据/>6GPU·h/claim推翻/投稿前。详见 DECISIONS.md。
