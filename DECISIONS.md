# DECISIONS.md — 自主决策日志（append-only）

> 格式：`日期 | 阶段 | 决策 | 理由 | 影响`
> 主窗口自主决策落此；你随时可回看/否决。被否决的也记（含新方向）。

---

## 2026-07-01 | P0 | 新建专用 `vtc` conda env（python 3.11 + torch cu128） | 三既有 env（fastv/qwen3vl/qwen3vl_clean）均缺 accelerate/pillow/mmcv 且绑别项目栈；base 待 P1 末定，复用会污染或被绑死；研究项目长期可复现性优先 | env 构建脚本 `scripts/build_env.sh`，~2.10/cu128；base 专属依赖（mmcv/flash-attn/qwen）延后 P1/P2 加 | 后台构建中

## 2026-07-01 | P0 | base 专属依赖（mmcv/flash-attn）不在 P0 安装 | base 尚未选定，mmcv 是 LLaVA 专用且编译重，flash-attn 编译耗时；P1 选定 base 后按需装 | P0 env 为"通用骨架"，P1/P2 再补 base 栈 | —

## 2026-07-01 | P0 | vtc 落地为 torch 2.11.0+cu128 / transformers **5.12.1**（比预期 4.x 新）；采用"双 env"策略 | cu128 index 当前发 torch 2.11；transformers 已进 5.x major，API 较 4.x 有变；既有 legacy `fastv` env（torch 2.0/transformers 4.31）保留用于在 LLaVA-1.5 上复现 FastV 原始基线 | 新方法与 Qwen2.5/3-VL 走 `vtc`；FastV@LLaVA-1.5 复现走 `fastv`；P1 文献定位时确认 transformers 5.x 下基座加载路径 | —

## 2026-07-01 | P1 | **选定 Gap A：serving-engine-aware 视觉 token 压缩**（部署级真实吞吐，集成进 vLLM 内部） | survey 23 法 0/23 在 serving engine 内测吞吐；Novelty5×Feasib4=20 最高；B/C/D 被 2025-26 SOTA 挤占；A 的对手是"空白"非强基线 | 方向锁定；详见 notes/positioning.md | —
## 2026-07-01 | P1 | 贡献框定为"方法 + 测量"，非纯测量 | 纯"塞 FastV 进 vLLM 量吞吐"易被审稿人判为工程 paper；故方法 = 与 serving 协同设计的压缩器（边界级 prune + 感知 KV-cache/batch），测量为支撑 | 决定 P2 方法设计取向 | —
## 2026-07-01 | P1 | 基座：LLaVA-1.5-7B 起步 → Qwen2.5-VL-7B-Instruct 泛化 | LLaVA-1.5 vLLM 支持最成熟、576 tok/img、可复用 FastV/SparseVLM/VisionZip ckpt→最快出结果；Qwen2.5-VL 的 M-RoPE/变长 token 是更难更可发的部署故事 | 双基座策略；先骨架后泛化 | —
## 2026-07-01 | P1 | **设 go/no-go 闸门为 P2 第一里程碑（claim-overturn 风险）**：LLaVA-1.5-7B@vLLM 边界级压缩器 {0,25,50,75}%，测 served tok/s/req/s/TTFT/KV-cache on GQA+TextVQA | 核心 claim 押在"压缩→真实 wall-clock 加速"，但 under continuous batching 极可能 wall-clock 几乎不动；重投入前必须先测 | GO(≥1.5×prefill & ≥1.2×e2e req/s @ ≤2%掉点)→设计方法；NO-GO(<1.2×e2e)→转 negative-result paper 或 Gap D；NO-GO 触发 charter §6 升级找人 | 项目最大风险点
## 2026-07-01 | P1 | 设 novelty 复核闸门（probe 前先跑） | survey 截止 2026-07-01，但领域迭代快，"0/23"是核心 novelty，若有竞品 landed 需重估 | Dev 在 P2 step-1 做 last-6mo 复扫 | —

## 2026-07-01 | P2-step1 | **三 env 策略落地：vtc / vtc_serve / fastv** | serving 必须用 vLLM，但 vLLM 与 vtc(torch2.11+cu128/tf5.12) 不完全兼容；FastV 复现需 legacy 栈。分 env 隔离依赖污染 | `vtc`（方法开发/CPU test，torch2.11+cu128/tf5.12.1）；`vtc_serve`（vLLM serving，**vLLM 0.10.2 + torch 2.8.0+cu128 + transformers 4.55.2 + numpy 2.2.6 + numba 0.61.2**）；`fastv`（legacy，torch2.0.1+cu118/tf4.31，复现 FastV 精度基线） | vtc_serve 已通过 1-image 烟测（runs/vllm_smoke.log，2.4s/答） |
## 2026-07-01 | P2-step1 | **vtc_serve 锁定 vLLM 0.10.2（非最新 0.24）** | vLLM 0.24 预编译 wheel 绑 CUDA 13（vllm._C_stable_libtorch 需 libcudart.so.13），本机 driver 560 仅支持 cu12.x；0.24 在 driver 560 上 EngineCore init 直接挂。0.10.2 是最后一个 cu12-native 大版本，torch2.8+cu128 在 driver 560 跑通。tf 必须 ≤4.x（vLLM 0.10.2 的 get_cached_tokenizer 调 all_special_tokens_extended，tf 5.x 已删该属性）→ 锁 tf 4.55.2（vLLM 0.10.2 的 floor，且保留该属性）。numpy 必须 ≤2.2（numba 0.61.2 上限）→ 锁 2.2.6 | vLLM 0.10.2 的 multimodal processor path 与 method-design.md §1b 描述一致（已核实源码 llava.py:96/649/660），hook 方案无需改 | 升级 driver 或换卡前不升级 vtc_serve 的 vLLM |
## 2026-07-01 | P2-step1 | **novelty 复核通过：Gap A 仍 OPEN，无 blocker** | 2026-H1 复扫：ElasticMM(2507.10069) 是 scheduling/disaggregation 非 compressor 且明确拒做压缩；survey 2507.20198 v5 独立证实 gap 存在并诊断 FlashAttention 不可得根因；EffiVLM-BENCH(2506.00479) 仅离线 HF latency、不入 engine；AgilePruner/VisionTrim/PRUNESID 挤占 accuracy/FLOPs 组合研究空间但不碰 serving-throughput；vLLM RFC#45098 仍是未完工 infra | differentiator 必须守住"serving-engine 真实吞吐"，不滑向又一篇 accuracy/FLOPs 组合研究 | 无 blocker，proceed to go/no-go probe |
