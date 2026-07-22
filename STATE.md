# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md
> 最近更新：2026-07-22 · **SOTA 决胜小矩阵完成**：ChartQA/OCRBench n=200 建成+全跑（21 cell 零失败）、VisionZip 官方 audit(不可跑)、三方法 audit、`drafts/v3_sota_matrix.md` FINAL

## ★ 当前主线：创新型论文 — V3 pre-merger pruning ✅ GO（2026-07-14）
方向：Qwen3-VL 原生 2×2 merger **之前**剪 token（prune merger input, preserve merger）—— 正交于全部失败尝试。Gate 全过：novelty=GO（VisionZip 源码核实为 POST-merger → pre-merger cell 空）② feasibility=GO ③ GPU go/no-go=GO 强结果。
**📌 SPINE（user 定 A，发现主导）**：lossy-merger 机制 + workload-conditional stage law + post-merger SOTA（VisionZip 类）text-dense 深压灾难崩溃、pre-merger 鲁棒修复。方法 = pre-merger pruning + adaptive stage selection（ptid、+2pp 如实报）。
**✅ SOTA 决胜矩阵（2026-07-22，n=200 iso-token, Qwen3-VL-8B, L2 selector）**：
- **OCRBench = text-dense 第三确证**：@25% pre **0.580** vs post/VZ 0.165（+41.5pp, 9.5σ）；@12.5% 0.380 vs 0.075（+30.5, 7.8σ）。子技能级：纯文字识别 pre 保 baseline/post 清零（Reg/NonSem/Irreg 1.0 vs 0.0）；手写公式同分（budget regime）。
- **ChartQA = 新第三 regime：budget-dominated**（非 bug，per-sample 核实）：@25% pre=post=VZ **同分 0.190**（失败集不同：各对 38 题仅 20 重叠；压缩下模型转 hedging 长答 1.1→7.7 词）；@50% 0.39/0.335、@12.5% 0.15/0.095（+5.5, 1.7σ 不显著）。论文报 stage law（何处 pre 胜）× budget（是否可行）双轴，不 overclaim 单调。
- **VisionZip-style ≡ post dom-only：11/11 cell**（含本轮补 post-mode TextVQA/DocVQA @25/@12.5）→ context tokens 零增益，崩溃纯由 stage 解释。
- **旧 bench 保持**：TextVQA +44pp(9.8σ) / DocVQA +33.5pp(7.0σ) @25%；GQA −6pp（object, post 胜）；误差棒 binomial。
**✅ audit 两则**：① VisionZip 官方（`JIA-Lab-research/VisionZip` CVPR'25）判定 (c) 本机不可跑（无 Qwen3-VL/vLLM；Qwen 变体需 attn 物化 OOM）；代码级再确认 post-merger；**作者 README Qwen2.5 OCRBench@50% 81.5→70.5（−13%）= 官方数字内 text-dense onset** → SOTA 列 = 我方 same-model port + 官方 mismatched reference（`drafts/visionzip_gap_report.md`）。② QuietPrune/Hi-Lo/IF-Prune **均不可同 budget 公平复现**（无码/空仓/需训 20–60 GPU·h）；IF-Prune InternVL recipe 弃（模型失配）；**Hi-Lo Prune 挂 watch**（`drafts/baseline_methods_audit.md`）。
**资产新增**：`eval/subsets/{chartqa,ocrbench}_200.jsonl` + lmms-eval 移植 scorer（commit f23841a）；cell `runs/v3_sota_matrix/`（21 json）；脚本 `src/v3_premerger/v3_sota_matrix{,_followup}.sh`；定性 10 例 `drafts/qualitative_examples.md`（"$1.3B→$1.3M" 单位错 post+VZ 同犯 pre 对）。
**▶ 下一步**：证据骨架已齐（`drafts/v3_sota_matrix.md` §0–6 + caveats 红线）→ **① 可起草论文**（nature-writing/nature-figure skills；stage_law.png 精修 + retention-vs-compression 曲线 + 子技能显微镜图）或 **② 先补机制可视化**（便宜、强化 §4）。写作红线：ChartQA/GQA gap 只报方向（≤1.7σ）；scope=Qwen3-VL 单架构；within-tier inversion 如实报。升级找人：投稿前。

## ★ method search 历史（均 GPU 证伪，作 negative 库）
selector 三连败（CLS/LLM-cosine/CLIP，OCR 失败）｜load-adaptive controller（n=500 null）｜ElasticVis allocator（EV-1e 负）｜EV-VAR variance（负，p=0.84）。pre-merger = 正交新维度。

## ★ v2 测量论文（fallback，drafts/paper_v2.md，ERA 后降级）
salvage = sharpen 到 continuous-batching/goodput/p99/TTFT。留作 fallback/支撑。

## 资产 + 约束
framework `src/serve_bench.py`+`compressors.py`；v3 `src/v3_premerger/`(runner: per_sample+vz-style+mnbt/mpix；router_probe*)；数据 eval/subsets/{textvqa,docvqa,gqa,mme,mmbench,scienceqa,chartqa,ocrbench}_{200}.jsonl + runs/；模型 Qwen3-VL-8B(-Instruct)；env `qwen3vl_clean`(vllm0.19 V1)。1× A40 串行共享机（跑前 nvidia-smi，勿 kill 他人）。提交用户名义 **Code-Yan-ZX** 禁 AI 署名。升级：凭据/>6GPU·h/claim 推翻/投稿前。详见 DECISIONS.md / HANDOFF.md。
