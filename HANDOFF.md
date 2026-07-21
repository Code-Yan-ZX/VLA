# HANDOFF.md — 给接手 agent 的交接包（2026-07-21）

> 本文件 = 上一会话（指挥窗口）的工作交接。先读 `CLAUDE.md`（项目铁律：lean 主窗口、派子 agent、禁 AI 署名、算力 1×A40 串行）→ 再读本文件 → 再读 `STATE.md`/`DECISIONS.md`/`drafts/v3_evidence.md`。
> 用户意图：**由你（新 agent）评估后续努力方向**，上一会话不替你做这个决定。

## 0. 一句话项目
VLM 视觉 token 压缩 → Q1/Q2 SCI。**主线 = pre-merger pruning**：在 Qwen3-VL-8B 原生 2×2 merger **之前**剪 token（preserve merger）。spine 已锁 = **发现主导**（见 §2）。

## 1. 本会话做了什么
- 摸清 `src/v3_premerger/`（runner 的 `--mode {none,post,pre}` 是**进程级锁死**、无 per-request 分支；pre hook `visual.merger.forward`+3 deepstack）。
- **Task4 adaptive router**：4a 探针 + 4b 全梯度离线路由分析完成。结论反直觉——廉价 router 仅 +2pp over always-pre；oracle 增益 73% 是 query-dependent、廉价信号够不到 → **router 当 headline 太弱**。
- 据上述把 spine 从"router 主线"重框为"发现主导"，**用户拍板 A**。
- 补 **VisionZip-style on DocVQA**（=0.390，与 post dom-only 同 → 崩）+ headline **binomial 误差棒** + **stage-law 核心图** + **证据整合文档**。
- 全部 commit + push origin/main，作者 Code-Yan-ZX，无 AI 署名。

## 2. 锁定的 spine（finding-led）
1. **机制**：原生 merger 是 lossy 聚合，破坏 text 高频 → post-merger 在退化特征上选 → text-dense 灾难掉点；pre-merger 在 raw patch 上选 → 保 text；object 上 merger helpful → post 胜。
2. **workload-conditional stage law**：pre/post 优劣随 text-density 呈**三层结构**（text-dense ≫ perception/MC ≈0 > object）。
3. **field-relevant 杀手锏**：post-merger SOTA（VisionZip 类，dom+ctx）在 text-dense 深压**同样灾难崩溃**（DocVQA/TextVQA 都 0.39/0.255 级），pre-merger 是鲁棒修复。
4. **方法**：pre-merger pruning（已核实空 cell = novelty）+ adaptive stage selection（ptid 廉价信号、workload 级）作**实用节**，+2pp 如实报、不夸大。
- 核心 claim 层级：机制 → stage law → post-merger 脆弱性 → pre 修复+adaptive。诚实护栏见 §6 / `drafts/v3_evidence.md` §2。

## 3. 关键数字（iso-token, keep 25%；binomial stderr；greedy temp=0 → seed 无方差，用 binomial）
| Bench | n | pre | post | VisionZip-style(dom+ctx) | gap pre−post | σ |
|---|---|---|---|---|---|---|
| TextVQA | 500 | 0.738±.020 | 0.272±.020 | 0.390(n200) | +46.6pp | 16.7 |
| DocVQA  | 200 | 0.725±.032 | 0.390±.034 | 0.390(n200,ptid1054) | +33.5pp | 7.2 |
- **stage law @25% (pre−post, n200)**：DocVQA +33.5 / TextVQA +44 / ScienceQA −0.5 / MMBench −2.5 / MME −0.5 / GQA −6.0。
- 深度：TextVQA keep50/25/12.5% → pre{.75,.70,.62} vs post{.51,.26,.18}；深点 pre 保 75% baseline vs post 21%（3.5× retention）。DocVQA @12.5% pre0.61 vs post0.135（+47.5，mnbt 修复后 clean）。
- **Task4 路由**（pooled N=774）：always-pre .634 / always-post .452 / oracle .702 / 廉价 router(ptid≥94) .655；分解 workload 27% / sample 73%；有效信号=**ptid**（OCR 关键词路由 0.539 更差）。
- ⚠️ 诚实 caveat：within-tier 小 inversion（DocVQA +33.5 < TextVQA +44；MME −0.5 > MMBench −2.5）→ 写作报 coarse 三层结构，**不 overclaim 完美单调**。suite_map 的 DocVQA @12.5% post=0.0 是 mnbt 修复**前**崩溃值，弃用。

## 4. 入口文件地图
- `STATE.md` — 当前阶段/下一步（≤30 行，每会话先读）。
- `DECISIONS.md` — 决策+理由（末尾：4a/4b 结果、spine 决策 user-A）。
- `drafts/v3_evidence.md` — **论文 results 骨架**（含误差棒/stage-law 表/机制/router 分解/待补清单）。
- `drafts/figures/stage_law.png` — 核心三层图（数值标签略压误差棒，投稿用 nature-figure 精修）。
- `src/v3_premerger/v3_premerger_runner.py` — 主 runner（含 `per_sample` 输出、`--visionzip-style --visionzip-dom-ratio`、`--max-num-batched-tokens`、`--max-pixels`、`--selector {l2,attn}`）。
- `src/v3_premerger/router_probe_full.py` — 4-bench 路由+workload/sample 分解。
- `runs/v3_router_probe/`（gitignored）— per-sample cells + 分析 json/png；`runs/v3_suite_map.json` — 6-bench suite 数。

## 5. 怎么跑实验（踩坑得来的命令模板）
env `qwen3vl_clean`（vllm0.19 V1），模型 `Qwen/Qwen3-VL-8B-Instruct` bf16 enforce_eager，1×A40 46GB `gpu_memory_utilization=0.9`。
```bash
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
python src/v3_premerger/v3_premerger_runner.py --mode {pre,post} --r 0.75 --selector l2 \
  --benchmark {bm} --subset eval/subsets/{bm}_200.jsonl --n 200 --max-num-seqs 16 \
  --out runs/.../{name}.json
```
- **DocVQA 必须**加 `--max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4`（否则 encoder_cache_size=max_num_batched_tokens 默认 8192 → 大图崩）。
- 跑前 `nvidia-smi` 确认空闲；**别动别的用户的进程**（共享机，曾遇 sues 的 YOLO 占卡 → 用 Monitor/后台 bash 轮询等待，不要 kill）。
- 子集：`eval/subsets/{bm}_{200}.jsonl`；**docvqa 只有 200，无 500**（n=200 已 7σ，非必需造 500）。
- 成本：textvqa/gqa/mme cell ~20s–4min；docvqa cell ~4–5min。

## 6. 待你评估的方向岔路（上一会话的暂定 lean，仅供起点，请独立判断）
- **起草优先 vs 补强优先**：证据已支撑 spine（lean=可起草，写中补缺）。补强候选=OCR-Bench/ChartQA 强化 text-dense tier + 机制/retention 图 + 定性例子（post 崩/pre 读对的文档样本）。
- **query-aware 重 router**：lean=**可能不值**（73% 增益 query-dependent，廉价信号够不到，投学习型 router 有过拟合 2–4 bench 风险且未必够到）。
- **跨架构 Qwen2.5-VL**：mrope 错位未解（grid_thw 未随 prune 同步 → 位置编码错位），现作 future work；lean=暂不投（预估 1–2 GPU·h，回报不确定）。
- **DocVQA n=500**：lean=非必需。
- **机制证据**：目前为断言+retention 数字；一个 attention/region 可视化或 retention-vs-compression 曲线会显著加强 §4（lean=值得做，便宜）。

## 7. ⚠️ 运维提醒
- 上一会话**子 agent 工具链连续 API 故障**：`API Error 400 InvalidParameter "Model not exist"`（默认模型+显式 `model=opus` 都被网关拒；主循环模型正常）。**先在你这会话测一次子 agent 是否恢复**，再决定是否依赖它；否则像上会话那样用**主窗口后台 bash / Monitor** 编排 GPU。
- 提交一律用户名义 **Code-Yan-ZX**，**禁任何 AI/Claude 署名、禁 `Co-Authored-By` 尾注**（push 前 grep commit 消息校验：`co-authored-by|claude|anthropic|generated with|🤖`）。
- 升级找用户：凭据 / 单次>6GPU·h / claim 被推翻 / 投稿前。
- GPU 被占时可用 Monitor 轮询 `nvidia-smi memory.used<4000` 持续确认后自动开跑（上会话验证可用）。
