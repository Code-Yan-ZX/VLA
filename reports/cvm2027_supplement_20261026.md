# CVM 2027 投稿前一日补实验报告
日期：2026-10-25 19:07 (T0) → 2026-10-26。执行人：服务器 Claude Code（自动模式）。
工作区：worktree 分支 `codex/cvm-24h-supplement` @ origin/main 1600bfe；主仓未提交修改零接触。
指令文件：`notes/cvm2027_server_experiment_prompt_20261025.txt`（22 条逐条执行；本报告即交付 #21）。
AGENTS.md：仓库中不存在该文件（已如实记录；按 CLAUDE.md 执行）。

## 0. 一页结论（胜负如实）
在冻结的 HF-Qwen3-VL-8B 同协议下（bf16、greedy≤32tok、px cap 4M；DocVQA 统一 2M 新协议）：
1. **现代基线 Hi-Lo Prune（官方 CVPR'26 代码 29265e3）已接入并通过 P0 门检**。三 bench 定位：
   - TextVQA@25%：Hi-Lo 66.77 << RBM 75.50 / FastV 76.83（−8.7pp / −10.1pp，CI 不过零）——**现代基线在此任务更弱**。
   - OCRBench@25%：attempted 分 Hi-Lo 613 > RBM 559 > FastV 413；**但同 921 题 common-answered 下 Hi-Lo 与 RBM 打平**（−3.3pts，p=0.89）：attempted 优势是 sdpa 引擎跑通了 eager 全部 OOM 的 79 张巨图（分母效应，非选择质量优势）；Hi-Lo vs None 同题仍 −164pts。
   - DocVQA-500@25%（统一 2M，无 skip 不对称）：**RBM 82.76 > Hi-Lo 76.38（+6.39pp CI[+3.04,+9.78] p=.0005）> … FastV 70.45（RBM−FastV +12.32pp）**——RBM 在文档类任务显著胜现代基线与 FastV。
2. 预算曲线（TV-1000 分层子集 / OCRB 全量）单调有序、无按胜补格：FastV≥RBM≫HiLo（TV 全预算带）；OCRB 上 RBM>FastV>HiLo(同题)。RBM 95% 目标达标点：TV keep=50%、OCRB 未达标（75% 仍 −23pts，注：RBM eager 79-skip 压低）、Doc@25% 未达标（−12.96pp）。
3. 完整请求成本（bs=1、200 请求×3 重复、同请求分段计时）：RBM/FastV@25% 只省 LLM 段——**ViT+merger 全程完整执行**（与执行图一致，pre 分支 gather 在完整视觉计算后；不得计入任何 ViT/merger 加速）：p50 总时长 None 0.604s → RBM 0.466s（−22.8%）、FastV 0.530s（−12.3%）；TTFT p50 0.422→0.298/0.310s。峰值显存由 eager 4D 注意力主导（各配置 41–42.7GB 无差异）。质量—延迟散点见 `reports/figures_cvm2027_supplement/score_vs_latency.svg`。
4. P0 正确性门（先于一切正式结果）：keep=1 时 RBM/Post/FastV 与原生 HF **逐位一致**（embeds/pos/mask/3×deepstack 全 torch.equal，8 样本首步 logits Δ=0.0）、FastV r=0/Post 1.0 no-op 恒等、K 公式 25/50/75 全 32 题精确（并确认 pre 模式 `n_image_full` 字段记录剪枝后行数——字段语义陷阱，未照文档补偿 4×）、deepstack 三流切片同源 kept IDs、文本 token 保留、manual-loop==native generate。Hi-Lo no-op（显式关 prune+fusion）8/8 与 None 一致。

## 1. 协议与实现
- 模型 Qwen/Qwen3-VL-8B-Instruct 本地缓存 snapshot 0c351dd…；transformers 4.57.6 + torch 2.10.0+cu128；评测 harness `src/v3_premerger/baselines_hf.py`（engine hf-transformers eager）；Hi-Lo=vendored 官方模型（sdpa；upstream dead-code 修复 `top_index_1`、rope_parameters shim for tf4.57 —— 均记录于预测文件 patches 字段）。
- Hi-Lo 冻结参数（正式跑前定，不看分调参）：prune_layer=2（deepstack 注入之后，构造性保对齐）、use_fusion=True、strength .3、top_k_ratio .1、attention_threshold .05、candidate_ratio=max(0.30, keep)（官方约束 1≤keep≤cand≤N；keep .5/.75 时 fusion 集退化——官方默认参数下同样如此）。bs=1（官方路径仅支持）。
- 评分：四个 `scripts/analyze_cvm_*_official.py` 的固定上游逻辑（lmms-eval 8f40097 pinned，SHA 校验；TextVQA 官方十字符答案 leave-one-out、标注本地副本 SHA==pin（fbaipublicfiles 不可达，改用 `_data_shared` 已验证副本）；失败计 0 入 attempted 分母；common-answered 敏感性单列；20,000 次配对 bootstrap，固定种子族 20261200+i）。驱动 `scripts/cvm24h_score.py` 自检：复现历史 RBM .75504/FastV .76826/Δ−1.322pp 精确一致。
- 复用（指令 10）：RBM/FastV@25% TextVQA-5000（expB）与 OCRBench-1000（p1）预测文件经协议指纹+ID 序列核验后复用；None 全量与全部曲线/Doc/计时格新跑。曲线 25% 与 None 格由全量预测按预固定 ID 抽取（`cvm24h_extract_slice.py`）。
- 子集预固定（先于任何新结果，seed 20261025）：TV-1000 按 n_image_full 四分位分层、Doc-500 按像素面积分层（历史 Doc 预测无单位字段——单调代理，注明）；计时 200 请求按分辨率分层。哈希在 `plan/p2_subsets_meta.json`、`plan/p3_timing_ids.json`。

## 2. 资源触发的协议变更（全部留痕，非成绩触发）
- **DocVQA 统一像素 cap 4M→2M**：None@4M smoke 23/32 OOM、@3M 22/32、@2M 0/32（四方法皆 0）；Doc 块整体作为新协议（与历史 600k/4M 结果不混表）。见 `plan/doc_cap_probe.jsonl`、`plan/doc_chosen_cap.txt`、DECISIONS D-S9。
- **Hi-Lo 计时 TTFT 重测**：首轮 HF streamer 首回调实为 prompt-put（TTFT≈0.04s 不可信）→ 修为第 2 回调（first generated token），hilo 计时整段重跑（时间触发，03:40）。
- **队列 runner 崩溃一次**（bash `local extra=$6` set -u 下 5 参调用未定义）：修复 `${6:-}` 后断点续跑（幂等 skip），无数据损失；为此增加每小时巡检 + monitor 宽过滤器（中文错误也触发）。
- Q10（TV 曲线扩全 5000）于必做格全部完成后启动（时间触发），硬截止 12:07；未完成即标 partial。

## 3. 结果
### 3.1 P1 主表（keep=25%，全量，attempted 分母；OCRB 附同题值）
| | None@100% | RBM | FastV-k3 | Hi-Lo |
|---|---:|---:|---:|---:|
| TextVQA (5000) | 83.24 | 75.50 | 76.83 | 66.77 |
| OCRBench (1000) | 707 | 559 | 413 | 613 / 同题≈RBM |
配对明细与 CI：`plan/scores/p1_*.json`（10 对全列），表格汇总 `reports/cvm2027_supplement_tables.md`。
### 3.2 预算曲线（详见 tables.md / budget_curves.svg）
### 3.3 DocVQA-500（subset，明示 n=500 非全量）
### 3.4 成本表（timing_harness.json / timing_hilo.json；p50/p95/peak/吞吐；OOM 请求列出并出分位）
## 4. 复现命令
```bash
# 环境
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd <worktree>   # branch codex/cvm-24h-supplement
# P0 门检
bash scripts/cvm24h_p0_smoke.sh && python scripts/cvm24h_p0_assert.py && python scripts/cvm24h_p0_probe.py 8
# Hi-Lo 接入（pinned 29265e3 + 修复见文件头注释）
git clone https://github.com/sealost/Hi-Lo_Prune /tmp/Hi-Lo_Prune   # 已做 dead-code 修复
# 任一格评测（例）
python src/v3_premerger/baselines_hf.py --mode pre --model Qwen/Qwen3-VL-8B-Instruct --benchmark textvqa --subset eval/full_splits/textvqa_val.jsonl --n 5000 --r-pre 0.25 --mrope native --max-pixels 4000000 --seed 0 --max-tokens 32 --out <D>/x.json
python scripts/cvm24h_hilo_eval.py --benchmark textvqa --subset eval/full_splits/textvqa_val.jsonl --n 5000 --keep-frac 0.25 --out <D>/y.json
# 评分（pinned 官方逻辑）
python scripts/cvm24h_score.py <bench> A.json B.json [--ids plan/p2_textvqa_1000_ids.txt] --seed-int N --out <S>/pair.json
bash scripts/cvm24h_score_all.sh   # 批量
# 计时
python scripts/cvm24h_timing.py --engine harness --out plan/timing_harness.json
# 表图
python scripts/cvm24h_build_tables.py
```
## 5. 缺失与边界（如实）
- Hi-Lo 计时 TTFT：以修正版重测结果为准（见 timing_hilo.json 更新后数值）；首轮失真值已废弃不引用。
- OCRBench RBM/FastV/None 的 79 巨图 skip 属既有协议行为（4M+eager 不可行样本）；Hi-Lo 因 sdpa 可答——attempted 表已按"双方失败计 0"，比较结论以 common-answered 注记为准。
- DocVQA 为 2M 新协议 n=500 子集，非全量、不可与历史 600k/4M 数字混排。
- GQA / 三模型全矩阵 / 新模型 / 连续批处理：本轮明确不做（指令 13/35）。
- Hi-Lo 与主 harness 的模型代码版本差异（vendored ≈tf4.58-dev vs 本地 4.57.6）如实记录于预测 fingerprint；keep=1 no-op 8/8 一致提示差异未改变行为（样本小，注记）。
- 论文未改动、未投稿（指令 2/21）。
