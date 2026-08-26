# paper_v4 改写地图（paper_v3 → submission-ready v4）

> 2026-07-24 · 写作规划子 agent（纯 CPU）· 不改代码/数据
> 证据截止 = J1 / J2 / J4(+step2_fix) / J5 digests + mechanism_verification_report + method_gate_report + DECISIONS 末 6 条（claim 终态与红线）
> **未完依赖（gating）**：J3（机制跨代复制 swap/ρ，Qwen2.5）· J4 余（DocVQA HF cell 降 max-pixels 重跑、Pyramid iso-25% [1,0,0,0] 试）· J6（效率表）· J7（官方完整 split 主表，~50–60 GPU·h）· J8（消融闭环）。下表所有数字标注 **[现成]** / **[待Jx]**。

---

## 1. Spine 声明（一段，英文 = 论文正文语言）

> In merger-equipped VLMs, the **stage** at which visual tokens are selected — before or after the native 2×2 merger — is a first-class design axis that dominates the choice of query-blind scorer on text-dense workloads. Under strict iso-model / iso-token / iso-selector control on **two Qwen-VL generations** (Qwen3-VL-8B, Qwen2.5-VL-7B), the same text-agnostic L2 scorer applied pre-merger beats post-merger selection by **+18.8 to +41.5pp on TextVQA / DocVQA / OCR-Bench** under official metrics (n=200, all ≥5σ; e.g. Qwen3-VL +38.3 / +26.5 / +41.5pp), ties on object-centric GQA with **no crossover**, and the gap **widens under deeper compression**. The mechanism is **causal**: the learned merger nearly re-shuffles unit saliency from scratch (M1: pre–post rank ρ = 0.14–0.36, worst on text-dense), it systematically demotes high-edge/text-stroke units (M2: DocVQA ρ(rank_shift, edge)=0.44; demoted group Sobel 0.64 vs 0.12), and a ranking-swap control — post-merger forward path, pre-merger ranking — recovers pre-merger accuracy **exactly** (M3: DocVQA 200/200 byte-identical, TextVQA 198/200 within ε-decode noise): the entire pre>post gap is a ranking effect, not forward-path information destruction. Consequently the **post-merger family is fragile under deep compression on text-dense data** — a faithful VisionZip dominant+contextual principle-port is byte-equivalent to plain post-L2 in every cell on both generations, and the official method's own published numbers show the onset at 50% retention — while **pre-merger selection (RBM, rank-before-merge) is the robust default**: it never collapses, and on OCR-Bench it beats even the query-conditioned competitor **FastV** by +42.5pp, because FastV's layer-2 attention pruning wins TextVQA (+8.2pp) and GQA (+7pp) precisely by being query-conditioned yet cannot rescue dense OCR already destroyed by the merger. **Three pre-registered negative results** — image-level routing, merger-aware hybrid masking, and query-level embedding blending (all λ>0 harmful, −1.7 to −5.2pp) — close the design space: pre-merger features are purely visual, so pre-merger selection **must remain query-blind**; the +8.2pp query-dependent oracle headroom is reachable only after the LLM cross-attention layers mix in the question, which is exactly where FastV operates.

**spine 组件对照（写作时逐项核验）**：① lossy-merger 机制 M1–M3 因果链 ② stage law（pre 弱占优 / text-dense 大胜 / GQA 平手级无 crossover / 深压拉大）③ 跨代一致（Qwen3+Qwen2.5）④ post-merger 族（VisionZip≡post）深压脆弱 ⑤ FastV = query-conditioned 竞争者定位（胜 TextVQA/GQA、OCR 崩）⑥ RBM = 鲁棒默认（方法冻结 plain L2 pre-merger ranking）⑦ 三连负结果（图像级 router / merger-aware hybrid / query 级 QA gate）→ 收束"pre-merger 必须 query-blind"。

---

## 2. 逐节改写地图

图例：✅保留 · ✏️改写 · ➕新增 · ❌删除/降级。红线编号：R1 不写"RBM 超过现有方法"（仅同模型同 harness 领先才可写"超过"，当前无处可写）；R2 不跨模型宣称 SOTA；R3 VisionZip 官方数只作 mismatched 锚；R4 GQA 只报"pre 弱占优/平手级/无 crossover"，Qwen3 n200 post +4.5pp(~1.3σ) 如实报方向、绝不写"post 胜 pre"（旧 −6.0pp 是 verbose-containment 假象）；R5 不显著 gap 只报方向；R6 只用官方指标（VQA-acc/ANLS/OCRBench 官方/GQA exact-match），旧 containment 数字一律替换；R7 披露 HF-baseline 引擎差异与 DocVQA 像素 cap。

### 全局横切（所有节适用）
- ✏️ **指标换代**：grep 替换旧 containment 数 → 官方（Qwen3-VL n200）：TextVQA none .820→**.858**、pre .695→**.598**、post .255→**.215**（gap +44.0→**+38.3pp**）；DocVQA .770→**.976** / .725→**.465** / .390→**.200**（+33.5→**+26.5pp**）；GQA pre .320→**.420**、post .380→**.465**（"−6.0pp post 胜"→ **−4.5pp post 弱领先 ~1.3σ，报方向**，R4/R6）。OCRBench(0.580/0.165) 与 ChartQA(0.190 tie) 的 lmms-eval 移植即官方口径，保留但改 scorer 标签措辞。n=500 TextVQA +46.6pp 是 containment → **[待官方重评或弃用]**。
- ✏️ 方法统一命名 **RBM (rank-before-merge)**；所有 `[CITE: …]` 占位 → 真实引文 pass（含 ERA arXiv 2606.31982 定位句）。
- ✏️ `[E: …]` 溯源标签全部更新到新 digest 路径。

### Title candidates —— ✏️改写
- 旧 #1 "Selection Stage Beats Scoring" **与新证据冲突**：FastV（更聪明的 scoring）在 TextVQA/GQA 胜 RBM（R1 相关）→ "beats scoring" 不再普适。
- 建议改用 #2 方向 *The Lossy Merger: Pre-Merger Selection as a Query-Blind Robust Default for VLM Visual-Token Compression*，或 *Where You Select Matters Most When the Merger Has Already Destroyed the Evidence*。候选 3 个重拟，核心词：lossy merger / pre-merger / robust default / text-dense。

### Abstract —— ✏️全改（≤250 词）
- 数字全部官方化 + 双模型；加 M3 因果句、FastV 竞争者句、三连负结果收束句；**删** "post 在 GQA 领先 −6.0pp"（改 tie/无 crossover）；删 router +2.1pp 的正面措辞（降为负结果）。R1/R4/R6 全触。

### §1 Introduction —— ✏️大改
- P1–P3（visual token 占比 / 未被质疑的 where 假设 / 受控实验设计）✅骨架保留，数字换官方。
- P4（stage law）✏️：+38.3/+26.5/+41.5pp（Qwen3）、+32.0/+18.8/+28.5pp（Qwen2.5）**[现成: rescore_rerun_report / j2 Table1]**；GQA 改"tie、无 crossover"（R4）。
- P5（机制）✏️升级为 M1–M3 因果（替换旧"two-layer 单图叙事"为主证据，单图降为插图）**[现成: mechanism_verification_report §1–3]**。
- P6（SOTA audit）✏️：加 FastV 同模型同预算已可跑且胜 TextVQA/GQA、OCR 崩（R1）；VZ≡post 升为**双代字节一致** **[现成: j2 Table1 末行 / j4 §3]**。
- ➕ P7（新）：三连负结果 → pre-merger 必须 query-blind（一句话收束）。
- Contributions 重列为 5 条：(1) 受控 stage effect + 跨代一致；(2) M1–M3 因果机制；(3) stage law（pre 弱占优/text-dense 大胜/GQA 无 crossover + ChartQA budget 第三轴）；(4) post-merger 族脆弱 + FastV 竞争者定位 + RBM 鲁棒默认；(5) 三连负结果封闭设计空间（query-blind 必然性）。R1/R2 触：不写 SOTA/超过。

### §2 Related work —— ✏️局部 + ➕
- "Pruning/selection" 段 ✅保留 + ➕一句 ERA（post-merger LLM-side、vLLM 内测吞吐——效率节定位用，非 stage 轴竞品）。
- "Dominant+contextual" 段 ✅保留（VZ 代码级 post-merger 核实）。
- "2026 三法 audit" 段 ✏️：**FastV + PyramidDrop 现为可跑同模型基线**（HF harness 验证链，见 §3 新增）；QuietPrune/Hi-Lo/IF-Prune 维持"不可公平复现"（baseline_methods_audit 延续）。
- "Empty cell" 段 ✅保留（novelty claim：pre-merger×native-merger cell 空）。

### §3.1 Problem setup —— ✏️
- 双模型（Qwen3-VL-8B + Qwen2.5-VL-7B-Instruct），同 vLLM 0.19 V1/enforce_eager/A40。
- ➕ mrope 跨架构机制注记：block-mrope [16,24,24] θ1e6（Qwen2.5，剪后尾部文本位置继承 2D 网格 → 崩）vs interleaved [24,20,20] θ5e6（Qwen3，容忍）——修后按实际 k 推进游标，r=0 逐位退化 **[现成: j1 §根因/修复]**。R7 相关（配置披露）。

### §3.2 L2 selector as control —— ✅ + ✏️命名
- 保留"最简 query-blind scorer 控变量"论证；方法定名 **plain RBM（冻结，不再有任何变体）** **[DECISIONS 07-24 J5]**。

### §3.3 Pre-merger pruning —— ✅小改
- 补双代实现一致性（deepstack 同 mask、mrope 重算 family 分支）。

### §3.4 VZ-style proxy —— ✅小改
- 等价结果升级为**双代字节一致**（Qwen2.5 0.415==0.415 等）**[现成: j2]**。

### §3.5 Adaptive stage selection via ptid —— ❌删除（折叠）
- router 正面方法叙事作废（属三连负结果之一）→ 折叠进新增 §4.12 方法空间封闭节。§3 不再出现 router。

### ➕ §3.6（新）Baselines and fairness protocol
- FastV（LLM layer-2 注意力一次性剪，HF transformers 4.57.6 eager）、PyramidDrop（层内渐进，ratios→等效均值）、VZ principle-port。
- **公平定义**：同 keep ratio（相对每图自身 full token）+ 报平均绝对 ptid + 按 family 统一 min/max pixels（patch 14 vs 16 校准 iso-token）+ Pyramid 折算等效均值；他文数字只作趋势锚（R3）。
- **HF harness 验证链**：r=0 锚 8/8 逐样本一致（剪枝路径 r→0 精确退化原生，含 deepstack/mrope/KV）+ HF-vs-vLLM none 16/16 等价 + 手动 pre-norm vs 原生 maxdiff=0 **[现成: j4_step2_fix §验证]**。R7：精度等价、**效率数字仍归 vLLM，HF 基线不报吞吐**。

### §4.1 Setup —— ✏️
- 双模型 + 官方 scorer 全列（TextVQA VQA-acc / DocVQA ANLS / OCRBench 五类 1000 分 / GQA word-normalized exact-match）+ 短答 prompt 烤入说明。
- 误差模型保留 binomial stderr + z 判据（z≥1.96 才 claim 领先，1.5–1.96 报方向）。
- ➕ 披露：DocVQA iso-config（max-num-batched-tokens 32768、max-pixels 1.5M cap）；subset n=200 vs 完整 split 状态（Table 1 **[待J7]**）。R7 触。

### §4.2 Main matrix —— ✏️重构为新 Table 1（见 §3 设计）
- 旧 Table 1（单模型 5 行 containment）作废；新表 = 双模型 × {full, RBM@25%, post≡VZ@25%, FastV@25%, Pyramid} × 4 基准官方。
-  interim 数字全现成（Qwen3: rescore_rerun + j4_probe Table1；Qwen2.5: j2 Table1），完整 split **[待J7]**。
- R1 触（最重）：FastV 列在 TextVQA/GQA 高于 RBM → 表注与正文**禁止**"our method beats existing methods"；加粗规则只允许 OCRBench 列 RBM 最优，措辞"robust, never collapses"。

### §4.3 Stage law —— ✏️
- 数字换官方 + ➕ Qwen2.5 行（跨代对照表）**[现成: j2 Table1–2]**。
- **删除"GQA ordering inverts / post leads −6.0pp"**（R4/R6）→ "GQA tie / pre 弱占优、无 crossover regime"；如实注 Qwen3 n200 post +4.5pp ~1.3σ 与 Qwen2.5 +1.0pp/@12.5% exact tie。
- 深度轴：@12.5% 用 **clean** +47.5pp（mnbt 修复后 n200，DECISIONS 07-16；旧 +61pp 脏数永不出现，R5）。
- 诚实 nuance 入稿：Qwen2.5 post 在 DocVQA@25% 较 Qwen3 稳健（0.499 vs 0.200）→ "lossy-merger 失真随压缩单调加重，跨代同向不同速率" **[现成: j2 §诚实 nuance]**。
- Figure 1 stage_law.png 需按官方数 + 双模型重绘 **[待绘图]**。

### §4.4 Scoring-invariance —— ✅ + ✏️扩
- 11/11 VZ≡post → 扩为双代全 cell（Qwen2.5 字节一致）**[现成: j2]**。
- cross-selector：attn 族官方口径 TextVQA +35.3pp（0.553 vs 0.200）**[现成: method_gate §5]** 补入 Table 4。

### §4.5 OCRBench subskill —— ✅（篇幅紧则降 appendix）
- 数字不变（官方口径）；定性机制证据定位不变。

### §4.6 ChartQA budget regime —— ✅（篇幅紧则降 appendix）
- 保留"两轴 taxonomy 第三轴"，R5 措辞保持（方向 only）。

### §4.7 Mechanism —— ✏️大改（核心节）
- 主证据 = **M1–M3 表**（见 §3 Table 2 设计），替换旧单图 edge-density 表为主、单图降插图。
- 三层叙事：M1 重排（ρ 0.14/0.33/0.36，Jaccard@25% 0.18/0.24/0.28，text-dense 最差）→ M2 反文本方向（DocVQA ρ(shift,edge)=0.44、group(a) Sobel 0.64 vs (b) 0.12、92% vs 35% 超中位；TextVQA 中等 0.16；GQA 近零 0.036）→ M3 因果交换（swap≡pre：DocVQA ANLS 0.465==0.465、200/200 字节一致；TextVQA 0.603 vs 0.598、198/200；swap-vs-post 仅 63/40 of 200）。**[现成: mechanism_verification_report §1–3]**
- 诚实 nuance 必写：docvqa 完整支持 / textvqa partial（M2 中等 + budget 层 0.60 vs 0.86）/ gqa 近零。
- Qwen2.5 复制列 **[待J3]**。

### §4.8 Efficiency —— ✏️扩为 Table 3
- 现有：stage 吞吐中性（±10%）、压缩 vs full +28~45% req/s **[现成: v3_sota_matrix §7]**。
- 扩 TTFT / p99-TTFT / req/s / peak VRAM × budget **[待J6]**；v2 轮 c64/p99/goodput 数字（LLaVA/Qwen3 @ 旧 harness）**只许进 appendix 并重标 harness**，不进主表。
- R7：效率只 claim vLLM 内我方方法；HF 基线无吞吐数。

### §4.9 Qualitative —— ✅精选
- 留 3 例：$1.3 BILLION→million（post+VZ 同犯 pre 对）、"text as absent"（07/10/2012）、GQA 诚实反例（pre 丢物体证据）。图路径核验过。

### §4.10 Official VisionZip comparison —— ✅保留
- mismatched 锚表（README OCRBench 81.5→70.5@50%，无 25% 行）保持，R3 标注不变；➕一句"VZ≡post 现双代字节一致 → 官方原则在 post stage 无处可救"。

### ➕ §4.11（新）Same-model baselines: FastV and PyramidDrop
- 数字 **[现成: j4_probe Table1]**：FastV@25% TextVQA **0.680**（胜 RBM +8.2pp）、GQA **0.490**（+7pp）、OCRBench **0.155**（崩至 post 水平，RBM 0.580 = **+42.5pp**）；Pyramid canonical keep_equiv 0.625（2.1× 我方预算）TextVQA 0.852 近无损 → retention 曲线点（iso-25% [1,0,0,0] **[待J4余]**）。
- 机制解读：layer-2 注意力**天然 query-conditioned** = oracle +8.2pp query 余量的实证捕获；dense OCR 在 merger 已毁、排序再聪明救不回 → pre-merger 不可替代。
- DocVQA HF cell 两法 skip 187/200 → 该格 **[待J4余 max-pixels 重跑]**，未完成前表中留空注。
- **R1 全触**：本节是红线最密处——只写"RBM 鲁棒（任何基准不崩、从不输 post 族）+ OCR 大胜"，明写 FastV 胜 TextVQA/GQA。

### ➕ §4.12（新）Method-space closure: three negative results
- (a) 图像级 router：pooled N=774 always-pre 0.634 / post 0.452 / oracle 0.702 / ptid-router 0.655；workload 级仅 26.9%、sample 级 73.1% query 依赖 **[现成: DECISIONS 07-21 4b]**；pooled n=192 disagreement-router 0.484 < always-pre 0.494 = ptid **[现成: method_gate §4]**。
- (b) merger-aware hybrid gate=FAIL：t=0.5 → TextVQA 0.560（≈pre 0.537 噪声内）、OCRBench 0.510（< pre 0.590，−8pp≈2σ）、GQA 0.500 且 pre==post==0.510；无单一 text-frac 过 gate **[现成: method_gate §2–3]**。
- (c) query 级 QA gate=NO-GO：λ∈{0,0.3,0.5,0.7} dev 均值 {0.5772, −1.7, −5.2, −3.3pp}，全负 → 按预注册选 λ=0，方法永久冻结 plain RBM **[现成: j5_qa_gate_result]**。
- 收束句：图像级 / merger-aware / query 级三连负 → pre-merger 特征纯视觉无 query 信息，**pre-merger 必须 query-blind**；query 余量只能在 LLM 交叉注意力后获得（与 §4.11 FastV 证据闭环）。写作定位 = bounded negative + 机制旁证（加固而非削弱 spine）。

### §5 Discussion —— ✏️
- "Why stage dominates" 升级为因果版（M3：ranking 效应 100%，forward path 脱罪）。
- ➕ "Why pre-merger must be query-blind"（三连负 + FastV 对照）。
- 保留 ChartQA 两轴、VZ 官方数从另一侧观察、"stage as first-class axis"；✏️ "scoring 无关" 措辞收紧为"**query-blind scoring 在 stage 间不变；query-conditioned scoring 捕获 scene-text/object 余量但救不回 post-merger OCR**"（R1 相关）。
- 加跨代 mrope 洞察一句（block vs interleaved = 架构诊断价值）。

### §6 Limitations —— ✏️（必含清单见 §5 检查表）
- 旧"Qwen2.5 被 mrope 阻塞"条目 ❌删（J1 已修通）→ 改为"单 LLM 族（Qwen-VL 两代），InternVL/LLaVA 族未验"。

### §7 Conclusion —— ✏️全改
- 对齐新 spine；删 router 正面句；加 FastV 定位与 query-blind 收束；next steps = Hi-Lo Prune（若放码）+ 跨族（InternVL3）。

---

## 3. 主表设计

### Table 1 —— 官方主表（headline）
- **行**：TextVQA (VQA-acc) / DocVQA (ANLS) / OCR-Bench (官方 /1000) / GQA (exact-match)。
- **列组**：Qwen3-VL-8B | Qwen2.5-VL-7B，每组 5 列 = full / RBM@25% / post≡VZ@25% / FastV@25% / Pyramid（见注）。
- **统计**：binomial CI ±；gap 列 z 值；z≥1.96 才称 lead、1.5–1.96 标方向。
- **来源**：[待J7] 官方完整 split；**interim [现成]**：Qwen3 = `drafts/rescore_rerun_report.md` + `experiments/j4_probe_qwen3vl.md` Table1（n200：full .858/.976/~.73/~.53；RBM .598/.465/.580/.420；post .215/.200/.165/.465；FastV .680/—/.155/.490；Pyramid .852@62.5%）；Qwen2.5 = `experiments/j2_crossgen_matrix.md` Table1（full .870/.975/.805/.585；RBM .735/.687/.465/.565；post .415/.499/.180/.555）。
- **已知空格**：FastV/Pyramid 的 DocVQA HF cell 坏（skip 187/200）[待J4余]；OCRBench/GQA none 列为既有 cell 参考值。
- **Pyramid 注**：canonical keep_equiv 0.625 = **不同预算点**（2.1×），独立脚注或进 retention 曲线；iso-25% [1,0,0,0] 试跑 [待J4余]，崩则只报 canonical 并如实注。
- **公平脚注**：同 keep ratio（相对每图自身 full）+ 报平均绝对 ptid + 按 family iso min/max pixels + Pyramid 折算等效均值；他文数字趋势锚（R3）。
- **VZ 官方锚**：不混入主表；脚注引 Table 9（README OCRBench 81.5→70.5@50%，model/stage-mismatched reference）。
- **红线表注（必印）**：FastV 在同模型同预算胜 TextVQA/GQA；本表不支持"RBM 超过现有方法"；RBM claim = 鲁棒（任何基准不崩）+ text-dense/OCR 大胜 post 族。

### Table 2 —— 机制（M1–M3）
- **Panel A · M1 重排**：Spearman ρ / Kendall τ / Jaccard@25%，bench = DocVQA/TextVQA/GQA × {Qwen3 [现成: mechanism §1：ρ 0.137/0.332/0.360，Jaccard 0.180/0.243/0.278]，Qwen2.5 [待J3]}。
- **Panel B · M2 反文本方向**：ρ(rank_shift,edge) + group(a)/(b) Sobel + frac>median，{Qwen3 [现成: mechanism §2：ρ +0.439/+0.155/+0.036；DocVQA (a)0.641 vs (b)0.124、92% vs 35%]，Qwen2.5 [待J3]}。
- **Panel C · M3 因果交换**：baseline / post / pre / swap 四列 + 字节一致计数。Qwen3 [现成: mechanism §3：TextVQA VQA-acc .858/.215/**.598**/**.603**（Δ+0.005, paired SE 0.005, 198/200 一致，swap-vs-post 40/200）；DocVQA ANLS .976/.200/**.465**/**.465**（Δ0.000 exact, 200/200 字节一致，swap-vs-post 63/200）]；Qwen2.5 [待J3]。
- **nuance 行**（必印）：docvqa 完整支持 / textvqa partial（M2 中等 + budget 层 0.60 vs 0.86）/ gqa 近零。

### Table 3 —— 效率 [待J6 为主]
- 维度：TTFT / p99-TTFT / req/s / peak VRAM × budget {full, 25%, 12.5%}（建议 + 并发 c1/c16/c64）。
- 现成底料：stage 吞吐中性 ±10%；压缩 vs full +28~45% req/s [现成: v3_sota_matrix §7]。
- 披露（R7）：效率数仅 vLLM 内我方方法；FastV/Pyramid（HF eager）**不报吞吐**（harness 注）。v2 轮 c64/goodput 仅 appendix + 重标 harness。

### Table 4 —— 负结果（设计空间封闭）
- 三行组 × 判据 × 结果 × 裁决：(a) 图像级 router [现成: method_gate §4 + 4b]；(b) merger-aware hybrid frac-sweep {0.0/.5/1.0} + gate FAIL [现成: method_gate §2–3]；(c) QA gate λ 梯度 {0/−1.7/−5.2/−3.3pp} NO-GO [现成: j5]。
- 结论列统一：图像级 ✗ / merger-aware ✗ / query 级 ✗ → **pre-merger 必须 query-blind**（+ FastV 证据：query 余量仅 LLM 交叉注意力后可得）。

---

## 4. Venue 候选（截稿日 2026-07-24 web 初核，**投稿前必须再核官网**）

**CCF-B 主目标（3 个，按推荐序）**
1. **Pattern Recognition**（Elsevier 期刊，CCF-B）—— **滚动投稿无截稿，本轮最稳主目标**。Fit：经验律 + 因果机制 + 大规模受控实验的方法型论文正中 PR 选题；期刊篇幅容得下双模型 + 四表 + 负结果。风险：审稿周期 6–12 月；审稿人或要求第三模型族（InternVL）——以 limitation 明 scope 应对。
2. **ICME 2027**（厦门，2027-07-13~17，CCF-B 会）—— 论文截稿**未官宣**，按往届约 2026-12~2027-01（ICME'25 ≈ 2024-12）；需盯 2027.ieeeicme.org。Fit：multimedia + 效率表（Table 3）匹配度好。风险：8 页压机制深度（M1–M3 + 三连负需 appendix）。
3. **EMNLP 路线**（CCF-B；EMNLP'26 commitment 2026-08-02 需已审 ARR 稿、ARR 新稿 2026-05-25 已截 → **本轮不可行**，现实目标 ARR'27 周期 → EMNLP'27/NAACL'27）。**注意：Findings 不入 CCF 检索**，计 CCF 只算主会。Fit：VLM 效率是 NLP 邻域热点、三连预注册负结果合 Findings 气质。风险：NLP 审稿人判 CV 新意轻。

**CCF-A 扩展（2 个）**
4. **ACM MM 2027**（香港，CCF-A）—— 截稿**未官宣**，按往届约 2027-04 上旬（MM'25 = 2025-04-11）；盯 acmmm.org。**CCF-A 首选**：multimodal + 效率 + 方法三维全中、9 页容全文；时间线（J7/J6 年内可完）宽裕。
5. **AAAI 2027**（蒙特利尔 2027-02-16~23，CCF-A）—— **abstract 2026-07-21 已过（3 天前）、full paper 2026-07-28 → 本轮无法投**（abstract 未注册即不能提交，且 J7 ~50–60 GPU·h 不可能 4 天内完成）。下一窗口 AAAI-28（约 2027-07/08 截稿）。本轮列为"观察，不备"。

**排序结论**：主投 Pattern Recognition（滚动、随时可投，待 J3/J7 补齐即投）；并行备 ICME'27（~12 月截稿，含精简 8 页版）；CCF-A 冲刺 ACM MM'27（~4 月截稿，全文 + 效率全表）。

---

## 5. Submission-ready 检查清单

**A. 数字一致性**
- [ ] grep 全文清旧 containment 数：0.695 / 0.255 / 0.725 / 0.390 / 0.320 / 0.380 / +44.0 / +33.5 / −6.0 / +46.6(n500) → 全部换官方或删除。
- [ ] abstract / §1 / Table 1 / §4.3 / §5 / §7 六处 headline 数（+38.3 / +26.5 / +41.5；Qwen2.5 +32.0 / +18.8 / +28.5）逐处核对同源。
- [ ] 所有 z 值按官方数重算；DocVQA 只用 clean +47.5pp（mnbt 修复后），旧脏数 +61pp 全文搜杀。
- [ ] OCRBench/ChartQA scorer 标签确认为官方口径移植（lmms-eval 逐式 + gt 自测 200/200）。
- [ ] Table 1 interim（subset n200）与 final（J7 完整 split）版本不得混用；投稿版只留 final。

**B. 红线审计（R1–R7，逐节签）**
- [ ] 全文无"RBM outperforms/beats existing methods/SOTA"字样；FastV 胜 TextVQA/GQA 如实写。
- [ ] 无跨模型 SOTA 宣称；VisionZip 官方数仅 Table 9 mismatched 锚注。
- [ ] GQA 表述 = "tie / pre 弱占优 / 无 crossover"；Qwen3 n200 post +4.5pp ~1.3σ 报方向（R4）。
- [ ] 所有不显著 gap（ChartQA、GQA 各档）标 direction only。

**C. 图表**
- [ ] Figure 1 stage_law 按官方数 + 双模型重绘；Figure 2 retention 曲线补 Qwen2.5；Figure 3/4 单图机制降插图。
- [ ] ➕ 新图：pipeline 示意（pre vs post hook + merger + deepstack）、M1–M3 机制图（已有 token_survival_m1/m2 png）、跨代对照条形图。
- [ ] 所有表脚注含 CI 口径、n、scorer、引擎。

**D. 代码与数据开放声明**
- [ ] Code availability：github.com/Code-Yan-ZX/VLA 公开（runner / official_scorers / mechanism 分析 / HF harness），投稿前清内部路径与日志。
- [ ] Data availability：subset 清单 + 完整 split 获取脚本；基准原始数据引官方来源。
- [ ] [CITE] 占位全部替换真实 bib（含 ERA 2606.31982、qwen25vl/qwen3vl、visionzip、fastv、pyramiddrop、lmms-eval）。

**E. Limitation 节必含（6 条）**
- [ ] ① 单 LLM 族（Qwen-VL 两代，同族非跨族；InternVL/LLaVA 未验）；② HF-baseline 引擎差异披露（FastV/Pyramid 走 HF eager、我方走 vLLM；r=0 锚 8/8 + 16/16 等价证精度可跨引擎比、**吞吐不可跨引擎比**）；③ DocVQA 同像素 cap 披露（max-pixels 1.5M + mnbt 32768 配置依赖，post 深压崩溃系环境非方法缺陷亦需如实注）；④ GQA n200 Qwen3 post 弱领先 +4.5pp ~1.3σ 如实报（不藏）；⑤ subset n=200 → 完整 split 的迁移状态；⑥ greedy 解码仅 binomial CI（无温度方差）+ 单 A40。

**F. 投稿前闸门（按序）**
- [ ] J4 余（DocVQA HF 重跑 + Pyramid iso-25%）→ J3（机制跨代）→ J7（完整 split 主表 ~50–60 GPU·h，逐 cell <6h 不触升级）→ J6（效率）→ J8（消融闭环）→ 全文 draft v4 → 自审（可 nature-reviewer 模拟）→ venue 定稿。
