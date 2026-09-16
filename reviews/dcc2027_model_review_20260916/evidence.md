# 实验证据与统计审稿（模型独立判断）

审稿对象：`drafts/dcc2027_submission_20260914/main.tex`，2026-09-16。未使用 skills、外部资料、既有审稿结论或运行新实验。为核对少量关键数字，读取了项目已有实验摘要；下列附属材料证据不等于原始逐样本数据已被重新审计。以下行号以当前 LaTeX 为准。

整体判断：已有三模型全量 pre/post 对照和诚实呈现的 GQA 反转，足以支持“某类 pre-merger 信号值得研究”。但稿件把 headline 方法的效果归为单一 merger 操作顺序效应，超过了当前对照所能区分的因果范围；RankBridge 的样本量与显著性报告也需要实质修正。建议大修后再投稿。

## 已确认的问题

### E1 — 重大：主表效果混合了特征深度与选择位置，纯 merger 效应没有被主表单独识别

- 正文证据：103–106 行称方法“changes only selection order”；186–203 行把分数定义为同一 merger 输入/输出的 L2；245–246 行称主表“isolates the operator-order effect”。主表 Qwen3 DocVQA 提升 24.3 pp（265），另有 4.59 pp 的 matched-merger-input 控制（286–289），两者为何不同没有解释。
- 关键附属证据：`reports/acmmm_final_controls.md:24` 明确 headline pre 在 ViT layer-8 / deepstack[0] 输入打分，post 在主 merger 输出打分，具有 stage 与 feature-space 两重混杂；该文件 38–40 行给出 matched-depth pre-final TextVQA .4985、DocVQA .2836、OCRBench 419，分别比 post 高 27.68 pp、4.59 pp、235 点。49 行明确 feature-depth 另有贡献。
- 影响：在 DocVQA，headline 24.3 pp 中仅有 4.59 pp 被现有同深度对照直接支持为前后 merger 选择差异，其余不能全部归到 merger。注意这不是说差额已被严格归因为“纯深度”，因为实现还涉及 deepstack 路径；它只是证明主表不能单变量归因。正文 343–358 的 rank/swap 说明选集重要，也不能自行排除早层表征贡献。
- 最小修复：明确每个模型 scoring tap、deepstack 处理与 post 定义；将 Qwen3 主表增加 pre-final 一列，或将已有四基准完整控制单列成表；把 headline RBM 与纯 stage control 分开命名。摘要和方法贡献以同深度控制支撑 operator-order 结论，并把更大的主表效果称作 merger-aware pipeline 效果。现有数字即可完成这一修复。

### E2 — 重大：RankBridge TextVQA 的“显著”依赖不充分报告的渐近二值检验

- 正文证据：293–295 行给出 +3.2 pp、z=2.11 并称其余提升不显著；392–393 行明确“only the TextVQA hybrid gain is significant”。232–235 行使用官方连续型 VQA accuracy。
- 附属证据：`experiments/rankbridge_gate.md:50–52` 给出 TextVQA z=2.11，discordant counts 为 b10=9、b01=2。因此 z=(9−2)/sqrt(11)=2.1106，是未作连续性校正的二值 McNemar 渐近量。
- 可直接核算：同一 9:2 不一致计数的双侧 exact McNemar p = 2×(C(11,0)+C(11,1)+C(11,2))/2^11 = 0.06543，超过 .05。即使采用 z 的正态近似 p≈.0348，若四个 hybrid-vs-FastV benchmark 属于同一检验族，最小 p 也过不了 Holm 第一阈值 .05/4=.0125。
- 影响：正文将官方 VQA-score 增益与一个二值正确性检验直接相连，而且未说是 nominal/unadjusted。不能据此宣布“唯一显著提升”。这里没有证明真实 VQA score 的 paired test 一定不显著；它尚未被报告，且不能由这两个计数恢复。
- 最小修复：对逐样本官方 VQA 分数差报告 paired bootstrap CI 和 permutation p，说明四探针的检验族并按事先规则校正；在此之前写成“+3.2 pp observed improvement; statistical evidence is exploratory”，删除确定显著措辞。小 discordant 数时同时报 exact McNemar，不能只报渐近 z。

### E3 — 中等：OCRBench 的 n=200 是计划量，表中实际有效分母为 181

- 正文证据：239、302–304 行统一标 n=200；313 行 .5801/.4586/.4641；295–296 行给出 RBM 高于 hybrid 11.6 pp。
- 附属证据：`experiments/rankbridge_gate.md:47` 与正文三个小数相同；53 行明确“ocrbench 181/181; 19 shared load skips”。三个数分别吻合 105/181、83/181、84/181。因此是 200 项计划、181 项有效共同样本，并非 200 项完整评测。
- 影响：有效 n 缩小与失效样本的排除未披露，读者会错误理解数据覆盖范围。这尤其涉及较难大图是否更容易被排除，以及与 full-split OCRBench 的分母可比性。不能据此推断数据造假或不配对，现有摘要反而明确共同 ID 配对。
- 最小修复：表注写清 attempted n=200，OCRBench paired n=181，19 个共同失效及原因；其余 n=200。若统一按 200、失效记零，分数应为 .525/.415/.420，hybrid 与 RBM 差 −10.5 pp、vs FastV +.5 pp。两种分母可选其一，但必须与全文统计和“official”定义一致。

### E4 — 中等：摘要 OCRBench 范围把纯阶段控制与九个主表比较混在一起

- 正文证据：55–58 行称九个 text-dense model–benchmark pairs 的提升为“11.0–38.4 percentage points and 235–432 OCRBench points”；247–248 行重复。主表 OCR 三项为 363（266）、298（272）、432（278）；235 出现在另外的 matched-input Qwen3 控制（288）。
- 影响：即使 235 是较保守数，也不属于这九格 headline 结果，范围的比较集合不清。
- 最小修复：九格主表写 OCR 298–432；另独立写 Qwen3 matched-depth OCR +235。不要为了维持较宽范围隐含合并不同实验定义。

### E5 — 中等：“12.5% gap widens”缺少适用任务范围，现有结果有反例

- 正文证据：247–250 行称更低 retention 时 reported text-dense cells 的 gap 增大。
- 可复核反例：`scripts/plot_dcc_rate_distortion.py:34–35` 的 OCRBench 点，25% RBM−post=.580−.165=.415，12.5% 为 .380−.075=.305；差距缩小。另 `reports/acmmm_final_controls.md:69–70` 的 P0-2 表给出匹配设置 Qwen2.5 OCRBench 差距从 25% 的 298 降到 12.5% 的 258。主稿也没有 12.5% 主表。
- 影响：把部分任务的趋势泛化为 text-dense 总规律。可能作者想说倍率增大，但当前句子紧承绝对 pp/points gap，读者会按绝对差理解。
- 最小修复：明确哪些 benchmark、哪个 split、哪个 gap 定义；若没有在稿内完整呈现，删除此句最省篇幅。若保留，准确写成有些任务扩大而 OCR 绝对差可能缩小。

### E6 — 中等：效率结论的工作负载与测量方式不足以复现

- 正文证据：382–387 行列 req/s、five repeats、0.36→0.23 s、InternVL3 1.8–2.5×，未给 benchmark、请求并发、output length、batch 设置、warmup、TTFT/总延迟定义或测量区间。
- 附属证据：`experiments/j6_efficiency.md:1–3` 指明 Qwen3 吞吐是 TextVQA n=200、vLLM offline 批推理、max_num_seqs=8；15 行延迟是单请求 n=1×5 的均值。该摘要不能证明吞吐也做了五次；不据此断言没有重复，仅要求正文澄清。
- 影响：offline batch 吞吐与单请求延迟来自不同场景；同样预算的准确率和吞吐若输出长度不同，端到端耗时还会混入生成内容/长度变化。数字算术本身没有问题（6.91/4.11≈1.68）。
- 最小修复：用一行表注交代 workload、n、并发、生成上限、warmup/repeats 和误差；明确 latency 定义。注明端到端自然生成结果，若要单独归因计算提速，补固定输出长度或 prefill 单独耗时。FastV/RBM 的最终预算相同并不等于前3层计算相同，当前“只在 vLLM 内比吞吐”的限制是正确的，应保留。

## 待澄清问题（不能由当前材料确认错误）

### E7 — dev 选择和 locked probe 是否独立

- 正文 210–211 行说 development gate 在 rho=.1/.2/.3 中选 .2，随后 locked n=200；没有 dev 大小、ID、是否与200项互斥。
- `experiments/rankbridge_gate.md:30–40` 给出 dev n=64、按 TextVQA/OCRBench 均值选择，但没有足以确认互斥的 ID/抽样说明。
- 影响：如果64项包含于200项，locked 仅表示参数后来冻结，不能把这200项称作独立测试；目前不能确认是否真的重叠。
- 最小修复：公开 dev/test ID hash 与交集计数，交集为零则一句话澄清；若重叠，报告排除开发 ID 后结果或标明探索性。

### E8 — 统计抽样单元与完整复现配置

- 正文 225–239 行未给模型具体 revision、各 family pixel cap/tiling、prompt、max_new_tokens、失败计分、样本选择 seed、bootstrap 以 question 还是 image/document 抽样；393 行只笼统说 family pixel caps 不同。
- 影响：一张图/文档可能对应多问题；若存在共享图像，按问题独立 bootstrap 可低估不确定性。当前未确认各split存在多少共享图像，不据此断言 p 值无效。缺少分辨率也使“25%”无法换算实际 LLM token 和计算量。
- 最小修复：提供一个紧凑 protocol 表/指向固定配置 manifest，逐模型报告 cap、实际 token 均值和解码规则；说明 bootstrap 单元，如多问题共享图则图像/文档 cluster bootstrap 作敏感性分析。九个 pre/post 效果很大，细节缺失不等于它们会失去显著性。

### E9 — 压缩器价值与机制对照应分开评价

- 正文主表只有 Post-L2；Table 2 自身显示 TextVQA .5967 vs FastV .7633，DocVQA .4239 vs .5863，GQA .415 vs .505，RBM 只有 OCRBench 更高（311–314）。320–322 行承认 FastV 的优势是优点。
- 影响：Post-L2 是合理机制控制，但“显著超过 Post-L2”不能直接证明是强实用压缩器；尤其“text-dense”本身不能预测 RBM 是否超过 attention baseline。
- 最小修复：明确主 claim 是 mechanism/order diagnostic 或 OCR niche。若要保留通用压缩算法定位，需要同模型、同数据、同像素/保留量的更强基线及随机/均匀/边缘简单基线；这是新增实验建议，并非当前审稿已授权运行。无需泛泛补很多SOTA名字，先解决最小简单基线能否解释优势。

## 需要保留的优点

1. 既有主表覆盖三模型、全数据集，对比较方法保持每图 token 数，且九个主比较进行了多重检验控制。
2. 主文主动呈现 GQA 反转、Qwen2.5 的 ordering confound、HF/vLLM 引擎限制、hybrid 未通过预设门槛，没有把所有任务包装成正结果。
3. 已有 matched-depth Qwen3 full-split 控制说明核心观察并未完全消失；把它提升为主证据即可大幅增强文章，无需为修复因果陈述立即启动训练。

优先修改顺序：E1 → E2/E3 → 图中指标/协议核对 → E4/E5 → 效率与复现表。上述建议均针对证据呈现与推断边界，不是对原始数据真实性作指控。

