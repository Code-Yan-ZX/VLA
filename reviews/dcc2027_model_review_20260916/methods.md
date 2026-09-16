# 方法与机制审稿（仅基于稿件）

审阅对象：`drafts/dcc2027_submission_20260914/main.tex`，共 412 行；另查看正文 Figure 1，以及父 agent 指定的既有控制记录 `reports/acmmm_final_controls.md` 中有关 scoring tap 的段落。未使用 skills、未检索外部论文、未运行代码或实验。以下“已确认”指稿件/既有记录呈现的证据；不能据此断言底层实验错误。

## 总体判断

本稿的可取之处是问题明确：在完整 native units、相同保留数量和固定模型下，比较两个具体的 query-blind 范数选择器。大幅且跨数据集的准确率差值得报告；固定 kept set 的 Qwen3-VL byte-identical 检查对排除某类实现差异很有价值；作者也明确披露 GQA 反转、hybrid 的局限及跨模型归因边界。

但目前最有说服力的结论是“选用不同表示上的范数可产生任务相关的选择差异”，并非已经充分证明“通用的 merger-induced text-evidence demotion 机制”。最关键的问题是主实验的实际 scoring tap 与真正 merger-input 控制不同：既有控制记录明确承认主表同时改变 stage 与 feature space，而正文没有解释这一差异。这直接影响方法定义及主表归因，而非小幅补充说明。以当前呈现，建议大修后再评估。

## M1 — 主要问题：主方法与真正 merger-input 控制的关系不清，影响核心归因

- **位置**：82–85、186–203、245–246、287–289 行。
- **证据**：前文称 RBM 在“merger input”打分，公式使用输入 patch vectors；主表被称为隔离 operator-order effect。但 287–289 行又报告一个“matched at the merger input”的 full-split Qwen3-VL control，其 TextVQA / DocVQA 差值为 +27.68 / +4.59 pp，与主表 +38.4 / +24.3 pp 明显不同。
- **附属记录核对**：`reports/acmmm_final_controls.md:24` 明确写 headline RBM 在 ViT layer-8 / deepstack[0]-input 打分，post 在 main-merger output 打分，并直接称其为 stage 与 feature space 的双重混杂；该文件 38–41 行的真正 pre-final 数字与本稿 287–289 行一致；49 行也说明早层 tap 带来额外收益。
- **状态**：已确认稿件省略了既有控制记录明确报告的 tap 差异与混杂；本审阅未独立复跑底层实现。
- **影响**：读者无法判断主表使用的是哪一层、哪一个归一化前后张量；若主方法并不位于实际 merger 输入，主表改变的可能包含 scoring tap、归一化或流选择，不能全部算作“只改变 selection order”。这项差异的量级尤其影响 DocVQA 机制结论。
- **最小修复**：加一张小表，逐模型列出 RBM 和 Post-L2 的具体张量位置、是否经过 LayerNorm、主/deepstack 流、单位顺序、merger 边界；明确主表与该控制唯一不同的因素，给出控制双方绝对分数。如果严格匹配版本才真正隔离该边界，应把它作为核心机制证据，把主表保留为算法表现证据。

## M2 — 主要问题：rank inversion 与 text-stroke demotion 的证据强度未分开

- **位置**：50–59、74–77、98–100、343–353、363–365 行。
- **证据**：Spearman/Jaccard 能支持范数排名和保留集合发生变化；Sobel edge energy 的相关性及组间差支持边缘丰富区域与选择差异相关。但正文从 Sobel 直接推到“text-stroke-rich”和“answer-bearing regions”，未给文字区域、文字笔画或答案区域的独立标注验证。
- **状态**：已确认论证缺口；不是证明这些区域并非文字。
- **影响**：边缘也来自纹理、边框及非文字物体。即使文本密集任务提升明显，也不能单凭 Sobel 指标认定被降权的内容就是答案文字，更不能认定 merger 的表示丢掉了该证据。
- **最小修复**：在当前证据下把机制描述收窄为“norm-rank rewriting associated with high-edge regions”，将文字解释明确标为假说/视觉案例观察。若保留强机制主张，在独立样本上按 OCR/text box mask 比较文字与非文字单位的 rank shift、retention 和答案相关性，且说明 edge energy 与 rank shift 的具体计算和汇总方法。

## M3 — 主要问题：swap 检查验证实现等价，尚不能独立证明 pre-rank 的语义优越性

- **位置**：151–171、201–203、355–358 行。
- **证据**：稿件已假设 merger 对 unit 独立，又定义 swap 使用与 RBM 完全相同的 kept set。在这些条件下，两条路径产生相同 retained features 本就是预期结果；正确保持位置与接口时，准确率恢复也随之而来。
- **状态**：已确认该控制的推断范围有限；控制本身合理且有用。
- **影响**：它有效排除“同一集合上的 merger 前向操作不同”，但没有区分“特定 pre-L2 score 比 post-L2 更适合 OCR”与“学习到的 merger 普遍破坏了可被下游选择器利用的显著性”。摘要 58–59 行把 rank、edge、swap 合并为较强因果解释，容易超过控制的识别范围。
- **最小修复**：明确将 swap 称为实现/归因一致性控制，结论限定为 Qwen3-VL 两个具体选择规则的差异经 kept-set 改变产生。若要解释为何 post-L2 失效，追加最少量的得分控制：例如 concat/group norm、归一化前后 norm，或在 post 表示上加入能恢复 pre-rank 的简单诊断。不要把后者写成当前已经证实的结果。

## M4 — 主要问题：跨模型机制表述宽于稿件自己承认的有效归因范围

- **位置**：56–59、103–106、245–248、389–390、399–402 行。
- **证据**：主表段落说隔离 operator-order effect，摘要随后用机制语言总结跨三模型结果；限制段却明确写 Qwen2.5-VL 有 ordering confound，InternVL3 只有 accuracy replication 与 bounded swap。
- **状态**：已确认表述范围不一致；ordering confound 的具体内容和大小未说明。
- **影响**：Qwen3-VL 上的严格控制不能自动赋予另外两模型同样的因果识别强度。ordering confound 若影响 token/position 对应，可能直接影响输出，不只是次要细节。
- **最小修复**：把“跨三模型准确率现象”与“Qwen3-VL 机制验证”分开陈述。用一两句精确定义 ordering confound，给 bounded swap 的样本量及结果/界限；否则删去无定量解释的“bounded”并明确尚未完成严格隔离。

## M5 — 中等问题：native contract 与 RankBridge 缺少足以独立复现的操作定义

- **位置**：186–213、225–230 行。
- **证据**：正文仅说保留完整 units、保持 positions/interface、propagate identities，并用 FastV rank 补齐。缺少排序后是否恢复原有空间/序列顺序、原始坐标如何对应到裁剪序列、deepstack 使用哪个流打分以及何时同步掩码。RankBridge 未明确从 protected set 的补集中补齐、attention score 的 query/head 聚合、K=3 是读取哪次 attention 后在哪一层裁剪。
- **状态**：已确认正文复现细节不足；不等于代码没有实现。
- **影响**：同样的公式可能实现为不同 token 序列与 attention 选择器；尤其稿件承认 ordering confound，使这些细节成为关键变量。
- **最小修复**：补 10–15 行伪代码，写出 original-index stable gather、原始 position ID 的处理、各流 mask、`P=Top_q(a)`、`K=P ∪ Top_(k-q)(attention restricted to complement(P))`，并明确 layer/head/query 约定。按模型给实际 tap/module 路径和 code commit，而非仅仓库首页。

## M6 — 中等问题：开发门槛与锁定验证的划分缺少定义

- **位置**：207–213、293–297 行。
- **证据**：作者在三个 rho 上做 development gate 后锁定 0.2，在四个 n=200 benchmark 上报告；但未说明 gate 的目标、开发样本量、开发与测试是否不重叠。“within-1 pp gate”到结果段才出现，未在方法段定义相对哪个比较对象、在哪些任务上要求成立。
- **状态**：已确认报告不完整；不能据此认定测试泄漏。
- **影响**：读者无法区分独立验证与调参集结果，也无法复现为何选择 0.2 或解释预设门槛失败。
- **最小修复**：给开发/锁定样本的 ID 或清单路径、随机种子和重叠检查，明确目标函数及所有门槛；若存在重用，按探索性结果报告，不继续称为独立锁定验证。

## M7 — 次要但应修正：数学措辞需要明确“集合相同”与“表示相同”

- **位置**：161–171、173–182 行。
- **证据**：“selection commutes ... exactly when K_pre=K_post”在定义为 unit identity 一致时成立；若把 commutes 理解为算子输出张量相等，则非单射 merger 可能把不同 units 映射为相同输出，这时集合相等并非必要条件。此外位置、顺序和并列分数的处理也是算子定义的一部分。`D(R)=1-S(R)/S(1)`本质是相对任务分数损失，不是直接测量信息存活；如果压缩提高分数它可以为负。
- **状态**：已确认数学表述需限定，不影响已报告分数。
- **最小修复**：将前者称为“identity-preserving selection agreement”，明确双方采用相同 tie-break 和原始索引顺序，或给出含索引的算子定义。把“more task-relevant information survives”改为“higher benchmark-score retention”，注明该 task-distortion proxy 可以为负、不同 benchmark 间不可直接比较。

## 建议的最小修改顺序

1. 先澄清 M1：真实 scoring tap 与主表/控制的映射。这决定是否需要重新组织主结论。
2. 在不增加实验的情况下即可修正 M2–M4 的措辞边界，并补 M5–M6 的已有实现和开发记录。
3. 若坚持强机制主张，再补文字区域独立验证及最少量得分/归一化控制；无需为了叙事强行增加更大范围的“通用优越”实验。

这些意见不否认 RBM 对报告的 Post-L2 对照存在较大实测优势，也不把未报告的细节自动当成错误。它们要求论文把“观察到的效应、排除的替代解释、尚未排除的替代解释”说清楚。
