# CVM 2027 写作与论证审查（2026-10-25）

审查对象：`F:\dcc2027_submission_20260916\main.tex`，行号以 203 行原稿为准。已读 README、EVIDENCE_AUDIT、FIGURE_AUDIT，并直接查看 fig1 与 fig4_word 图像。本文不修改原稿。另结合 `/root/evidence_review` 核实的 `origin/main:reports/dcc_presubmit_expA.md` 后续证据。

**收尾更新，优先于下文对旧稿的原句引用：** 本地 TextVQA scorer 的共识公式、EvalAI 正常化及答案分号边界均有问题，ExpB 又误用 binary containment。现已从原始预测与官方十答案 annotation 对 14 格完成 CPU 官方重评分，0 missing，参见 [证据报告](cvm2027_evidence_review_20261025.md) 和 [权威新评分表](cvm2027_textvqa_rescore_20261025.md)。三模型 text-dense operational 优势方向保留；同 HF TextVQA 则 RBM .75504、FastV .76826，差 −1.322 pp，95% CI [−2.30205, −.32000]。下文引号中的旧分仅定位原稿，不可复制进最终表；新稿应统一使用新评分表。本报告不替代该独立数据审核。

**GQA 最后一轮 CPU 核验也已完成：** 固定 lmms-eval 函数只忽略大小写和标点，不使用旧 scorer 额外的冠词/数字/缩写处理。11 格完整重评、5 组 20,000 次配对 bootstrap 后方向均不变；Post-main − Pre-final 更新为 +6.18540 pp，95% CI [5.43012, 6.95659]。完整新分及来源见 [GQA 重评分表](cvm2027_gqa_rescore_20261025.md)。

## 1. 先判断问题：可删的是重复防守，不是反证

原稿确有明显的防御写作：摘要连续交代三种归因限制，引言先讲“我们把准确率和归因分开”，多张图注再解释它们不能证明什么，结果、诊断、成本、结论又重复同一限制。读者很难迅速看到研究问题、方法和证据链。这些可以大幅压缩。

但其来源不是单纯“作者不够自信”：三个架构实际有不同的评分、流聚合、顺序、位置与配额设置。9 月审计已承认这些差异。更关键的是，后续已完成的 Post-main 控制使 DocVQA 上 pre-vs-post 的方向反转。删掉相关限定而维持“pre-merger 对三类文字任务普遍更好”的叙事，会实质改变证据含义。

建议主张由“先排再合并更优”转为：**原生视觉单元的 norm 选择高度依赖评分表征、辅助流聚合与具体实现；一个便宜的 pre-input 选择方案在所测部署路径上提供了有用结果，但最佳表征随任务而变。** 这是写作决策建议，不是已经得到新方法新颖性证明。

### 最新证据必须进入正文

ExpA 主分支输出评分（Post-main）与 Pre-final 的全量比较如下。TextVQA、GQA 已替换为本次官方 CPU 重评分；其他行供定位旧结果，最终以各任务最新核验表为准。主窗口应把准确协议及样本清单同步到新稿。

| 任务 | Pre-final | Post-main | Post-main − Pre-final | 已核验的配对 95% CI |
|---|---:|---:|---:|---:|
| TextVQA | .49170 | .31230 | −17.94 pp | [−19.45, −16.45] pp |
| DocVQA | .2836 | .3735 | +9.00 pp | [7.84, 10.14] pp |
| OCRBench | 419 | 356 | −63 points | [−99, −26] points |
| GQA | .41978057 | .48163460 | +6.18540 pp | [5.43012, 6.95659] pp |

**区分旧限制与新反证：** Qwen3 聚合、Qwen2 顺序、Intern tile 配额都是原审计已发现并且现稿已局部承认的混杂；并非本次突然发现全部实验无效。新增的是 Post-main 的实测结果，它表明 DocVQA 的“Pre-final 比 post 好”不能从 Post-concat 基线推广到 main-only post 基线。两者都应讲清，但不需要每段重复。

ExpB 旧报告中的 `.8374 / .8494 / +1.20 pp` 使用了 `correct` 的 binary containment 值，不能直接称官方 TextVQA VQA accuracy。本次已从原始答案离线重评为 RBM .75504、FastV .76826（全 attempted 5000 分母，共同 5 个失败记零）。这属于评分正确性修复，没有重新跑 GPU。

## 2. 标题：保留方法名可以，避免标题替读者预设结论

现题（L14–15）：**Rank Before You Merge: Native-Unit Selection for Visual Token Compression**。

优点是容易记住，能定位“选择原生单元”。风险是“before merge”在 InternVL3 实际是 pixel shuffle 之后、projector 之前；最新控制又显示 pre 优势非普遍成立。标题本身没有字面声称 universal SOTA，因此不属于必须撤回的假结论，但它会让审稿人用“pre 比 post 更好”的标准评判全文，而现有证据支撑不了这一宽泛机制。

推荐标题（研究性质更准确）：

> **Where to Score Native Visual Units? A Study of Norm-Based Token Compression**

若继续以方法为主：

> **Rank Before You Merge: Native-Unit Norm Selection for Text-Dense VLM Tasks**

后一标题仍须在摘要明确流聚合与任务依赖，也不能把“text-dense”理解为所有文字任务、所有实现都占优。不要使用“Text-Preserving”“Evidence-Preserving”“Robust”“Universal”之类尚未实证的修饰。

## 3. 建议采用的证据组织

1. **问题**：同样的最终 token 预算下，用哪一种特征表示给原生单元打分？为什么表征与 native interface 值得关心？
2. **实现**：RBM 如何选择完整单元；三个模型具体在哪一层/算子前、怎样打分与分配配额。
3. **部署比较**：当前三模型矩阵是实际实现结果；不把其中全部差异归结为 merger 一项。
4. **分解比较**：Qwen3 的 early-tap、Pre-final、Post-concat、Post-main 四个变体合在一张控制表，直接展示 DocVQA 反转。
5. **竞争力与成本**：同一 HF 协议的 FastV OCRBench；其余 FastV 指标在官方重评前不挪用。200 样本吞吐只给具体 workload 结论。
6. **解释**：排序重排、边缘关联、固定集合路径检查；每项只回答它能回答的问题。

把共同边界放在“变体/协议”中一次说完。结果段写“什么现象 + 哪组实验支持”，不再每段追加“this is not ...”的重复句。

## 4. 摘要、引言、贡献和结论的可执行改写

### 4.1 摘要（替换 L28 整段；推荐包含 ExpA、并完成 TextVQA 官方重评后使用）

原段关键负担：`under model-specific taps and allocation rules`、`reduces the feature-depth confound`、`not a pure merger-order intervention`、`bound the interpretation`、`not a universal selector` 连续出现，方法与关键结果被审核旁白淹没。

建议英文：

> Visual token compression requires both a token budget and a representation for ranking candidate units. We study norm-based selection at the inputs and outputs of native visual mergers and projectors. Rank-Before-Merge (RBM) selects complete native units without training or changing the learned projection weights. At 25\% retention, its deployed implementations outperform their Post-L2 baselines on nine text-dense model--benchmark pairs across Qwen3-VL, Qwen2.5-VL, and InternVL3. Full-split Qwen3-VL controls show that the preferred representation depends on the task and scoring streams: final-input selection exceeds main-output selection on TextVQA and OCRBench, while main-output selection favors DocVQA and leads by 6.19 percentage points on GQA. Matched-harness comparisons favor RBM over FastV on OCRBench and FastV over RBM on TextVQA. These results identify scoring representation and auxiliary-stream aggregation as consequential choices in native-unit compression.

**使用条件**：仅 Qwen3 concatenate 了 main+deepstack，故用 `their Post-L2 baselines`，后文统一定义；不得为了顺口把 Qwen3 特性推广到 Qwen2/Intern。上文 `nine` 及涉及 TextVQA 的方向也必须在官方重评后再确认。重评前的内部排版稿可只概述已核验的 DocVQA、OCRBench、GQA，不把占位数据称最终结果。

这版摘要有意不列吞吐与所有 CI，把有限字数用于“问题—做法—部署结果—控制发现”。若标题仍按方法论文，须补一句真正区别现有方法的新颖性；不能用“简单/无训练”自动当新颖性证明。

### 4.2 引言（替换 L32、L34、L36、L50，保留引用后插入）

P1（L32 现有第一段方向正确，可扩成研究问题）：

> High-resolution vision--language models convert many image patches into visual inputs for a language model. Native spatial mergers and projectors impose a model-specific unit structure, so additional compression must choose both how many units to retain and which representation to use for selection. Feature magnitudes can change across these operators, giving different retained sets at the same final token budget. We ask how this choice affects text-dense and object-centered visual question answering.

P2（替换 L34；删除 `deliberately`）：

> Rank-Before-Merge (RBM) ranks complete native units using feature norms at the inputs of the native mergers or projectors. It reuses their learned weights and output dimensions and introduces no training. The implementation uses a shared unit mask across Qwen3-VL's main and auxiliary streams and model-specific adapters for Qwen2.5-VL and InternVL3.

P3（替换 L36；以实验设计解释证据，而不是先说自我约束）：

> We evaluate three model families on full TextVQA, DocVQA, OCRBench, and GQA splits at 25\% retention. We then decompose the Qwen3-VL design into an early-input selector, a final-input selector, and two output selectors that score either the main stream alone or its concatenation with auxiliary streams. This comparison exposes the role of scoring depth and stream aggregation in the observed accuracy differences.

P4（替换 L50；加入新增反转）：

> The deployed RBM implementations exceed Post-L2 on all nine text-dense comparisons. The controlled Qwen3-VL results reveal a more specific pattern: final-input selection favors TextVQA and OCRBench, whereas main-output selection favors DocVQA and GQA. Thus, representation choice matters, and its preferred setting depends on the workload. Our contributions are a native-unit norm selector, a full-split comparison across three model families, and controls that quantify the effects of scoring representation and auxiliary-stream aggregation.

注意：最后一句 `quantify the effects` 必须限定为上述对应变体对比，不要说 clean causal effects of every architecture。更稳妥可替换为 `controls that characterize scoring representation and auxiliary-stream aggregation`。

引言贡献建议使用 3 项；不要把 `full-split`、`audited`、`confidence intervals` 单独包装成创新：

> (i) a training-free native-unit norm selector integrated with three merger/projector interfaces; (ii) full-split accuracy comparisons at matched final token budgets; and (iii) Qwen3-VL controls and spatial diagnostics that distinguish task-dependent input/output scoring behavior from the deployed implementation gains.

### 4.3 结论（替换 L198；新增控制结果进入后才采用）

> Native-unit selection offers a simple way to reduce visual inputs while reusing a VLM's existing merger and projector interfaces. At 25\% retention, the tested RBM implementations improve text-dense accuracy over their Post-L2 baselines. Qwen3-VL controls show that the preferred scoring representation changes with both the task and auxiliary-stream aggregation: final-input scoring favors TextVQA and OCRBench, while main-output scoring favors DocVQA and GQA. Visual token compression should therefore evaluate the scoring representation together with the retained budget, native unit structure, and target workload.

不要继续用现稿结论 `the useful signal persists at a later pre-merger tap` 作为整体主句：它对现稿 Post-concat 比较成立，但新增主分支控制表明这不是所有文本任务上的统一优势。

## 5. 逐段删改清单：原句、处理理由与替换文本

### A. 可以删除或缩短的重复防御（不改变结果含义）

| 原稿位置 | 原句/片段 | 建议处理与替换 |
|---|---|---|
| L34 | `deliberately simple` | 改为 `training-free` 或直接删 `deliberately`。简单是性质，不必提前为简陋辩护。 |
| L36 | `We separate operational accuracy from attribution.` | 删；用上节的具体四变体设计接替。 |
| L41 | `The illustrated pre-merger location is conceptual ...` | 重绘正确位置后改成正面图注；不能在错误位置示意上无穷补限定。 |
| L41 | `The diagram's ... denotes retention, written ... in the text.` | 统一图文符号，删除此句；这是排版工作，不是科学内容。 |
| L47 | `The cases illustrate different spatial allocations and do not replace the full-split comparison.` | 改为 `Selected cases illustrate the retained spatial units.`；`selected` 足以说明案例性质。若答复与mask来自不同重捕获记录，则在方法/补充材料交代来源。 |
| L50 | `Token retention denotes a task budget, not bit rate or total cost.` | 从引言删除；方法中定义 retained fraction，成本节一次交代即可。CVM 版本不必沿用面向 DCC 的 bit-rate 辩护。 |
| L53 | `equal final token count need not imply equal computation.` | 从 related work 删，放入成本定义。 |
| L59 | `rather than a claim that downstream selectors generally assume rank preservation or that RBM ranks the full design space above.` | 全删。原稿并未提出这种问题，这句把不存在的 straw man 引到读者面前。 |
| L79 | `they are not identical scoring functions applied to identical vectors.` | 删；两条公式和明确流定义已说明。 |
| L81 | `but low overlap alone predicts neither utility nor accuracy.` | 改为 `We use Jaccard overlap to measure selection disagreement.`，准确且正面。 |
| L88 | `it is neither a pure stage intervention nor an efficiency test.` | 定义四变体之后可删；明确它比较的输入/输出表征与执行策略。新增 Post-main 后尤应避免沿用旧三变体括号。 |
| L91 | `since small output checks do not establish full-split equivalence.` | 改成 `We report vLLM and HuggingFace results in separate comparisons.`；不同后端比较原则一次说明即可。 |
| L97 | `Full-split scores and skipped-example handling are distinguished below.` | 删，信息量为零。 |
| L105 | `this table does not isolate scoring stage.` | 如果表头已标 `deployed implementations`，方法差异已列，则可以删此重复句。 |
| L136 | `prevents attributing that headline contrast to the merger boundary alone; residual ...` | 用新的四变体具体结果替换，不继续以长防守句描述本已可实测的问题。 |
| L161 | `These HF scores are a separate experiment and must not be substituted ...` | 改为 `The HF experiment uses its own shared resolution and backend settings.`，并在表注明确即可。 |
| L185 | `rather than aggregate accuracy or a merger-only causal effect` | 图改干净、旧汇总分删除后，改为 `Selected spatial allocation examples at 25\% retention.`；若仍保留旧n200指标则不能删分母说明。 |
| L188 | `not that norm or edge energy identifies causally relevant text` | 改为正面发现：`For a fixed retained set, the verified Qwen3-VL path returns identical representations and predictions.`；text-specific 因果限制在诊断定义中说一次即可。 |
| L191 | `not an encoded bit rate` | CVM 稿删除；定义 budget 是最终视觉序列长度即可。 |
| L193 | `the proxy is not a serving-latency distribution` | 直接删整个 n=1 proxy 的 .23/.36 句，比保留弱证据再加免责声明更好。 |
| L195 | `Our strongest controlled conclusion ...` | 删除“最强证据”自我评级，直接报告新增控制结果。 |

### B. 必须保留的信息，用一次清晰定义替代反复降调

**L79/L86/L88，流聚合与路径差异。** 原句：`Qwen3-VL ... layer-8 ... concatenated return`；`Qwen2.5-VL ... different ordering conventions`；`InternVL3 ... per tile ... pools image tiles`。不能一删了之。建议改成三行模型适配表，列“input score/tap / output score / allocation & ordering / position convention”。英文说明：

> The deployed adapters differ in score depth, stream aggregation, allocation, and token ordering, as summarized in Table X. The Qwen3-VL controls hold the final-input depth and retained counts fixed while comparing main-only and main-plus-auxiliary output scoring.

第二句不要被理解为“所有控制都只剩聚合一因素”；表中要具体指明每两个 arms 的差异。

**L84，位置语义。** 原句：`Backends follow their compressed-position convention, which need not gather original spatial coordinates.`。其必要性来自审计记录：Qwen2 vLLM 给保留 token first-k grid positions，不是 gather 原坐标。建议正面写为：

> Each backend uses its adapter's compressed-position convention; the Qwen2.5-VL vLLM path assigns the first k grid positions to the retained sequence.

如果新稿要说“保留原生几何位置”，必须先修实现并重评，现有数据不能证明。

**L95/L161，失败样本与分母。** 不能删 `79/88` shared failures、None anchor 的 `2/18` 失败以及 full-split denominator。建议将所有失败处理集中到 protocol 表，正文只说一次：

> Scores use the complete attempted split as the denominator, with failed samples assigned zero. Table X reports attempted, completed, and failed counts for every comparison.

若不同实验实际采用不同分母，必须逐项列出，不可用以上统一句替代真实差异。

**L133/L149，GQA 负结果。** 保留具体值和 CI。它是结果，不是防御性写作。新增 Post-main 同时表明 DocVQA 也反转，需要把“双任务边界”纳入论证。

**L180，Sobel 不等于文字。** 原句：`Because Sobel energy also responds to borders, texture, and contours ...`。可以短写为：

> Sobel energy measures local edges; we use it to characterize spatial selection rather than text localization.

保留这一科学定义即可，无需在后续每个图注再否认一次因果。

**L193，吞吐计时边界。** 保留 A40、n=200、offline batch、max concurrency 8、warmup、max output 32、排除 engine/image loading。建议改成：

> On an A40, a 200-example TextVQA offline workload achieves 6.91 requests/s with RBM, 6.89 with Post-L2, and 4.11 without compression. Runs use native resolution, at most eight concurrent sequences, greedy decoding capped at 32 tokens, and one untimed warmup; timing covers generation after engine construction and initial image loading.

可以计算并说该 workload 的 `1.68x` 加速（6.91/4.11），不能延伸到普遍 serving TTFT/p95 或 selector 本身更快。若 retention curves 只有旧未核验 n200 数字，不纳入准确率曲线。

### C. 必须修正的表述/自洽性

**L63–71 与 L86：统一公式没有覆盖 InternVL3。** 原文将所有模型都写成 `a_u=(1/4) sum_j ||h_uj||_2`；L86 仅说 `after pixel shuffle and before its projector`，并没有告诉读者其评分聚合不同。EVIDENCE_AUDIT §1 已明确，Intern 实际是 4c 拼接向量 L2，即 `sqrt(sum_j ||h_uj||_2^2)`，不是 norm 均值。这是已有实现事实在正文数学描述中的缺口，**不是新发现 RBM 完全失效或主张被推翻**。

建议将式(2)前改为 `For the Qwen adapters, ...`；并新增：

> InternVL3 scores the concatenated vector after pixel shuffle, giving a_u = (sum_{j=1}^{4} ||h_{u,j}||_2^2)^{1/2}. Both implementations rank complete native units, but their within-unit norm aggregation differs.

统一给 Qwen 公式后暗示三模型使用同一 scorer 会影响复现与解释，必须修，不是 optional polishing。

**L81：秩关系说明可以更有效。** `A strictly increasing b_u=g(a_u) is sufficient at all budgets` 是充分条件，不是针对真实学习 merger 的有内容机制证明。建议压成：

> Input and output norms can induce different top-k sets. We quantify this disagreement and test its task-level consequences in Sections X–Y.

如果保留理想化推导，应明确是 fixed set / per-unit independent merger 的事实，不把它包装成理论新颖性。

**L84：`heap top-k costs O(N log k)`。** 未在本次独立核查实际 top-k 内核；torch.topk 不等同于承诺 heap 实现。安全写法：`Norm scoring costs O(Nd), followed by top-k selection.`。时间复杂度附加精确数据结构无助于本稿论证。

**L101：九个 text-dense cells 的计数与指标。** 原句容易读成九个 VQA/ANLS cell。建议：

> Across the six TextVQA/DocVQA cells, deployed RBM gains 11.0--38.0 percentage points over Post-L2; the three OCRBench gains are 298--432 points out of 1,000.

其后保留绝对退化例子，但无需 `yet remains lossy` 的情绪词：

> On Qwen3-VL DocVQA, RBM scores .481 compared with .956 without compression.

**L136/L140–149：增加 Post-main 列。** 不应仅在 limitation 中悄悄提 DocVQA 反转；必须在主控制表中让读者看到。推荐同表列 `Pre-final / Post-main / Post-concat`，CI 可另列两组差值或用森林图。继续把只有 Pre-final vs concat 的表称“strongest controlled conclusion”会产生选择性呈现。

**L195：`FastV shows that RBM is not uniformly strongest`。** 原稿仅展示 OCRBench 两个 RBM 胜的 FastV cell，这句话和可见证据之间有缺口。早期审计记录有 TextVQA/GQA FastV，但跨后端不能当严格 head-to-head；新增同 HF TextVQA 旧分又有 metric bug。现已完成官方 CPU 重评，推荐用具体比较填上这条证据链：

> The same-harness comparisons favor RBM on OCRBench and FastV on TextVQA. On the complete 5,000-example HF TextVQA split, RBM scores .75504 versus .76826 for FastV, a difference of −1.322 percentage points (95\% paired-bootstrap CI [−2.302, −.320]).

应在结果表纳入该负结果，不要只放审计附件；它使 OCR 与一般场景文字任务之间的边界可直接评估。

## 6. 图的问题比图注更影响信任

### Figure 1（L40–41）

直接查看 `figs/fig1_user_20260915_v2.png`，图内有三处需要处理：

1. 底部：**`Only the selection domain changes`**。这与现稿已承认的 score tap、aux aggregation、顺序/分配差异不一致；建议改为 `Same final token budget; model-specific scoring representations`，或者按新控制具体画主分支对比。
2. 右上：**`text evidence demoted`**。没有文本区域/关键证据干预，不足以作为通用机制标签。改为 `different units selected`。
3. 中部：**`native model unchanged`**。权重/算子不变，不意味着 position/order/runtime path 全不变。改为 `native merger weights reused`。

此外图画的是 encoder 结束后分支，实际 headline Qwen3 的打分 tap 在 layer-8 deepstack input；新图至少用 dashed tap 明确深度和四 stream，避免把 early 与 final 混成一条线。

用户历史上选择过此图，FIGURE_AUDIT 要求保留是当时的编辑决策；本次用户请求整理论文时应向主窗口报告图内强命题，不能悄悄删 caption 限定后原样复用。重绘是最清楚的解决办法。

### Figure 2（L46–47）

是有选择的成功案例（manifest 专门挑 RBM 正确、Post/FastV 错误），不是 unbiased qualitative sample；`selected` 应保留。mask 是重捕获、答案记录来自历史运行，不能写成完全同步同一 inference 的 causal trace。本次未发现正确/错误勾选与审计记录矛盾。若保留，图注写选例与保留单元示意即可。

### Figure 4（L184–185）

直接查看 companion `fig4_word.png`，确认：

- 顶部写 `TextVQA-200 ... pre .695 / post .255`；底部写 `DocVQA-200 ... pre .725 / post .39`。这是历史 n=200 诊断，**与全量主表不同本身不是数字错误**；当前审计没有足够指标来源支持把它们当现稿官方分数。
- 灰框两处写 **`(+ => merger demotes high-edge/text)`**。Sobel 只是 edge，且 pre 是 block-8、post 是 main+deepstack concat；不能称“merger 导致 text demotion”。
- TextVQA 案例实际 `rho(rank_shift, edge)=-0.0593`，DocVQA 案例为 `+0.6919`；二者并不构成每个成功案例都支持统一边缘 demotion 的示例。

最安全的编辑是按现有留存 mask 重绘干净图：保留图像/网格/各方法保留集合，删去不可直接对接现稿的历史汇总分与因果解释。不应用 PS 隐去不利结果；图中若留 rank-edge 数字，正负号都保留并清楚写 `edge association`。如果无重绘资源，暂时移除该图也优于旧图配很长免责声明。

## 7. 实质接收风险及工作优先级

### P0：提交前必须修复，措辞无法解决

1. **纳入 Post-main 新控制与 DocVQA 反转。** 它直接影响核心论证，不应只改语气。
2. **离线修正全部 TextVQA 官方评分。** 原 scorer 的 `min(total_matches/3,1)` 与官方 leave-one-out 不同，ExpB 又有 `correct` containment 冒称 VQA 的额外问题；旧分数与对应 CI 都须同步更新，不能公布已知指标混淆。
3. **公式与真实实现一致。** Qwen norm 均值与 Intern concat-L2 的差异须写清楚。
4. **图文去矛盾。** Fig1 的 `Only ... changes`、`text evidence demoted` 与 Fig4 强因果词必须重绘/改字。
5. **区分 operational 与 isolated representation 对比。** 三模型每个适配器差异已知，应以小表明示，而不是用“same model, same K”掩盖。

### P1：对接收率影响大；需要实验或进一步分析

1. **竞争基线太弱且过窄。** 主表比较的是 Post-L2；实际下降很大，胜过它未必证明有用的压缩器。至少补随机、空间均匀/分层采样、强度匹配的 main-only norm，并在同后端接入 1–2 个相关现代方法。`notes/token_compression_landscape_20260920.md` 已列候选，具体选型由代码兼容性与会议范围决定。
2. **只有 25% operating point。** 必须有官方计分、相同设置的 50/75% retention 等少量点，才能判断何时是质量/成本可用的方案。不能使用 FIGURE_AUDIT 已否决的旧硬编码 rate-distortion 图。
3. **绝对质量损失重。** Qwen3 DocVQA .956→.481、Pre-final .2836，本身不支持“高保真/实用默认”。按 iso-accuracy 比吞吐，比“75% token 少了所以快”更有说服力。
4. **norm 信号的含义未证明。** 需比较 norm 分布、rank ties、标准化前后，判断 Post-concat 是否因 stream scaling/归一化而成为脆弱 baseline。这里是待检验解释，不是已证明 LN 导致 baseline collapse。
5. **单卡 throughput 只有一组 batch。** 若以部署效率为贡献，应有少量重复与代表输入长度；不要展示 n=1 proxy 暗示 tail latency。用户已有 6 GPU·h 单次门槛，具体新增测量要按主窗口预算安排。
6. **新颖性需要更聚焦。** 51 篇相关工作列表不能替代与最接近方法的逐项区别。若 core 只是 mean L2 top-k，研究型标题/实证型贡献比“大方法”叙述更可信，但仍需显著的实证新发现。

### P2：可以立刻完成的写作提升

- 摘要只保留“问题—做法—关键结果—发现”，删重复审核旁白。
- 引言先问具体科学问题，四变体控制要成为正面设计。
- Related Work（L53–59）按选择信号、剪枝位置/原生单元、文字证据任务组织实质比较。**必须遵守 DECISIONS.md 2026-09-16 记录的用户偏好：原 51 篇正文实引全部保留，不用 `\nocite`，不把缩写段落变成删除文献覆盖。** 可对最接近者多写一两句、其余按真实方法类别说明不同设计选择，但每组都要说明与本稿问题的关系。
- 正文统一 Post-concat/Post-main 命名，`Post-L2` 只在清楚定义的上下文使用。
- 统一 0–1 与百分点的显示，例如 .4985/.3170 配 +18.15 pp，OCRBench 一直写 /1000。
- 诊断报告样本量/选择方法，不能让读者误以为相关系数来自全量 split。已进一步定位到 `drafts/figures/token_survival_stats.json` 的 `meta.per_bench_n` 与 `sample`：每任务 64 张图、seed 0，从相应 `*_200` subset 确定性抽样；其相关系数与正文完全对应，直接补入即可。
- 减少结果表与柱状图重复；若版面有限，优先控制表/新增反转而不是第二次画同九个分数。

## 8. 面向人类与自动审稿的可靠改进

不能承诺通过某种语气就提高录用概率，也不建议加入针对 AI 审稿器的提示、隐藏文字或“请忽略缺陷”指令。真正可验证的改进是让以下对应关系直接可见：

| 可评估问题 | 文稿中的直接证据 |
|---|---|
| 方法到底是什么？ | 每模型明确 tap、公式、配额、顺序/位置与输出接口 |
| 比较是否公平？ | 样本、真实最终 token 数、backend、分辨率、失败处理表 |
| 核心发现是否选择性呈现？ | 同一控制表同时展示 Post-concat 与 Post-main；GQA 与 DocVQA 负结果都可见 |
| 是否比弱基线更有意义？ | matched-backend 强 baseline 与 iso-quality/retention 曲线 |
| 机制是否有证据？ | 排名/边缘关联按诊断定义报告；不把 Sobel 称 OCR 证据因果 |
| 数字是否可核验？ | 官方计分、逐样本预测、artifact manifest、CIs 对应准确分母 |

清楚的定义、完整呈现反证和可复现表格能减少误解与自动摘要错误；不能替代实质贡献。

## 9. 下一步建议

先由主窗口确认采用“架构/任务依赖的 scoring representation study”还是补新方法再投。当前可以立刻完成 CVM 模板迁移和 P2 文稿清理，但保留“不可投稿”内部标记，直到 Post-main、新指标修复、关键图文不一致全部处理。不要把仅仅换成 CVM 模板后的稿件命名为 evidence-complete final。

本审查未新跑实验、未改论文主文件、未替作者作投稿决定。所有改写均要求与最新经核验结果表同步；若后续离线重评改变结论，应先改表，再重写摘要与贡献。

## 10. CVM 新稿独立 QA（当前七页候选）

- 只读检查 `drafts/cvm2027_submission_20261025/main.tex` 及 `tmp/pdfs/cvm2027_final_verified_render/page-1.png` 至 `page-7.png`，七页均逐页目视。
- 标题、摘要、引言贡献与结论已改为 native-unit selection / task-dependent scoring，未发现新增超出表格的优势主张。
- 三模型 TextVQA 新官方分、Post-main 的 DocVQA 反转、同 HF FastV TextVQA 小胜均进入正文，正负结果可见。
- Qwen 的 mean patch norm 与 InternVL3 的 concatenated-vector norm 分式正确区分；image/tile 配额与适配表一致。
- 完整 ViT 计算、Qwen2 顺序/位置与 Intern tile 分配集中定义；没有恢复“只变 stage”的错误概括。
- 诊断 n64/benchmark、seed0 来源明确；swap 分别报告 31/30 probe 与 198/200、200/200 答案一致数，泛化 byte-exact 说法已删。
- Fig1 的强因果/模型完全不变标语已中性化；原 Fig4（新 Fig3）保留数据面板，旧分数及因果灰框已去除。
- 未见新增裁切、重叠、公式缺字或控制表不可读；旧六页 render 的 Table3 图注左裁问题在新目录中不存在。
- 第七页参考文献偏疏、Fig3 浮在 references 中可后续美化；不构成本轮的提交阻断。
- 收尾时主窗口仍在独立核查 DocVQA 历史 normalizer 对分数的影响；若改数，应同步刷新全文相关数值并复查最终渲染，本次 QA 不替代该核验。
- 后续 GQA 官方 CPU 重评已完成 11 格/5 组 CI，方向保持；最新 GQA 的 +6.19 pp 与相关表格须随 release 统一更新。
