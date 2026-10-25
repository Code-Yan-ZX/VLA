# CVM 2027 投稿前科学证据审查（2026-10-25）

审查对象：`F:\dcc2027_submission_20260916\main.tex`；对照本地 drafts、Git `origin/main`（84f0168）、既有审计、9 月 19 日新增实验和保存的逐样本预测。只做读取与 CPU 重评分，没有启动 GPU、修改原论文或覆盖历史评分。以下“严重”指投稿前必须处理的证据问题，不等于对接收结果作概率判断。

## 1. 版本结论

用户给出的目录确是最新**论文文本**，但没有包含最新实验证据。

| 文件 | 用户目录与本地 drafts SHA256 | 与 origin/main |
|---|---|---|
| main.tex | `84c2283d2706f1268805df73ab0dab60470b7b7f78aa07b985e956701b0f1d3c`，逐字节相同 | 仅 CRLF/LF 差异，LF 归一化后完全相同 |
| references.bib | `6f8d2db09b79f94ce02ef70e76f725a7fbcdc1ab7eb42f7621847112b93dfe08`，逐字节相同 | 仅 CRLF/LF 差异，LF 归一化后完全相同 |
| EVIDENCE_AUDIT.md | `3bc23e2e3e01b6dfd0212d27f4c478080e267cc396a2631ddbeb2292841e684a` | 三方逐字节相同 |
| FIGURE_AUDIT.md | `9e61e6e147d0f0c700f82aef50deff40ff945fc9fe6ba8c22444cf76438cab9b` | 三方逐字节相同 |

正文引用的四个图文件也在用户目录、本地 drafts 与 origin/main 三方逐字节相同。图 1/2/3/4 的 SHA256 分别为 `01545bcc…bbe4c71`、`4ce58deb…82322d`、`c83d41dd…49fca`、`8d61ef06…24e9c`。

9 月 19 日新增的 `reports/dcc_presubmit_expA.md`（b73ddc9）和 `reports/dcc_presubmit_expB.md`（6cdb2dd）未被整合进该稿。当前本地旧分支落后 origin/main 六个提交，不能只查看本地 reports 就断言这是全部证据。

## 2. 已证实的严重问题

### A. DocVQA 的 final-input 归因在更恰当对照下反转

9/16 表格比较 Pre-final 与 main+deepstack 拼接范数 Post-full，得到 DocVQA +4.59 pp。9/19 新 Post-main 仅用 main-merger 输出列打分，但保留完整 main+deepstack 行，保持相同最终预算。新结果：

| Qwen3-VL 全量控制 | Pre-final | Post-main | Pre-final − Post-main，95% 配对 CI |
|---|---:|---:|---:|
| TextVQA（本次纠正官方评分） | .49170 | .31230 | +17.94 pp [16.45, 19.45] |
| DocVQA（本次纠正的 pinned lmms-eval ANLS） | .28369566 | .37368419 | −9.00 pp [−10.17, −7.82] |
| OCRBench /1000 | 419 | 356 | +63 [27, 100] |
| GQA | .4207 | .4818 | −6.11 pp [−6.89, −5.33] |

TextVQA 本次使用官方正常化与共识评分，DocVQA 用固定版本 lmms-eval 的上游 ANLS 函数统一重评分，OCRBench 已直接执行上游函数核验全部15格逐样本结果完全不变并给出本次固定种子的 CI；GQA 的最终独立核验由另一审查 agent 负责。不得把旧 TV/DocVQA 的 CI 或分数与新分数混用。

结论：仍然可说 operational early-tap RBM 在该配置的三个文本密集任务上优于 Post-full；不能再把“三个文本密集任务都支持 final-input pre 优势”写成方法机制。DocVQA 反转直接显示，拼接输出的打分空间对结论起决定作用。即便 main-only 也不应叫完全纯因果干预：两边评价的仍是不同向量与不同范数组合。

### B. ExpB 把二元包含匹配误标为官方 TextVQA VQA-acc

`scripts/analyze_dcc_expB.py` 直接把 `per_sample[].correct` 当成官方分数。`baselines_hf.py:251` 的在线 `score_textvqa` 实际是二元子串/词形匹配，`correct` 写成 `int(c)`；其第 164 行已明确规定权威分数必须离线重新评价 answer/GT。故 ExpB 报告的 .8374/.8494、−1.20 pp 与对应 CI 不是官方 VQA 共识分数，不可直接搬进投稿稿。

已从 A40 只读取回两个预测 JSON，按真正官方规则重评分。最终可入稿的同 HF、同 25% 实际视觉 token 预算结果为：

| Qwen3-VL TextVQA 全 5000，双方同 5 个失败按 0 | RBM | FastV-k3 | RBM − FastV，95% 配对 CI |
|---|---:|---:|---:|
| 全 attempted 分母 | .75504 | .76826 | −1.322 pp [−2.30205, −.32000] |
| common answered 敏感性，n=4995 | .75579580 | .76902903 | −1.32332 pp [−2.33033, −.32628] |

20,000 次 paired bootstrap，全 attempted 种子 20261101；官方逐样本分数后作配对，5 个共同失败两臂差为零。两种分母均指向 FastV 略优，不能只展示 OCRBench 的 RBM 胜利而省略该同后端结果。

### C. 全部旧 TextVQA “official scorer” 还有三层实现错误

1. `official_scorers.py:score_textvqa_vqaacc` 用 `min(total_matches/3,1)`，而 lmms-eval 是逐一留出一个标注者再平均。十答案时 1/2/3/4 次匹配正确分值为 .3/.6/.9/1，旧函数为 1/3、2/3、1、1。
2. 旧正常化并不等于完整 EvalAIAnswerProcessor，含 contraction/所有格等差异。
3. 13/5000 题的原答案文本本身含分号。项目将十答案用分号拼接后再 split 会产生 11 个或更多片段，原标注边界丢失。已按官方 question_id 恢复原答案数组，且核验每个保存 GT 都等于官方数组逐项 strip 后拼接值。

本次没有改旧 scorer，以免静默改变历史结果。新稿的 TextVQA 权威来源改为新增独立 CPU 脚本和本次轻量结果表。覆盖三模型 none/pre/post 九格、Qwen3 pre-final/post-main 两格、新 HF 两格、历史 FastV 复现一格，共 14 格，缺失 0 格。各操作性差值方向保留，但所有 TextVQA 绝对分数、差值和相关图必须刷新。

## 3. 可直接替换的 TextVQA 数据

下表全部用 5000 attempted 分母；失败记零。三模型操作性表仍是相同后端内的比较。

| 模型 | None | RBM | Post-full/Post-L2 | RBM − Post，pp |
|---|---:|---:|---:|---:|
| Qwen3-VL | .83430 | .59784 | .21810 | +37.974 |
| Qwen2.5-VL | .85286 | .69410 | .43604 | +25.806 |
| InternVL3 | .82512 | .78042 | .40926 | +37.116 |

Qwen3 Pre-final − Post-full 为 +27.36 pp，95% CI [25.88195, 28.80605]。Qwen3 None 及 Qwen2.5 None 各有 2 个失败；把“失败按零”写入协议后必须使用 attempted 分母，不能继续使用历史 answered-only anchor 均值。

完整分母、14 个预测源路径与每个源的未压缩 SHA256、每题改变计数、全部配对统计见 `notes/cvm2027_textvqa_rescore_20261025.json`；可读表见同名 `.md`。这些是轻量审计结果，不含逐样本回答。

## 4. 评分复现与独立验证

新增脚本：`scripts/analyze_cvm_expB_official.py`。其默认读取 `tmp/cvm_cpu_rescore_20261025/` 的保存预测、历史本地 artifact 和 gzip 预测；只做 CPU 字符串评价与配对统计。支持 `--cache`、`--annotations`、`--out`、`--resamples`。

- 固定 lmms-eval 提交：`8f40097532a7bf4db6123713f1a905e02c8f5971`。
- 官方正常化文件 SHA256：`4e4154c6c7e4657e6c44aa08e184ce5ae0e872665744c6cb684cf645e4647ce0`。
- 官方 textvqa/utils.py SHA256：`4d1e10d2a72136e2fb11486827f29d955f4782b51e0cafe7228b1d53da14d405`。
- 官方 TextVQA v0.5.1 val annotation SHA256：`4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d`。
- 直接用 AST 隔离执行固定版本上游 `textvqa_process_results`，与本次 helper 在 HF RBM 的全部 4995 个非失败样本逐项核验，0 mismatch。
- 0/1/2/3/4 次标注匹配、冠词、数字词、contraction、所有格和拒绝二元子串匹配的针对性检查全部通过。ID 无重复、配对 GT 一致；HF 4995 个已完成样本 `n_image_kept` 全部相同。
- 预测、下载 annotation 和参考源码均在 tmp，不应进入匿名投稿 zip 或版本控制。重评没有修改服务器数据、启动模型或耗用 GPU。

官方一手依据：[固定版本 TextVQA task](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/textvqa/utils.py)、[固定版本 EvalAIAnswerProcessor](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/_task_utils/vqa_eval_metric.py)、[官方数据下载](https://textvqa.org/download/)。

## 5. 真实但尚未解决的科学证据缺口

### 新颖性和强基线

“训练免、图像侧、原生组结构”不足以单独证明新颖性。现稿方法本体主要是 L2 TopK 加三个架构适配，最有价值的可验证贡献是打分表征依赖、不同任务的符号差异、以及适配细节审计。强于 Post-L2 不等于优于当前压缩方案，FastV 只有一个 2024 年注意力基线；没有同后端同实际预算下的随机/均匀空间、现代视觉多样性、组内覆盖等基线，难以量化 L2 选择本身的贡献。

一手相关证据：[VisPruner ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.pdf) 使用视觉侧重要性与去重；[CVPR 2026 Information Horizon](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html) 直接研究层位、OCR/VQA差异及随机 pruning。推断：把“预先排序一定更好”当普遍新发现已不成立，需要围绕本文实测的原生接口和表征对照明确区分。此次未做全领域优先权证明，也没有伪称已跑现代强基线。

### 公平预算不等于单变量机制

- Qwen3 operational RBM 在 deepstack_0/layer-8 输入打分，Post-full 在 main+三 deepstack 拼接输出打分，同时改变深度、空间和执行路径。
- Qwen2.5 pre 路径跳过 reverse_indices 恢复，保留 window order，post 是恢复后的空间顺序；还存在位置语义差异。因此该族不是干净的“只换排序位置”验证。
- InternVL3 pre 每 tile TopK，post 汇总所有 image tiles 做全图 TopK。最终 token 总数相等，但空间预算分配不同，不能声称跨模型统一因果机制。
- InternVL3 实际打分是 pixel shuffle 后 4c 拼接向量范数，即 `sqrt(sum_j ||h_j||^2)`；它不是论文统一公式中的 `mean_j ||h_j||`。应在方法公式或实现表里明确区别，否则算法定义与代码不一致。
- HF native mRoPE 与 vLLM compressed-position convention 不相同。HF TextVQA RBM 官方 .75504，vLLM operational .59784，相差约 15.72 pp；这首先是适配/后端/位置路径差异信号，不能未经对照就宣布全部由某一种 mRoPE bug 导致。

### 统计和效率

全量 paired bootstrap 对“这些题目上的均值差”有帮助，不能替代独立任务、不同预算、分辨率和模型规模上的复现。主文仅一个 25% operating point，不能支持普遍质量—成本 Pareto 优势。

J6 的 6.91 vs 4.11 req/s 是一个 A40、TextVQA n200 offline batch、最多 8 concurrent sequences 的 timed generation；Post-L2 6.89 几乎相同。它证明该工作负载压缩加速，不能证明 selector 独有提速。n=1×5 的 .23/.36 秒不是代表性服务延迟分布；没有 p95 TTFT、连续到达吞吐、实测 peak memory 或吞吐重复试验汇总。无需虚构这些指标。现有“timing excludes engine/image loading”应保留一次集中说明。

### 图、机制和数据泄漏

- 图 2 是事先按 RBM 对、其余错挑出的两个 OCRBench案例；应称 selected examples。mask 是后采集的，有些再生成回答与展示历史回答不字节相同，不能当因果 trace。
- 图 4 内嵌历史 n200 数字不等于本次官方重评分或 full split。仅靠 caption 弱化仍会造成视觉上的数字冲突；优先重绘或去掉旧分数、text demotion 等文字。
- Sobel energy 对边、纹理、轮廓均响应。它与 rank shift 相关不等于证明文字因 merger 被压低，也不是文本区域保真测量。诊断样本为各基准从 n200 subset 抽取 n64、seed0，不能省略样本量。
- swap 资料不支持泛称“所有保留表征字节相同且所有预测相同”。`mechanism/swap_control.json` 记录 Qwen3 Jaccard probe TextVQA n31、DocVQA n30 均1；分开的 n200 预测一致率为 198/200 和200/200。`drafts/mechanism_verification_report.md` 承认 subset/full GEMM 数值差异，未找到更强后续 tensor-byte 比较。
- 没有发现本文主实验使用测试答案训练 scorer 的直接证据；query-blind L2 本身无训练参数。RankBridge 的 dev/test ID 不重叠未证明，但当前稿没有 RankBridge 主方法，不能把该历史探索证据混进正式独立测试。选图偏置与数据泄漏应严格区分。

## 6. 截止前可做的实质修正

1. 用本次官方 TextVQA 全覆盖表更新全部主文、摘要差值和控制图；明确失败与分母。
2. 将 Post-main 四任务结果纳入正文，正面陈述 DocVQA 反转；把贡献定位为原生单元选择与表征依赖实证，而非统一 merger 因果规律。
3. 同时呈现 HF TextVQA 的 FastV 小胜和 OCRBench 的 RBM 胜利，保持同后端、同分辨率、同实际预算；不用跨后端大差值构造优越性。
4. 集中一张模型实现表说明 tap、范数聚合、tile预算、顺序/位置 convention，修正 InternVL 公式；删除反复自我辩护，保留会改变结论的事实条件。
5. 将历史图中旧分数/机制标语撤掉；swap 用真实分母和一致率，Sobel只说边缘相关。
6. 复现材料应提供命令、环境、模型revision、提示词、processor/pixel cap、retention与r的换算、官方评分来源和源码hash、完整分母及失败处理。公开链接可匿名性应由投稿打包统一处理。

这些修正能减少审稿人需要自行猜测的内容，但无法用编辑替代缺失的现代强基线、无混杂适配对照和更完整效率测量。无论审稿是否使用 AI，可核验数据、明确贡献、对称报告正负结果才是可操作的改进，不应向稿件注入评审提示或推测接收率。

## 7. 最终复核增补：DocVQA 精确 ANLS 统一与其余任务协议边界

静态检查发现 local DocVQA ANLS 只 lower/strip，和固定版本 lmms-eval 的内部空白折叠不同。不能仅靠换一句措辞假定实际影响为零，因此追加了有界 CPU 检查。ExpA Post-main 有 8/5349、Pre-final 有5/5349 样本发生非零改变；均值分别由 .373538678→.373684188、.283574233→.283695660。方向不变，但保留四位小数的控制分数及配对 CI 必须更新。

已统一重评三模型 DocVQA none/pre/post 九格和 Qwen3 Pre-final/Post-main 两格，共11格；全部5349 attempted IDs，0 skipped。直接 AST 隔离执行固定版本 `lmms_eval/api/metrics.py` 的 `anls` 和 `levenshtein_distance`，没有再次推理。参考源码 SHA256 为 `eaf48195af80a46d22e545a03a1bd7e4cad4fcbadeb3d2d817ac7d148390d8f4`。算法按上游原样：编辑距离前 lower/strip/折叠空白，长度分母使用原始 reference/prediction 字符串长度，最高相似度低于 .5 则零。不能把它误写成“归一化后长度分母”。原始独立 DocVQA 答案数组未重新恢复；使用保存的分号分隔 GT variants，这一数据来源边界已写入脚本与结果元数据。

| DocVQA 模型 | None | RBM | Post-L2 | RBM − Post，pp |
|---|---:|---:|---:|---:|
| Qwen3-VL | .95633616 | .48089675 | .23780170 | +24.30950 |
| Qwen2.5-VL | .94915529 | .63612581 | .52582762 | +11.02982 |
| InternVL3 | .92234990 | .72849848 | .38203448 | +34.64640 |

Qwen3 Pre-final .28369566，Post-main .37368419。Pre-final − Post-concat +4.58939593 pp，95% CI [3.41125931, 5.76785687]；Post-main − Pre-final +8.99885281 pp，95% CI [7.82219184, 10.16572876]。20,000 次 paired bootstrap，完整 attempted 分母。操作性“九个文本密集单元格均为正”的方向保持。

新增可复现脚本：`scripts/analyze_cvm_docvqa_official.py`；轻量表与源哈希：`notes/cvm2027_docvqa_rescore_20261025.{md,json}`。原预测与临时上游源码仍只在 tmp。

最初仅对 GQA/OCRBench 追加静态检查，随后追加了§8的有界 OCRBench 全覆盖核验；GQA交由另一审查agent：

- GQA 的本地指标是 VQA-normalized exact match，额外处理冠词、数字词和缩写；固定版本 [lmms-eval GQA 配置](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/gqa/gqa.yaml) 仅声明忽略大小写与标点。因此可明确报告本地正常化定义，不能笼统说逐位复现上游。
- OCRBench 的旧本地指标是小写/strip/换行处理后的包含匹配，HME 分支区分大小写且去空格；固定版本 [lmms-eval OCRBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/ocrbench/utils.py) 对非 HME 还执行全角 ASCII 折叠，本地旧 scorer 没有该步。静态差异本身不能证明实际分数改变；§8已量化此次全部单元格确为零差异。

因此协议应分别写清四个任务的指标和正常化，指向实际使用、且经过此次核验的固定实现，避免不经逐项验证就统一套用“所有分数均为官方精确实现”。

## 8. OCRBench 全覆盖有界 CPU 核验（静态差异的最终量化）

已直接执行固定 lmms-eval 提交的 `_fold_fullwidth_ascii` 和 `ocrbench_process_results`，覆盖全部15格：三模型none/pre/post九格、Qwen3 Pre-final/Post-main两格、同HF两模型RBM/FastV四格。完整1000题的本地 `eval/full_splits/ocrbench.jsonl` 提供 `dataset`、`question_type` 与GT，HME100k共100题、分类无缺失。每个预测GT均与该map一致；原始未拼接答案数组未独立取回，脚本/结果元数据保留此来源边界。

**15格每个样本的最终分数与历史 scorer 完全一致，changed_samples 全为0。** 因此主矩阵760/547/184、817/480/182、852/753/321，Qwen3控制419/356，以及HF559/413、382/295均保持。静态发现的全角折叠缺失在这些既有预测上没有实际分数影响。

本次以完整1000 attempted分母、失败置零，20,000配对bootstrap（各pair固定种子20261120起）重算的控制区间为：Pre-final−Post-concat +235 [200,271]；Pre-final−Post-main +63 [27,100]。HF Qwen3 +146 [118,174]、Qwen2.5 +87 [54,120]。由于逐样本数组相同，历史CI同样是有效的旧seed bootstrap估计；一分左右的端点差来自随机重采样种子，不是评价规则改变或新模型实验。新稿若统一此次审计应使用本次轻量结果表对应区间。

新增 `scripts/analyze_cvm_ocrbench_official.py` 和 `notes/cvm2027_ocrbench_rescore_20261025.{json,md}`；上游参考文件 SHA256：`0baaa04eed6ba5c610b7d2260a5a44c9fa76cd06331cbd86817064d882ac3557`。全角ASCII匹配、HME去空格且区分大小写、HME不应用全角折叠的针对性检查通过。没有GPU任务或服务器写入。
