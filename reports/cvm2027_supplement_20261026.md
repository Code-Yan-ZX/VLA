# CVM 2027 补实验与写作交付报告

时间：2026-10-25 19:07:06 开始；2026-10-26 接管审计和收束（Asia/Shanghai）。隔离分支 `codex/cvm-24h-supplement` 基于含 `1600bfe` 的 main。原始执行代码至 `90a5b73`，随后修正与论文版本见 Git 历史。

**当前状态：新增统一attention完整主表8/8于17:56:51完成，全部样本零失败，正式评分与协议核验通过；修正版P3成本10组和预定预算曲线齐全。论文正在按新结果定稿，可选旧协议Hi-Lo75全5000于17:57:25开始补跑，19:00前硬停。下面旧表保留为历史操作性结果。** 原工作区修改、历史预测与日志保留。用户10/26已追加授权补论文，覆盖原提示词“本轮不改论文”；不包含外部投稿。

## 0. 统一attention完整主表（8/8完整、零失败）

用户批准为新主表延长原15:07 GPU停线，要求仍在20:00前交付。预冻结配置和顺序在 `plan/matched_main_plan.json`，实际完整矩阵14:51:25开始、17:56:51完成评分/汇总。四臂均使用同一vision SDPA/text eager、模型权重、processor/prompt、4M cap、native位置、bf16和greedy≤32；None保留100%，压缩臂25%。Hi-Lo仍使用作者vendored模型，保留其实现边界。

| 任务与官方单位 | None100% | RBM25% | FastV-k3 25% | Hi-Lo25% | 每方法成功/尝试 |
|---|---:|---:|---:|---:|---|
| TextVQA全5000，accuracy% | 83.45 | 75.58 | 76.90 | 66.75 | 5000/5000 |
| OCRBench全1000，points/1000 | 765 | 610 | 453 | 612 | 1000/1000 |

- **TextVQA：FastV领先RBM，RBM领先Hi-Lo。** RBM−FastV为−1.32pp，20,000次配对bootstrap 95%CI[−2.32,−0.34]；RBM−Hi-Lo为+8.83pp，CI[+7.63,+10.04]。不得据此声称RBM是最强选择器。
- **OCRBench：RBM领先FastV，与Hi-Lo的差值不确定。** RBM−FastV为+157分，CI[+128,+187]；RBM−Hi-Lo为−2分，CI[−31,+27]。CI跨0不证明两方法等价。此次全部1000题都成功，旧表79个OOM及其完成率解释不适用于新主表。
- **25%并未在完整任务上保持95%质量。** 三个压缩臂均低于各自任务同协议None参考的95%；第3节100题/任务成本工作负载的达标点只适用于这些固定子集，不能外推为full-split质量保持。

全部24,000个尝试、完整ID顺序、官方TextVQA十答案/image ID、共享输入hash/grid/N与实际K均通过核验。每格100个P3重合请求逐一盲核对完整输出及N/K，共800个；未用中途分数改变配置。完整12对比较、primary/common分母、固定seed、20,000次CI和全部源SHA在 `reports/cvm2027_matched_main_20261026.{md,json}`；独立重算审计另见 `reports/cvm2027_matched_main_audit_20261026.{md,json}`。旧50/75%曲线保持原mixed-attention协议，不重标为新统一协议。

## 1. 初轮质量结论（历史mixed-attention协议）

同一 Qwen3-VL-8B 权重、processor、prompt、像素上限与 greedy≤32 tokens 下，25%最终视觉保留率的结果如下。Hi-Lo使用官方模型适配和SDPA，其他三组使用统一HF eager harness；因此它是操作性实现比较，不能称完全相同计算实现的隔离实验。

| 任务/协议 | None100% | RBM25% | FastV-k325% | Hi-Lo25% | 完成样本数（相同列顺序） |
|---|---:|---:|---:|---:|---|
| TextVQA全5000，4M cap，官方accuracy% | 83.24 | 75.50 | 76.83 | 66.77 | 4995/4995/4995/5000 |
| OCRBench全1000，4M cap，points/1000 | 707 | 559 | 413 | 613 | 921/921/921/1000 |
| DocVQA固定500子集，统一2M cap，ANLS% | 95.73 | 82.76 | 70.45 | 76.38 | 500/500/500/500 |

主分母始终是全部尝试样本，失败记0。TextVQA从原始官方十标注按question_id评分，禁止使用旧online correct/acc或拼接GT重建。所有新比较使用20,000次配对bootstrap和固定种子；轻量表列每个差值、CI、分母与失败集合，见 `reports/cvm2027_supplement_tables.md`。

- **TextVQA不支持RBM最强。** FastV−RBM = +1.322pp，新配对重采样95%CI[+0.338,+2.326]；RBM−Hi-Lo = +8.73pp，CI[+7.51,+9.94]。RBM仍比None低7.74pp。10/25历史重评使用另一固定bootstrap seed，边界略有Monte Carlo差异，点估计相同。
- **OCRBench的Hi-Lo attempted领先主要来自完成覆盖。** RBM/FastV两份原日志各79个OOM，ID集合与预测skip完全一致；Hi-Lo全答。共同921题上，RBM=559/921=60.69%，Hi-Lo=556/921=60.37%，差−0.33pp且CI跨0；不能把attempted的+54分解释为选择质量更强。共同样本敏感性仍不能消除attention与vendored模型差异。
- **DocVQA500存在RBM优势，但不是95%质量保持。** RBM−Hi-Lo = +6.39pp CI[+3.04,+9.78]；RBM−FastV = +12.32pp CI[+8.15,+16.47]。所有压缩臂都低于该Doc500实验块所报告None参考的95%；Hi-Lo与该参考仍有attention实现差异。此500题子集不能称full split，也不能与历史600k或4M数据混排。500题映射到417张原始文档页；另做20,000次页面聚类bootstrap，RBM−Hi-Lo CI[2.91,9.87]、RBM−FastV CI[8.05,16.61]仍为正，主表按问题重采样的CI不变。来源映射和复现见 `reports/cvm2027_docvqa_cluster_sensitivity_20261026.md`。

## 2. 25/50/75%预算曲线

TextVQA为seed=20261025预固定、按视觉token数代理（旧pre字段为25%保留数，与原始数成比例）分层的1000题；OCRBench使用全1000。25%及None从对应P1预测按固定ID抽取，其他点新跑。每个方法/预算都保留，不按胜负补格。

| 任务/分母 | keep | RBM | FastV | Hi-Lo |
|---|---:|---:|---:|---:|
| TextVQA1000，accuracy% | 25% | 75.29 | 76.95 | 66.20 |
| 同上（None82.93） | 50% | 79.65 | 82.24 | 75.64 |
| 同上 | 75% | 81.73 | 82.57 | 81.24 |
| OCRBench1000，points/1000 | 25% | 559 | 413 | 613 |
| 同上（None707） | 50% | 659 | 569 | 709 |
| 同上 | 75% | 684 | 676 | 740 |
| OCRBench共同921题，归一化×1000 | 25% | 606.95 | 448.43 | 603.69 |
| 同上（None767.64） | 50% | 715.53 | 617.81 | 711.18 |
| 同上 | 75% | 742.67 | 733.98 | 741.59 |

旧曲线的质量阈值取各mixed-attention实验块所报告None参考的95%；Hi-Lo与该参考后端不同，因此只是该操作性比较的描述阈值，不是完全相同attention下的质量保持证明。只判断已测离散点，不插值、不把“不显著”当等价：TextVQA1000上最小达标keep为RBM50/FastV50/Hi-Lo75；OCRBench attempted为75/75/50，共同921题则均为75。**这里是最小达标保留率；最低实际成本必须另用同协议计时选点，不能假定更小keep必定更快。** 旧报告把RBM OCR75说成未达标是算术错误，已更正（684≥671.65）。

表图：`reports/figures_cvm2027_supplement/budget_curves.{pdf,svg}`；实际K图 `score_vs_actual_k.{pdf,svg}`。后者基于共同样本真实计数，四路特征宽度不计作四倍LLM token。

Q10可选全5000扩展已完成RBM/FastV50/75及Hi-Lo50，共5/6格，独立表与CI见 `reports/cvm2027_q10_extension_20261026.{md,json}`。对应分数依次为80.75/82.36、82.26/83.00及76.86%，均按5000尝试题官方评分；Hi-Lo75缺失，不能称完整全量曲线。修正P3与统一attention全量主表已优先完成，固定1000题曲线始终是完整的预定结果。最后Hi-Lo75按17:42预先记录的纯资源门于17:57:25启动，保守预留3445秒、19:00前硬停；最终成功或停止状态另记，不把partial计完整。

## 3. 成本审计：旧结果撤回与新测量边界

接管发现旧P3脚本存在缺失inference/no-grad、累积MergerTap hook、FastV漏传deepstack、Hi-Lo外部config更新未作用于实际模型、不同total时间边界等问题。旧41–43GiB显存结论和全部速度百分比无效；旧raw不删除、不作为论文数据。详见 `reports/cvm2027_timing_audit_20261026.md` 和 `reports/cvm2027_execution_audit_20261026.md`。

修正脚本 `scripts/cvm24h_timing_v2.py` 统一vision SDPA/text eager、bf16、bs1、greedy≤32、4M cap。固定200个请求（TV100/OCR100）每个重复3次，每配置预热；TV采样以历史25%保留计数为原始数量代理且排除旧5个OOM，OCR采样按全部1000图像面积，均在计时前冻结，不是总体无偏随机样本；先取每请求三次中位数，再取200请求的p50/p95。直接测完整请求和TTFT，保留CPU processor、传输、完整视觉、选择/融合、LLM prefill、decode、actual K、逐层token轨迹、allocated/reserved峰值、输出长度与全部原始重复。

单次请求从已经解码的RGB PIL图像开始，不含磁盘、图像解码、模型加载、队列或预热。workload wall包含整个循环、每ID一次本地解码、同步、记账与失败恢复；其吞吐分母不可与单请求边界混淆。选择/融合可能包含在prefill段，禁止重复求和。当前HF RBM确实完整执行ViT与所有merger后才gather，因此不能计入任何未实际省下的视觉加速。修正版手工循环仅对最后prefill token做词表投影，与原生generate的`logits_to_keep=1`一致，并以完整greedy输出核对原helper；这是新的成本测量实现；输出一致的证据限于已经测试的请求，不把历史harness原样时延与优化后结果混称，也不外推全split逐位等价。

新统一attention不同于旧full-quality路径，**质量—成本图只用本次200题产生的答案重新官方评分，按任务各100题与相同请求的成本配对**，不把旧全量分数冒充新协议质量。新文件为 `plan/timing_harness_corrected.json` 和 `plan/timing_hilo_corrected.json`；完成后 `reports/cvm2027_corrected_quality_cost.{md,json}` 给出完整表、同协议质量与95%目标下最低测得p50成本。

修正版于14:48完成：每配置200个共同ID、各3次，共6000次完整请求，零失败，重复答案全部一致，actual K/36层轨迹/原始中位数与峰值检查通过。10组完整workload wall合计4009.6秒，不含模型加载/预热。全部表、分段、p50/p95、原始峰值与配对CI已写入上述质量成本报告。下表仅列达到预定95%点估计目标的最低实测p50，并非质量等价证明。

| 方法 | TextVQA100最低达标点：keep / score% / p50秒 | OCRBench100最低达标点：keep / score归一化×1000 / p50秒 |
|---|---|---|
| 同协议None参考 | 100% / 81.70 / 0.396 | 100% / 780 / 0.713 |
| RBM | 50% / 80.50 / 0.295 | 50% / 760 / 0.642 |
| FastV-k3 | 25% / 79.20 / 0.269 | 75% / 790 / 0.729 |
| Hi-Lo | 75% / 79.30 / 0.494 | 75% / 750 / 0.978 |

目标分别为77.615%与归一化741分。Hi-Lo在这两个固定子集的达标点均慢于None，FastV的OCR达标点也未缩短p50；不能把减少token当作必然加速。RBM OCR75的p50=0.730秒，高于OCR50的0.642秒，最低成本的选择确实来自实测而非keep单调性假设。25%保留时跨全部200题/3次重复的峰值allocated为None22.071、RBM17.268、FastV21.586、Hi-Lo21.610 GiB。输出长度与decode成本也随方法变化，应和TTFT分别解读；不能将生成更短直接称为prefill加速。全部点的score-vs-latency图为 `reports/figures_cvm2027_supplement/score_vs_latency.{pdf,svg}`。

## 4. 正确性、实现与复用

- 模型snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；transformers4.57.6、torch2.10.0+cu128；无环境改装、无重复权重下载。
- Hi-Lo上游commit `29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6`。仅移除引用未定义`top_index_1`的转换guard，并提供4.57版本所需rope配置shim。官方选择/融合代码保留。`scripts/cvm24h_prepare_hilo.py` 可重建精确修复并输出源码SHA。
- 官方仓库两个CPU测试文件共5个测试全部通过（CUDA设备屏蔽）：候选/保留/融合集合划分、确定性、非法预算拒绝、merged图格对齐及视频输入拒绝；未修改既有环境。逐条执行证据与边界另见 `reports/cvm2027_instruction_audit_20261026.md`。
- 冻结Hi-Lo参数：prune_layer=2、fusion=.3、top_k_ratio=.1、attention_threshold=.05、candidate=max(.30,keep)、fusion_layers=[0,1]。50/75%时candidate与keep相同，融合候选集合退化为空；不是根据验证成绩调参。
- 实际K已逐共同样本核验为round(N×keep)，Hi-Lo融合不额外保留token。RBM同kept IDs对齐主流及三条deepstack；不能照旧CLI描述再乘4补偿。
- 旧P0检查涵盖32题no-op答案/K和8题部分张量检查；其logits probe没有完整传入deepstack，Post/FastV/Hi-Lo未形成全面的logits/位置/cache证明，不能统称“全部P0逐位等价”。新P3补真实生成parity、live配置与逐层actualK门检；BF16误差按实值记录。
- 新native/vendored共享attention无剪枝门检32/32通过：显式关闭选择与融合，input/mRoPE/main+三DS完整hash及greedy相同，首步logits误差0。另一个原eager协议真实生成门检中，pre1/post1/FastV-drop0各32/32完整greedy和保留集合/位置/三DS一致，但首步logits最大绝对误差各0.125（BF16），不得写成logits逐位相同。二者是不同检查，原始证据分别在修正版timing和 `plan/p0_noop32_verified.json`。
- native/mimic位置规则在32题中有7题答案不同；这说明位置语义会影响结果，仍不能据此完全解释历史HF/vLLM全量差距。
- RBM/FastV25% TextVQA来自`results/dcc_presubmit/expB/`，OCR来自`results/acmmm_final_controls/p1/`；记录配置与ID核验。预测和所有代码/评分文件的完整SHA见 `reports/cvm2027_artifact_manifest.json`，复用来源本地保留。
- DocVQA统一2M是显存触发协议变更：高cap smoke失败后四方法一起改。四预测同序等于预固定500行JSONL，集合等于预固定IDs，无样本替换或失败分母不对称。
- 评分上游源码、官方标注SHA与10/25审核依据在四个`notes/cvm2027_*_rescore_20261025`文件和`analyze_cvm_*_official.py`中固定。本轮不再使用旧online correct/acc作为权威。

## 5. 复现

从此worktree根目录执行，模型和数据缓存路径保持服务器现状。完整GPU运行仅在A40无其他compute任务时串行进行。

```bash
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$P scripts/cvm24h_prepare_hilo.py --out results/cvm2027_supplement_20261026/plan/hilo_source_manifest.json
# 完整冻结质量矩阵：不依赖历史T0/deadline，输出独立reproduction目录
bash scripts/cvm24h_reproduce_quality.sh all
# 历史输出的CPU评分
bash scripts/cvm24h_score_all.sh
bash scripts/cvm24h_score_oc_curve.sh
bash scripts/cvm24h_score_q10.sh
# 修正成本：新文件，先harness后hilo；deadline只用于本次收束
$P scripts/cvm24h_timing_v2.py --engine harness --out results/cvm2027_supplement_20261026/plan/timing_harness_corrected.json
$P scripts/cvm24h_timing_v2.py --engine hilo --out results/cvm2027_supplement_20261026/plan/timing_hilo_corrected.json
$P scripts/cvm24h_score_timing.py
$P scripts/cvm24h_build_tables.py
# 统一attention完整主表：先核验P3，再做eager32/no-op、8格smoke和全部8格full。
# 新复现时显式给未来4小时截止；已有同指纹完整cell会幂等复用，不覆盖历史raw。
CVM_MATCHED_DEADLINE_EPOCH="$(( $(date +%s) + 14400 ))" bash scripts/cvm24h_run_matched_main.sh
$P scripts/cvm24h_score_matched.py
# 文档页相关性敏感性与最终所有产物哈希
$P scripts/cvm24h_doc_cluster.py
$P scripts/cvm24h_manifest.py
```

单格示例和冻结配置见原`plan/task_queue.json`与对应运行脚本；最终manifest只导出轻量哈希、协议及失败ID，原始预测、timing repeats、标注、数据及权重均留本地且不进入Git/匿名源包。

需要在新目录从零重新推理统一attention单格时，直接运行以下命令（其他mode取`none/pre/fastv`；None的keep须为1，OCRBench用对应full JSONL与n1000）；不要删除本轮原始预测：

```bash
$P scripts/cvm24h_matched_quality.py --mode hilo --benchmark textvqa \
  --subset eval/full_splits/textvqa_val.jsonl --n 5000 --keep-frac .25 \
  --out results/cvm2027_reproduction_matched/hilo_textvqa.json \
  --deadline-epoch "$(( $(date +%s) + 14400 ))"
# 8格均在独立目录完成后做同样的完整性/官方评分核验；会刷新标准报告，建议在复现checkout执行。
$P scripts/cvm24h_score_matched.py --prediction-dir results/cvm2027_reproduction_matched
```

## 6. 写作与未覆盖范围

论文权威路径 `drafts/cvm2027_submission_20261025/main.tex`。补入四方法主表、预算表/图、Doc500表与质量—实际成本；保留原三模型矩阵和用户选定的图，明确历史vLLM与新HF协议边界。摘要/引言/结论不再推“先选普遍更好”，同时报告FastV的TextVQA优势和RBM的Doc子集优势。

全部正式结果完整后，运行 `python scripts/cvm2027_build_release.py` 生成匿名论文PDF与最小源ZIP。该入口会拒绝pending稿或不完整实验，核对源数据SHA、51个实引、匿名性、字体、页数/大小及TeX错误，并解包源ZIP到独立目录复编；全部机器检查成功后才替换canonical PDF/ZIP。可用 `--check-source-only` 做只读稿件门检，或 `--engine /path/to/tectonic` 指定隔离的TeX引擎。全页PNG在 `output/qa/cvm2027_review_20261026/pages/`，机器检查和人工目视结果在 `reports/cvm2027_build_qa_20261026.{json,md}`；渲染成功本身不等于目视通过。

本轮不做训练、新模型、Doc50/75、空间均匀/随机控制或连续批处理；其中额外控制按原指令为可选任务。新增现代baseline全split比较已统一attention，Hi-Lo的vendored模型代码仍是实现边界；旧曲线/Doc500的attention差异另行披露，不把小样本no-op或common-answered包装为全量等价。未代作者投稿。
