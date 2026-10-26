# CVM 2027 补实验与写作交付报告

时间：2026-10-25 19:07:06 开始；2026-10-26 接管审计和收束（Asia/Shanghai）。隔离分支 `codex/cvm-24h-supplement` 基于含 `1600bfe` 的 main。原始执行代码至 `90a5b73`，随后修正与论文版本见 Git 历史。

**当前结论：质量主实验及预定预算曲线齐全；旧成本实现无效，已撤回，修正版 P3 在独立文件重测。** 原工作区修改、历史预测与日志保留。用户10/26已追加授权补论文，覆盖原提示词“本轮不改论文”；不包含外部投稿。

## 1. 可成立的质量结论

同一 Qwen3-VL-8B 权重、processor、prompt、像素上限与 greedy≤32 tokens 下，25%最终视觉保留率的结果如下。Hi-Lo使用官方模型适配和SDPA，其他三组使用统一HF eager harness；因此它是操作性实现比较，不能称完全相同计算实现的隔离实验。

| 任务/协议 | None100% | RBM25% | FastV-k325% | Hi-Lo25% | 完成样本数（相同列顺序） |
|---|---:|---:|---:|---:|---|
| TextVQA全5000，4M cap，官方accuracy% | 83.24 | 75.50 | 76.83 | 66.77 | 4995/4995/4995/5000 |
| OCRBench全1000，4M cap，points/1000 | 707 | 559 | 413 | 613 | 921/921/921/1000 |
| DocVQA固定500子集，统一2M cap，ANLS% | 95.73 | 82.76 | 70.45 | 76.38 | 500/500/500/500 |

主分母始终是全部尝试样本，失败记0。TextVQA从原始官方十标注按question_id评分，禁止使用旧online correct/acc或拼接GT重建。所有新比较使用20,000次配对bootstrap和固定种子；轻量表列每个差值、CI、分母与失败集合，见 `reports/cvm2027_supplement_tables.md`。

- **TextVQA不支持RBM最强。** FastV−RBM = +1.322pp，新配对重采样95%CI[+0.338,+2.326]；RBM−Hi-Lo = +8.73pp，CI[+7.51,+9.94]。RBM仍比None低7.74pp。10/25历史重评使用另一固定bootstrap seed，边界略有Monte Carlo差异，点估计相同。
- **OCRBench的Hi-Lo attempted领先主要来自完成覆盖。** RBM/FastV两份原日志各79个OOM，ID集合与预测skip完全一致；Hi-Lo全答。共同921题上，RBM=559/921=60.69%，Hi-Lo=556/921=60.37%，差−0.33pp且CI跨0；不能把attempted的+54分解释为选择质量更强。共同样本敏感性仍不能消除attention与vendored模型差异。
- **DocVQA500存在RBM优势，但不是95%质量保持。** RBM−Hi-Lo = +6.39pp CI[+3.04,+9.78]；RBM−FastV = +12.32pp CI[+8.15,+16.47]。所有压缩臂都低于95%×同协议None。此500题子集不能称full split，也不能与历史600k或4M数据混排。

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

预定义质量目标为同协议None的95%，只判断已测离散点，不插值、不把“不显著”当等价：TextVQA1000上最小达标keep为RBM50/FastV50/Hi-Lo75；OCRBench attempted为75/75/50，共同921题则均为75。**这里是最小达标保留率；最低实际成本必须另用同协议计时选点，不能假定更小keep必定更快。** 旧报告把RBM OCR75说成未达标是算术错误，已更正（684≥671.65）。

表图：`reports/figures_cvm2027_supplement/budget_curves.{pdf,svg}`；实际K图 `score_vs_actual_k.{pdf,svg}`。后者基于共同样本真实计数，四路特征宽度不计作四倍LLM token。

Q10可选全5000扩展：接管时RBM/FastV50/75四格完成，Hi-Lo50在跑，Hi-Lo75未启动。修正P3优先；只有六格全部完成并评分才将全量扩展表称完整曲线。固定1000题曲线始终是完整的预定结果。

## 3. 成本审计：旧结果撤回与新测量边界

接管发现旧P3脚本存在缺失inference/no-grad、累积MergerTap hook、FastV漏传deepstack、Hi-Lo外部config更新未作用于实际模型、不同total时间边界等问题。旧41–43GiB显存结论和全部速度百分比无效；旧raw不删除、不作为论文数据。详见 `reports/cvm2027_timing_audit_20261026.md` 和 `reports/cvm2027_execution_audit_20261026.md`。

修正脚本 `scripts/cvm24h_timing_v2.py` 统一vision SDPA/text eager、bf16、bs1、greedy≤32、4M cap。固定200个请求（TV100/OCR100）每个重复3次，每配置预热；先取每请求三次中位数，再取200请求的p50/p95。直接测完整请求和TTFT，保留CPU processor、传输、完整视觉、选择/融合、LLM prefill、decode、actual K、逐层token轨迹、allocated/reserved峰值、输出长度与全部原始重复。

单次请求从已经解码的RGB PIL图像开始，不含磁盘、图像解码、模型加载、队列或预热。workload wall包含整个循环、每ID一次本地解码、同步、记账与失败恢复；其吞吐分母不可与单请求边界混淆。选择/融合可能包含在prefill段，禁止重复求和。当前HF RBM确实完整执行ViT与所有merger后才gather，因此不能计入任何未实际省下的视觉加速。

新统一attention不同于旧full-quality路径，**质量—成本图只用本次200题产生的答案重新官方评分，按任务各100题与相同请求的成本配对**，不把旧全量分数冒充新协议质量。新文件为 `plan/timing_harness_corrected.json` 和 `plan/timing_hilo_corrected.json`；完成后 `reports/cvm2027_corrected_quality_cost.{md,json}` 给出完整表、同协议质量与95%目标下最低测得p50成本。

## 4. 正确性、实现与复用

- 模型snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；transformers4.57.6、torch2.10.0+cu128；无环境改装、无重复权重下载。
- Hi-Lo上游commit `29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6`。仅移除引用未定义`top_index_1`的转换guard，并提供4.57版本所需rope配置shim。官方选择/融合代码保留。`scripts/cvm24h_prepare_hilo.py` 可重建精确修复并输出源码SHA。
- 冻结Hi-Lo参数：prune_layer=2、fusion=.3、top_k_ratio=.1、attention_threshold=.05、candidate=max(.30,keep)、fusion_layers=[0,1]。50/75%时candidate与keep相同，融合候选集合退化为空；不是根据验证成绩调参。
- 实际K已逐共同样本核验为round(N×keep)，Hi-Lo融合不额外保留token。RBM同kept IDs对齐主流及三条deepstack；不能照旧CLI描述再乘4补偿。
- 旧P0检查涵盖32题no-op答案/K和8题部分张量检查；其logits probe没有完整传入deepstack，Post/FastV/Hi-Lo未形成全面的logits/位置/cache证明，不能统称“全部P0逐位等价”。新P3补真实生成parity、live配置与逐层actualK门检；BF16误差按实值记录。
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
# 原始质量队列和评分；幂等跳过已完成格
bash scripts/cvm24h_queue_runner.sh
bash scripts/cvm24h_score_all.sh
bash scripts/cvm24h_score_oc_curve.sh
bash scripts/cvm24h_score_q10.sh
# 修正成本：新文件，先harness后hilo；deadline只用于本次收束
$P scripts/cvm24h_timing_v2.py --engine harness --out results/cvm2027_supplement_20261026/plan/timing_harness_corrected.json
$P scripts/cvm24h_timing_v2.py --engine hilo --out results/cvm2027_supplement_20261026/plan/timing_hilo_corrected.json
$P scripts/cvm24h_score_timing.py
$P scripts/cvm24h_build_tables.py
$P scripts/cvm24h_manifest.py
```

单格示例和冻结配置见原`plan/task_queue.json`与对应运行脚本；最终manifest只导出轻量哈希、协议及失败ID，原始预测、timing repeats、标注、数据及权重均留本地且不进入Git/匿名源包。

## 6. 写作与未覆盖范围

论文权威路径 `drafts/cvm2027_submission_20261025/main.tex`。补入四方法主表、预算表/图、Doc500表与质量—实际成本；保留原三模型矩阵和用户选定的图，明确历史vLLM与新HF协议边界。摘要/引言/结论不再推“先选普遍更好”，同时报告FastV的TextVQA优势和RBM的Doc子集优势。

本轮不做训练、新模型、Doc50/75、空间均匀/随机控制或连续批处理；后两类是原指令可选任务。完整同实现现代baseline全split比较仍有模型代码/attention混杂，报告明确留边界，不把小样本no-op或common-answered包装为全量等价。未代作者投稿。
