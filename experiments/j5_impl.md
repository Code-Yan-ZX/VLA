# J5 — query-aware pre-merger saliency (QA-pre) 实现摘要（2026-07-24，纯 CPU 实现，未跑 GPU）

设计=notes/j5_qa_gate_design.md；判据锁死于 DECISIONS 2026-07-23（未改）。本文件=实现+执行手册。
**只写代码/脚本，未加载任何模型、未占 GPU**；GPU 验证（dev 选 λ + gate）由主窗口执行（见末「执行清单」）。

## 方法（qsim selector）
对每个 merge-unit i：`qsim_i = max_{t∈question tokens} cos( merger(unit_feat_i), embed(q_t) )`
- `merger(unit_feat_i)`=**原生 main merger 单次前向**（复用 PreMergerPruner 已注册的 merger_origs["main"]，无新 hook）→ LLM 空间行向量；
- `embed(q_t)`=问题 token 经 **LLM 词嵌入层**（共享空间）；
- 组合（逐图像）：`s_i = (1−λ)·minmax(sel_i) + λ·minmax(qsim_i)`，两路**各自 per-image min-max 到 [0,1] 后**加权（top-k 是秩选择，绝对尺度无关；λ 是真权衡旋钮）。仅 pre 模式、stage 秩、dominant-only 路径生效。

## runner 改动（src/v3_premerger/v3_premerger_runner.py，+337/−5）
| 函数/位置 | 行 | 作用 |
|---|---|---|
| `--qa-lambda` (default 0.0) / `--qa-embed-cache` | 595 / 606 | 新 CLI。λ=0 → qsim 路径**从不进入**（零开销、逐位=plain RBM） |
| `_minmax` | 337 | per-image min-max→[0,1]，常向量→0（安全除） |
| `_find_embed_tokens` | 345 | 定位 LLM 词嵌入层（路径+fallback，见下） |
| `_qa_tokenize_question` | 380 | 问题原文 tokenize（add_special_tokens=False） |
| `_qa_offline_unit_counts` | 391 | HF image processor 离线算每样本 unit 数（CPU，复用 attach_hybrid 同款 guard） |
| `_qa_precompute` | 418 | 生成前嵌入全部问题（可 string 缓存）+按 unit 数键控队列；含 warmup 额外副本 |
| `attach_qa_per_sample` | 480 | per-image mean qsim 按 unit 数 best-effort 回填 per_sample（分析用，非 gate 关键） |
| `PreMergerPruner.__init__`(+qa_lambda/qa_state) | 1220 | |
| `slice_input` QA hook | ~1340 | `if qa_lambda>0 and qa_state and embed:` → `_qa_blend_scores`（else 分支/stage 秩内；swap/vz 已 guard） |
| `_qa_pop_embedding` | 1350 | 按 unit 数出队问题嵌入（抗 vLLM 批重排+warmup encoder-cache replay），缺失→有序 FIFO fallback |
| `_qa_blend_scores` | 1368 | merger_orig(hs)→[U,hidden]，逐图 cos-max→qsim，min-max 组合；**任何异常→返回 base_scores 不崩溃** |
| `setup_pre_merger`(+qa params) | 1572 | 传参；注册 `qa_state["main_merger_orig"]=merger_origs["main"]`（1611） |
| main(): guards | 1897 | λ>0 需 mode=pre & mask_ranking≠swap & ¬visionzip，否则 SystemExit |
| main(): qa_state 构建 | 1978 | `llm.get_tokenizer()`（fallback `llm_engine.tokenizer.tokenizer`）+ `_qa_precompute` |
| main(): per_sample/result 附加 | 2071/2097 | λ>0 时每样本 `qa_lambda`+`qa_mean_qsim`；result 附 `qa`(诊断计数) |

## qsim 取嵌入的确切路径与 fallback（`_find_embed_tokens`）
主路径（**vLLM 0.19 已核对**，双族通用）：`model.language_model.model.embed_tokens`
（qwen3vl: Qwen3LLMForCausalLM.model→Qwen3LLMModel.embed_tokens，qwen3_vl.py L1396/1402；
qwen2vl: language_model.model，qwen2_5_vl.py L1143/1472）。
fallback 候选路径：`model.language_model.embed_tokens`、`model.model.embed_tokens`、`model.embed_tokens`、`embed_tokens`；
最终 fallback=`named_modules()` 扫**最大 num_embeddings 的 nn.Embedding**（文本词表 151k+ ≫ 视觉位置嵌入）。
仍找不到→qsim 禁用、回落 plain selector（λ 等效 0），诊断 `qa.embed_found=false`。**enforce_eager=True 下进程内可读权重。**

## 脚本用法
**dev 选 λ** `scripts/j5_qa_dev_select.py`（`<conda>/bin/python`）：
- `--build-only`=CPU 建 dev slice（full_splits 中 id∉_200 各 32，random.Random(0)；落 j5/dev_ids.json 审计 + dev_{textvqa,docvqa}.jsonl）；
- 无参=建 slice→等 GPU（≥30000MiB，≤6h，同 j2/j3）→对 λ∈{0,0.3,0.5,0.7} 跑 pre r=0.75 n=32（8 cell，shell 出 runner；textvqa STD/docvqa DOC 旗标）→官方指标（textvqa VQA-acc/docvqa ANLS）均值增益选 λ（全≤0→0）→`j5/lambda_selection.json`；
- `--score-only`=跳过 GPU 只评分选 λ（cell 已有时）。幂等（cell 存在即 skip）。
**gate** `runs/v3_merger_aware/j5_gate.sh`（bash）：读 lambda_selection.json 的 λ→QA-pre(λ) 与 plain(λ=0) 在 4 子集 n=200 r=0.75 Qwen3-VL 成对跑（8 cell，幂等）→内嵌 python 官方评分（textvqa VQA-acc/docvqa ANLS/ocrbench 官方含 question_type/gqa exact-match）+ 预注册判据→打印 **GO/NO-GO**→`j5/j5_gate_result.json`（存在即 skip）。
判据：(i) 官方均值≥plain+1.0pp；(ii) ≥3/4 基准≥plain−0.5pp；(iii) pooled paired **McNemar z**=(c−b)/√(b+c)≥1.5（按 id 配对 per_sample correct，b=plain对QA错/c=反之，pooled 4 基准；附 paired-diff z 备查）。
注：λ=0 被选中时 plain 与 QA cell tag 相同(qa0.00)→自动只跑 4 cell、增益≈0→NO-GO（正确）。

## CPU 验证（已通过，未碰 GPU）
runner py_compile✓；dry-check 双族 ALL PASS（λ=0 路径未坏）；单测✓：_minmax、λ=0 guard=False（逐位隔离）、_qa_pop_embedding（键控+fallback+miss）、_qa_blend_scores 数学 λ∈{0.3,0.5,0.7,1.0}、merger 异常→返回 base、_find_embed_tokens（路径优先+最大嵌入 fallback+none）、_qa_precompute（真图离线计数+归一+warmup 副本）、attach（8/8）、slice_input 集成（iso-token、mask 缓存、qsim 改选择、λ=0 逐位一致）；dev select 切片不相交/确定/选 λ 逻辑✓；gate 判定 GO/NO-GO 双场景内部一致✓。
**已知（与本次无关）**：runner `--help` 崩溃=**既有** bug（旧 help 文本含裸 `%`，如 "70% dominant"，argparse `% params` 误展开）；未修（不在本任务范围，不影响任何脚本/运行）。

## 执行清单（给主窗口，串行 A40）
1. **dev 选 λ**（GPU ~20min）：`/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/j5_qa_dev_select.py`
   → 产出 `runs/v3_merger_aware/j5/lambda_selection.json`（看 `selected_lambda`）。可先 `--build-only` 查 dev_ids。
2. **gate**（GPU ~1h）：`bash runs/v3_merger_aware/j5_gate.sh`（自动读选定 λ，等 GPU，跑 8 cell）
   → 屏幕打印 GO/NO-GO，落 `runs/v3_merger_aware/j5/j5_gate_result.json`。
3. **判定**：读 j5_gate_result.json 的 `verdict`。
   - **GO**→QA-pre 进主表/消融（J7/J8 带 λ）。
   - **NO-GO**（大概率，诚实预期）→方法**冻结 plain RBM**；QA 作 bounded negative 如实写入论文；**不再搜任何 hybrid/router 变体**（预注册）。
4. GPU 等待礼仪同 j2/j3（≥30000MiB，≤6h 止损）；两脚本均幂等，可断点续跑。
预算：dev(~20min)+gate(~1h)=~1.3 GPU·h ≤ 预注册 2 GPU·h 止损线。
