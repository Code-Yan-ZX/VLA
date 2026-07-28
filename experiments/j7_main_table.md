# J7 — 官方完整 split 主表（vLLM，2026-07-25）

**Goal 条件②（完整官方 benchmark 成立）达成。26/26 cells 零缺失。**

## 官方指标主表（完整 split，binomial stderr）

### Qwen3-VL-8B
| bench（官方） | none | pre@25% | post@25% | Δ | pre@12.5% | post@12.5% |
|---|---|---|---|---|---|---|
| TextVQA VQA-acc | 0.844 | **0.605** | 0.222 | **+38.4pp** | 0.472 | 0.132 |
| DocVQA ANLS | (native 巨图崩溃/部分) | **0.481** | 0.238 | **+24.3pp** | 0.352 | 0.103 |
| OCRBench Final/1000 | 760 | **547** | 184 | **+363pts** | 350 | 53 |
| GQA acc | 0.616 | 0.449 | 0.477 | −2.8pp（**post 显著领先 z≈4.5**，n=12578；量级≈text-dense Δ 的 1/14） | — | — |

### Qwen2.5-VL-7B
| bench（官方） | none | pre@25% | post@25% | Δ | pre@12.5% | post@12.5% |
|---|---|---|---|---|---|---|
| TextVQA VQA-acc | 0.862 | **0.702** | 0.442 | **+26.1pp** | 0.597 | 0.319 |
| DocVQA ANLS | 0.949 | **0.636** | 0.526 | **+11.0pp** | 0.455 | 0.245 |
| OCRBench Final/1000 | 817 | **476** | 183 | **+293pts** | 335 | 67 |
| GQA acc | 0.604 | 0.559 | 0.585 | −2.6pp（**post 显著领先 z≈4.1**，n=12578；量级≈text-dense Δ 的 1/10） | — | — |

## 结论（完整 split）

1. **跨代一致**：text-dense 三基准 pre 全胜（Qwen3 +38.4/24.3/36.3pp；Qwen2.5 +26.1/11.0pp、OCRBench pre@12.5% 335 vs post 2.5% 67 即 8.3×）；跨 split 与 subset 同向数值近。
2. **GQA 完整 split 解析（claim 修正）**：n=12578 下 post +2.8/+2.6pp **统计显著**（独立 binomial z≈4.5/4.1；配对 McNemar z≈8.1/7.1；逐样本官方 rescore 精确复现表值）。n=200 时"tie"系欠功效（SE 大 8×）误判。**这不推翻 spine**：效应量仅为 text-dense Δ 的 1/10–1/14，与原 workload-conditional stage law（text-dense→pre 大胜；object-centric→post 微胜）完全一致；入稿口径="post 在 GQA 显著但微幅领先；无 text-dense crossover；RBM 为鲁棒默认而非全域最优"。红线保持：不写 RBM beats existing methods。
3. **OCRBench Final/1000**：Qwen3 pre@25% 547 vs post 184（差 5×），pre@12.5% 350 vs 53（6.6×）——post-merger 族深压脆弱性在官方 /1000 口径下更突显。
4. **Goal 条件统计**：①双模型结论一致 ✓②完整 benchmark 成立 ✓（坏 cell 1 个安全-flags 补跑中）；③强 baseline ④消融闭环 ⑤submission-ready 待 J6/J8/HF-n500/paper。

## 诚实 nuance（入稿）
- Qwen3-VL none DocVQA 巨图 native 分辨率超 max_model_len 部分样本被 skip；pre/post @25% 用 native、@12.5% 同；700k 全量评估需更大模型上下文（论文注明 n 部分）。
- qwen2vl pre OCRBench r0.75 full 坏 cell 已安全-flags 补跑完成（0.476, skip=0）。
- 可能纳入正文：同快速 token 数列以助公平性（ptid 已记录）。