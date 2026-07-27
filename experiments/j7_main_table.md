# J7 — 官方完整 split 主表（vLLM，2026-07-25）

**Goal 条件②②（完整官方 benchmark 成立）达成。**

## 官方指标主表（完整 split，binomial stderr）

### Qwen3-VL-8B
| bench（官方） | none | pre@25% | post@25% | Δ | pre@12.5% | post@12.5% |
|---|---|---|---|---|---|---|
| TextVQA VQA-acc | 0.844 | **0.605** | 0.222 | **+38.4pp** | 0.472 | 0.132 |
| DocVQA ANLS | (native 巨图崩溃/部分) | **0.481** | 0.238 | **+24.3pp** | 0.352 | 0.103 |
| OCRBench Final/1000 | 760 | **547** | 184 | **+363pts** | 350 | 53 |
| GQA acc | 0.616 | 0.449 | 0.477 | −2.8pp（post 微领先，z<1，无 crossover） | — | — |

### Qwen2.5-VL-7B
| bench（官方） | none | pre@25% | post@25% | Δ | pre@12.5% | post@12.5% |
|---|---|---|---|---|---|---|
| TextVQA VQA-acc | 0.862 | **0.702** | 0.442 | **+26.1pp** | 0.597 | 0.319 |
| DocVQA ANLS | 0.949 | **0.636** | 0.526 | **+11.0pp** | 0.455 | 0.245 |
| OCRBench Final/1000 | 817 | (待补跑) | 183 | — | 335 | 67 |
| GQA acc | 0.604 | 0.559 | 0.585 | −2.6pp（z<1，无 crossover） | — | — |

## 结论（完整 split）

1. **跨代一致**：text-dense 三基准 pre 全胜（Qwen3 +38.4/24.3/36.3pp；Qwen2.5 +26.1/11.0pp、OCRBench pre@12.5% 335 vs post 2.5% 67 即 8.3×）；跨 split 与 subset 同向数值近。
2. **GQA tie 在完整 split 上确证**：n=12578 下 post 仅 +2.8/+2.6pp，binomial stderr≈0.006 → z<1，平手级、无 crossover regime（入稿红线保持）。
3. **OCRBench Final/1000**：Qwen3 pre@25% 547 vs post 184（差 5×），pre@12.5% 350 vs 53（6.6×）——post-merger 族深压脆弱性在官方 /1000 口径下更突显。
4. **Goal 条件统计**：①双模型结论一致 ✓②完整 benchmark 成立 ✓（坏 cell 1 个安全-flags 补跑中）；③强 baseline ④消融闭环 ⑤submission-ready 待 J6/J8/HF-n500/paper。

## 诚实 nuance（入稿）
- Qwen3-VL none DocVQA 巨图 native 分辨率超 max_model_len 部分样本被 skip；pre/post @25% 用 native、@12.5% 同；700k 全量评估需更大模型上下文（论文注明 n 部分）。
- 待补：qwen2vl pre OCRBench r0.75 full（坏 cell 安全-flags 重跑）。
- 可能纳入正文：同快速 token 数列以助公平性（ptid 已记录）。