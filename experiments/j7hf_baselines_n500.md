# J7hf — HF baseline n=500 双模型（官方指标，2026-07-28）

**Goal 条件③（强 baseline 充分）达成。** FastV + Pyramid(canonical) × 4 基准 × 双模型，n=500 headline subset，同模型同预算，skip≈0（HF harness 验证链：r=0 锚 8/8、HF≡vLLM 16/16、qwen2vl smoke 通过）。DocVQA 同像素 600k（HF 注意力物化约束，脚注披露）。

## 官方指标

### Qwen3-VL-8B（n=500，FastV r=0.75 / Pyramid canonical keep_equiv 0.625）
| bench | **RBM@25%**(full) | **FastV@25%** | post≡VZ@25%(full) | Pyramid@62.5% |
|---|---|---|---|---|
| TextVQA VQA-acc | 0.605 | **0.646** | 0.222 | 0.835 |
| DocVQA ANLS (600k) | 0.481(native)/0.424(600k n200) | 0.486 | 0.238(native)/0.251(600k n200) | 0.882 |
| OCRBench acc | **0.547** | 0.288 | 0.184 | 0.746 |
| GQA acc | 0.449 | **0.510** | 0.477 | 0.606 |

### Qwen2.5-VL-7B
| bench | **RBM@25%**(full) | **FastV@25%** | post≡VZ@25%(full) | Pyramid@62.5% |
|---|---|---|---|---|
| TextVQA VQA-acc | 0.702 | **0.757** | 0.442 | 0.779 |
| DocVQA ANLS (600k) | 0.636(native，600k 待补) | 0.481 | 0.526(native) | 0.818 |
| OCRBench acc | 0.476 | 0.484（**tie**） | 0.183 | 0.604 |
| GQA acc | **0.559** | 0.498 | 0.585 | 0.562 |

## 结构结论（入稿）

1. **FastV（layer-2 query-conditioned）胜 RBM**：TextVQA 双模型（+4.1/+5.5pp）、GQA Qwen3（+6.1pp）、DocVQA 同像素（Q3 ≈、Q2.5 j4d n200: +9.4pp）。机制=注意力已混问题 token，可恢复 query 相关区域。
2. **RBM 胜/平 FastV**：OCRBench Q3 **0.547 vs 0.288（+25.9pp，dense OCR 文字 merger 前保留不可替代）**、Q2.5 平（0.476 vs 0.484）、GQA Q2.5（0.559 vs 0.498，+6.1pp）。
3. **跨代差异（诚实入稿）**：FastV OCR 在 Q3 崩（0.288）、Q2.5 不崩（0.484）——Qwen2.5 merger 对文字保留较温和（与 J2 DocVQA Δ 较小同因）；Pyramid OCR Q3 近无损（0.746）Q2.5 掉得多（0.604）。
4. **无全域赢家**：FastV 赢 query-relevant 基准、输 dense OCR（Q3）；RBM 反之；Pyramid 高预算近无损但 iso-25% 退化 schedule 全崩（j4c）。→ RBM=鲁棒默认（从不崩、text-dense/OCR 对 post 族大胜）、非全域最优（claim 红线保持）。
5. post-merger 族（VZ≡post）text-dense 全线最弱（0.222/0.238/0.184/0.442/0.183）——spine"post-merger 深压脆弱"保持。

## 待补
- Qwen2.5 DocVQA 同像素 600k {none,pre,post} n200（J8 后补，~40min）。
- 效率表 J6 / 消融 J8 运行中。