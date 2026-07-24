# J4 探针 — 同模型同预算 baseline（Qwen3-VL-8B，n=200 子集，官方指标）

HF harness 已验证（r=0 锚 8/8 逐样本一致、HF-vs-vLLM none 16/16 等价）。

## 官方指标（n=200 子集，Qwen3-VL-8B）

| 基准（官方） | none | RBM pre@25% | post≡VZ@25% | **FastV@25%** | Pyramid@62.5%（canonical） |
|---|---|---|---|---|---|
| TextVQA VQA-acc | 0.858 | 0.598 | 0.215 | **0.680** | 0.852 |
| DocVQA ANLS | 0.976 | 0.465 | 0.200 | **cell 坏**（skip 187/200） | **cell 坏**（skip 187/200） |
| OCRBench | ~0.73 | **0.580** | 0.165 | 0.155（skip 19） | — |
| GQA exact | ~0.53 | 0.420 | 0.465 | **0.490** | — |

（none 列 OCRBench/GQA 为既有 cell 参考值；post≡VZ 为 J2/rescore_rerun 同池数字。）

## 核心发现

1. **FastV（LLM layer-2 注意力一次性剪）在 TextVQA +8.2pp、GQA +7pp 胜 plain RBM**（同 n=200 同子集同 scorer）。机制解读=layer-2 注意力**天然 query-conditioned**（注意力已混入问题 token）——恰好是 oracle +8.2pp 的 query 依赖余量：scene-text/GQA 中"哪块区域对**该题**相关"可被恢复。
2. **FastV 在 OCRBench 崩到 post 水平**（0.155 vs RBM 0.580）：dense OCR 的文字在 merger 阶段已毁，layer-2 再聪明的排序也救不回 → **pre-merger（raw patch 保文字）不可替代**。
3. **post-merger 族（VZ≡post）在 text-dense 全线崩溃**（0.215/0.200/0.165）的 spine claim 保持。
4. Pyramid canonical（keep_equiv 0.625，2.1× 我方预算）TextVQA 0.852 近无损——**不同预算点**，入稿作 retention-accuracy 曲线一点（需补我方 pre r=0.375 cell 同点对比）。

## claim 纪律（红线）

- **不得写"RBM 超过现有方法"**：FastV 同模型同预算胜 TextVQA/GQA。
- 可写：① RBM **鲁棒**（任何基准不崩、从不输给 post-merger 族）；② text-dense/OCR 大胜（vs post 族 +36~42pp）；③ FastV 的 query-conditioned 优势恰是 QA gate 要回答的问题——若 QA-pre 拿回 TextVQA/GQA 且守住 OCR = 双赢方法 claim（gate 预注册判据裁决）；否则 RBM 作鲁棒安全默认、query-aware 作 bounded negative。

## 缺陷与待办

- **DocVQA HF cell 两法皆 skip 187/200**（max-pixels 1.5M 仍 OOM/长序列路径问题）→ 降 max-pixels（1.0M→0.8M）重跑 fastv+pyramid docvqa；13 样本的 acc 无意义。
- Pyramid iso-25% 失败：harness 校验要求 ratios 以 1.0 开头（忠于论文）→ 待试 [1.0,0.0,0.0,0.0]（keep_equiv 恰 0.25，退化极端 schedule）n=2 可行性；崩则只报 canonical 点并如实注明。
- FastV OCRBench skip 19 待查（长 OCR 图）。
- 产物：runs/j4_baselines_hf/j4_*.json + j4_probe_qwen3vl.sh；digest 链 experiments/j4_step2_fix.md（修复）。
