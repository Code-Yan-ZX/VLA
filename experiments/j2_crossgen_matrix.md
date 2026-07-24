# J2 — Qwen2.5-VL-7B 跨代矩阵（官方指标，n=200，2026-07-24）

**结论：跨代一致性成立。** pre-merger 在 text-dense 基准大胜、GQA tie/弱占优无 crossover、VisionZip-style≡post——全部在 Qwen2.5-VL 复现。

## 官方指标汇总（Qwen2.5-VL-7B-Instruct，n=200 子集，短答烤入）

| 基准（官方） | none | pre@25% | post@25% | Δ | pre@12.5% | post@12.5% | Δ |
|---|---|---|---|---|---|---|---|
| TextVQA VQA-acc | 0.870 | 0.735 | 0.415 | **+32.0pp** | 0.618 | 0.318 | **+30.0pp** |
| DocVQA ANLS | 0.975 | 0.687 | 0.499 | **+18.8pp** | 0.476 | 0.253 | **+22.3pp** |
| OCRBench 官方 | 0.805 | 0.465 | 0.180 | **+28.5pp** | 0.335 | 0.060 | **+27.5pp** |
| GQA exact-match | 0.585 | 0.565 | 0.555 | **+1.0pp (tie)** | 0.505 | 0.505 | **0 (exact)** |

显著性：text-dense 三基准 Δ 均 ≥5σ（paired SE≈0.035 @n=200）；GQA Δ z≈0.3（tie）。

## 与 Qwen3-VL 对照

| | Qwen3-VL（官方，n=200） | Qwen2.5-VL（官方，n=200） |
|---|---|---|
| TextVQA Δ@25% | +38.3pp | +32.0pp |
| DocVQA Δ@25% / @12.5% | +26.5pp / +47.5pp | +18.8pp / +22.3pp |
| OCRBench Δ@25% | +41.5pp | +28.5pp |
| GQA | n100 tie 0.470/0.470；**n200 pre 0.420 vs post 0.465（post +4.5pp，~1.3σ 不显著）** | n200 tie +1.0pp / @12.5% exact tie |
| vz-style ≡ post | 11/11 cell 一致 | ✓ 0.415==0.415 字节一致 |

**诚实 nuance**（如实入稿）：① Qwen2.5-VL 的 post 在 DocVQA@25% 比 Qwen3 稳健（0.499 vs 0.200），差距在 12.5% 深压才完全展开（机制=lossy merger 失真随压缩单调加重，跨代同向不同速率）；② Qwen3 GQA n=200 post 弱领先 4.5pp（n=100 曾 exact tie）——n 依赖的噪声级波动，只报方向不 claim crossover；stage law 表述维持"pre 弱占优、GQA 平手级、无 crossover regime"。

## 产物
runs/v3_crossarch_cells/j2_*.json（24 cell）+ j2_official_summary.json + j2_qwen2vl_matrix.sh。下一步：J4 baseline 等价性 gate 运行中 → J5 QA gate → J7 完整 split。
