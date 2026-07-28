# J8 — 消融闭环（n=200 子集，官方指标，2026-07-28）

**Goal 条件④消融部分达成。**

## A. Budget 曲线（L2，pre vs post，官方）

| 基准·模型 | 75% 保留(r.25) | 50% 保留(r.5) | 25% 保留(r.75) |
|---|---|---|---|
| TextVQA Q3 | pre 0.740 / post 0.653（**+8.7**） | 0.670 / 0.380（**+29.0**） | 0.605 / 0.222（**+38.4**） |
| TextVQA Q2.5 | 0.870 / 0.800（**+7.0**） | 0.813 / 0.660（**+15.3**） | 0.702 / 0.442（**+26.1**） |
| DocVQA Q3 | 0.687 / 0.700（−1.3 ≈tie） | 0.589 / 0.531（**+5.9**） | 0.481 / 0.238（**+24.3**） |
| DocVQA Q2.5 | 0.946 / 0.967（−2.1 ≈tie） | 0.875 / 0.889（−1.4 ≈tie） | 0.636 / 0.526（**+11.0**） |

**模式：pre 优势随压缩深度单调扩大**——浅压（75% 保留）两法近似（DocVQA 甚至 post 微胜 1–2pp 噪声内），深压（25%）pre 大胜。与 lossy-merger 机制直接对应：merger 的文字信息销毁只在深压缩显著，浅压缩下 post 的"全局池化特征"尚有可救信息（object/GQA 同因）。入稿=retention-vs-gap 曲线图（Fig.）。

## B. Selector 不变性（r.75，官方）

| | L2 pre−post | attn(质心) pre−post |
|---|---|---|
| TextVQA Q3 | +38.4 | +35.5（0.577/0.222） |
| DocVQA Q3 | +24.3 | +24.4（0.451/0.207） |
| TextVQA Q2.5 | +26.1 | pre 0.673（l2 0.702）；sign 同 |

**Q3：sign+量级双不变 ✓**（attn 代理绝对值略低但 stage gap 几乎同）。Q2.5：L2 sign 不变（J2 全谱）；attn 代理在 Q2.5 上绝对值低（J3b：docvqa attn 曾 sign 反转）——入稿口径="selector invariance 在 Q3 双 selector 成立；Q2.5 的 L2（论文 selector）sign 不变、质心代理弱"。

## C. Mask 粒度
unit=2×2（native merger 粒度）为方法定义；token 级（unit 前）不可行——merger 输入即最小语义单元，pre 的"merge 前"恰指 unit 级。入稿作方法定义说明（非消融维度）。

## 产物
runs/full_matrix/ablations/j8_*.json + j8_summary.json + j8_ablations.sh。待补：Qwen2.5 DocVQA 同像素 600k {none,pre,post} n200（Table 1 脚注 ⁱ 互验）。