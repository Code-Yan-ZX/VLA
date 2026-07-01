# DECISIONS.md — 自主决策日志（append-only）

> 格式：`日期 | 阶段 | 决策 | 理由 | 影响`
> 主窗口自主决策落此；你随时可回看/否决。被否决的也记（含新方向）。

---

## 2026-07-01 | P0 | 新建专用 `vtc` conda env（python 3.11 + torch cu128） | 三既有 env（fastv/qwen3vl/qwen3vl_clean）均缺 accelerate/pillow/mmcv 且绑别项目栈；base 待 P1 末定，复用会污染或被绑死；研究项目长期可复现性优先 | env 构建脚本 `scripts/build_env.sh`，~2.10/cu128；base 专属依赖（mmcv/flash-attn/qwen）延后 P1/P2 加 | 后台构建中

## 2026-07-01 | P0 | base 专属依赖（mmcv/flash-attn）不在 P0 安装 | base 尚未选定，mmcv 是 LLaVA 专用且编译重，flash-attn 编译耗时；P1 选定 base 后按需装 | P0 env 为"通用骨架"，P1/P2 再补 base 栈 | —

## 2026-07-01 | P0 | vtc 落地为 torch 2.11.0+cu128 / transformers **5.12.1**（比预期 4.x 新）；采用"双 env"策略 | cu128 index 当前发 torch 2.11；transformers 已进 5.x major，API 较 4.x 有变；既有 legacy `fastv` env（torch 2.0/transformers 4.31）保留用于在 LLaVA-1.5 上复现 FastV 原始基线 | 新方法与 Qwen2.5/3-VL 走 `vtc`；FastV@LLaVA-1.5 复现走 `fastv`；P1 文献定位时确认 transformers 5.x 下基座加载路径 | —
