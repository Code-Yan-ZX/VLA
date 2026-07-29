# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：**Rank-Before-Merge → ACM MM'27**（user 批准强化包）
> 最近更新：2026-07-30 · **InternVL3-8B 主矩阵完成（官方指标，泛化 claim 成立）**；FastV k3 起跑中 → 其后幂等补 none_docvqa 1 cell。

## ★ 实验已完成
- **Qwen 主表 26/26（官方·完整 split）**：text-dense pre 全胜（Q3 TextVQA +38.4/DocVQA +24.3/OCRBench +363pts；z≥12）；GQA post 显著微胜 +2.6–2.8pp（量级 1/10，无 crossover）。digest j7_main_table.md
- **InternVL3-8B 主矩阵（官方·full split·15/16 + backfill 1）**：pre−post text-dense **+37.4pp(TextVQA 0.789/0.415) / +34.6pp ANLS(DocVQA 0.728/0.382) / +432pts(OCRBench 753/321) —— 与 Qwen3 量级同构**；GQA tie（0.599/0.603，红线持）；none 锚 0.834/852/0.629；r0.875 pre 0.723/0.505 vs post 0.306/0.245；pre ptid≡post ptid（同预算）。缺 wave1 none_docvqa（bug 已修，待幂等补跑）。official: runs/internvl3/internvl3_official_summary.json
- **Baseline/机制/效率**：FastV 胜 TextVQA/GQA/DocVQA-600k、RBM 胜 OCRBench；Jaccard≡1.000 双架构（因果=选择级）；25% 保留 +68% 吞吐。digest j7hf_baselines_n500.md / r1_1_swap_jaccard.md
- **Cascade gate = NO-GO**：cas 8/8 cell 落后 max(pre,fst) 8–18pp，两 training-free 选择器串联不互补（pre 不动点第四证）→ 方法冻结 plain RBM、入负结果节、full_splits 不执行。digest cascade_gate.md
- **Bug 修复**：runner 对 internvl3 关 max_pixels kwarg 转发（cap 走既有 PIL 预缩放），pre/post docvqa 线上验证通过（f2ff06d）。

## ★ 待办（按序，GPU 串行）
1. **FastV k3 同 scope（进行中）**：agent 起 `r2b_fastv_k3.sh`（仅 qwen3vl+qwen2vl）→ digest r2b_fastv_k3.md
2. **none_docvqa 补跑**：k3 后 `bash scripts/internvl3_main_matrix.sh` 幂等只补 1 cell → internvl3 digest 补完整表（experiments/internvl3_main_matrix.md）
3. **论文重构 ACM MM 版**：约束见 notes/acm_mm27_cfp.md（≤8+2pp、acmart sigconf、双盲、截稿 ~2027-03/04、track=Foundation Models area、**framing 锚多模态贡献禁纯效率**）；内容=方法泛化 merger-equipped VLMs + InternVL3 泛化表 + §6 cascade NO-GO；审稿模拟二轮 nature-reviewer
4. **投稿前升级 user**（charter 强制）

## ★ 红线（判据锁定 DECISIONS.md）
不跨模型宣 SOTA、不写 beats existing methods；GQA 只报微胜/tie·无 crossover；VZ 官方数仅 mismatched 锚；预注册判据不改（cascade/QA gate 已兑现 NO-GO）；claim 标配置边界。

## ★ 约束/资产
env qwen3vl_clean（vllm 0.19 V1）；权重齐（Qwen3/2.5/InternVL3）；runner=src/v3_premerger/v3_premerger_runner.py；HF harness=baselines_hf.py；GPU 空闲只认 mem<6000；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim 推翻/投稿前。手册 ORCHESTRATION.md。
