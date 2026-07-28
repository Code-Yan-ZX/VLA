# J8b — Qwen2.5-VL DocVQA 同像素 600k（n=200，官方 ANLS，2026-07-28）

Table 1 脚注 ⁱ 的 Q2.5 角补齐：none 0.9624 / pre 0.4239 / post 0.5044（ptid 777→226，skip=0）。

**分辨率×模型交互**：Q2.5 native（ptid 4786）pre 0.636 > post 0.526（+11.0pp）；600k cap 下 **post 0.504 > pre 0.424（+8.0pp，反转）**。Q3 同像素仍 pre 0.424 ≫ post 0.251（+17.3pp，j4d）。解读=低分辨率低文字密度下 pre-L2 文字区分力弱、post 池化相对鲁棒；与 budget 曲线（75% 保留 ≈tie、25% pre 大胜）、GQA post 显著微胜同因（信息密度低→merger 损失小→post 不劣）。入稿=Table 1 脚注+§5.4 段，headline 维持 native full split。
产物：runs/full_matrix/ablations/j8b_*.json + scripts/j8b_q2vl_docvqa_cap.sh。
