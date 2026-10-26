# 2026-10-26 接管核验：一天补实验

接管时间约 13:04（Asia/Shanghai）。用户要求补完实验与论文，今天 20:00 前交付；这覆盖旧提示词“本轮不改论文”的范围限制，仍不授权投稿。保留原 T0=10/25 19:07:06、GPU 截止 10/26 15:07:06 和总截止 19:07:06。

## 接管时已经完成

- 隔离 worktree 从含 `1600bfe` 的 main 建立，最新原执行提交 `90a5b73`。主工作区已有未跟踪材料不动。
- P1：None、RBM、FastV、官方 Hi-Lo adaptation，TextVQA 全 5000 / OCRBench 全 1000；旧预测复用及官方 CPU 评分已做。
- P2：预先固定的 TextVQA1000 子集和 OCRBench1000 的 25/50/75% 全部预算点；DocVQA500 统一 2M cap 四组齐全。
- Q10 可选 TextVQA 全量扩展：RBM/FastV 的 50/75% 四格已落盘；Hi-Lo50 正在跑。
- P0：32 题预算/no-op答案检查、8 题张量与 deepstack 检查；Hi-Lo此前仅有8题no-op答案检查，不能概括为完整的张量/位置等价证明。

## 接管发现与修正

| 问题 | 证据与影响 | 处理 |
|---|---|---|
| 成本 harness 未使用 inference/no-grad | 与正式质量路径不同；不必要的 autograd 图导致显存/时延失真 | 全部旧成本结论撤回，另建 v2，不覆盖原 raw |
| 每请求新建 MergerTap 未移除 | hook 累积并持有历史特征，不同配置出现不同幸存集 | 仅安装一次、每请求 reset；记录失败与统一样本 |
| FastV 计时漏传 deepstack | 计时副本传 None，正式质量路径传三流 | v2 全程传三流，并与正式 generate_pruned 比较 token IDs |
| Hi-Lo 修改的是传入前的外部 config | from_pretrained deepcopy 后 live 配置未变，旧 50/75% 标签不证明实际预算 | 修改实际 model config；逐请求断言 actual K、保存逐层轨迹 |
| Hi-Lo 旧 TTFT 表失真 | 首轮 streamer 把 prompt 回调当首输出；raw 已重测但表未刷新 | v2 按实际 prompt tensor 识别回调，直接记录首 token 时间 |
| 不同 total 边界与 attention | 旧 harness 排除 processor，Hi-Lo 包含；eager vs SDPA | v2 从已解码 RGB 图像开始，统一 vision SDPA / text eager |
| 质量—成本跨协议混配 | 新计时 attention 与旧完整质量评测不同 | 保存新200题的答案并官方评分，只在同200题上画匹配曲线 |
| 主表漏 FastV TextVQA | 缺 p1_tv_fv_none，而非预测缺失 | 补CPU评分及表生成映射 |
| OCRBench 95% 目标误判 | 684 ≥ 0.95×707=671.65 | 改为 RBM75 达标；共同921题三法均75才达标 |
| OCRBench skip不对称 | RBM/FastV 原日志各79个OOM，ID与预测skip集合完全一致；Hi-Lo全答 | attempted与共同921题并列，不能将54分提升归因选择更强 |
| Hi-Lo和harness实现不同 | vendored模型+SDPA vs TF4.57.6+eager | 主表标明操作性adaptation比较，no-op小样本不代表全量等价 |

DocVQA500 四组 ID 同序等于固定 JSONL，集合等于预定500 IDs；`head500` 文件没有换成全split前500。所有四组0失败。Hi-Lo融合不额外增加保留token：三预算实际K与RBM/FastV相同。

## 资源决策

Q10 父进程 PID1927473 于约13:06被有意 SIGSTOP，当前 Hi-Lo50 子进程继续运行。这样在其结束后优先跑必须的修正 P3，再依据预估耗时决定能否恢复可选 Hi-Lo75；不根据分数决定补哪格。子集完整曲线始终保留，只有六个扩展格全部完成才将全量曲线作为完整结果。

修正 P3：同一200个固定请求（TV100+OCR100）×3重复；先对每请求三次取中位数，再报请求间p50/p95。记录processor/transfer/视觉/选择/融合/TTFT/decode/total、allocated/reserved峰值、整个workload wall及失败。新脚本为 `scripts/cvm24h_timing_v2.py`，旧raw文件仅用于审计，禁止恢复为有效成本数据。

## 原22条指令的边界

P1/P2已实做，但“现代方法完全相同实现/attention”的强条件未达到，报告保留限制；Hi-Lo是官方算法适配，不能据小样本no-op说全量等价。旧P3不满足正确性要求，只有v2门检及完整原始数据通过后才可进入论文。native/mimic smoke显示位置规则能改变7/32答案，不能据此把HF/vLLM全量差异完全归因于位置。空间均匀/随机控制为可选未做，不影响预定主终点；Doc50/75、新模型和训练未做。总报告将保留所有不利结果与未完成格。

复现所需Hi-Lo精确commit及undefined-variable修复已封装在 `scripts/cvm24h_prepare_hilo.py`；协议、预测、固定子集、评分与脚本哈希由 `scripts/cvm24h_manifest.py` 导出轻量清单。数据、权重、完整预测及日志均留本地，不随源包提交。
