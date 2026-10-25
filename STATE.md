# STATE.md — 当前项目状态（主窗口维护，≤30 行）
> 项目：VLM 视觉 token 压缩；最近更新：2026-10-25。

## 当前任务
- 用户指定 `F:\dcc2027_submission_20260916` 准备 CVM 2027 投稿，要求清理防御性写作并审查真实缺陷。
- 已核验：该目录与仓库/远端最后论文文本一致；9/19 新实验报告尚未纳入原稿，故不是最终证据版本。
- 新匿名审阅稿：`drafts/cvm2027_submission_20261025/`；旧 DCC 稿保持原样，尚未对外投稿。
- 官方审稿模板；≤14页含参考文献；摘要/全文截止北京时间 2026-10-27 07:59。

## 已核验的关键问题
- ExpA Post-main 使 DocVQA 方向反转；精确ANLS重评后 .3737 > Pre-final .2837，Δ+9.00pp CI[7.82,10.17]。
- TextVQA 旧评分公式/归一化及分号拼接GT需纠正；已用保存预测、官方10答案和固定版本 EvalAI CPU重评。
- 同HF TextVQA全5000失败记零：RBM .75504，FastV .76826；不可转抄ExpB二值correct作为官方分。
- 逐段改写、图内因果词、模型适配方程、强基线/预算曲线不足均已审查；不以“AI多数审稿”为事实。

## 交接材料
- 总体判断：`notes/cvm2027_revision_plan_20261025.md`。
- 写作审计：`notes/cvm2027_writing_audit_20261025.md`；科学审计：`notes/cvm2027_evidence_review_20261025.md`。
- 官方重评：`notes/cvm2027_textvqa_rescore_20261025.md`；脚本 `scripts/analyze_cvm_expB_official.py`。
- DocVQA 11格精确ANLS重评：`notes/cvm2027_docvqa_rescore_20261025.md`；脚本 `scripts/analyze_cvm_docvqa_official.py`。
- GQA 11格精确重评、OCRBench 15格零差核验完成；同名报告在notes，脚本为 `scripts/analyze_cvm_{gqa,ocrbench}_official.py`。

## 下一步 / 人工决策
- 四任务更正合入最终8页审阅稿，51实引；PDF/源包独立编译、全部8页目视QA、字体嵌入及匿名检查通过。
- 核心claim受新增控制削弱，按ORCHESTRATION §6向用户升级：是否以native-unit方法+条件依赖实证研究继续投CVM。
- 未启动GPU补实验；未提交会议。实际投稿前必须人工确认最终稿/版权（P5）。
- 独立科研主线仍保留9/20的P1方向：同模型同预算细粒度保真与服务性能诊断；未擅自恢复RBM为长期主线。
