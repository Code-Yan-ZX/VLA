# ACM MM'27 CFP 核实（2026-07-29）
**结论：MM'27 官方 CFP 未发布 → 全部要求字段以 MM'26 官方 CFP 为 proxy。**

## MM'27-official（已确认的仅有地点）
- 地点：Hong Kong；日期 "to be announced"；CFP 未出。[acmmm.org](https://acmmm.org/)
- 惯例会期：10 月下旬–11 月上旬；截稿惯例 ~每年 4 月初 → 预计 MM'27 截稿 ≈ 2027-03/04。

## MM'26-proxy（官方 2026.acmmm.org）
1. **时间线**（Technical Track，[cfp-guidelines](https://2026.acmmm.org/site/cfp-guidelines.html)）：abstract 2026-03-25（OpenReview 注册 03-26）→ paper **2026-04-01 23:59 AoE** → supp 04-08 → rebuttal（可选，dataset 页示 06-04 AoE，"勿在 rebuttal 要新结果"）→ notification **2026-07-07** → 会议 2026-11-10~14 里约。
2. **页数**：正文 ≤8 页 + 参考 ≤2 页（参考页仅含参考）；超页 desk-reject；正文后禁附录，supp 可选（≤50MB）。
3. **模板**：`\documentclass[sigconf, screen, review, anonymous]{acmart}` 双栏 + CCS concepts/keywords（[author-instructions](https://2026.acmmm.org/site/author-instructions.html)）。
4. **双盲**：Technical track 强制 double-blind，blinding 不足 desk-reject（去作者名/自引措辞/致谢/链接/supp）。
5. **可复现性**：regular track **无强制 checklist**；reviewer 须 "note if resources (code/data) were released"（[review-process-guidelines](https://2026.acmmm.org/site/review-process-guidelines.html)）；另有独立 Reproducibility Track（≤4+2 页）。
6. **主题 areas**（[cfp-guidelines](https://2026.acmmm.org/site/cfp-guidelines.html)）：强调 "inherently multimedia/multimodal"，out-of-scope desk-reject。与我们最贴合：**"Multimedia Generative and Foundation Models"**（首选），备选 "Multimodal Fusion" / "Multimedia and Language"。无 "efficient inference/token compression" 专列 area。
7. **其他硬约束**：全程 OpenReview；所有作者须有完整 OpenReview profile（不完成即 desk-reject）；author list 自 abstract 截止后锁定；Technical+BNI+Datasets 合计 ≤10 篇/组；投稿即志愿审稿。

## 对我们 paper 的建议
投 **Main Technical Track, area = "Multimedia Generative and Foundation Models"**；framing 须锚定多模态贡献（VLM 视觉 token 压缩 = foundation model 效率），避免被当纯 CV/NLP 效率工作 desk-reject；效率+机制双贡献契合 reviewer "claims supported + reproducibility" 维度 → 主动开源代码。
