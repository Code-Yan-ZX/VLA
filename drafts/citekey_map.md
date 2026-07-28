# citekey_map.md — [CITE:*] → @key 映射（paper_v4）

共 16 个占位符（去重），全部已查证（arXiv ID / venue / 年份 / 作者经 WebSearch+arXiv abs 页核实，2026-07）。
bib 文件：`drafts/references_v4.bib`。主窗口按此表统一替换 paper_v4.md 正文。

| [CITE:*] 占位 | 推荐 @key | 出处与选择理由（确认状态） |
|---|---|---|
| [CITE: fastv] | `chen2024fastv` | ✅ **ECCV 2024 (Oral)**，arXiv:2403.06764，Chen et al.（PKU NLP）。验证：arxiv.org/abs/2403.06764；ecva.net/papers/eccv_2024 |
| [CITE: visionzip-yang2024] | `yang2024visionzip` | ✅ **CVPR 2025**，arXiv:2412.04467，Yang et al.（CUHK/JIA-Lab）。注意标题是 "Longer is Better **but Not Necessary**"。验证：dblp.org/rec/conf/cvpr/YangCTWL0J25.html；openaccess.thecvf.com CVPR2025 |
| [CITE: pyramiddrop] | `xing2024pyramiddrop` | ✅ **CVPR 2025**，arXiv:2410.17247，Xing et al.（Shanghai AI Lab）。⚠ 正式发表标题改为 "Conical Visual Concentration for Efficient LVLMs"（bib note 已注明；正文仍用 PyramidDrop 名称，建议保留 arXiv 标题引用）。验证：github.com/Cooperx521/PyramidDrop（CVPR 2025 徽章）；cvpr.thecvf.com/virtual/2025/poster/33817 |
| [CITE: fastervlm] | `zhang2024fastervlm` | ⚠ **arXiv 预印本**（无已确认会议 venue），arXiv:2412.01818，Zhang et al.（PKU）。⚠ arXiv v2 改名为 "Beyond Text-Visual Attention…"，方法名 FasterVLM 来自 v1（bib note 已注明双标题）。验证：arxiv.org/abs/2412.01818；theia4869.com/FasterVLM |
| [CITE: sparsevlm] | `zhang2024sparsevlm` | ✅ **ICML 2025**（PMLR vol.267, zhang25s），arXiv:2410.04417，Zhang et al.（PKU/Panasonic）。验证：proceedings.mlr.press/v267/zhang25s.html；icml.cc/virtual/2025/poster/46297 |
| [CITE: prumerge] | `shang2024prumerge` | ✅ **ICCV 2025**（DOI 10.1109/ICCV51701.2025.02122），arXiv:2403.15388，Shang, Cai, Xu, Lee, Yan（UW-Madison）。验证：arxiv.org/abs/2403.15388；oamonitor OpenAIRE DOI 记录 |
| [CITE: fitprune] | `ye2024fitprune` | ✅ **AAAI 2025**，arXiv:2409.10197，Ye, Wu, Lin, Zhou。标题 "Fit and Prune: Fast and Training-free Visual Token Pruning for MLLMs"。验证：ojs.aaai.org/index.php/AAAI/article/view/34366 |
| [CITE: glimpseprune] | `zeng2025glimpseprune` | ⚠ **IEEE TCSVT accepted**（官方 repo 徽章 [TCSVT]），arXiv:2508.01548，Zeng et al.（NKU HVision）。[UNVERIFIED volume/pages]——卷期未公布，bib 以 article+note 标注。验证：github.com/HVision-NKU/GlimpsePrune；arxiv.org/abs/2508.01548 |
| [CITE: vscan] | `zhang2025vscan` | ⚠ **arXiv 预印本（under review）**，arXiv:2505.22654，Zhang et al.（Tencent AI Lab）。标题 "VScan: Rethinking Visual Token Reduction for Efficient LVLMs"。验证：arxiv.org/abs/2505.22654；openreview.net/forum?id=h6UiecXxlV |
| [CITE: lmms-eval] | `zhang2024lmmseval` | ✅ **Findings of NAACL 2025**（pp.881–916），arXiv:2407.12772，Zhang, Li, Zhang, Pu 等。验证：aclanthology.org/2025.findings-naacl.51/ |
| [CITE: qwen3vl] | `bai2025qwen3vl` | ✅ 技术报告，arXiv:2511.21631（2025-11），Bai, Cai, Chen et al.（Alibaba Qwen）。验证：arxiv.org/abs/2511.21631 |
| [CITE: qwen25vl] | `bai2025qwen25vl` | ✅ 技术报告，arXiv:2502.13923（2025-02），Bai, Chen, Liu et al.（Alibaba Qwen）。验证：arxiv.org/abs/2502.13923 |
| [CITE: textvqa] | `singh2019textvqa` | ✅ **CVPR 2019**，arXiv:1904.08920，Singh et al.（FAIR）。标题 "Towards VQA Models That Can Read"（兼 VQA-accuracy 指标来源）。验证：dblp.org/rec/conf/cvpr/SinghNSJCBPR19 |
| [CITE: docvqa] | `mathew2021docvqa` | ✅ **WACV 2021**（pp.2200–2209），arXiv:2007.00398，Mathew, Karatzas, Manmatha, Jawahar。**ANLS 指标亦出自此文**（正文 [CITE: docvqa] 兼指 ANLS，正确）。验证：openaccess.thecvf.com WACV2021 |
| [CITE: ocrbench] | `liu2024ocrbench` | ✅ **Science China Information Sciences** 67(22):220102 (2024)，DOI 10.1007/s11432-024-4235-6，Liu et al.（HUST）。验证：link.springer.com/article/10.1007/s11432-024-4235-6 |
| [CITE: gqa] | `hudson2019gqa` | ✅ **CVPR 2019**，arXiv:1902.09506，Hudson & Manning（Stanford）。验证：arxiv.org/abs/1902.09506 |

## 统计
- 条目数：16（@inproceedings ×10，@article ×2，@misc ×4）
- 已确认正式 venue：**12**（ECCV'24、CVPR'25 ×2、ICML'25、ICCV'25、AAAI'25、WACV'21、CVPR'19 ×2、NAACL'25 Findings、SCIS'24、TCSVT-accepted）+ 技术报告 ×2
- 未确认 venue（arXiv-only）：**FasterVLM、VScan** → bib 以 @misc 收录，**正文措辞无需改**（均属预印本可引）
- [UNVERIFIED] 明细：`zeng2025glimpseprune` 卷/期/页未公布（仅确认 TCSVT 录用）

## 主窗口替换提示
1. 直接 `[CITE: xxx]` → `[@key]`（或按目标模板 `\cite{key}`）。
2. `[CITE: visionzip-yang2024]` 出现 2 处（§相关工作、§消融 309 行），同一 key。
3. PyramidDrop：若审稿要求引用正式标题，改用 CVPR camera-ready 标题（bib note 已含）。
4. vLLM 0.19（221 行）正文无占位符，未新增条目；如需可补 @misc{vllm2023}（SOSP'23, arXiv:2309.06180）。
