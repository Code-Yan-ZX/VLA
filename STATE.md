# STATE.md — 当前项目状态（主窗口维护，≤30 行）
> 项目：VLM 视觉 token 压缩；当前目标：DCC 2027。
> 最近更新：2026-09-16；证据审计版 DCC 投稿稿已完成，尚未实际投稿。

## 当前论文
- DCC 权威稿：`drafts/dcc2027_submission_20260916/main.tex`；编译 PDF 与投稿源包同目录。
- 标题：*Rank Before You Merge: Native-Unit Selection for Visual Token Compression*。
- 主线：RBM 是 model-specific operational design；三模型主表不再称纯 stage 证据。Qwen3 final-input control 单列，且披露 post score 拼接 main/deepstack，避免 merger-only 因果归因。
- RankBridge 降为 exploratory probe：TextVQA/DocVQA/GQA n=200、OCRBench n=181；删除显著性宣称并披露预设 gate 失败。
- Fig.1 保留用户指定 v2；Fig.2 改为全量 Qwen3 三臂控制图；Fig.3 保留审计案例图；旧 RD/强机制/混合口径图从正文移除。
- 作者：Zhengxing Yan（SUES）、Hao Sun（NJUST）、Chen Guo（ECNU）、Jianpeng Hu（SUES，通讯作者）；单位与邮箱已写入首页。
- 摘要末尾已加入公开代码仓库：`https://github.com/Code-Yan-ZX/qrbm-vlm`；首页作者区与摘要间距已核正。
- 新稿加入 same-HF FastV OCRBench、可复现效率协议、模型特定 tap/顺序/配额差异及明确 limitations；未新增实验或数据。

## DCC 合规与验证
- 官方 `dccpaper.cls`、单栏 12pt、US Letter；总长 10 页，19 篇正文实引，无 `\nocite`。
- DCC 2027 为 single blind；四位作者、三家单位、四个邮箱和通讯作者信息已填写。
- `latexmk` 通过；0 undefined citation/reference、0 overfull/underfull；全部字体嵌入；10 页均已渲染检查，无裁切、重叠或图文穿插。
- DCC 2027 截止：2026-10-02 23:59 US Pacific；实际投稿须 user 明确确认。

## 既有权威材料
- TCSVT 稿：`drafts/ieee_tcsvt_submission_20260904/main.tex`；实验、native-mRoPE 修正与 artifact 结论继续有效。
- Word 图源：`drafts/RBM_merger_aware_中文_Fig1更新版.docx`。

## 下一步
- 四位作者逐一确认姓名、单位、邮箱、通讯作者；核对 EasyChair 关键词、利益冲突与版权。
- 实际外发投稿须 user 明确确认；未经确认不上传。
