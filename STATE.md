# STATE.md — 当前项目状态（主窗口维护，≤30 行）
> 项目：VLM 视觉 token 压缩；当前目标：DCC 2027。
> 最近更新：2026-09-15；DCC 初稿与投稿源包已生成，尚未实际投稿。

## 当前论文
- DCC 权威稿：`drafts/dcc2027_submission_20260914/main.tex`；编译 PDF：同目录 `main.pdf`。
- 标题：*Rank Before You Merge: Training-Free Machine-Centric Visual Token Compression for Efficient Vision-Language Model Inference*。
- 主线：native merger 改写 saliency ranking；RBM 在 merger 前选择完整 native units，并保持 merger、位置、deepstack 与 LLM 接口不变。
- 第二贡献：RankBridge 用 20% 最终预算保护 pre-merger rank，其余由 FastV-k3 选择；Qwen3-VL 锁定 n=200 四基准均较 FastV 提升 0.5–3.2pp。
- claim 边界：只有 TextVQA +3.2pp 达显著；OCRBench 上 RankBridge 仍落后纯 RBM 11.6pp，故写作 complementary signal，不宣称 universal hybrid winner。
- Fig.1 已按用户指定图片替换；Fig.2、Fig.4、Fig.5 使用项目原始 PDF 矢量源，Fig.3 为矢量 task rate--distortion 图。
- 作者：Zhengxing Yan（SUES）、Hao Sun（NJUST）、Chen Guo（ECNU）、Jianpeng Hu（SUES，通讯作者）；单位与邮箱已写入首页。

## DCC 合规与验证
- 官方 `dccpaper.cls`、单栏 12pt、US Letter；总长 10 页，原库 51 篇全部在正文实质引用，无 `\nocite`。
- DCC 2027 为 single blind；四位作者、三家单位、四个邮箱和通讯作者信息已填写。
- `latexmk` 通过；0 undefined citation/reference、0 overfull/underfull；全部字体嵌入；10 页均已渲染检查，无裁切、重叠或图文穿插。
- DCC 2027 截止：2026-10-02 23:59 US Pacific；实际投稿须 user 明确确认。

## 既有权威材料
- TCSVT 稿：`drafts/ieee_tcsvt_submission_20260904/main.tex`；实验、native-mRoPE 修正与 artifact 结论继续有效。
- Word 图源：`drafts/RBM_merger_aware_中文_Fig1更新版.docx`。

## 下一步
- 投稿前由四位作者逐一确认英文姓名拼写、单位和邮箱；需要时补 ORCID。
- 公平的多方法 SOTA 表仍缺同模型/同 rate 运行；现有可比强 baseline 为 FastV。新增 full-split/多方法实验可能超过 6 GPU·h，执行前升级确认。
- 投稿前核对 EasyChair 表单、关键词、利益冲突与最终 PDF；获得明确确认后再实际提交。
