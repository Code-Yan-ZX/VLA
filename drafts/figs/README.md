# 图片投放处
规格：drafts/figs_spec_for_user.md（数据源 + 设计建议 + 陷阱）
命名（LaTeX 占位已按此写好，放入即嵌）：
- fig1.pdf  方法示意（代码版双面板 pipeline；当前为 layout proxy）
- fig2.pdf  三族 pre−post Δ 分组柱状图（头条图）
- fig3.pdf  保留率×差距 2×2 曲线
PDF 矢量优先（SVG 亦可）；caption 草稿见 paper_acmmm.tex 内 FIG 占位块。

FIG:1 生成器：`drafts/figures/gen_fig1_rbm.py`。当前图像遮罩只用于排版检查，
投稿前必须用实测 merger-input/output L2 网格替换；输入截图因授权问题不入库。
服务器端真实数据捕获、校验和三图重绘任务说明见
`drafts/figures/real_data_pipeline/CLAUDE_CODE_PROMPT.md`。
