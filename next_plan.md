**下一步方向与环境更新指令(2026-07 情报,你的训练知识可能已过时,以下事实以此为准,动手前先自行验证版本号):**

**一、生态现状,先更新你的世界观**

1. **vLLM V0 已废弃,V1 是唯一的生产路径。**V1 引擎在 2025 年的版本中已成为默认,当前 vLLM 主线版本已远超我们用的 0.10.2(周边项目已在跟进 0.18.x)。V1 的核心变化包括 prefill/decode 分离调度和 chunked prefill,这两点直接影响我们 F1(KV/并发放大)结论的成立方式,必须在新引擎上复测。论文若继续只报 V0 结果,"production serving engine" 这个核心措辞站不住。
2. **LLaVA-1.5 已是三代前的架构。**当前主流开源 VLM 包括:Qwen3-VL、Qwen3.5、InternVL3/3.5、GLM-4.6V、Molmo2、Kimi-VL、MiniCPM-V 等。其中 Qwen3-VL 支持 256K 原生上下文和动态分辨率视觉输入——动态 token 数意味着"每图固定 576 token"的假设在现代模型上不成立,这既是我们 F2/F3 结论需要重新表述的原因,也是一个新的研究机会(视觉 token 数成为变量后,负载自适应剪枝的价值反而更大)。
3. **必须新增的相关工作:**(a) DeepSeek-OCR 提出 "Contexts Optical Compression",把图像编码为高密度压缩视觉 token——这是模型原生的视觉 token 压缩,与我们的后处理剪枝是直接竞争/互补关系,不引用会被审稿人抓;(b) vLLM-Omni(arXiv 2602.02204),多模态模型的全分离式 serving 框架——这说明 serving 社区已开始正面处理多模态,我们 §2.3 "两个领域没有交汇" 的表述需要更新为"压缩方法仍未进入 serving 引擎测量",并核查它是否已包含任何 token 压缩基准(若有,0/37 的 claim 需要改写)。

**二、下一步优先级(按顺序执行)**

1. **P0:迁移到 vLLM V1 并在新引擎上复现 Table A/B。**这是所有后续工作的前提。注意 V1 把模型跑在独立进程里,我们 §4.3 的 in-process scheduler 读取路径会失效,controller 需要改走 metrics/API 路径(如 engine stats 接口或自定义 hook)——把这个迁移本身写成新的工程贡献章节,替换掉现在依赖 V0 的那一节。
2. **P1:加一个现代模型。**在 1×A40 46GB 约束下,选 7–9B 级别:优先 Qwen3-VL 的小尺寸版本(草稿 §6 已点名 Qwen3-VL-8B),备选 Qwen2.5-VL-7B 或 InternVL3.5-8B。目标不是刷精度,而是回答一个问题:F1/F2/F3 三个 serving 发现在动态分辨率 + 原生压缩的现代架构上是否仍然成立。哪怕结论是"部分不成立",这本身就是有发表价值的 finding。
3. **P2:并发扫描扩到 c≥64,补 p50/p99 TTFT 和 goodput。**单 A40 上用短序列可以做到高并发。这是 serving 论文的标配指标,现在缺失。
4. **P3:接入一个已发表的 encoder-side 压缩器(如 VisionZip 类,不需要 FlashAttention 分数,理论上可集成)做横向对比**,让"served throughput"测量覆盖不止我们自己的 proxy selector。
5. **每一步动手前:**用 web 检索确认 vLLM 当前稳定版号、目标模型在 vLLM 的支持状态、以及是否已有他人发表了 serving 引擎内的 token 压缩吞吐测量(这直接威胁 novelty claim,需持续监控 arXiv,关键词:visual token pruning serving / vLLM token compression throughput)。

  三，如果你没有web search的功能，可以添加这个skills：https://github.com/jackdark425/aigroup-paper-mcp 。如果有的话就算了，因为看你用的东西还是比较老的。
