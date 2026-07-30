# Round-2 mock peer review — paper_acmmm.md (ACM MM'27 submission draft)

> 2026-07-30 · review sub-agent（nature-reviewer skill；referee 视角，非作者 rebuttal、非编辑决定）。
> 审稿对象：`drafts/paper_acmmm.md`（"Rank Before You Merge: A Stage Law for Training-Free Visual
> Token Compression in Merger-Equipped Vision-Language Models"，Main Technical Track / area
> "Multimedia Generative and Foundation Models"，≤8+2pp，双盲）。
> Grounding 材料：`experiments/{internvl3_main_matrix, r2b_fastv_k3, cascade_gate, j7_main_table,
> j6_efficiency}.md`、`notes/acm_mm27_cfp.md`、`STATE.md` 红线节、第一轮审稿 `drafts/pre_submission_review.md`
> （2026-07-28，针对 paper_v4）。三位审稿人仅 **emphasis** 不同，共享同一事实基（skill role-boundaries）。
> 评分校准锚：ACM MM 录用率约 1/4；本稿是单卡 A40、training-free、finding-driven 方法论文，不得按
> "新架构 beats SOTA" 标尺审（STATE 红线：不跨模型宣 SOTA、不写 beats existing methods）。

---

## Review setup

- **Input scope.** 完整投稿稿（markdown 形态，含 SOURCE MAP / 红线自检 / TODO 作者注——作者注明示
  不入正文）+ 5 份实验 digest + venue 约束 + 第一轮审稿。**Round-2 性质**：第一轮 22 项意见的落实复核
  + 重构稿新增主张（三族泛化、FastV-k3、cascade NO-GO、Jaccard≡1.000）的技术审查。
- **Assessment boundary.** 审稿人**未运行代码、未见 per-sample JSON、未见实际图**（稿内全为
  `[FIG:]` 占位）；M1/M2 机制数、router/hybrid/QA-gate 数、逐格 McNemar z 仅经 SOURCE MAP 溯源到
  `drafts/paper_v4.md`，**不在本次提供的 digest 集内**，标记 `Not independently verified from digests`。
  数字抽查为"稿内数 vs digest 数"比对，非"稿内数 vs 原始 run" 比对。
- **Shared manuscript claim summary.** 在 merger-equipped VLM 上把"选择发生在 native merger 之前
  还是之后"隔离为实验变量（iso-model/iso-token/iso-selector），主张：**(1) stage law**——text-dense
  上 query-blind pre-merger 选择系统性胜 post：Qwen 两代 +11.0~+38.4pp（full split，McNemar |z|≥14.6），
  第三族 InternVL3-8B 胜更强的 query-conditioned 层内剪枝 +34.6~+432pts；无 text-dense crossover，
  gap 随压缩深度单调扩大，GQA 至多小幅 post 领先（Qwen +2.6–2.8pp）或 tie（InternVL3 −0.4pp）；
  **(2) 选择级因果机制**——ranking-swap 控制精确复现 pre 精度，两路径 kept-set Jaccard≡1.000（双架构
  四格），gap 100% 归于 merger 重写的 ranking（M1 重排 ρ=0.14–0.36、M2 反文本降权、M3 swap≡pre）；
  **(3) 泛化+效率+负结果闭包**——三族复现、25% 保留 stage-neutral 提速 +68%/1.8–2.5×、四个预注册扩展
  （QA-gate/hybrid/router/cascade）全败 → pre-merger 选择是不动点。方法 RBM = 故意极简的 query-blind
  L2 top-κ；定位"robust default, not uniformly optimal"（FastV-k3 胜 TextVQA/GQA/DocVQA，RBM 唯胜 OCR）。
- **Visible evidence base.** Table 1（Qwen full split 26/26）、Table 2（InternVL3 16/16 zero-missing）、
  Table 3（FastV 最优 K=3 同 scope，Qwen3 两格 full split + 两格 n=200，Qwen2.5 四格 n=200 且 paired Δ
  缺）、Table 4（cascade gate 8/8 NO-GO）、Table 5（效率，单引擎）；§5 机制（M1/M2 n=64、M3 n=200、
  kept-set n=32）；§6 四负结果。
- **Missing materials affecting confidence.** ① InternVL3 无 post-merger-L2 格（post 列算子≠Qwen 表 post
  列算子）；② Table 3 Qwen2.5 paired Δ 与 RBM 同格数缺（`[TODO]`）；③ 全部图为占位；④ InternVL3 bibkey
  待定；⑤ 机制/负结果逐格数无独立 digest（经 paper_v4 间接溯源）；⑥ Qwen2.5-VL 效率未测。
- **第一轮意见落实情况（复核摘要）.** 22 项中：**已解决 15**（R1-1 kept-set Jaccard 已补且双架构≡1.000、
  R1-3 ToMe/AdaptMerge 已补、R1-4 贡献重述为 stage law、R1-5 PatchMerger=concat+MLP 已更正、R1-6/7/8/9、
  R2-2 Qwen3 none-DocVQA 锚已补 0.956@49152/0-skip、R2-5 McNemar 升为主检验、R2-6 FastV K 扫描且用最优
  K=3、R3-1 脚手架清除、R3-2 GQA 收敛为单完整陈述+回指、R3-4 abstract 改 full-split 区间）；**部分解决 5**
  （R1-2 M1/M2 仍仅 Qwen3——以收窄措辞代替补数，属 R1-2 选项(b)；R2-3 feature tap 已逐族说明但**未做**
  1 格敏感性；R2-4 600k 反转仍在 §8 limitations 而非正文 boundary 段；R2-7 引擎等价无容差附录表；
  R3-3 图仍占位）；**未解决 2**（R1-10 第二文本代理、Qwen2.5 效率）。另：第一轮 meta 建议的 cascade
  正向实验**已执行且预注册判据判 NO-GO**（8/8 败）→ 转为第四负结果——科学上诚实，但意味着第一轮设想的
  "正向方法增量"路径实证失败，novelty 辩护更依赖机制与闭包。

---

## Reviewer 1

**Emphasis：technical soundness / technical failings**（仍覆盖全部五轴）。

### Overall assessment
重构稿相对 v4 是一次实质性升级：第一轮最致命的 R1-1（swap 控制从未在 kept-unit 层验证、且被 Qwen2.5
反例证伪一次）已用四格 Jaccard≡1.000（n=32，chance κ/N=0.25）正面解决，且 Qwen2.5 残差被收窄为
"selection-level causal + 排序混杂残差"，措辞诚实。统计协议（paired McNemar 主检验、full split、
官方 scorer、预注册 gate）在同规模工作中属上乘。但**重构稿用 InternVL3 换泛化叙事时，悄悄替换了
对比算子**：Qwen 主表的 post = post-merger L2（iso-selector 成立），InternVL3 表的 post = FastV 式
layer-2 attention 剪枝（query-conditioned，scorer 与 stage 同时变化）——§3.2 引以为傲的 iso-selector
控制在第三族**不成立**，而 abstract/§1/§9 仍写"the law reproduces across three model families"。
这是当前稿最大的技术缺陷。次要缺陷：abstract 的"no text-dense crossover"无限定，与 §8 自报的
Qwen2.5 DocVQA 600k-cap 反转（+8.0pp post 领先）直接矛盾。

### Who would be interested in the results, and why
多模态基础模型的推理系统研究者、VLM 压缩/部署从业者、以及关心"训练好的 merger 到底丢了什么"的
表征学习读者。stage≻scorer 的发现若成立，对所有"在 merger 后做 token 选择"的既有工作（VisionZip、
GlimpsePrune、VScan 等）是一次有解释力的重估，读者面超出纯效率优化。

### Major strengths
1. kept-set identity 是全文最硬的证据：两架构四格 Jaccard≡1.000，chance 0.25——把 M3 从"近乎同义反复
   的结构推论"（第一轮 R1-1 原话）升级为真正的经验判别，且把 Qwen2.5 反例收编为"排序混杂"而非"证伪"。
2. cascade NO-GO 的预注册判据逐字锁定、8/8 失败、退化自测 8/8 与引擎一致性 14/16 先行验证——负结果
   的证据链质量达到"可当正向结果审"的程度。
3. 对照的诚实度罕见：FastV 用最优 K=3、full-split 上承认 FastV 三胜（+17.2/+8.9/+16.2pp）、自己只守
   OCR（+16.5pp）；GQA 单次完整统计陈述；OCRBench skip 方向性注明保守。
4. 效率 stage-neutral（pre/post req/s ≤3%）把"pre 的优势纯在精度/鲁棒"与压缩收益干净分离——
   5/5 抽查数（4.11→6.91 req/s、+68%、0.36→0.23s、InternVL3 2.2/2.5/2.4/1.8×）与 digest 完全一致。

### Major concerns
1. **第三族对比非 iso-selector（致命）。** InternVL3 的 +34.6~+432pts 同时改变了 stage（pre-merger vs
   LLM layer-2）**与** scorer（L2 vs attention）。因此该格无法把 gap 归因于 stage——完全可能是
   query-blind L2 与 query-conditioned attention 的 scorer 差。稿内 §4.3 标题句仍写"beats the stronger
   query-conditioned in-layer pruning stage"，把两个变量并列进一个动词；§8 item 1 虽承认 InternVL3 是
   "accuracy-level corroboration"，但 abstract/contribution 1/结论仍是无定语的"三族确立"。
2. **"no text-dense crossover"与自报反转矛盾。** §8 item 3：Qwen2.5 DocVQA 在 600k cap 下 post 领先
   +8.0pp（0.504 vs 0.424）——这是一次真实的 text-dense crossover，只是发生在非原生分辨率。abstract、
   contribution 1、结论三处"no text-dense crossover"均无限定词。审稿人交叉核对 §8 会直接抓到。
3. **Qwen2.5 swap 残差 +6.7pp 被"解释"而非"排除"。** reverse_indices 排序混杂说明在 Qwen2.5 上
   *相同 kept 集合、不同物理排序* 可移动精度 6.7pp——这反过来说明"kept 单元的 merged 表征无
   stage-dependent loss"在 Qwen2.5 上并不干净（排序本身成了第二变量）。n=32 也偏小。
4. **不动点闭包的证据是单族的。** 四负结果全部/主要在 Qwen3-VL；"pre-merger selection is a fixed
   point; selectors do not stack"是普适句式，证据覆盖 1.x 个族（InternVL3 无任何负结果实验）。

### Technical failings that need to be addressed before the case is established
- **F1 [must-fix]** 给 InternVL3 补 **post-merger L2** 一格（同 25% 保留、同 4M-cap，text-dense 三基准
  即可）恢复 iso-selector；若算力不允许，则把"三族 stage law"降级为"pre-merger L2 在第三族同时击败
  两个不同的 post 侧算子（iso-scorer 的 post-L2 于 Qwen、query-conditioned 层内剪枝于 InternVL3）——
  后者是更强对比但非 stage 单独归因"，并在 abstract/contribution 同步加该限定。
- **F2 [must-fix]** "no text-dense crossover"全面加限定："at each family's native/standard imaging
  configuration"；600k 反转从 §8 脚注级提升为 §5.4 boundary 一句（shallow-regime×mild-merger 交互）。
- **F3 [must-fix]** 不动点闭包句加证据边界："on the families tested"；并明说 InternVL3 未做装饰实验。
- **F4 [must-fix]** Qwen2.5 机制措辞再收一档：+6.7pp 残差下，Qwen2.5 的结论只能是"kept **set** 相同，
  排序混杂未排除"；"merged representations carry no stage-dependent loss"限定为 Qwen3-VL（byte-exact）。
- **F5 [nice-to-have]** kept-set Jaccard 从 n=32 扩到 M3 同调用的 n=200（同一 run 出索引即可，近零成本），
  堵住"为何硬证据 n 小于精度证据 n"。
- **F6 [nice-to-have]** feature-tap 1 格敏感性（Qwen3 block-8 vs final-layer，TextVQA @25% n=200）——
  第一轮 R2-3 只完成了一半（说明未消融）；若惰性则一句话关闭。
- **F7 [nice-to-have]** M2 加第二文本代理（Laplacian 方差），把"单 Sobel 代理"升级为"对代理稳健"
  （第一轮 R1-10，仍未做）。

### Assessment against Nature-style criteria
- **originality**：stage 作为被隔离变量 + 选择级因果归因，在 merger-equipped VLM 压缩文献中确属新问法；
  方法本身（top-k L2）无新颖性，稿内已正确地把新颖性押在轴与机制上。
- **scientific importance**：若 F1 解决，"merger 系统性反文本重排 saliency"是一个会被引用的机制发现；
  当前第三族证据尚不能独立支撑该普适度。
- **interdisciplinary readership**：对多模态/系统/表征三类读者均有钩子，达标。
- **technical soundness**：Qwen 内部链条（Table 1 + M3 + Jaccard + cascade）坚实；跨族链条（F1）与
  跨配置边界（F2）是当前结构性弱点。
- **readability for nonspecialists**：iso-selector 等控制术语密集但均有就地解释；"stage law / fixed
  point"的命名对跨领域读者友好。

### Recommendation posture
**倾向：weak reject（就当前证据形态）**——不是因为结论错，而是第三族主张超出了控制实验能支撑的范围，
且 abstract 与 §8 自相矛盾（F2）。**置信度：4/5**（熟悉该文献与 digest 全链）。F1–F4 全部为措辞级 +
至多 ~2 GPU·h（InternVL3 post-L2 格），**解决后转 weak accept**。

---

## Reviewer 2

**Emphasis：originality + scientific importance**（仍覆盖全部五轴）。

### Overall assessment
本稿最可能的拒稿向量就是"方法不过是 top-k L2 范数，四个负结果说什么都不 work——技术贡献在哪？"。
重构稿对这一攻击的防御**构造正确**：§3.2 把极简性定义为控制设计（"a strong task- or query-aware scorer
would confound scoring quality with stage"），贡献列表三条全是轴/机制/闭包而非方法，§6 把负结果写成
"design space closure"而非"我们失败了"。Jaccard≡1.000 + swap≡pre 的因果归因、以及 cascade 预注册
NO-GO 构成的"实验证明的不动点"，在同量级会议论文中不常见。**但防御有一处承重裂缝**：不动点闭包与
机制 M1–M3 的证据高度 Qwen3 中心化，而普适措辞没有跟上证据边界——novelty 辩护的强度被措辞稀释了。
另一个 novelty 风险：第三族对比算子不一致（与 R1 的 F1 同一事实，但从 significance 角度看更伤——
"三族确立"是 significance 的支柱叙事）。

### Who would be interested in the results, and why
token 压缩方法社区（需要重估"在 merger 后选择"的默认范式）、VLM 架构设计者（merger 训练目标是否
应保留高频信息）、以及负结果/实验方法论读者（预注册闭包作为"用实验逼近不动点定理"的范例）。

### Major strengths
1. **问法新于方法**：把社区默认滑过的 stage 变量隔离出来，且配了因果级控制（swap + kept-set identity）——
   这是"发现型论文"的正典结构，比又一个 +0.5pp 的 scorer 更适合 MM 的 foundation-models area。
2. **负结果即内容**：四连败 + 预注册判据逐字锁定，给出"为什么方法必须这么简陋"的科学解释（pre-merger
   特征纯视觉、query 信息只在 cross-attention 后可得）——把第一轮的"方法太薄"攻击转化为证据本身。
3. **与 AdaptMerge 的关系处理得当**（第一轮 R1-3 已修）：引为"lossy-merger 论题的佐证而非竞品"，
   ToMe→PruMerge→AdaptMerge 谱系与本文"冻结 native merger、只动 stage"的正交性交代清楚，
   QuietPrune 式 in-ViT 剪枝也被显式划界。novelty 侦察兵式审稿人几分钟内能找到的漏洞基本被堵。
4. 数字纪律：5/5 抽查（+38.4pp、+34.6pp/−19.4pp、0.7771 vs 0.5801、cascade 0.595/0.597/0.680、
   6.91 req/s）与 digest 逐一吻合；−19.4pp vs none 主动写进表注与 §8，未藏进 robust 话术。

### Major concerns
1. **"三族确立"是 significance 的承重墙，但第三族是 accuracy-level 旁证而非受控复现。** 若审稿人认定
   InternVL3 对比混入 scorer 变量，contribution 1 的 significance 从"law"缩回"两代一族内的 law +
   第三族同向旁证"——仍是可发表的贡献，但叙事档位要整体下调，否则是 overselling。
2. **不动点闭包的普适度。** "selectors do not stack"作为全文 closing sentence，证据是 Qwen3 的四个
   实验（cascade 仅 Qwen3、hybrid/router/QA-gate 同）。一个跨族装饰实验（哪怕只 cascade 一格
   InternVL3）都不存在。普适句 + 单族证据 = significance 审稿人的标准攻击位。
3. **正向方法增量的缺席。** 第一轮 meta 曾设想 cascade 若 Pareto 胜可成为 CCF-A 级正向贡献；实验
   判 NO-GO，诚实入稿——但这意味着相对 v4，本稿的**正向**贡献清单没有变长，变长的是证据质量与
   泛化 breadth。对 MM 而言这可能刚好够，但评审池里"so what's the method"的声音需要 §3.2 + §6 的
   防御每句都站住。
4. **abstract 的 pp/pts 混排区间**（"+34.6 to +432 pts"）读起来像挑选了最大数——significance 观感
   受损，建议分基准陈述。

### Technical failings that need to be addressed before the case is established
- **F1 [must-fix]**（与 R1-F1 联动）把"三族 stage law"改写为**算子分层表述**：一族内 iso-selector
  受控（Qwen 两代，post-L2）、第三族对更强 query-conditioned 算子的 accuracy-level 胜利（非同算子
  复现）。Contribution 1 与 abstract 同步。若补得 InternVL3 post-L2 格则可用最强措辞。
- **F2 [must-fix]** 闭包与 fixed-point 句加"on the families/configurations tested"；deployment
  guidance 已是正确收口，保留。
- **F3 [must-fix]** Table 3 补完：Qwen2.5 四格 paired Δ（同 run 内配对，禁跨 run 相减）、ptid 对齐
  核验（稿内自报 r2b 的"690"行与 Qwen3 ptid 215.8 冲突的 TODO 必须落地为"≈214 vs 213，approximately
  matched"的确定表述）。**最强基线比较表带 [TODO] 入审 = 直接给 novelty 审稿人递刀**。
- **F4 [nice-to-have]** §3.2 "minimalism-as-control" 再加一句显式所有格（第一轮作者注已计划，未落地）。
- **F5 [nice-to-have]** abstract 区间拆为按基准陈述，消除 pp/pts 混排。

### Assessment against Nature-style criteria
- **originality**：问法与因果控制新；方法不新且稿内承认。辩护成立度 ~80%，裂缝在证据-措辞边界。
- **scientific importance**：text-dense/OCR 工作负载下的 stage 主导性若被三族受控证实，具领域级
  重要性；当前证实度为"两代一族受控 + 一族旁证"，半档。
- **interdisciplinary readership**：foundation-model 效率 + 机制科学的组合对 MM 评审池适配良好。
- **technical soundness**：见 R1；significance 审稿人视角下，F1/F3 是承重项。
- **readability for nonspecialists**："robust default, not uniformly optimal"的自限式定位比 v4 成熟很多。

### Recommendation posture
**倾向：borderline weak accept**——novelty 防御构造正确、证据质量高于同档均值，但 F1–F3 未解决时
会被同池 novelty 审稿人拉回 reject。**置信度：3/5**（对评审池"方法太简"倾向的判断含主观成分）。

---

## Reviewer 3

**Emphasis：interdisciplinary readership / venue fit + readability**（仍覆盖全部五轴；承载用户指定
检视点⑤ ACM MM 多模态 framing）。

### Overall assessment
out-of-scope desk-reject 风险**已基本解除**：标题锚 "Merger-Equipped Vision-Language Models"，
CCS = Multimodality / Vision and language models，关键词含 multimodal foundation models，开篇
即 "Multimodal inference cost is a visual-token problem"，§7 首句把效率显式降级为 "supporting
evidence … not the claim"，area 选择 "Multimedia Generative and Foundation Models" 与 CFP
"inherently multimedia/multimodal" 约束吻合（notes/acm_mm27_cfp.md）。机制叙事（merger 反文本重排）
正是把"纯效率工作"改写成"foundation-model 科学"的正确手段。**残余风险在形态而非 framing**：
全部图为占位、Table 3 带 [TODO]、supp 迁移清单未执行——按 MM'26 proxy 规则（超页/双盲不足 desk-reject、
正文后禁附录）这些是硬门槛。可读性上，稿内仍有少量防御性重复（robust-default 句在 abstract/§1/§4.4/
§8/§9 出现 5 次），但较 v4 的 10 次 GQA 反复已大幅收敛。

### Who would be interested in the results, and why
MM 评审池中最可能给高分的画像：做多模态系统/部署的审稿人（直接受益）、做 VLM 评测的审稿人
（官方 full-split + 预注册纪律的示范价值）；最可能给低分的画像：纯生成模型审稿人（会问"与我的
diffusion/LLM 何干"）——需要 discussion 首段"merger-equipped VLMs treat their native merger as
lossless; it is not"这种跨子领域可引用的结论句来钩住，目前 §8 有，合格。

### Major strengths
1. **多模态锚定到位**：title/CCS/keywords/§1/§8 五处一致地把贡献定位为"多模态基础模型的结构性发现"，
   效率仅作支撑——符合 CFP 防 desk-reject 的首要建议。
2. **claim 纪律可视化**：红线自检清单（虽不入正文）对应的正文痕迹——GQA 单完整陈述 + 简洁回指、
   无 SOTA/beats 措辞、配置边界逐表注——审稿人读到的是一篇"知道自己不能说什么"的稿子，信任感强。
3. **双盲卫生干净**：无作者名/仓库 URL/机构/commit；"code will be released"；`anonymous` acmart。
4. Table 2 表注 (i)(ii) 把跨模型 DocVQA 不可比与 −19.4pp 直接写在结果旁边——比绝大多数同档稿的
   "robust" 话术诚实得多（用户检视点③：未见 robust 话术藏 regression；abstract 亦未暗示 DocVQA 无损）。

### Major concerns
1. **形态上不可投**（与科学无关但致命）：3 图全占位（v4 有生成图但数字已更，须按官方数重绘 QA）；
   Table 3 四个 [TODO]；supp 迁移未执行（stderr 表、McNemar b/c 表、n=200 一致性表、selector
   invariance 表——CFP 禁正文后附录，只能进 ≤50MB supp）；InternVL3 bibkey 待定。8 页密排下
   Table 1（7 数据列 + 5 表注）与 Table 4 并存的空间需模板验证。
2. **defensive 重复残留**："robust default, not uniformly optimal" 5 次、"+16.5 pp over best-K
   FastV" 4 次。第一轮教训（10 次 GQA → 引审稿人盯败绩）应同样适用于 OCR 胜场：陈述一次 +
   回指即可，反复强调"我们没输 OCR"反而暗示心虚。
3. **abstract 信息密度过载**：190 词里塞了三族、pp/pts 区间、Jaccard、fixed point、四负结果、
   效率——非专门读者第一遍抓不住主干。建议"钩子（merger 非无损）→ 定律 → 机制 → 闭包"四句骨架，
   数字降级为各句从句。
4. **跨族对比的披露对非专门读者不够显眼**：Table 2 caption 已写 post = FastV-style 层内剪枝（诚实），
   但正文 §4.3 "cross-family isomorphism" 小节紧接着用"same order and direction"叙述同构性——
   一般读者会把 Table 1 与 Table 2 的 post 列当作同一算子。需要一句"两表的 post 列是**不同的**
   对比算子；第三族用的是更强的那个"放在 §4.3 开头而非仅 caption。

### Technical failings that need to be addressed before the case is established
- **F1 [must-fix]** 投稿形态闭环：3 图按官方数生成并 QA、Table 3 补完或删 Qwen2.5 行只报 Qwen3
  full-split 对比、supp 清单落地、bibkey 定稿、模板渲染验证（超页即 desk-reject）。
- **F2 [must-fix]** §4.3 开头加"两表 post 算子不同"显式句（与 R1-F1 协同，framing 侧）。
- **F3 [must-fix]** abstract 按四句骨架重写；robust-default/OCR 胜场各收敛到 1 主陈述 + 回指。
- **F4 [nice-to-have]** §8 首段补一句面向生成模型读者的可引用结论（"native merger is not lossless"
   已有，提到段首）。
- **F5 [nice-to-have]** Table 5 列头 "mean vis. tokens/req" 与 j6 digest 原表头"平均 prompt tokens"
   不一致——核验 766/582/397/213 究竟是视觉 token 还是含文本 prompt，定稿前二选一（抽查发现，
   数值本身与 digest 一致，唯标签存疑）。

### Assessment against Nature-style criteria
- **originality**：framing 已把原创性押在"结构性发现"，正确；不重复 R1/R2 评估。
- **scientific importance**：对 MM 评审池"刚好够"——前提是形态闭环让审稿人看到完成品。
- **interdisciplinary readership**：多模态锚定充分，out-of-scope 风险低；跨子领域钩子句可再锐化。
- **technical soundness**：接受 R1/R2 结论；本视角补充披露显眼度问题（F2）。
- **readability for nonspecialists**：术语就地解释合格；abstract 与重复度是主要扣分项。

### Recommendation posture
**倾向：weak accept conditional on form closure**——framing 与 claim 纪律已过线，科学问题交给 R1/R2；
但当前稿**以现有形态投出会被形态问题（非科学）拖入低分**。**置信度：3/5**。

---

## Cross-review synthesis

### Consensus strengths
1. **因果证据链是全文资产**：kept-set Jaccard≡1.000（双架构四格）正面解决第一轮最致命项 R1-1；
   swap≡pre（DocVQA 200/200 字节一致）+ M1/M2 方向性 + 引擎自测构成完整归因。
2. **诚实度与预注册纪律**：FastV 最优 K、GQA 单陈述、OCRBench skip 保守方向、−19.4pp 不藏、
   cascade 判据锁定后认败——三人一致认为是同档罕见的可信度来源。
3. **数字可溯源**：5/5 抽查（Qwen3 TextVQA +38.4pp / InternVL3 DocVQA +34.6pp·−19.4pp /
   FastV-k3 0.7771·OCR 0.415 vs 0.5801 / cascade 0.595·0.597·0.680 与 12.5% GQA 0.350·0.395·0.455 /
   效率 4.11→6.91·+68%·0.36→0.23s·InternVL3 2.2/2.5/2.4/1.8×）与 digest 逐一精确吻合，
   稿内派生算术（3.0×/2.6×/6.6×、−4.5/−3.0/−99）复核无误。
4. **venue framing 过线**：多模态锚定五处一致，效率降级为支撑，desk-reject 风险基本解除。

### Consensus technical risks
- 第三族对比非 iso-selector（R1-①、R2-F1、R3-F2 同一事实，三人独立触及）——**最高权重**。
- abstract 无限定 vs §8 自报的 600k text-dense 反转（R1-F2）——内部矛盾，最易被抓。
- 普适闭包句 vs 单族负结果证据（R1-F3、R2-F2）。
- 投稿形态未闭环（R3-F1；Table 3 [TODO] 同时是 R2-F3 的科学问题）。

### Where emphasis differs across reviewers
- R1 视第三族问题为**技术缺陷**（归因无效），给 weak reject；R2 视同一事实为 **significance 档位**
  问题（叙事降档即可，不必降结论），给 borderline weak accept；R3 认为披露存在但**显眼度不足**，
  属 framing 修复。三人修复方案收敛到同一动作（补 InternVL3 post-L2 格 或 算子分层表述），
  分歧仅在"未修时是否足以拒"。
- 对 novelty 攻击的评估：R2 认为防御构造正确（80% 成立），R1 认为防御强度被普适措辞稀释，
  R3 不担心 novelty 而担心形态让审稿人根本看不到防御。

### Broad-interest / significance readout
对 MM foundation-models area 的读者面适配良好：结论（"stage 主导 scorer"、"native merger 非无损且
系统性反文本"）是可被跨子领域引用的命题式发现，而非仅某方法的 +X pp。广域重要性判断**谨慎给出**：
若 F1 以补格方式解决，达 MM 中位偏上竞争力；若仅以措辞降档解决，则为诚实但半档的泛化主张。
（最终广域兴趣判断属编辑职责，审稿人仅就证据-主张匹配发言。）

### ★ Top-5 致命问题（排序）+ 防御策略
1. **第三族对比非 iso-selector → "三族确立"超出受控证据。**
   防御：首选补 InternVL3 post-merger-L2 格（text-dense 三基准 @25%、同 4M-cap，≤2 GPU·h）恢复最强
   措辞；次选算子分层表述（"一族受控 + 一族对更强算子的 accuracy-level 胜利"），abstract/contribution/
   结论三处同步降档，§4.3 开头显式声明两表 post 算子不同。
2. **abstract "no text-dense crossover" 无限定 vs §8 的 600k Q2.5 DocVQA 反转（+8.0pp post 领先）。**
   防御：全部 crossover 表述加 "at each family's native/standard configuration"；反转提升为 §5.4
   boundary 一句（resolution×generation shallow-regime 交互，Q3 不反转），不再只放 §8。
3. **投稿形态未闭环（图占位 / Table 3 [TODO] / supp 未迁移 / bibkey）——非科学但可直接致死。**
   防御：v4 图按官方新数重绘 QA 后接入；Table 3 补 Q2.5 同 run paired Δ 与 ptid 核验（或删 Q2.5 行
   只报 Q3 full-split）；supp 清单（stderr/b-c/一致性/invariance 表）落地；模板渲染 + 超页检查。
4. **不动点闭包普适句 vs 单族负结果证据 + "top-k L2 太简单"攻击面。**
   防御：闭包句加 "on the families/configurations tested"；§3.2 minimalism-as-control 加一句显式
   所有格；若有余量，InternVL3 上跑一格 cascade（NO-GO 亦为跨族闭包证据，与主实验同 harness 低成本）。
5. **机制泛化缺口 + 溯源卫生：M1/M2 仅 Qwen3、Q2.5 swap +6.7pp 残差被解释未排除、kept-set n=32、
   机制/负结果数仅经 paper_v4 间接溯源、Table 5 列头标签与 digest 存疑。**
   防御：Q2.5 结论收窄为 "same kept set, ordering confound not eliminated"，"no stage-dependent loss"
   限定 Q3；Jaccard 扩 n=200（同 run 出索引）；把 M1/M2/M3、router/hybrid/QA-gate、逐格 McNemar z
   从 paper_v4 提取为独立 experiments/ digest 使每个正文数可直查；核验 766/582/397/213 的口径标签。

### must-fix 总数（去重并集）
**8 项**：① InternVL3 iso-selector（补格或降档+显式句）② no-crossover 限定 + 反转提升 ③ 闭包/
fixed-point 证据边界 ④ Q2.5 机制措辞收窄 ⑤ Table 3 补完（paired Δ + ptid）⑥ 投稿形态闭环
（图/supp/bibkey/渲染）⑦ abstract 重写 + robust-default 收敛 ⑧ 溯源 digest 化 + Table 5 标签核验。
nice-to-have 6 项（Jaccard 扩 n、feature-tap 敏感性、M2 第二代理、§3.2 显式句、abstract pp/pts 拆分、
§8 段首钩子句）。

### 距可投稿还差什么（综合判断）
**科学层面：已非常接近。** 第一轮 22 项解决 15/部分 5/未 2，最致命的 R1-1 已正面硬化；新增三族、
FastV-k3、cascade NO-GO 与 Jaccard 证据把 v4 的 CCF-A 距离评估（15–20%）实质拉近——若 top-5 中的
①以补格方式落地，校准区间约 **30–40%**（MM 中位偏上竞争力；仅措辞降档则约 25–30%）。**形态层面：
尚不可投**——图、Table 3、supp、bibkey 四项硬门槛未完成。**剩余工作量 ≈ 2–3 天集中 CPU 工作 +
≤4 GPU·h**（InternVL3 post-L2 格 + Jaccard n=200 + Q2.5 paired Δ + 可选 InternVL3 cascade 一格），
不需要新科学，需要一次收尾冲刺。建议顺序：F①补格 → F⑤补表 → F②③④⑦措辞批处理 → F⑥形态 → F⑧溯源。

---

## Risk / unsupported claims

- **Not independently verified from provided digests**（经 SOURCE MAP 溯源到 paper_v4.md，未直接核对）：
  逐格 McNemar z（43.0/34.7/16.7/−8.0/30.8/15.6/14.6/−5.7）；M1 ρ=0.137/0.332/0.360 与 Jaccard@25%
  0.180/0.243/0.278；M2 ρ=+0.439/+0.155/+0.036 与 Sobel 0.641 vs 0.124、92% vs 35%；§6(a)(b)(c)
  router/hybrid/QA-gate 全部数值；kept-set n=32 与残差 +6.7pp 的原始 run。→ AUTHOR_INPUT_NEEDED：
  上述数应各有独立 experiments/ digest（top-5 第 5 项）。
- **可能的标签不一致（抽查副产物）**：Table 5 "mean vis. tokens/req" 766/582/397/213 与 j6 digest
  表头"平均 prompt tokens"口径存疑（数值一致；标签需核验，非确认性错误）。
- **证据不支持的普适表述**（稿内现有、本审要求收窄）：abstract/contribution/结论的"the law
  reproduces across three model families"（第三族非 iso-selector）；"no text-dense crossover"
  （600k cap Q2.5 DocVQA 反转）；"pre-merger selection is a fixed point; selectors do not stack"
  （证据单族）；"the merged representations of kept units carry no stage-dependent loss"
  （Q2.5 +6.7pp 排序残差未排除）。
- **不可评估项**：实际成图质量（全占位）；8+2pp 排版可行性（markdown 态无法判定）；rebuttal 阶段
  评审池对 "method = top-k L2" 的最终接受度（主观，置信度已标注）。
- **本审稿的边界**：未运行代码、未访问 per-sample JSON、未做 run-level 复算；5/5 数字抽查为
  稿-vs-digest 一致性核验，非稿-vs-实验真值核验。审稿人仅就证据-主张匹配发言，录用概率区间为
  校准性判断而非编辑决定。
