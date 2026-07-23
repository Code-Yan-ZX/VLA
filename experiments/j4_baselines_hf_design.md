# J4 — FastV / PyramidDrop baseline harness (HF-transformers)

**状态**: 代码完成 + CPU 自检全过（无需 GPU）。GPU 等价性/冒烟待 A40 空闲（`baselines_hf_check.sh`）。
**产物**: `src/v3_premerger/baselines_hf.py`（harness，未改 runner）、`src/v3_premerger/baselines_hf_check.sh`（等价性+冒烟）。

## 1. 实现摘要
- **为什么 HF**：vLLM V1 的 attention metadata 固定全栈序列长，无法在 LLM 层间改 token 数；FastV（层 K 一次性剪）与 PyramidDrop（层间分段丢）官方实现均为 HF-transformers modeling patch。本 harness 对 **同模型/同权重/同 prompt/同采样/同 scorer** 复刻 runner 协议；论文披露引擎差异（HF eager vs vLLM flash），效率数字仍走 vLLM。
- **协议复刻**（对齐 `v3_premerger_runner.py:main`）：prompt=`question` 原样（短答指令已烤入 subset）经 Qwen chat template + `add_generation_prompt`（== runner `llm.chat` make_msgs）；greedy（runner temp=0 → HF `do_sample=False`/argmax）；max_tokens=32；`--max-pixels>0`→processor max_pixels、`==0`→processor 默认（runner `mm_processor_kwargs` 仅在 >0 传，逐一对齐）；输出 JSON 同字段（model/mode/benchmark/r/acc/n_skipped/mean_ptid_len/per_sample[{id,question,gt,answer,correct,prompt_token_ids,n_image_full,n_image_kept,n_text,(ocrbench:question_type)}]）→ `official_scorers.py` 离线重评分无缝。
- **实现结构**：`mode=none` 走原生 `model.generate()`（HF 标准路径，即 vLLM 等价比对目标）；`fastv/pyramid` 走手写 prefill+decode：
  - *capture*：stub 掉 `model.model.language_model.forward`，让原生外层（视觉编码+merger+`get_rope_index`，族特异含 Qwen3-VL deepstack）备好 `inputs_embeds`+`position_ids` 后在 LM 边界截获——**不复刻族特异 embedding，零族差异风险**。
  - *pruned prefill*：手动遍历原生 `LM.layers`（复用其 layernorm/MLP/RoPE/lm_head，仅自驱层序与层间剪枝），`attn_implementation="eager"` 取 `output_attentions`；剪枝层用 `rank_keep_indices` 选 top-k，`index_select` 切 hidden/position_ids/image_mask + **切已写入的 KV cache（0..idx 层）**，重建 causal mask（Qwen2.5-VL-7B `use_sliding_window=False`、Qwen3-VL-8B 无 sliding → 纯因果，启动 assert 护栏）。
  - *decode*：KV cache 已裁到 L′，逐 token 追加，position=max+1 三轴广播，greedy 至 EOS/32。
- **CPU 自检（`--dry-check`，tiny 随机模型，无权重/GPU）全过**：纯函数单测（ranking/prune-plan/keep_equiv）；**手写 loop @ r=0 与原生 forward 逐位一致 maxdiff=0.00**（最强正确性证据）；FastV r=0.5 img6→3/L12→9 且各层 cache 裁切正确；Pyramid 分段 keep[4,3,2] fired@[1,3,5] 正确；端到端 greedy 解码通；capture 路径（原生视觉+merger+get_rope_index→scatter）通。

## 2. 与官方 repo 逐式对应
| | 官方（URL） | 本 harness |
|---|---|---|
| **FastV** | `github.com/DL-Prism/FastV` `src/transformers/.../modeling_llama.py` INPLACE 分支：`avg=mean(attn,dim=1)[0]; last=avg[-1]; img=last[SYS:SYS+IMG]; top=img.topk(RANK).indices+SYS; keep=cat(text,top,text).sort(); hidden=hidden[:,keep]` | `rank_keep_indices`：`attn[0].mean(0)[-1]`（head 均值+末 query 行）→ image 列 → `topk(round(n_img*(1-r)))` → `keep=sort(非image ∪ top-image)`。SYS/text 推广为「所有非 image 位恒留」。**差异已注**：官方读层 K−1 attention 并在层 K 前剪（off-by-one）；本实现按论文「层 2 后剪」用层 K 自身 attention（task 指定）。`fastv_config` 默认 K=2、r 可配（`--fastv-k/--r`）。 |
| **PyramidDrop** | `github.com/Cooperx521/PyramidDrop` `llava/model/modeling_llama_pdrop.py` `pdrop_rank_drop`：`layer_list=[8,16,24]`（=L 的 25/50/75%）；`image_tokens=int(N*ratio[cur]); keep=int(N*ratio[cur+1])`；attention（末 instruction token→image key，head 均值）`topk(keep)`；`new_embeds=cat(pre,top,post)` | 段界 `round({.25,.5,.75}*L)`（28 层→[7,14,21]，36 层→[9,18,27]）；段 0/1/2 末层（b−1）用该层 attention 同式 ranking 丢到 `round(n_img0*ratio[1/2/3])`；`--pyramid-ratios` 默认 `1.0,0.75,0.5,0.25`。**差异已注**：官方用**下一层** Q/K 重算 ranking、用 floor（int）；本实现复用**当前层** output_attentions（同 text-query→image-key 语义）、用 round（对齐 runner `round(full*(1-r))` 口径）。官方默认 λ=0.5→[1.0,0.5,0.25,0.125]，本用 task 指定公平预算表。 |

## 3. 等效 keep 折算（PyramidDrop 公平口径）
`keep_equiv = Σ_s(ratio_s · L_s) / Σ_s(L_s)`，`L_s`=第 s 段层数；等宽 4 段 → `mean(ratios)`。
默认 [1.0,0.75,0.5,0.25] → **keep_equiv=0.625 → r_equiv=0.375**（即等效于 uniform 方法 keep 62.5%/r=0.375 的 LLM token 预算）。代码 `pyramid_keep_equiv()` 带注释；pyramid 模式输出 `r`:=r_equiv、`pyramid_keep_equiv`、`pyramid_ratios`。每图 `prompt_token_ids`:=`round(n_text + n_img_full·keep_equiv)`（层均有效 token），与 fastv(`L_after`)/none(`L`)同义，`mean_ptid_len` 跨法可比。

## 4. 等价性 / 冒烟执行步骤（GPU 空闲后跑 `baselines_hf_check.sh`）
1. **STEP 0（CPU，随时可跑，秒级）**：`python baselines_hf.py --dry-check` — 逻辑自检（已本地通过）。
2. **STEP 1（GPU，等价性 gate）**：runner(vLLM) `--mode none` 与 HF `--mode none` 同跑 GQA subset **n=16**（同 model/prompt/sampling/pixels/seed0），内联 python 按 id 比对逐样本：`answer_match` 应 **≥15/16**（残差=引擎 ε）。脚本打印 `[equiv] ... PASS/CHECK`。**这是信任 HF harness 的 go/no-go**。
3. **STEP 2（GPU，冒烟占位）**：FastV `--mode fastv --r 0.5 --fastv-k 2` n=8、Pyramid `--mode pyramid --pyramid-ratios 1.0,0.75,0.5,0.25` n=8，确认真权重端到端跑通+出兼容 JSON（`[done] ... acc/ptid/keep_equiv/skip`）。**不计时不 claim**（效率用 vLLM）。
4. 命令：`bash src/v3_premerger/baselines_hf_check.sh`（幂等、无 set -e、sparse 行可喂 Monitor；默认 Qwen3-VL，跨族改脚本内 `MODEL=Qwen/Qwen2.5-VL-7B-Instruct`）。

## 5. 已知风险 / 估算（A40-46G, bf16, batch1, eager）
- **显存**：权重 8B≈16GB / 7B≈14GB；KV（GQA，L≈1500）≈0.5GB；**output_attentions 物化 attention 阵（fp32 softmax）= heads·L²·4B**：L≈1000→128MB、L≈1500→288MB（单层，可接受）；**但 L≈4000→2GB、L≈16000→~32GB 会 OOM** → **大图基准（DocVQA）必须 `--max-pixels 1500000`**（同 runner，post-merger L≈1500）。合计 ≈18–22GB，A40 充裕。PyramidDrop 官方 README 亦注「依赖显式 attention 阵，不兼容 FlashAttention」。
- **速度**（量级，correctness-first，效率非卖点）：decode 受显存带宽限 ≈16GB/~500GB/s≈32ms/tok→32tok≈1s；prefill L≈1000 eager≈1–2s。GQA/TextVQA（L≈600–900）**≈2–4s/样本 → n=200 ≈ 7–13min/cell**；DocVQA（max_pixels，L≈1500）**≈4–8s/样本 → n=200 ≈ 13–27min/cell**。mode=none 为单加载统一也走 eager（略慢于 sdpa，等价性 ε 不受影响）。
- **确定性**：temp=0 greedy，跨引擎 ε 来自 kernel（eager vs flash），非随机 → 等价性 ≥15/16 阈值即为此设。
- **待 GPU 验证项**（dry-check 覆盖不到）：真权重数值、真 processor 图像展开计数、M-RoPE 真图 position、bfloat16 下 attention 阵精度、generate() 与手写 decode 的 EOS 行为一致性。
- **scope 红线**：单架构分别验证（Qwen3-VL / Qwen2.5-VL），不跨模型宣 SOTA；baseline 数字仅同模型同 harness 对比。
