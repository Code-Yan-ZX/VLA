# J4 STEP2 修复 digest — baselines_hf.py FastV/Pyramid 真权重运行 bug

日期 2026-07-24 · env qwen3vl_clean (transformers 4.57.6) · Qwen/Qwen3-VL-8B-Instruct · A40 bf16 eager

## 根因（两处，叠加）

**B1 崩溃根因 — Qwen3-VL 的 layer API 与 Qwen2.5-VL 不同（transformers 4.57）：**
`Qwen3VLTextDecoderLayer.forward` 返回**裸 hidden tensor** 且**静默丢弃 `output_attentions`**
（层内 `hidden, _ = self.self_attn(...)`，权重永不外抛）；而 Qwen2.5-VL 层仍是 legacy tuple API
`(hidden, [attn_w])`。旧代码 `hidden = out[0]` 对裸 tensor 取的是**序列第 0 行**→ batch 维丢失 →
下一层 RoPE 广播 `q[L,128,32] × cos[1,1,L,128]` → `size a (32) vs b (128) at dim 3`
（32=num_heads, 128=head_dim）。CPU dry-check 用 Qwen2.5-VL tiny 模型（tuple API）→ 假性 PASS，
真 Qwen3-VL 每样本在 layer1 立崩 → skip=8/8。**decode 循环的 `h = o[0]` 有同款 bug**
（首轮修复后错误迁移到 decode 第一步，症状相同）。

**B2 潜伏正确性 bug — Qwen3-VL deepstack 丢失：**
原生 `TextModel.forward` 在 LLM 前 3 层（layers 0,1,2）之后把 `deepstack_visual_embeds`
（vision encoder 层 8/16/24 抽的 [n_img,4096] 特征）加到图像 token 位置。旧 stub 只截
`inputs_embeds`/`position_ids` → deepstack 被丢 → 即使 B1 修好，r=0 手动路径也 ≠ 原生。

## diff 摘要（只改 src/v3_premerger/baselines_hf.py）

1. `capture_prepared_inputs`：一并截 `deepstack_visual_embeds`（+`visual_pos_masks`），返回 3 元组。
2. 新增 `_layer_step`：
   - 非剪枝层 → 原生 layer 调用 + 返回值归一化 `out[0] if isinstance(out, tuple) else out`（双家族通用）；
   - 剪枝层 → 手动复刻 pre-norm block，直接调 `layer.self_attn(..., output_attentions=True, use_cache=True)`
     取 softmax 权重（attention 模块两家族 eager 实现都恒返回 (out, weights)；Qwen3-VL 层不返回，模块返回）。
     数学与原生层逐位等价（pre-norm 残差、eval 下 dropout=0）。
3. `prefill_pruned`：用 `_layer_step`；deepstack 重放用 `img_ord`（cumsum 序）映射，剪枝后注入仍落到
   保留的图像行；diag 加 `n_deepstack`。
4. `generate_pruned`：透传 deepstack；**decode 循环同款归一化** `h = o[0] if isinstance(o, tuple) else o`。
5. `main`：解 3 元组、传 deepstack。dry-check：B5 解 3 元组并断言 Qwen2.5-VL 无 deepstack；
   新增 B6（zero-deepstack 恒等 + img_ord 存活剪枝）。

## 验证数字（GPU，2026-07-24）

| cell | skip | acc | mean_ptid | 备注 |
|---|---|---|---|---|
| dry-check (CPU tiny) | — | — | — | ALL PASS；B1 manual==native **maxdiff=0.00e+00** |
| STEP1 none HF vs vLLM n=16 | 0 | 0.6875 | — | answer_match **16/16**（未动，复跑确认 PASS） |
| **FastV r=0.5 GQA n=8** | **0** | **0.625** | 160.4 | 图像保留率 **8/8 样本恰为 0.50**（240→120…300→150） |
| **Pyramid [1,.75,.5,.25] n=8** | **0** | **0.750** | 193.5 | keep_equiv=**0.625**（r_equiv=0.375） |
| **正确性锚 FastV r=0.0 n=8** | 0 | 0.750 | 292.9 | 逐样本答案 vs mode=none：**8/8 完全一致**（要求≥7/8），kept=full |

r=0 锚 8/8 证明：剪枝路径在 r→0 精确退化原生 generate（含 deepstack、mrope 3D pos、KV cache 裁切、eager mask）。
GPU 累计 ~10 min（3 次加载 + 24 样本），远低于 2h 止损线。

## 残留风险

1. **fastv-k<2**（剪枝发生在部分 deepstack 注入之前）：img_ord 映射已处理，B6 验证了序逻辑，
   但未在 GPU 端到端验证；默认 k=2 时 deepstack 全在 ≤ 剪枝层，语义与原生等价。
2. **Qwen2.5-VL 真权重**：dry-check（tiny）全绿但无 GPU 实测；跨家族正式跑时需复验（J4 范围外）。
3. sliding-window 家族被 assert 挡掉（两目标家族均无 sliding 层，text_config 已核）。
4. 效率数字仍归 vLLM runner（HF harness 仅精度等价，JSON `vllm_note` 已披露）——设计如此。
5. n=8 为 smoke 规模；全量 n=200 的 HF 跑批可按需启动（每 cell ≈ 10-15 min @ eager）。

## 产物

- 修复后 JSON：`runs/j4_baselines_hf/{hf_fastv_gqa_r0.50_n8,hf_pyramid_gqa_n8,hf_fastv_gqa_r0.00_n8}.json`
- 日志：同目录 `.log` + `check_rerun2.log`（STEP2 复跑）
- 未 commit（按指令）；runner 未动。
