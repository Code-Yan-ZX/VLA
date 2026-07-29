# InternVL3-8B 第三模型族接入 digest（rank-before-merge 架构泛化）

> 目的：在 Qwen-VL 两代之外，用第三种视觉-语言架构（InternVL3，pixel-shuffle merger，
> 无 mrope）验证 pre-merger（rank-before-merge）> post-merger 的机制泛化。
> 状态：Part A（核实+下载+实现）完成且 CPU 全绿；Part B smoke 已启动（等卡）；Part C 主矩阵脚本已写好待启动。

## 1. vLLM 支持核实（vLLM 0.19.0 / transformers 4.57.6 / python 3.10.20）
- `vllm/model_executor/models/internvl.py` 提供原生 `InternVLChatModel`；registry 映射
  `"InternVLChatModel" -> ("internvl","InternVLChatModel")`，**同时覆盖 InternVL2/InternVL3**
  （同架构）。无需 InternVL2 兜底。
- vLLM 用**自带** processor：`vllm/transformers_utils/processors/internvl.py` 的
  `InternVLProcessor`（含 `get_image_repl`/`ctx_image_token_id`/`image_seq_length`），
  与 transformers 原生 `InternVLProcessor`（无这些方法）无关 → 自洽。
- **config 兼容性坑（已解决）**：InternVL3-8B 的 config.json 是 legacy `internvl_chat` 格式
  （`auto_map` 自定义代码 + LLM 子配置存于 `llm_config`）。`trust_remote_code=False` 被拒
  （auto_map 强制自定义代码）；而 vLLM 的 `internvl.py` 读 `config.text_config`（line 576/591），
  remote `InternVLChatConfig` 只有 `llm_config` → 直接挂。
  **修复**（runner `_prepare_internvl3_config`，LLM() 前调用）：以 `trust_remote_code=True`
  加载一次 remote config 类（缓存进 sys.modules），给该类加 `text_config -> llm_config`
  **property**。vLLM 后续 AutoConfig 复用同一缓存类 → 每个实例都有 text_config，模型仍解析到
  vLLM 原生、可 hook 的 `InternVLChatModel`（非 repo 远端 modeling 代码）。CPU 验证：
  `text_config→Qwen2Config(arch=Qwen2ForCausalLM, hidden=3584)`，且 downsample_ratio=0.5 /
  ps_version=v2 / force_image_size=448 / select_layer=-1 / use_thumbnail=True /
  max_dynamic_patch=12 / vision_config(image_size=448,patch=14,hidden=1024) 全部就位。
  `image_token_id` 不在 config 上无碍（由 processor 的 ctx_image_token_id=151667 运行时注入）。
- runner `trust_remote_code` 改为 `(family=="internvl3")`：Qwen 两族仍 `False`（逐位不变）。

## 2. 权重
- **主用（已完成）**：默认 HF cache `~/.cache/huggingface -> /data/models/huggingface/hub`
  下 `models--OpenGVLab--InternVL3-8B`，**15GB / 4 shards 齐 / 0 incomplete**（含远端 *.py）。
  下载经 `huggingface.co` + 超时重试 supervisor 完成（直连大 shard 多次挂死，加
  `HF_HUB_DOWNLOAD_TIMEOUT=20`+重试后续传成功；hf-mirror 触发 hf_hub metadata 校验失败弃用）。
- **为何不在 /media/disk2/.../hf_cache**：运行时矩阵脚本均 `HF_HUB_OFFLINE=1` 且不设 HF_HOME，
  靠默认 cache 解析；Qwen 权重也在 `/data/models/huggingface/hub`。放到默认 cache 才能与
  offline 运行时一致（/media/disk2 的 hf_cache 是**数据集** cache）。
- **备份（下载中）**：全局 `VLLM_USE_MODELSCOPE=True`（Qwen 走 modelscope），故另起 modelscope
  下载 `OpenGVLab/InternVL3-8B` 到 `/data/models/modelscope`（慢，~2.7MB/s，作冗余）。
  **internvl3 运行时用 `VLLM_USE_MODELSCOPE=False` 指向已完成的 HF cache**（局部覆盖，不动 Qwen）。

## 3. Hook 点与占位符机制
- **merger 结构**：ViT(InternViT-6B-448px) → drop CLS → reshape `[B,h,w,c]`（h=w=32）→
  `pixel_shuffle(0.5)`（无参 2×2→1，c→4c）→ `mlp1`（LN+Linear+GELU+Linear，4c→llm_hidden）。
  每图动态分辨率切 T 个 448×448 tile（+thumbnail），每 tile `U=(32/2)²=256=num_image_token` 个 merged token。
- **PRE**（`setup_pre_merger_internvl`，wrap `model.extract_feature`）：复用原生 `pixel_shuffle`
  得 `[B,U,4c]`，对每个 2×2 unit（4c 向量）打 L2 分（= 4 patch 的聚合 L2），**每 tile top-κ**，
  仅对 survivor 跑原生 `mlp1` → `[B,k,llm_hidden]`。真 rank-before-merge（mlp1 少算 token）。
- **POST**（`setup_post_merger_internvl`，wrap `model._process_vision_input`——InternVL 无
  `_process_image_input`，此为对位方法，返回每图 split）：原生全量后对 merged token L2 top-κ。
- **占位符**：`<img>` + `<IMG_CONTEXT>×N` + `</img>`（token id：`<img>`=151665 /
  `</img>`=151666 / `<IMG_CONTEXT>`=151667；vocab=151674）。replacement 是 `PromptUpdateDetails`
  （`.full`=该字符串，`.is_embed` 仅标记 `<IMG_CONTEXT>`）。`_make_internvl_prompt_patch` 缩放
  **内层 `<IMG_CONTEXT> run**（保留两个 wrapper），用 `PromptUpdateDetails.select_text` 重建。
- **计数精确对齐（任意 r）**：三处统一用每-tile keep `round(U·(1−r))`：
  processor=`num_patches×round(256·(1−r))`；pre=每 tile `round(256·(1−r))`；post=`(n//U)×round(256·(1−r))`。
  主矩阵 r∈{0.75,0.875} 时 256·(1−r)∈{64,32} 为整数 → 与 `round(T·U·(1−r))` 也精确相等。
- **无 deepstack / 无 mrope**：LLM=Qwen2（1-D RoPE），mrope 修复 family-gate 到 qwen2vl，
  internvl3 永不触发；占位符收缩**无需位置修复**（mrope 无关性，Part B.3 待 smoke 再确认）。
- **guard**：internvl3 拒 `--mask-ranking swap` / `--visionzip-style` / `--mode hybrid` /
  `--qa-lambda>0`（皆 Qwen merger/deepstack/mrope 专属机制，未移植）。

## 4. CPU 验证（零回归）
- `py_compile` OK。
- dry-check **internvl3 ALL PASS**：processor patch 落 `InternVLMultiModalProcessor`；
  占位符 N=512→128（wrapper 保留）；pre `[2,16,16]→[2,4,16]`（1 个 extract_feature wrap，r=0 原样直通）；
  post `[32,16]→[8,16]`（1 个 _process_vision_input wrap）；期望 wrap 数 pre=1/post=1，无 deepstack/mrope。
- dry-check **qwen3vl ALL PASS**（4 merger hooks，deepstack INCLUDED）与 **qwen2vl ALL PASS**
  （1 merger hook，deepstack OMITTED）——mask/swap/hybrid 逻辑与期望 hook 数与改动前一致，Qwen 两族零回归。

## 5. Smoke（Part B，GPU）
- 状态：`scripts/internvl3_smoke.sh` 已启动，轮询等卡（阈值 >40000MiB free 自串行化）。
  当前 GPU 被 `qwen3vl --benchmark docvqa --mode none` full-split cell 占满
  （PID 1873918，43.4GB，util 100%，~0.7GB free）——即 R2/J7 campaign；smoke 排队等其释放。
- **CPU 预去险（已过）**：runner 的 `_prepare_internvl3_config` patch 后，vLLM `ModelConfig`
  接受该 config：`architectures=['InternVLChatModel']`、`is_multimodal=True`、
  `hf_config.text_config=Qwen2Config(hidden=3584)`、downsample_ratio=0.5、ps_version=v2
  → 架构解析到 vLLM 原生可 hook 的 internvl.InternVLChatModel。GPU 端仅剩权重装载/hook 运行时/
  embed-scatter 计数匹配待 smoke 验证。
- 数字：**待补**（none gqa n16 / pre·post gqa n16 r0.75 / pre·post textvqa n16 r0.75）。
- gate：none gqa ≥0.45（预期 ~0.6+）；pre/post ≥0.3 不崩；textvqa 上 pre>post = 机制泛化最小信号
  （若 post≈pre 或 post 胜，如实收窄架构泛化声明）。

## 6. 主矩阵（Part C，GPU）
- `scripts/internvl3_main_matrix.sh` 已写好（+x）：none/pre/post × {textvqa,docvqa,ocrbench,gqa}
  @ r0.75 + textvqa/docvqa @ r0.875；full split（textvqa 5000 / docvqa 5349 / ocrbench 1000 /
  gqa 12578）；docvqa `--max-pixels 4000000` 防巨图 OOM；mns4/chunk500 稳健参；幂等断点续跑
  （skip_ratio≤0.25 跳过，>0.25 safe-flags 重试）；末尾官方 rescore（official_scorers；OCRBench
  `category`/`question_type` 取 jsonl **顶层字段**）→ `runs/internvl3/`。
  自含：先等 4 shards（HF cache）再等空闲 GPU。
- 状态：**待 smoke 通过后启动**。

## 7. 硬约束遵守
- 未改 `baselines_hf.py`（cascade agent 在用）。Qwen 两族 dry-check 零回归、行为 guard 隔离。
- **未做任何 git commit**。GPU 轮询自串行化。下载断点续传。

## Smoke 结果（2026-07-29，n=16，官方 containment，gmu 0.55）

| cell | acc | skip | ptid |
|---|---|---|---|
| none gqa | 0.500 | 0 | 2227 |
| pre gqa r0.75 | 0.562 | 0 | 607 |
| post gqa r0.75 | 0.563 | 0 | 607 |
| pre textvqa r0.75 | **0.750** | 0 | 614 |
| post textvqa r0.75 | 0.375 | 0 | 614 |

**判读**：sanity 过（none ≥0.45）；**模式与 Qwen 家族完全同构**——text-dense pre 大胜（+37.5pp，post 毁文字同 pattern）、object-centric GQA tie（0.562 vs 0.563）。pre-merger selection 泛化到第三架构族（pixel-shuffle merger，无 mrope）。InternVL3 token 数更多（dynamic tiling 256/tile，native ptid 2227）。注：gqa pre 首跑 hang（疑并发争用），mns2/chunk8 重跑正常。主矩阵待 cascade gate 后启动（GPU 优先级=cascade 裁决）。
