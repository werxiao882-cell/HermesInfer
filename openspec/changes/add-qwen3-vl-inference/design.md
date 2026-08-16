# 设计:Qwen3-VL 生成式推理 + VL 专项优化

针对 `proposal.md` 的技术设计。架构参数同已确认的 `Qwen3VLForConditionalGeneration`
(与 Qwen3-VL-Embedding-2B 共享 vision+text 配置,仅多 lm_head 且走 decode)。

## 1. 背景

### 1.1 模型

`Qwen/Qwen3-VL-2B-Instruct`(及 8B),`architectures: ["Qwen3VLForConditionalGeneration"]`,
chat 模型。文本+图像输入,自回归生成文本。架构(与 Embedding 同源):

| 部件 | 值 |
|---|---|
| 文本 decoder | dim 2048 / 28 层(8B 为 36 层)/ 16 头 / 8 KV 头(GQA)/ head_dim 128 |
| MRoPE | theta 5e6, `mrope_section=[24,20,20]`,interleaved |
| Q/K norm | 有(`attention_bias=false`) |
| vision tower | depth 24, hidden 1024, patch 16, temporal_patch 2, spatial_merge 2, out 2048 |
| deepstack | 抽取于 ViT [5,11,17],注入文本层 [0,1,2] |
| VAE | 无(这是 VLM,不是视频) |
| 文本编码 | 输入用 tokenizer + chat_template(无 T5,文本就是 BPE token) |
| lm_head | 有(生成;`tie_word_embeddings=true`) |

### 1.2 与已实现 Embedding 路径的关系

Embedding(pooling)已实现且**单文件自包含**(`models/qwen3_vl.py`):VisionTower、
PatchEmbed3D、compute_mrope_positions、build_multimodal_inputs、EmbeddingHead、
Qwen3VLForEmbedding;`loader_vl.py`(TP-aware);`layers/` 的 MRotaryEmbedding /
VisionRotaryEmbedding / IS_CAUSAL flash。本变更**复用**这些,只新增"生成式"任务层:

| 维度 | Embedding(已实现) | 生成式(本变更) |
|---|---|---|
| runner_type | pooling | generation |
| prefill | 单次,不写 KV | 写 KV cache(图像+文本 token 的 K/V 入 paged cache) |
| decode | 无 | 有(paged attention 走 cache) |
| CUDA graph | 跳过 | 用(decode batch) |
| 输出 | embedding 向量 | token 序列 → 文本 |
| 头 | EmbeddingHead | lm_head + sampler |

### 1.3 为什么不能直接复用 pooling 的 prefill

pooling 的 `_prepare_prefill_vl` 设了 `slot_mapping=None`(让 Attention 跳过 KV 存储)。
生成式必须**写 KV** —— 图像 token 与文本 token 一样进 paged cache,decode 时 paged
attention 按位置读回。故生成式 prefill 要:复用 pooling 的多模态构造(vision tower /
MRoPE 位置 / scatter / image_token_mask),**但补上 slot_mapping 与 block_tables**,
让 `Attention.forward` 走 `store_kvcache` 分支。

## 2. 目标与非目标

### 目标
- 文本/图像/图像+文本输入 → 自回归生成文本,数值与 transformers / vLLM 在 FP 容差内一致。
- 复用 Embedding 路径已建的 vision/MRoPE/multimodal,不重写。
- decode 走现有 paged attention + 连续批 + CUDA graph(纯文本那套直接套,因为图像 token 的
  K/V 在 prefill 已写进 cache,decode 与纯文本无差异)。
- 5 项 VL 优化(§4)至少落地视觉 prefix 缓存与连续批;分离式编码、CUDA graph 扩展按优先级。

### 非目标(v1)
- **视频**:Qwen3-VL 支持视频,v1 只图像。
- **强化 distill/少步**:标准 chat 生成。
- **生成路径 TP>1 加载修复(R-7)**:沿用现状,留后续(见风险)。
- **CPU/MPS**:CUDA-only。

## 3. 提议架构

### 3.1 生成式 prefill 数据流(写 KV)

```
prompt+image ─► build_multimodal_inputs(复用) → input_ids + mm_data
Scheduler.add_sequence() ─► Sequence(token_ids, mm_data, block_size) ─► waiting
Scheduler.schedule()  (连续批,prefill 分支) ─► (seqs, is_prefill=True)
ModelRunner.prepare_prefill_vl_gen(seqs):
  ├── 打包 input_ids + cu_seqlens_q(复用)
  ├── VisionTower(pixel_values, grid_thw) → visual_emb + deepstack(复用,复制)
  ├── scatter visual_emb 进 <|image_pad|> 位置(复用)
  ├── compute_mrope_positions → positions_3d(复用)
  ├── 【新增 vs pooling】block_manager.allocate(seq) → block_table + slot_mapping
  │   (图像 token 与文本 token 同样占 KV slot;num_blocks 按 token 总数算)
  ├── set_context(is_prefill=True, cu_seqlens_q/k, slot_mapping=<非None>,
  │                block_tables, positions_3d, runner_type='generation')
  └── 返回 input_ids
ModelRunner.run_model()  → Qwen3VLForCausalLM.forward(input_ids) → hidden → logits
Sampler(logits, temperature)  → next token  (rank 0)
Scheduler.postprocess()  → append token,检查 eos/max_tokens,未完则回 running
```

### 3.2 生成式 decode 数据流(与纯文本一致)

decode 阶段每步喂 `last_token`,`prepare_decode` 构造 `slot_mapping` / `context_lens` /
`block_tables`,paged attention 从 cache 读 K/V。**图像 token 不再出现**(decode 只生成新
文本 token),故 decode 路径与纯文本 Qwen3 完全一致,直接复用 `prepare_decode` +
`capture_cudagraph` + `paged_attention_decode`。这是 VL 生成式能"免费"复用文本引擎的关键。

### 3.3 新模型 `Qwen3VLForCausalLM`

在 `models/qwen3_vl.py` 增一个类(或新文件 `qwen3_vl_gen.py`),复用 `Qwen3VLForEmbedding`
的 vision tower + 文本 decoder + deepstack,但:

- 把 `EmbeddingHead` 换成 `lm_head: ParallelLMHead`(词表分片,与 Qwen3 一致)。
- `tie_word_embeddings=True` 时 `lm_head.weight = embed_tokens.weight`。
- `forward(input_ids, ...)` 返回 `hidden_states`(hidden),新增 `compute_logits(hidden)`
  走 `lm_head`——对标 `qwen3.py` 的 `Qwen3ForCausalLM` 接口(供 `run_model` 调用)。
- prefill 写 KV(经 context.slot_mapping);Attention 的 `is_causal=True`(文本 decoder 因果)。

复用度:~90% 代码与 `Qwen3VLForEmbedding` 共享(vision/MRoPE/decoder/deepstack),只差头与
forward 返回。

### 3.4 引擎改动

- `ModelRunner.__init__` 分发:新增 `case "Qwen3-VL-2B-Instruct"` / `"Qwen3-VL-8B-Instruct"`
  → 构造 `Qwen3VLForCausalLM`。`runner_type` 仍是 `"generation"`(默认),但配置带 multimodal。
- `ModelRunner.run()`:生成式 + 多模态时,prefill 调 `prepare_prefill_vl_gen`(写 KV),
  decode 调现有 `prepare_decode`。判定:序列有 `mm_data` → prefill 走 VL 分支,否则走原
  `prepare_prefill`(纯文本)。
- `LLMEngine`:新增 `chat(prompts_images, sampling_params)` 入口(对标 `generate`),内部对
  每条输入调 `build_multimodal_inputs` 构造带 mm_data 的 Sequence,再走 `step()` 循环到完成,
  decode 出 token → tokenizer.decode → 文本。
- `Scheduler`:连续批处理容纳带图序列——`add_sequence` 对带 mm_data 的序列走原 KV 容量
  校验(图像 token 计入 num_tokens/num_blocks);`schedule` 不变(prefill 优先、decode 次之)。
  预算:图像 patch 数纳入 `max_num_batched_tokens` 的 token 预算(每图像 token 占 1 KV slot)。

### 3.5 权重加载

生成式 VL 模型走 `utils/loader.py`(生成路径加载器,绕过 weight_loader)。需让它的
名 remapping 认 `Qwen3VLForCausalLM` 的结构(visual.* / language_model.* → custom)。
**最简方案**:把 `loader_vl.py` 的 `_candidate_custom_names` / `_load_param` 抽成共用,
生成路径加载器对 VL 名走 remap + 直接 copy(现状绕过 weight_loader,TP>1 受 R-7 影响)。
v1 单卡可用;多卡 TP 走 `loader_vl`(但生成路径目前调 `load_weights_from_checkpoint`,
需让它对 VL 模型改调 `load_weights_vl` —— 见 tasks T-7)。

## 4. VL 专项优化

### O1. 视觉 prefix 缓存(v1,高价值)

同一图像被多请求复用时(如检索增强、多轮带图),重复跑 vision tower 浪费。按
`pixel_values` 内容哈希(`xxhash.xxh64`,复用 `block_manager.py` 的哈希思路)缓存
`visual_emb` + `deepstack_features`,命中直接取,跳过 vision tower 前向。

- 缓存键:`xxh64(pixel_values.tobytes() + grid_thw)`。
- 命中:`prepare_prefill_vl_gen` 跳过 `self.model.visual(...)`,直接用缓存的 visual_emb。
- 未命中:跑 vision tower,结果入缓存。
- 容量:LRU,按 visual_emb 字节数限(如 2GB)。
- 注:与 `block_manager.py:62-73` 的文本 prefix 缓存"故意禁用"不冲突——那是文本 KV 的
  跨序列复用(因 prefill kernel 忽略 block_tables);视觉特征缓存的是 vision tower 的
  **输出 embedding**(在 scatter 之前),不涉及 KV/paged kernel,可安全复用。

### O2. 分离式视觉编码(v2,中价值)

vision tower 计算重(24 层 ViT × 大图),若与文本 prefill 同批跑,会拖慢 decode 批。
分离式:把图像编码成独立"encode 批",提前跑完入视觉缓存,文本 prefill/decode 批只做轻量。

- 调度器新增 `encode` 队列:带图序列先入 encode(跑 vision tower),完成后入 `waiting`。
- 文本 prefill 批不被 vision 阻塞,decode 批连续不被打断。
- v2:实现复杂度高于 O1,先做 O1 拿大部分收益。

### O3. VL 连续批处理(v1,基础)

现有 scheduler 已连续批(prefill 优先 + decode 次之)。VL 只需:带图序列的图像 token 计入
token 预算与 KV block 预算(图像 token 也占 KV slot)。`schedule` 的 `max_num_batched_tokens`
预算对带图序列按 `len(seq)`(含图像 token)计,自然容纳。无需改调度逻辑,只确认预算口径。

### O4. decode CUDA graph(v1,基础)

`capture_cudagraph` 按文本 decode batch 形状捕获。VL decode batch 与纯文本同形(每序列一个
last_token),图直接复用。唯一差异:VL 的 `block_tables` 含图像 token 的 block(decode 时
context_lens 更长),但 `run_model` 的图重放分支已按 `context.block_tables` 拷贝,无改动。

### O5. TP(v1,基础)

vision 复制、text 分片,与 Embedding 同。生成式 decode 的 KV cache 按 `num_kv_heads//P`
分片(现有 `allocate_kv_cache` 已做)。lm_head 词表分片 + gather 到 rank 0 采样(现有)。

### O6. chunked prefill(可选,v2)

长 图像+文本 prompt 超 `max_num_batched_tokens` 时分块 prefill。现有 `prepare_prefill` 支持
prefix cache 的 `num_cached_tokens`;VL 需保证 chunk 边界不切在图像 token span 中间
(图像 token 必须整块进同一批,否则 scatter 与 MRoPE 位置会错)。

## 5. 测试计划

| 测试 | 文件 | 守护 |
|---|---|---|
| 生成式 VL 一次 prefill 写 KV:图像 token 的 K/V 进 cache | `tests/test_vl_prefill.py` | §3.1 slot_mapping 非 None |
| VL decode:paged attention 读回图像+文本 K/V 出 token | `tests/test_vl_decode.py` | §3.2 |
| `Qwen3VLForCausalLM.compute_logits` 形状 `(total_tokens, vocab)` | `tests/test_vl_model.py` | §3.3 |
| `chat()` 端到端:文/图/图文 → 文本(mock,小 dim) | `tests/test_vl_chat.py` | API 形状 |
| 视觉 prefix 缓存:同图二次命中,跳过 vision tower(计次) | `tests/test_visual_cache.py` | O1 |
| VL 连续批:带图 + 纯文本混合调度不丢序列 | `tests/test_vl_scheduler.py` | O3 |
| 数值对齐 transformers/vLLM(@gpu,小分辨率短输出) | `tests/test_parity_vlgen.py` | FP 容差 |
| 现有 Embedding / 纯文本生成测试全绿 | (不变) | 回归 |

## 6. 风险与开放问题

- **R-1 —— 生成路径 VL 权重加载(TP>1)**。生成路径加载器绕过 weight_loader(见 AGENTS.md
  gotcha 与 Embedding R-7),VL 名 remapping 也需加。v1 单卡用 `load_weights_vl` 改写生成路径
  可行;多卡 TP>1 的生成路径加载修复留后续。
- **R-2 —— 图像 token 与 chunked prefill**。若开 O6,chunk 边界切在图像 span 中间会破坏
  scatter 与 MRoPE 位置。v1 关 O6 或保证 span 不被切。
- **R-3 —— 视觉缓存键**。`pixel_values` 的 tobytes 在大图上较重;可用 grid_thw + 尺寸哈希
  近似(碰撞风险低)。validate。
- **R-4 —— deepstack 在 decode**。deepstack 只在 prefill 注入文本层 [0,1,2];decode 不注入
  (只生成新 token,无新图像特征)。确认 transformers 行为一致。
- **R-5 —— 显存**。图像 token 占 KV slot,长图 + 长上下文 → KV 涨;`gpu_memory_utilization`
  预算需含图像 token。2B 单卡够,8B 需 TP。

## 7. 启动命令与调用方式

```bash
uv sync
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*'])"
uv run python main_vl_chat.py            # 单卡 chat
uv run pytest tests/test_vl_*.py -v
```

```python
config = {"model_name_or_path":"Qwen/Qwen3-VL-2B-Instruct","world_size":1,
          "runner_type":"generation","block_size":256,...,
          "multimodal":{"max_image_patches":16384}}
engine = LLMEngine(config)
out = engine.chat([
    {"image":"./a.jpg","text":"描述这张图"},
], sampling_params=SamplingParams(temperature=0.6, max_tokens=256))
print(out["text"])
```

## 8. 被否决的替代方案

- **新写一套 VL 生成引擎**:否决,复用 Embedding 已建的 vision/MRoPE/multimodal + 文本
  decode 路径,只加任务层。
- **v1 即上分离式编码(O2)**:否决,先 O1(视觉缓存)拿大部分收益,O2 留 v2。
- **生成路径 TP>1 加载修复**:否决(本次),沿用 R-7 现状,单卡优先。
