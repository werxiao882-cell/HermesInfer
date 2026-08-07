# 设计:Qwen3-VL-Embedding-2B 推理支持

针对 `proposal.md` 提出的变更做技术设计。设计锚定在当前代码库;`file:line` 引用指向被改
动的版本。

## 1. 背景

### 1.1 模型

`Qwen/Qwen3-VL-Embedding-2B`(2.1B BF16 参数,Apache-2.0,由
`Qwen/Qwen3-VL-2B-Instruct` 微调而来)为检索 / 相似度产出稠密多模态 embedding。其
`config.json` 标注 `architectures: ["Qwen3VLForConditionalGeneration"]`、
`pipeline_tag: sentence-similarity`——它复用 Qwen3-VL 的 *生成* 拓扑,但以纯 prefill +
pooling head 的方式使用。

架构参数(来自 `config.json`):

| 组件 | 字段 | 值 |
|---|---|---|
| Vision (ViT) | `depth` | 24 |
| | `hidden_size` | 1024 |
| | `intermediate_size` | 4096 |
| | `num_heads` | 16 |
| | `patch_size` / `temporal_patch_size` | 16 / 2 |
| | `spatial_merge_size` | 2 |
| | `in_channels` | 3 |
| | `out_hidden_size` | 2048 |
| | `hidden_act` | `gelu_pytorch_tanh` |
| | `deepstack_visual_indexes` | `[5, 11, 17]` |
| | `num_position_embeddings` | 2304 |
| Text (LLM) | `num_hidden_layers` | 28 |
| | `hidden_size` | 2048 |
| | `num_attention_heads` / `num_key_value_heads` | 16 / 8(GQA) |
| | `head_dim` | 128 |
| | `intermediate_size` | 6144 |
| | `vocab_size` | 151936 |
| | `tie_word_embeddings` | `true` |
| | `rope_theta` | 5,000,000 |
| | `rope_scaling` | `mrope_interleaved=true`,`mrope_section=[24,20,20]`(T,H,W) |
| | `rms_norm_eps` | 1e-6 |
| | `max_position_embeddings` | 262144(实际上下文 32K) |
| Embedding | 输出维度 | 2048(MRL 可截断到 64–2048) |
| | pooling | last-token(instruction-aware) |
| | 归一化 | L2 |
| 特殊 token | `image_token_id` / `video_token_id` | 151655 / 151656 |
| | `vision_start` / `vision_end` | 151652 / 151653 |
| | `eos`(`<|im_end|>`)/ `pad` | 151645 |

推理是 **单次前向(纯 prefill)**:对 chat-templated 的 prompt(系统指令
`Represent the user's input.` + 带 `<|vision_start|><|image_pad|>×N<|vision_end|>`
占位符的用户内容)跑一次 VL 模型,池化最后一个 token,L2 归一化,可选截断到
Matryoshka 子维度。无自回归 decode、无采样。

### 1.2 当前代码库

`myvllm` 是 vLLM 的教学复刻(见 `HowToApproachvLLM.md`)。与本设计相关的事实:

- **面向生成的生命周期**(`llm_engine.py:68` `step()` → `scheduler.schedule()` →
  `model_runner.run()` → `scheduler.postprocess()`)。Decode 是一等公民:
  `scheduler.py:57-93`、paged decode kernel `attention.py:283`、CUDA graph 捕获
  `model_runner.py:406`、带 preemption 的连续批处理 `scheduler.py:96`。
- **可复用的 prefill 核心**:`flash_attention_varlen_kernel`(`attention.py:112`)与
  `prepare_prefill`(`model_runner.py:265-314`)构造变长批处理 prefill 用的
  `cu_seqlens_q/k` 和 `slot_mapping`。这正是一个 embedding 批处理需要的。
- **模型分发**基于 HF checkpoint 的 *目录基名* 做 `match`(`model_runner.py:30-69`)
  ——只支持 `Qwen3-0.6B` 和 `Llama-3.2-1B-Instruct`。
- **RoPE**(`rotary_embedding.py:48`)支持 Llama-3 scaling;**无 MRoPE**。
- **TP 线性层**(`linear.py`)带 per-parameter `weight_loader` 可调用对象——但 `loader.py`
  自己用 `torch.cat` 做 QKV/gate_up merge 后直接 `param.data.copy_(full)`,**绕过**了这些
  callable(`loader.py:76,103,165`)。`ColumnParallelLinear`(`linear.py:84`)把 param 建成已
  分片形状,其 `weight_loader`(`linear.py:97-107`)才会按 `tp_rank` 正确切片——现状下它根本
  没被调用,故 `world_size>1` 在加载期就把 rank-0 切片装到每个 rank 上。对 VL embedding 的
  TP 支持是必须修正的(见 §3.8.2 与 R-7:只改 VL 路径,不动生成路径)。
- **KV cache 池**(`model_runner.py:247`)按 warmup 后的空闲显存定容。对 embedding 来说
  **不需要** KV cache(无 decode、无复用)——pooling 模式应整体跳过,把显存让给 vision
  激活。
- **`Context` 单例**(`context.py:6`)按迭代携带注意力元数据;由 `ModelRunner` 调
  `set_context()` 写入,由 `Attention` / `ParallelLMHead` 读取。
- **无量化、无 MPS/CPU 路径。**
- **公开 API** 走子模块(`__init__.py` 为空),无扁平接口面。

### 1.3 架构错配(以及它迫使我们做的决策)

Embedding 是纯 prefill。项目的招牌优化——PagedAttention **decode**、**CUDA graph**
捕获/重放、**带 preemption 的**连续批处理——在本工作负载下全是死代码。这不是副作用,
而是一条结构性接缝。vLLM 用 `RunnerType`(`GENERATION` vs `POOLING`)共享同一个 prefill
引擎来形式化它。我们采用同一条接缝,理由:

- 把 prefill 引擎(flash attention、varlen、TP、weight loader)作为可复用核心保留;
- 把 decode-only 代码干净地收到 `runner_type == "generation"` 守卫之后。

被否决的替代方案——一个没有 scheduler 的并行 `EmbeddingEngine`——见 §8:它会重复 prefill
批处理与 TP 管线,且对未来 reranker / 分类 pooler 不具扩展性。

## 2. 目标与非目标

### 目标
- 在单次 prefill 内批处理混合 `(文本 | 图像 | 图像+文本)` 输入,每条输入返回归一化的
  `(embed_dim,)` 向量。
- 复用现有 Flash-Attention varlen prefill 路径、TP 线性层、RMSNorm、`torch.compile` 热路径,
  对生成路径行为零改动。
- 支持 last-token 池化(instruction-aware 默认)、mean 池化,以及 Matryoshka(MRL)子维度
  截断(64–2048)。
- 跨 `world_size` GPU 张量并行(vision tower 复制;文本 decoder 按现有方式分片)。
- 在 Qwen 参考样例上与 vLLM/SGLang 数值输出在 FP 容差内一致(使下游余弦相似度能复现模型卡)。

### 非目标(v1)
- **视频**输入(模型支持;v1 只出图像)。MRoPE 的时间轴与 `temporal_patch_size=2` 接好线,
  但只在 T=1 下被实际执行。
- **Reranker**(`Qwen3-VL-Reranker-*`)——输出头不同,单独的变更。
- **权重量化**(INT8/FP8/W4A16)。模型卡的"quantization support"指的是对 *输出 embedding*
  的事后量化(二值/hash),v1 同样 **不** 实现。
- **跨请求的视觉特征 prefix 缓存**——受 `block_manager.py:62-73` 记录的同一设计告警约束,
  以及 prefill kernel 忽略 `context.block_tables`。只有在设计了视觉特征缓存之后才回头。
- **CPU/MPS** 路径——出范围(按 `AGENTS.md`,CUDA-only)。
- **在线服务**——仅离线批 API,无 HTTP server。

## 3. 提议架构

### 3.1 数据流(单条 embedding 请求)

```
AutoProcessor.apply_chat_template([{system: 指令}, {user: [image?, text]}])
        │  prompt_ids,带 <|image_pad|>×N 占位符
        ▼
MultimodalData (pixel_values, image_grid_thw, image_token_spans)
        │
        ▼
Scheduler.add_sequence() ─► Sequence(prompt_ids, mm_data) ─► waiting
Scheduler.schedule()  (纯 prefill) ─► (seqs, is_prefill=True)
        │
        ▼
ModelRunner.prepare_prefill():
  ├── tokenize 已完成;在 packed prompt token 上构造 cu_seqlens_q
  ├── 跑 VisionTower(pixel_values, image_grid_thw) → visual_emb  (B, P', 2048)
  ├── 构造 input_emb:文本 token embedding;把 visual_emb scatter 到图像 span
  ├── 计算每 token 的 (T, H, W) 位置(MRoPE)
  └── set_context(Context(is_prefill=True, cu_seqlens_q/k, slot_mapping,
                          positions_3d, image_token_mask, ...))
        │
        ▼
ModelRunner.run_model()  (pooling 模式下不走 cuda graph)
  └── Qwen3VLForEmbedding.forward(input_emb, positions_3d):
        ├─ for layer idx in 0..27:
        │    emb = decoder_layer(emb, positions_3d)
        │    if idx in deepstack_indexes: emb += deepstack_proj(visual_feat[idx])
        ├─ 最终 RMSNorm
        └─ EmbeddingHead:按 seq 取最后一个 token → (B, 2048)
        │
        ▼
L2 归一化  →  可选 MRL 截断(dim)  →  返回 (B, dim)
Scheduler.postprocess():标记 FINISHED、释放
```

### 3.2 新增子包

#### `src/myvllm/vision/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 重导出 `VisionTower`、`PatchEmbed3D`、`SpatialMerger`、`DeepstackProj` |
| `patch_embed.py` | `PatchEmbed3D`:在 `(T, H, W)` patch 上做卷积,`temporal_patch_size=2`、`patch_size=16`;产出每 patch 一个维度 `hidden_size=1024` 的 token。加 3D 位置嵌入(`num_position_embeddings=2304`)。 |
| `vit.py` | `Qwen3VLVisionBlock` ×24:自注意力(无 causal mask)+ MLP(`gelu_pytorch_tanh`)+ 带 QK-norm 的 RMSNorm(Qwen3-VL ViT 约定)。复用 `layers/attention.py:Attention` 跑 prefill kernel——ViT 序列对 varlen-flash 友好。 |
| `merger.py` | `SpatialMerger`:把 `2×2` 相邻 patch 通过线性 `4*hidden_size → out_hidden_size`(每 patch 4×concat 后 1024→2048)合并。 |
| `deepstack.py` | `DeepstackProj`:线性 `vision_hidden → text_hidden`(1024→2048),作用于选定的 ViT 中间特征,注入 LLM decoder 层 `[5,11,17]`。 |
| `vision_tower.py` | `VisionTower`:编排 patch embed → 24 个 block → 返回中间特征列表(给 deepstack)+ 最终 merged embedding。 |

ViT 注意力是 **对一张图的所有 patch 双向**,且无 KV 复用——因此每张图是一个自包含的
varlen prefill。现有 `flash_attention_varlen_kernel`(`attention.py:112`)直接复用,用
`cu_seqlens` 分隔不同图像(当前没有 causal mask 开关;见 §6 风险 R-3)。

#### `src/myvllm/multimodal/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 重导出 `MultimodalData`、`compute_mrope_positions`、`build_image_inputs` |
| `processor.py` | `build_image_inputs(raw_image)`:按 `preprocessor_config.json`(HF `AutoImageProcessor`)做 resize/normalize,patchify 成 `image_grid_thw`,产出 `pixel_values`(pinned,H2D non-blocking)。 |
| `positions.py` | `compute_mrope_positions(seq_ids, image_token_spans, image_grid_thw)`:按 `mrope_section=[24,20,20]` 的 interleaved 切分,产出三组位置数组 `(T, H, W)`,shape `(num_tokens,)`。文本 token 取自身位置的 T;图像 token 从 patch 网格派生单调递增的 T/H/W。 |
| `registry.py` | `MultimodalRegistry`:每 `Sequence` 的挂载点,存 `pixel_values` + `image_grid_thw` + `image_token_spans`。对标 vLLM 的 `MultiModalDataDict`。挂在 `Sequence` 上(新增可选字段)。 |

#### `src/myvllm/pooling/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 重导出 `EmbeddingHead`、`Pooling`、`MRLTruncate` |
| `pooling.py` | `Pooling` 枚举/模式:`LAST_TOKEN`(默认)、`MEAN`、`CLS`。`last_token` 用 `cu_seqlens_q[1:]-1` 索引取(与 `embedding_head.py:75-76` 的 prefill-gather 同一招)。 |
| `embedding_head.py` | `EmbeddingHead(nn.Module)`:可选线性投影(若模型带 embedding projector;该 checkpoint 看似直接池化原始 hidden states——见 §6 R-1 验证),随后池化、L2 归一化、可选 MRL 截断。若存在分片投影,沿用 `ColumnParallelLinear` 语义。 |
| `mrl.py` | `mrl_truncate(emb, dim)`:切前 `dim` 维并重新归一化。 |

### 3.3 层扩展

#### MRoPE 进 `layers/rotary_embedding.py`

现状:`RotaryEmbedding`(`rotary_embedding.py:48`)+ Llama-3 scaling 分支
(`rotary_embedding.py:69-87`)、`@torch.compile` forward(`rotary_embedding.py:100`)、
`cos_sin_cache` buffer。

改动:新增 `rope_type` 判别(构造参数)→ `"default" | "llama3" | "mrope"` 三选一。`mrope`
分支:
- 接收 **3D 位置** `(T, H, W)` 而非 1D,shape `(B, seqlen, 3)` 或三个 `(seqlen,)` 张量。
- 按 `mrope_section=[24,20,20]` 把 `head_dim=128` 的 rotary 空间切成三段连续区段:T 取
  `[0:24]`、H `[24:44]`、W `[44:64]`;**interleaved** 标志意味着每段内部以
  `(cos0, sin0, cos1, sin1, …)` 交替对应用(Qwen3 约定),而 Qwen2.5-VL 用的是
  `(cos0..n, sin0..n)` 的非交错布局。
- 三段拼合后产出 shape `(seqlen, head_dim)` 的 `cos`/`sin`。

`apply_rotary_pos_emb`(`rotary_embedding.py:4`)当前支持 3D varlen 与 4D batched——扩展为
同时接收 3D-position 形式,按 shape 的 rank 分发。

#### `Attention.forward`(`layers/attention.py:472`)

`Attention.forward` 先存 K/V,再调 `flash_attention_prefill`(prefill)或
`paged_attention_decode`(decode)。prefill 路径已通过 `Context.cu_seqlens_q` 支持 varlen。
**ViT 双向注意力**不需要 causal mask——而当前 flash kernel 在 prefill 时 **不施加**
intra-sequence causal mask(它依赖 `cu_seqlens` 划分序列,序列内部无 mask)。请验证(R-3);
若属实,ViT 与 LLM-prefill 逐字共用该 kernel。新增一个 `Context.is_causal: bool` 字段,LLM
默认 `True`、ViT 默认 `False`,仅在将来加 kernel 因果变体时才用(v1 下若 kernel 本就无 mask
则是 no-op)。

### 3.4 新模型 `src/myvllm/models/qwen3_vl.py`

`Qwen3VLForEmbedding(nn.Module)`,decoder 栈结构与 `Qwen3ForCausalLM`(`qwen3.py:285`)相同,
但:

- 没有 `lm_head` / `ParallelLMHead` 出 logits;改为 `pooling.EmbeddingHead`。
- `Qwen3VLDecoderLayer` 在索引 `{5, 11, 17}` 处新增可选 `deepstack_proj`,在 decoder block
  的残差之后:`emb = emb + deepstack_proj(visual_feat[idx])`。
- decoder 用 **MRoPE**(传 3D 位置)替代 1D。
- 保留 Q/K-norm(`qwen3.py:50-51`)(Qwen3-VL 文本配置与 Qwen3 在此一致——
  `attention_bias=false`,`qkv_bias` 推断为无)。
- `packed_module_mapping` 扩展加入 `visual.*` 条目。

权重 tying(`tie_word_embeddings=true`)对 embedding 路径无关(无 unembed head),但 loader
仍需对缺失的 `lm_head` / `embed_tokens` tied 参数优雅跳过。

### 3.5 引擎改动

#### 3.5.1 配置(`runner_type`)

扩展 `LLMEngine(config)`(`llm_engine.py:25`)消费的配置 dict:

```python
config = {
    ...
    "runner_type": "pooling",          # 默认 "generation"
    "pooling": {"mode": "last_token", "normalize": True, "mrl_dim": None},
    "multimodal": {"max_image_patches": 16384},   # 预算守卫
}
```

`runner_type` 由 `Scheduler` 与 `ModelRunner` 读取。

#### 3.5.2 `Scheduler` 纯 prefill 模式

`Scheduler.schedule()`(`scheduler.py:35`)当前有 prefill 分支(`scheduler.py:42-54`)与
decode 分支(`scheduler.py:57-93`)。在 `runner_type == "pooling"` 下:

- 始终走 prefill 分支;把 `running` 视为空。
- 调度完一批后,所有序列在一步内完成(无 decode 后续),由
  `postprocess()`(`scheduler.py:104`)释放。
- `preempt()`(`scheduler.py:96`)不可达——对其加 assert。
- 绕过 no-progress guard(`scheduler.py:80-91`);改为把"waiting 与 running 都空 → 完成"
  的终止信号返回给 `LLMEngine` 的循环。

批处理策略:按现有 `max_num_batched_tokens` 与 `max_num_sequences` 打包 prompt。对多模态,
额外对整批施加 `max_image_patches` 预算(各 `prod(thw)` 之和),避免 vision tower OOM。

#### 3.5.3 `ModelRunner` pooling 路径

- `__init__`(`model_runner.py:16`):在分发的 `match`(`model_runner.py:32-69`)里加
  `case "Qwen3-VL-Embedding-2B"`;构造 `Qwen3VLForEmbedding`。
- `allocate_kv_cache()`(`model_runner.py:197`):pooling 模式下 **整体跳过**——把池内存让给
  vision 激活。(decode 才需要;embedding 永不 decode。)`BlockManager` / `Scheduler` 仍跑
  (用于 prompt 批处理元数据),但不分配物理 block——或者更简单:每序列分配一个 dummy block
  且永不分页。在 §6(R-2)定夺。
- `capture_cudagraph()`(`model_runner.py:406`):pooling 模式下 **跳过**。`run_model()` 的
  图重放分支(`model_runner.py:354-378`)改为直接 `model(**inputs)`。理由:embedding 批大小
  随图像 patch 数变化;在由此产生的 (batch × seqlen) 网格上做 CUDA graph 捕获 v1 不划算。
- `run()`(`model_runner.py:386`):pooling 模式下返回池化后的 embedding(仅 rank-0,对标
  `model_runner.py:394` 的 rank-0 采样)——若 embedding head 被分片,则用 `dist.gather` 从非零
  rank 收集(由于输出维 2048 很小,通常不分片;保持 `EmbeddingHead` 复制)。
- `prepare_prefill()`(`model_runner.py:265-314`):新增多模态分支:

  1. 把批内 prompt token 拼接 → `flat_ids`,构造 `cu_seqlens_q`(现有逻辑)。
  2. 对每个带 `mm_data` 的 `Sequence`:在其 `pixel_values` 上跑 `VisionTower` → `visual_emb`
     (按 image-order 在批内拼接)。构造 `image_token_spans`,把 `visual_emb` 各行映射到
     `flat_ids` 中的位置。
  3. 构造 `input_emb = embed(flat_ids)`;把 `visual_emb` scatter 进 `image_token_spans` 行
     (覆盖 `<|image_pad|>` 的 embedding——标准 Qwen3-VL 行为)。
  4. 由 `compute_mrope_positions` 计算 3D 位置。
  5. `set_context(Context(is_prefill=True, cu_seqlens_q=..., cu_seqlens_k=...,
       slot_mapping=None, positions_3d=..., image_token_mask=...))`。

- `prepare_decode()` 在 pooling 模式不可达——加 assert。

#### 3.5.4 `LLMEngine.encode()` 公开 API

```python
def encode(self, inputs: list[dict], /) -> list[torch.Tensor]:
    """inputs: [{"text": "..."} | {"image": 路径} | {"text":..,"image":..},
                 可选每条带 {"instruction": "..."}]"""
```

内部:对每条输入,用默认 `Represent the user's input.` 系统提示(或每条各自的 instruction)
跑 `AutoProcessor.apply_chat_template`,构造 `MultimodalData`,包成 `Sequence`,`add_prompt()`。
然后循环跑 `step()` 直到 scheduler 给出完成信号。按输入顺序返回 embedding,L2 归一化,可选
MRL 截断。

`generate()`(`llm_engine.py:95`)不变。

### 3.6 权重加载器(`utils/loader.py`)

`load_weights_from_checkpoint()`(`loader.py:16`):

- 扩展对 `state_dict` 的遍历,识别 `visual.*`、`merger.*`、`deepstack.*` 前缀 → 按名路由到
  vision tower 子模块。这些是标准 `q/k/v/o` + `mlp.{gate,up,down}` 布局,故现有
  `default_weight_loader` 路径即可(`loader.py:76` 的 QKV merge 仅当 *目标* 模块是
  `QKVColumnParallelLinear` 时触发;ViT 的 Q/K/V 是分开的,不合并)。
- 文本 decoder 的 QKV 与 gate_up merge(`loader.py:76,103`)照常适用。
- `EmbeddingHead` 投影权重(若有)走标准路径。
- 复用现有详细加载摘要打印(`loader.py:179-213`);新增"vision tower"小节。

### 3.7 入口脚本 `main_embedding.py`

对标 `main.py:43` 结构:配置 dict(带 `runner_type="pooling"`,目标
`Qwen/Qwen3-VL-Embedding-2B`),构造 `LLMEngine`,对一组 文本/图像/图像+文本 输入调
`encode()`,打印余弦相似度矩阵。复现 README 的参考相似度值作为冒烟测试。

### 3.8 张量并行(TP)

本变更要让 embedding 路径在 `world_size>1` 下**真正可用**(生成路径的 TP 加载现状见
§1.2 与 R-7,不在本次修复范围)。

#### 3.8.1 拓扑:谁分片、谁复制

| 部件 | TP 策略 | 理由 |
|---|---|---|
| `VocabParallelEmbedding`(embed_tokens) | **词表分片**,all_reduce 出 full hidden | 现有 `embedding_head.py:12` |
| 文本 decoder 的 `QKVColumnParallelLinear` | **列分片**(GQA 感知,按头切:Q 用 `num_heads`、K/V 用 `num_kv_heads`) | `linear.py:152`;`num_kv_heads//world_size` 见 `model_runner.py:208` |
| 文本 decoder 的 `MergedColumnParallelLinear`(gate_up) | **列分片** | `linear.py:113` |
| 文本 decoder 的 `RowParallelLinear`(o_proj、down_proj) | **行分片**,forward 末尾 `all_reduce(SUM)` | `linear.py:199,224-225` |
| 文本 decoder 注意力头 | 每 rank 负责 `num_heads//world_size` 个 Q 头、`num_kv_heads//world_size` 个 KV 头 | `model_runner.py:208` |
| **Vision tower(ViT)** | **整塔复制**:每 rank 各自跑完整 ViT,产出相同的 `visual_emb` | ViT 小(1024 hidden ×24 层),复制成本低;避免 ViT 内部 all_reduce;与 vLLM 一致 |
| `DeepstackProj`(1024→2048) | **复制**(`ReplicatedLinear`) | 输入视觉特征已复制、输出接残差流(残差流是 full hidden) |
| 最终 RMSNorm | 复制 | 残差流在每 rank 上是 full |
| `EmbeddingHead`(pooling + L2 + MRL) | **复制**,仅 rank 0 返回 | 输出 2048 维很小;对标 `model_runner.py:394` rank-0 采样 |
| MRoPE 位置 / `cos_sin_cache` | **复制**:由序列结构派生,与权重无关,每 rank 相同 | 无分片需求 |
| KV cache pool | **pooling 模式整体跳过**(§3.5.3),故无 KV 分片问题 | decode 才需要 |

关键不变量:**残差流(hidden_states)在每 rank 上都是 full 2048 维**——只有权重矩阵分片,
激活不分片。因此视觉特征复制进 `input_emb`、DeepstackProj 加进残差、EmbeddingHead 取最后
token,三者都与 TP 兼容,无需额外 gather/scatter。

`input_emb` 的产生:`VocabParallelEmbedding` 已 `all_reduce`(embedding_head.py),输出 full,
随后把复制来的 `visual_emb` scatter 进图像 token 位置——每 rank 独立做,结果一致。

#### 3.8.2 加载器必须走 per-param `weight_loader`(关键修正)

现状(§1.2 / `AGENTS.md` gotcha):`loader.py` 对 QKV/gate_up 做完 `torch.cat` merge 后直接
`param.data.copy_(merged_full)`(`loader.py:94,119,165`),**绕过** `linear.py` 里挂在 param 上的
`weight_loader` callable。对 `world_size=1` 无影响;对 `world_size>1`,`ColumnParallelLinear` 的
param 是已分片形状(`output_size//tp_size`),`copy_(full)` 触发分支 5 的形状不匹配兜底,把
**rank-0 的切片装到每个 rank 上**——TP 错。

VL 路径必须修正这一点,且**不动生成路径**(R-7)。做法:在 `utils/loader.py` 新增一个
`_load_param(model, hf_name, hf_weight, loaded_weight_id=None)` 分发函数,VL 分支调用它:

```python
def _load_param(model, hf_name, hf_weight, *, merged_id=None):
    param = model.get_parameter(hf_name)
    if hasattr(param, "weight_loader"):
        # 走 per-param 切片:ColumnParallelLinear 按 tp_rank 切;
        # QKVColumnParallelLinear 传 merged_id ('q'/'k'/'v') 做 GQA 感知切;
        # MergedColumnParallelLinear 传 merged_id (0/1) 切 gate/up;
        # ReplicatedLinear 的 weight_loader 就是 copy 全量
        if merged_id is not None:
            param.weight_loader(param, hf_weight, merged_id)
        else:
            param.weight_loader(param, hf_weight)
    else:
        default_weight_loader(param, hf_weight)   # 形状一致直接 copy
```

VL 加载流程(替换 `loader.py:73-168` 的 VL 部分;生成路径分支不动):

- **文本 decoder QKV**:对 HF 的 `q_proj`/`k_proj`/`v_proj` 各调一次
  `_load_param(..., "model.layers.N.self_attn.qkv_projection.weight", hf_weight, merged_id='q'/'k'/'v')`。
  不再做 `torch.cat`——`QKVColumnParallelLinear.weight_loader`(`linear.py:152`)会按
  GQA 头边界把对应分量写到合并 param 的正确槽位。这样每 rank 拿到自己的 Q/K/V 头切片。
- **文本 decoder gate_up**:对 `gate_proj`/`up_proj` 各调一次,`merged_id=0/1`。
- **ViT(复制)**:ViT 的 Q/K/V 在 HF 是分开的 `q/k/v` 且本项目 ViT 用 `ReplicatedLinear`
  (不分片)→ 直接 `_load_param` 走 `default_weight_loader` 复制全量到每 rank。
  注意:不要把 ViT 的 q/k/v 合并成 qkv_projection(ViT 模块结构与 LLM 不同)。
- **merger / deepstack / RMSNorm / EmbeddingHead 投影(若有)**:复制全量。
- **embed_tokens**:`VocabParallelEmbedding` 的 `weight_loader` 负责词表分片——VL 也走它。

生成路径的 `torch.cat` merge + `param.data.copy_(full)` 分支(`loader.py:76-146`)保持原样
不动,TP>1 在生成路径的现状(R-7)不在本次范围。

#### 3.8.3 多 GPU 执行与跨进程数据

- 复用 `LLMEngine` 的 worker spawn(`llm_engine.py:29-37`,"spawn" 上下文):rank 0 跑引擎,
  rank≥1 跑 `worker_process`→`ModelRunner.loop()`(`llm_engine.py:13`)。
- 通信:`SharedMemory(name='myvllm')` + `multiprocessing.Event`,pickle `(method_name, *args)`
  带 4 字节长度前缀(`model_runner.py:125-143`);NCCL 进程组在
  `tcp://localhost:12345`(`model_runner.py:26`)。
- **多模态跨进程序列化**:`Sequence.__getstate__`/`__setstate__`(`sequence.py:88-114`)
  现在只传 `last_token`(decode)或全 `token_ids`(prefill)。VL 需要把 `mm_data`
  (`pixel_values` + `image_grid_thw` + `image_token_spans`)一并 pickle 到 worker。
  `pixel_values` 可能较大(单图最高 ~1024 patch × 3 × 16×16 ≈ 数 MB),需 pin_memory + 在
  pickle 协议里走 `__reduce__` 零拷贝或 `shared_memory.SharedMemory` 大块传输,避免每步
  pickle 开销。新增 `Sequence.__getstate__` 在 prefill 分支附带 `mm_data` 字段。
- ViT 复制执行:每 rank 各自对相同 `pixel_values` 跑 ViT → 相同 `visual_emb`,无需 NCCL
  同步视觉特征。
- EmbeddingHead 仅 rank 0 返回(`model_runner.py:394` 的 rank-0 模式);若 `EmbeddingHead`
  被分片(本设计不分片),则用 `dist.gather` 到 rank 0。

## 4. 模块交互图

```
LLMEngine ──generate()──► (不变) 生成路径
          └──encode()────► Scheduler(纯 prefill) ─► ModelRunner(pooling)
                                                          │
                       ┌──────────────────────────────────┤
                       ▼                                  ▼
                VisionTower                         Qwen3VLForEmbedding
                 (vision/)                            (models/qwen3_vl.py)
                   │  visual_feat[idx] (deepstack)     │  decoder 层
                   └──────────────────────────────►───┤  + DeepstackProj@{5,11,17}
                                                      │
                                    MRoPE 位置 ──► Attention (flash prefill)
                                                      │
                                          EmbeddingHead (pooling/)
                                                      │
                                            L2 归一化 + MRL 截断
```

## 5. 测试用例

测试沿用 `tests/test_scheduler.py` 的回归导向风格(每个测试类守护一个具体行为)。用例分六组;
A–D 组在 CPU + mock ViT 上跑(CI 默认),E 组带 `@pytest.mark.gpu` + 联网标记(需真实 GPU
+ 2B checkpoint),F 组守护生成路径回归。每组给出测试函数名、输入、断言。

### A. MRoPE 单元测试 — `tests/test_mrope.py`(§3.3,R-4)

```python
# conftest:在 src 加入 sys.path(沿用 tests/test_scheduler.py:4 写法)
HEAD_DIM = 128
SECTION = [24, 20, 20]          # T, H, W

def test_mrope_section_split():
    # head_dim=128 按 [24,20,20] 切段:T 占 [0:24]、H [24:44]、W [44:64],合计 64(=head_dim/2)
    rope = RotaryEmbedding(head_dim=HEAD_DIM, rope_type="mrope",
                           mrope_section=SECTION, mrope_interleaved=True)
    cos, sin = rope.cos_sin_cache  # 或 forward(zero positions)
    assert cos.shape[-1] == HEAD_DIM
    # 各段 cos 的模长在该段内相等;跨段不串扰
    assert torch.allclose(cos[..., 0:24].abs(), cos[..., 0:1].abs())
    assert torch.allclose(cos[..., 24:44].abs(), cos[..., 24:25].abs())
    assert torch.allclose(cos[..., 44:64].abs(), cos[..., 44:45].abs())

def test_mrope_interleaved_layout():
    # interleaved=true:段内 (cos0,sin0,cos1,sin1,...) 交替;非交错是 (cos0..n, sin0..n)
    rope = RotaryEmbedding(head_dim=HEAD_DIM, rope_type="mrope",
                           mrope_section=SECTION, mrope_interleaved=True)
    cos, sin = rope._compute_for_positions(positions_3d)
    # 段内相邻奇偶下标应分别为 cos/sin(即 sin 段 = 1 - cos 段的对应旋转)
    seg = slice(0, SECTION[0])
    assert cos[..., seg][..., 0::2].shape == sin[..., seg][..., 1::2].shape

def test_mrope_text_only_positions():
    # 纯文本 "Hello world" token 化后 T 单调递增,H=W=0(或全图共享基准)
    pos = compute_mrope_positions(seq_ids=[1,2,3], image_token_spans=[], grid_thw=None)
    T, H, W = pos
    assert T.tolist() == [0, 1, 2]
    assert H.tolist() == [0, 0, 0]
    assert W.tolist() == [0, 0, 0]

def test_mrope_image_positions():
    # 一张图 grid_thw=(T=1,H=4,W=4) → 16 patch 的 token,T 恒定(同帧),H/W 按 patch 坐标派生
    pos = compute_mrope_positions(seq_ids=[IMG_PAD]*16,
                                   image_token_spans=[(0,16)], grid_thw=(1,4,4))
    T, H, W = pos
    assert set(T.tolist()) == {0}                 # 同帧 T 相同
    assert sorted(set(H.tolist())) == [0,1,2,3]  # H 在 [0,3]
    assert sorted(set(W.tolist())) == [0,1,2,3]  # W 在 [0,3]

def test_mrope_mixed_positions():
    # 文本+图像+文本:位置连续,不出现空洞或重复(跨模态 T 单调)
    pos = compute_mrope_positions(seq_ids=[1,2, IMG_PAD, IMG_PAD, 3],
                                   image_token_spans=[(2,4)], grid_thw=(1,2,1))
    T, H, W = pos
    assert T.tolist() == [0, 1, 2, 2, 3]          # 图像两 token 共享 T=2
```

### B. 视觉塔单元测试 — `tests/test_vision.py`(§3.2,R-3)

```python
def test_patch_embed_output_shape():
    # (B=1,C=3,T=2,H=512,W=512),patch=16,temporal_patch=2 → patch 数 = 1*(512/16)*(512/16)=1024
    pe = PatchEmbed3D(hidden=1024, patch_size=16, temporal_patch_size=2, in_ch=3)
    out = pe(torch.zeros(1,3,2,512,512))
    assert out.shape == (1024, 1024)

def test_spatial_merger_shape():
    # spatial_merge_size=2 → 4 邻 patch 合一,(P=1024,1024) → (256,2048)
    m = SpatialMerger(in_hidden=1024, out_hidden=2048, spatial_merge_size=2)
    out = m(torch.zeros(1024, 1024))
    assert out.shape == (256, 2048)

def test_deepstack_injection_only_at_layers():
    # deepstack 索引 {5,11,17} 恰好在这三层注入残差,其余层为 no-op
    model = Qwen3VLForEmbedding(config, deepstack_indexes=[5,11,17])
    feats = [torch.zeros(1, P, 1024) for _ in range(24)]
    injected = []
    for idx in range(28):
        emb = torch.zeros(1, S, 2048)
        if idx in {5,11,17}:
            emb_new = model.layers[idx].inject(emb, feats[idx])
            assert not torch.allclose(emb_new, emb)   # 有变化
            injected.append(idx)
        else:
            assert model.layers[idx].inject is None
    assert injected == [5,11,17]

def test_vit_attention_bidirectional(R-3):
    # ViT 自注意力无 causal mask:对调两个 patch,输出对应对调(双向)
    vit_block = Qwen3VLVisionBlock(...)
    x = torch.randn(4, 8, 1024)          # 4 patch
    y1 = vit_block(x)
    x_rev = torch.flip(x, dims=[0])
    y2 = vit_block(x_rev)
    assert torch.allclose(torch.flip(y2, dims=[0]), y1, atol=1e-5)
```

### C. Scheduler pooling 测试 — `tests/test_pooling_scheduler.py`(§3.5.2,R-2)

沿用 `tests/test_scheduler.py:12` 的 `make_scheduler` fixture 与
`tests/test_scheduler.py:27` 的 `inject_running` 思路,但 mock 一个 pooling 模式的
scheduler。

```python
def make_pooling_scheduler(max_batched_tokens=8192, max_image_patches=1024):
    sch = Scheduler(..., runner_type="pooling",
                    max_num_batched_tokens=max_batched_tokens,
                    max_image_patches=max_image_patches)
    return sch

def test_every_seq_finishes_in_one_step():
    # 3 条 prompt → schedule 一次全部进入 prefill,postprocess 后全部 FINISHED
    sch = make_pooling_scheduler()
    for p in ["a b c", "d e", "f g h i"]:
        sch.add_sequence(make_seq(p))
    seqs, is_prefill = sch.schedule()
    assert is_prefill is True
    assert len(seqs) == 3
    sch.postprocess(seqs, [embed_for(s) for s in seqs])  # mock 输出
    assert all(s.status == SequenceStatus.FINISHED for s in seqs)
    assert sch.waiting == deque() and sch.running == deque()

def test_max_image_patches_budget():
    # 两条图各 600 patch,预算 1024 → 只能调度第一条;第二条留 waiting
    sch = make_pooling_scheduler(max_image_patches=1024)
    sch.add_sequence(make_seq("img", mm_patches=600))
    sch.add_sequence(make_seq("img", mm_patches=600))
    seqs, _ = sch.schedule()
    assert len(seqs) == 1
    assert len(sch.waiting) == 1

def test_preempt_unreachable():
    # pooling 模式下 preempt() 不可达
    sch = make_pooling_scheduler()
    with pytest.raises(AssertionError):
        sch.preempt(make_seq("x"))

def test_termination_signal():
    # waiting 与 running 都空 → schedule 返回 (None, False) 表示完成
    sch = make_pooling_scheduler()
    seqs, is_prefill = sch.schedule()
    assert seqs is None and is_prefill is False
```

### D. 端到端 encode(mock ViT) — `tests/test_encode_e2e.py`

用 monkeypatch 把 `VisionTower` 替成返回固定张量的 stub,跑 CPU 前向。

```python
def test_encode_text_returns_2048_dim():
    emb = engine.encode([{"text": "a woman on a beach"}])
    assert len(emb) == 1 and emb[0].shape == (2048,) and emb[0].dtype == torch.float32

def test_encode_image_returns_2048_dim():
    emb = engine.encode([{"image": "tests/data/beach.jpg"}])
    assert emb[0].shape == (2048,)

def test_encode_mixed():
    emb = engine.encode([{"text": "sunset", "image": "tests/data/beach.jpg"}])
    assert emb[0].shape == (2048,)

def test_l2_normalization():
    emb = engine.encode([{"text": "x"}])
    assert abs(float(emb[0].norm()) - 1.0) < 1e-5

def test_mrl_truncation():
    emb = engine.encode([{"text": "x"}], pooling={"mrl_dim": 512})
    assert emb[0].shape == (512,)
    assert abs(float(emb[0].norm()) - 1.0) < 1e-5   # 截断后重新归一化

def test_batch_preserves_order():
    inputs = [{"text":"a"},{"text":"b"},{"image":"tests/data/beach.jpg"}]
    embs = engine.encode(inputs)
    assert len(embs) == len(inputs)
```

### E. 数值对齐(真实 GPU) — `tests/test_parity_qwen.py`

```python
@pytest.mark.gpu
@pytest.mark.network
def test_reproduce_readme_similarity():
    # 复现 model card README 的 4 查询 × 3 文档样例,余弦容差 1e-3
    engine = LLMEngine(config_qwen3_vl_embed())
    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        {"text": "Pet owner training dog outdoors near water."},
        {"text": "Woman surfing on waves during a sunny day."},
        {"text": "City skyline view from a high-rise building at night."},
    ]
    docs = [
        {"text": "A woman shares a joyful moment with her golden retriever ..."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {"text": "A woman shares ...", "image": "https://.../demo.jpeg"},
    ]
    emb_q = engine.encode(queries)
    emb_d = engine.encode(docs)
    sim = torch.stack(emb_q) @ torch.stack(emb_d).T
    expected = torch.tensor([
        [0.8160, 0.7155, 0.7054],
        [0.5173, 0.3295, 0.4446],
        [0.3863, 0.2987, 0.3312],
        [0.1061, 0.0433, 0.0839]])
    assert torch.allclose(sim, expected, atol=1e-3)
```

### F. 生成路径回归 — `tests/test_scheduler.py`(不变)

`tests/test_scheduler.py:39`(`TestBug2TokenLimitBreak`)、`:100`
(`TestBug1CanAppendFailure`)、`:146`(`TestSchedulerHappyPath`)必须仍全绿——
证明纯 prefill 模式是超集而非替换。

> 数值对齐测试(E)需真实 GPU + 联网下载 2B checkpoint;CI 默认跑 A–D、F。

### G. 张量并行测试 — `tests/test_tp.py`(§3.8,真实多 GPU)

需 `world_size>=2`,用 `torchrun --nproc_per_node=2`(对标 `linear.py:231-485` 的
`__main__` TP 测试 harness 风格)。脚本以 spawn 方式起 2 进程,各跑一遍断言。

```python
@pytest.mark.gpu
@pytest.mark.tp(world_size=2)
def test_tp_text_decoder_qkv_shard():
    # world_size=2 下,每 rank 的 qkv_projection param 形状 = (q+2*kv 分片后)
    # q 头 = num_heads//2 = 8,kv 头 = num_kv_heads//2 = 4
    init_process_group("nccl", "tcp://localhost:12345", world_size=2, rank=rank)
    model = build_vl_text_decoder(tp_size=2)              # QKVColumnParallelLinear
    _load_param(model, "...qkv_projection.weight", q_full, merged_id="q")
    _load_param(model, "...qkv_projection.weight", k_full, merged_id="k")
    _load_param(model, "...qkv_projection.weight", v_full, merged_id="v")
    param = model.get_parameter("...qkv_projection.weight")
    # 断言:每 rank 切片正确,不全是 rank-0 的切片(复现 §1.2 的 bug 守卫)
    assert param.data.shape[0] == (num_heads//2 + 2*(num_kv_heads//2)) * head_dim
    assert not torch.allclose(param.data, rank0_slice_at_all_ranks_bad)

def test_tp_vision_tower_replicated():
    # ViT 复制:world_size=2 下两 rank 的 ViT 权重逐元素相等
    vit_a = build_vit(tp_size=2, rank=0); vit_b = build_vit(tp_size=2, rank=1)
    for (n0,p0),(n1,p1) in zip(vit_a.named_parameters(), vit_b.named_parameters()):
        assert n0 == n1 and torch.allclose(p0.cpu(), p1.cpu())

def test_tp_residual_full_hidden():
    # 残差流在每 rank 是 full 2048:跑一次 decoder layer,检查输出 shape
    layer = build_decoder_layer(tp_size=2)
    x = torch.zeros(1, S, 2048, device=f"cuda:{rank}")
    out = layer(x, positions_3d)
    assert out.shape == x.shape                  # full hidden,未分片

def test_tp_embedding_identical_across_ranks():
    # end-to-end:world_size=2 跑 encode(),两 rank 的 embedding 逐元素相等
    # (EmbeddingHead 复制,残差 full,所以每 rank 结果一致;rank 0 返回)
    eng = LLMEngine(config_tp2())
    emb0 = eng.encode([{"text": "a woman on a beach"}])
    # worker rank 的本地 embedding 用 dist.all_reduce 等价核对
    assert_allreduce_equal(emb0)

def test_tp_vs_single_gpu_numerical_parity():
    # world_size=1 与 world_size=2 的 embedding 余弦相似度 > 1-1e-4
    eng1 = LLMEngine(config_tp1()); eng2 = LLMEngine(config_tp2())
    e1 = eng1.encode([{"text": "x", "image": "tests/data/beach.jpg"}])
    e2 = eng2.encode([{"text": "x", "image": "tests/data/beach.jpg"}])
    assert torch.allclose(e1[0], e2[0], atol=1e-4)
```

> 数值对齐(E)与 TP(G)需真实 GPU(+G 需 ≥2 卡与 `torchrun`);CI 默认跑 A–D、F。
> 所有测试遵循 `AGENTS.md` 的 `uv run pytest` 约定,无 lint/typecheck 配置。

## 6. 风险与开放问题

- **R-1 —— embedding head 组成。** 该模型是用于 embedding 的
  `Qwen3VLForConditionalGeneration`;checkpoint 是否带专门的投影层(如 `embed_proj.*`)
  还是直接池化原始 `hidden_states`,尚未确认。HF 仓库里有 `1_Pooling/config.json` 和
  `sentence_bert_config.json`——定稿 `EmbeddingHead` 前先读这两个。若直接池化:无投影权重,
  只 gather + 归一化。若有投影:加载 `Linear(2048, 2048)`(可能与 `lm_head`/`embed_tokens`
  tied)。
- **R-2 —— pooling 模式下的 `BlockManager`。** `Scheduler.schedule()` 的 prefill 分支会调
  `block_manager.allocate()`(`scheduler.py:25-31`),`postprocess()` 释放。对 embedding 没有
  KV cache,但 scheduler 需要 block-table 管线。最省事的修法:每序列分配一个 dummy block
  (block table 长度 1、无实际存储)。更干净的修法:把 `allocate`/`deallocate` 用 `runner_type`
  守卫。在现有测试被证明仍通过后,优先选干净修法。在任务 T-9 定夺。
- **R-3 —— prefill flash kernel 的 causal mask。** `flash_attention_varlen_kernel`
  (`attention.py:112`)用 online softmax + `cu_seqlens` 边界,但实现必须确认它 **不施加**
  intra-sequence causal mask(LLM prefill 因果;ViT prefill 非因果)。若当前施加了 causal
  mask,新增 `is_causal` 标志并分支 mask 计算。这是 vision tower 的承重正确性校验。
- **R-4 —— MRoPE 位置语义。** Qwen3-VL MRoPE 即便对文本 token 也分配 T 位置(全局单调
  计数器),对图像 token 按 patch 坐标给 H/W。确切规则(文本 token 是否每个都让 T 前进?
  每个图像 token 是共享 T 还是让 T 前进?)实现前必须与 `transformers` 的 `Qwen3VLModel`
  源码交叉核对。风险:位置错 → 检索质量静默下降,而非硬失败。
- **R-5 —— 图像预算 vs `gpu_memory_utilization`。** 一张 1024-patch 图的 vision tower
  激活约 `1024 × 24 × 1024 × BF16 ≈ 50 MB/层` 的中间注意力张量——单图不大,但批内多图会尖峰。
  `max_image_patches` 预算 + 释放 KV 池(`model_runner.py:247`)应能覆盖;用 README 最大样例
  图验证。
- **R-6 —— `transformers` 版本锁。** 模型卡要求 `transformers>=4.57.0`。当前
  `pyproject.toml:7-12` 无上界 `transformers`;锁版可能影响文本路径的 tokenizer 行为。锁版后
  跑一遍 `main.py` 冒烟。
- **R-7 —— 生成路径 TP>1 加载现状(本次不修)。** §1.2 已述:`loader.py` 绕过 per-param
  `weight_loader` 直接 `copy_(full)`,在 `world_size>1` 下会把 rank-0 切片装到每个 rank 上。
  本次只对 **VL 路径**改走 `weight_loader`(§3.8.2),**不动生成路径**的 `torch.cat` merge 分支
  (`loader.py:76-146`)。即:`main.py` / `main_llama32.py` 在 `world_size>1` 下的现状(可能
  不正确)保持不变,留作后续变更。`tests/test_scheduler.py` 是单进程 mock,不会暴露此问题。

## 7. 迁移 / 兼容性

- `runner_type` 默认 `"generation"`;`main.py` / `main_llama32.py` 配置不动 → 行为一致。
- `pyproject.toml` 新增 `Pillow`、`qwen-vl-utils>=0.0.14`;锁 `transformers>=4.57.0`。
  `uv.lock` 重新生成。
- `AGENTS.md` 在 Architecture 下新增"Pooling 模式"小节,列出被跳过的子系统与 `encode()`
  API。
- `README.md` 新增多模态 quickstart 块。

## 8. 被否决的替代方案

- **单独的 `EmbeddingEngine`,不要 scheduler。** 否决:重复 prefill 批处理与 TP 管线,且对未来
  reranker pooler 不具扩展性。vLLM 的 `RunnerType` 接缝才是正确模型。
- **强行 decode 出 embedding(跑生成,取最后 hidden state)。** 否决:浪费(白跑 paged
  decode + 采样),且模型未被训练来生成;"embedding" 是最后一个 prefill token 的 hidden
  state,不是生成的 token。
- **ViT 用 `flash_attn` 库。** 否决:项目定位就是从零用 Triton 复刻;复用自己的 kernel。
- **实现 MRL 训练期支持。** 超出范围;此处的 MRL 仅为推理时截断 + 重新归一化。
- **ViT 也做列/行分片(而非复制)。** 否决:ViT 体量小(1024 hidden ×24 层),分片带来的
  `all_reduce` 与加载复杂度收益不大;复制执行简单且与 vLLM 一致(§3.8.1)。仅在 8B VL
  embedding(ViT 更大)时回头考虑。

## 9. 启动命令

所有命令遵循 `AGENTS.md`:包管理器 `uv`,Python `>=3.11,<3.12`,无 lint/typecheck
配置,验证只用 `pytest`。命令一律在仓库根目录 `D:\PerJoker\code\HermesInfer` 下执行。

### 9.1 环境与依赖

```bash
# 首次安装 uv(PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 安装依赖(uv.lock 已锁,新增 Pillow / qwen-vl-utils / transformers>=4.57.0)
uv sync
```

### 9.2 模型获取

`Qwen3-VL-Embedding-2B`(2.1B BF16,~4GB)首次运行由 `huggingface_hub.snapshot_download`
自动拉取(`loader.py:39`)。预下载离线跑:

```bash
# 设 HF 端点(国内可切镜像)
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-Embedding-2B', allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*','merges.txt'])"
```

### 9.3 跑 demo

```bash
# 单卡,embedding demo(文本/图像/图像+文本 编码 + 余弦相似度矩阵)
uv run python main_embedding.py

# 回归:生成路径不变
uv run python main.py
uv run python main_llama32.py
```

### 9.4 跑测试

```bash
# 全量(默认 CI 集:CPU mock ViT 的 A–D、F + 生成路径回归)
uv run pytest tests/ -v

# 仅 MRoPE 单测
uv run pytest tests/test_mrope.py -v

# 仅 scheduler pooling 单测
uv run pytest tests/test_pooling_scheduler.py -v
```

### 9.5 多卡 / TP 启动

本项目 TP 通过 `LLMEngine` 内部 `multiprocessing` "spawn" 起 worker + NCCL over
`tcp://localhost:12345`(`llm_engine.py:29-37`,`model_runner.py:26`)——**不需要**
`torchrun`,改配置里的 `world_size` 即可:

```bash
# 2 卡 embedding demo(单进程入口,LLMEngine 自己 spawn worker)
$env:WORLD_SIZE=2
uv run python main_embedding.py        # config["world_size"] 读 2
```

若要复用 `linear.py:231-485` 的 `torchrun` harness 风格跑 TP 单测(G 组):

```bash
# G 组 TP 测试,需真实 ≥2 卡
uv run pytest tests/test_tp.py -v -m "gpu and tp"
# 或脚本化:
uv run python -m torch.distributed.run --nproc_per_node=2 tests/test_tp.py
```

### 9.6 数值对齐(真实 GPU + 联网)

```bash
uv run pytest tests/test_parity_qwen.py -v -m "gpu and network"
```

## 10. 调用方式

公开 API 是 `myvllm.engine.llm_engine.LLMEngine.encode()`(对标 `generate()`);输入是
list[dict],每条 dict 描述一条 query/document,可选自定义 instruction。

### 10.1 最小示例(对标 README 的 sentence-transformers 用法)

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams   # pooling 模式不用,仅占位 import 习惯

config = {
    "model_name_or_path": "Qwen/Qwen3-VL-Embedding-2B",
    "world_size": 1,
    "runner_type": "pooling",
    "pooling": {"mode": "last_token", "normalize": True, "mrl_dim": None},
    "multimodal": {"max_image_patches": 16384},
    # 其余字段(max_num_batched_tokens 等)沿用 main.py:15-41 的习惯
}
llm = LLMEngine(config)

queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."},
]
documents = [
    {"text": "A woman shares a joyful moment with her golden retriever ..."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares ...", "image": "https://.../demo.jpeg"},
]

q_emb = llm.encode(queries)
d_emb = llm.encode(documents)

import torch
sim = torch.stack(q_emb) @ torch.stack(d_emb).T
print(sim)   # (4, 3),复现 README 的相似度矩阵
```

### 10.2 三种输入形态

```python
# 纯文本
llm.encode([{"text": "a cat sitting on a mat"}])

# 纯图像(本地路径 / http(s) / oss)
llm.encode([{"image": "./assets/demo.jpeg"}])
llm.encode([{"image": "https://.../demo.jpeg"}])

# 图文混合(图像在前,文本在后——与 chat_template 一致)
llm.encode([{"text": "describe the scene", "image": "./assets/demo.jpeg"}])
```

### 10.3 自定义 instruction(instruction-aware)

默认系统提示 `Represent the user's input.`。可对每条输入覆盖:

```python
q = llm.encode([{"text": "surfing", "instruction": "Retrieve relevant documents for the query."}])
```

### 10.4 MRL(Matryoshka)子维度截断

```python
# 输出截断到 512 维并重新 L2 归一化,shape (B, 512)
emb = llm.encode(inputs, pooling={"mrl_dim": 512})
```

### 10.5 批处理与吞吐

`encode()` 把全部输入一次性入队 `waiting`,`Scheduler` 按
`max_num_batched_tokens` / `max_num_sequences` / `max_image_patches` 自动分批 prefill,
循环到全部完成。调用方按输入顺序拿到 embedding。大批量无需手动切分:

```python
embs = llm.encode(huge_input_list)   # 内部自动分批
```

### 10.6 多卡(TP)调用

只改配置,API 不变:

```python
config = {**..., "world_size": 2, "runner_type": "pooling", ...}
llm = LLMEngine(config)              # 内部 spawn 1 worker + NCCL
emb = llm.encode([{"text": "x", "image": "./a.jpg"}])
# 单卡与 2 卡结果在 1e-4 内一致(test_tp_vs_single_gpu_numerical_parity 守护)
```

### 10.7 与 sentence-transformers / vLLM 的等价对照

`encode()` 的输出等价于:
- sentence-transformers:`SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B").encode(...)` 的
  L2 归一化向量;
- vLLM:`EngineArgs(model=..., runner="pooling").llm.embed(...)` 的输出。
容差:余弦 `1e-3`(E 组守护)、单卡与多卡 `1e-4`(G 组守护)。

## 11. file:line 引用索引(被改动的当前代码)

- 引擎:`llm_engine.py:25,68,88,95` · `scheduler.py:6,25,35,42-54,57-93,80-91,96,104` · `model_runner.py:16,32-69,197,247,265-314,354-378,386,394,406`
- 层:`attention.py:112,283,455,472` · `rotary_embedding.py:4,48,69-87,100` · `linear.py:6,84,113,152,199` · `embedding_head.py:64,75-76` · `sampler.py:5,16-18`
- 模型:`qwen3.py:285,50-51,129,159,231,286-292,333-334` · `llama.py:266,8,50`
- 工具:`context.py:6,16,18-26` · `loader.py:16,76,103,179-213`
- 配置:`pyproject.toml:2,3,6,7-12,14-19,21-29,31-35` · `sampling_parameters.py:5,12`
- 测试:`tests/test_scheduler.py:39,100,146`
