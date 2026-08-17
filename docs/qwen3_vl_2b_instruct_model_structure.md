# Qwen3-VL-2B-Instruct 模型结构全流程 (HermesInfer 实现)

> 基于 Qwen3-VL 架构，适配生成任务 (Prefill + Decode 自回归)。
> 与 Qwen3-VL-Embedding-2B 共享 Vision Tower + 文本 Decoder 拓扑，
> 但输出端改为 LM Head + 采样，推理模式改为自回归生成。
> 每个模块均标注 shape 信息。

## 模型配置参数

```
# ---- 文本 Decoder ----
vocab_size         = 151936
hidden_size        = 2048
num_heads          = 16        (Q/O heads)
head_dim           = 128
num_kv_heads       = 8         (KV heads, GQA)
intermediate_size  = 6144
num_layers         = 28
tie_word_embeddings = True
rope_theta (base)  = 5_000_000
rms_norm_epsilon   = 1e-6
mrope_section      = [24, 20, 20]   (T, H, W)
mlp_bias           = True

# ---- Vision Tower (ViT) ----
vision_depth       = 24
vision_hidden_size = 1024
vision_intermediate_size = 4096
vision_num_heads   = 16
patch_size         = 16
temporal_patch_size = 2
in_channels        = 3
out_hidden_size    = 2048
spatial_merge_size = 2
num_position_embeddings = 2304  (48x48 学习位置网格)
deepstack_visual_indexes = [5, 11, 17]

# ---- 生成相关 ----
block_size         = 256       (KV cache block)
max_position       = 32768
eos                = 151645    (im_end)
```

---

## 一、与 Qwen3-VL-Embedding-2B 的关键差异

| 特性 | Qwen3-VL-Embedding-2B | Qwen3-VL-2B-Instruct |
|------|----------------------|----------------------|
| **任务** | 多模态 Embedding | 多模态生成 (VQA/对话) |
| **推理模式** | 纯 Prefill (单次前向) | Prefill + Decode (自回归) |
| **输出** | embedding 向量 (2048维) | token 序列 (自回归采样) |
| **输出层** | EmbeddingHead (last-token+L2) | LM Head (ParallelLMHead) |
| **KV Cache** | 无 | 有 (PagedAttention) |
| **CUDA Graph** | 无 | 有 (decode 加速) |
| **采样** | 无 | Gumbel-max 采样 |
| **Vision Tower** | 相同 | 相同 |
| **文本 Decoder** | 相同 (28层) | 相同 (28层) |
| **MRoPE** | 相同 | 相同 |
| **DeepStack** | 相同 | 相同 |

---

## 二、端到端总览 (Engine Pipeline)

```
+---------------------------------------------------------------------+
|                    LLMEngine.generate()                              |
|                    llm_engine.py:105                                 |
+---------------------------------------------------------------------+
|                                                                      |
|  prompts: list[str]  (含图像描述或纯文本)                            |
|       |                                                              |
|       v                                                              |
|  +------------------------------------------+                        |
|  |  AutoProcessor / AutoTokenizer           |                        |
|  |  apply_chat_template + encode            |                        |
|  |  + process_vision_info (有图像时)         |                        |
|  |  prompt str -> token_ids + mm_data       |                        |
|  +--------------------+---------------------+                        |
|                       v                                              |
|  +------------------------------------------+                        |
|  |  Scheduler.add_sequence()                |                        |
|  |  Sequence(token_ids, mm_data) -> waiting |                        |
|  +--------------------+---------------------+                        |
|                       v                                              |
|  +---------- while not is_finished() --------------------------+     |
|  |                                                              |     |
|  |  +------------------------------------+                      |     |
|  |  |  Scheduler.schedule()              |                      |     |
|  |  |  waiting -> prefill (is_prefill=T) |                      |     |
|  |  |  running -> decode  (is_prefill=F) |                      |     |
|  |  |  BlockManager.allocate/append      |                      |     |
|  |  |  return (seqs, is_prefill)         |                      |     |
|  |  +----------------+-------------------+                      |     |
|  |                   v                                          |     |
|  |  +------------------------------------+                      |     |
|  |  |  ModelRunner.run()                 |                      |     |
|  |  |  Prefill:                          |                      |     |
|  |  |    prepare_prefill_vl()            |                      |     |
|  |  |    VisionTower -> visual_emb       |                      |     |
|  |  |    scatter + 28层decoder           |                      |     |
|  |  |    compute_logits (LM Head)        |                      |     |
|  |  |    sampler -> token_ids            |                      |     |
|  |  |  Decode:                           |                      |     |
|  |  |    prepare_decode()                |                      |     |
|  |  |    CUDA Graph replay               |                      |     |
|  |  |    compute_logits (LM Head)        |                      |     |
|  |  |    sampler -> token_ids            |                      |     |
|  |  +----------------+-------------------+                      |     |
|  |                   v                                          |     |
|  |  +------------------------------------+                      |     |
|  |  |  Scheduler.postprocess()           |                      |     |
|  |  |  append_token -> check EOS/max_len |                      |     |
|  |  |  FINISHED -> deallocate blocks     |                      |     |
|  |  +------------------------------------+                      |     |
|  |                                                              |     |
|  +--------------------------------------------------------------+     |
|                       v                                              |
|  output: {'text': list[str], 'token_ids': dict}                      |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## 三、Prefill 阶段 — 数据准备 (含图像)

```
model_runner.py:463  _prepare_prefill_vl()

输入: seqs: list[Sequence]  (每条可能带 mm_data)

多条序列打包为 varlen 格式:

  input_ids:     [seq0_tokens... | seq1_tokens... | seq2_tokens... ]
                  <-- 1D tensor, 所有 token 拼接 -->

  cu_seqlens_q:  [0, len(seq0), len(seq0)+len(seq1), ...]
                  shape: (num_seqs + 1,)  dtype: int32

  pixel_values:  cat([seq.pixel_values for seq with image])
                  shape: (total_patches, 1536)

  grid_thw:      cat([seq.grid_thw for seq with image])
                  shape: (num_images, 3)

  image_token_spans: [(s+offset, e+offset), ...]
                      全局坐标的图像 token 区间

  positions_3d:  compute_mrope_positions(...)
                  shape: (3, total_tokens)   [T, H, W]

  image_token_mask: (total_tokens,) bool

  Context 写入:
    is_prefill      = True
    cu_seqlens_q    = (num_seqs + 1,)
    slot_mapping    = (N,)
    positions_3d    = (3, N)
    image_token_mask = (N,)

其中 N = sum(len(seq) for seq in seqs) = 总 token 数
```

---

## 四、Decode 阶段 — 数据准备

```
model_runner.py:365  prepare_decode()

输入: seqs: list[Sequence]  (每条只取 last_token)

  input_ids:     [seq0_last, seq1_last, ..., seqB_last]
                  shape: (B,)  dtype: int64

  context_lens:  [len(seq0), len(seq1), ..., len(seqB)]
                  shape: (B,)  dtype: int64

  slot_mapping:  [slot_0, slot_1, ..., slot_{B-1}]
                  shape: (B,)  dtype: int64

  block_tables:  [[b0_0, b0_1, ..., -1, -1],
                  [b1_0, b1_1, b1_2, -1, -1], ...]
                  shape: (B, max_num_blocks)  dtype: int32

  Context 写入:
    is_prefill   = False
    slot_mapping = (B,)
    context_lens = (B,)
    block_tables = (B, max_num_blocks)

注意: Decode 阶段不再跑 Vision Tower (图像特征已在 Prefill 时写入 KV cache)
```

---

## 五、Qwen3VLForCausalLM 模型前向 (完整结构)

```
基于 qwen3_vl.py:431 Qwen3VLForEmbedding 改造为生成版本

=======================================================================
                  Qwen3VLForCausalLM.forward()
=======================================================================

  +---------------------------------------------------------------------+
  |  input_ids                                                          |
  |  Prefill: shape (N,)        N = total tokens (varlen packed)        |
  |  Decode:  shape (B,)        B = batch_size                          |
  +-------------------------------+-------------------------------------+
                                  |
                                  v
  +---------------------------------------------------------------------+
  |  VocabParallelEmbedding (embed_tokens)                              |
  |  weight: (num_embeddings_per_partition, 2048)                       |
  |                                                                     |
  |  Prefill: (N,) --> (N, 2048)                                       |
  |  Decode:  (B,) --> (B, 2048)                                       |
  +-------------------------------+-------------------------------------+
                                  |
                                  |  x: (N, 2048) or (B, 2048)
                                  v
  +---------------------------------------------------------------------+
  |  Vision Tower + Scatter (仅 Prefill 且有图像时)                     |
  |                                                                     |
  |  visual_emb, deepstack = VisionTower(pixel_values, grid_thw)        |
  |    visual_emb: (merged_patches, 2048)                               |
  |    deepstack: list of 3 x (merged_patches, 2048)                   |
  |                                                                     |
  |  x[img_idx] = visual_emb   <-- 覆盖 image_pad 位置                |
  +-------------------------------+-------------------------------------+
                                  |
                                  |  residual = None
                                  v
  +----------- x 28 layers -------------------------------------------+
  |                                                                    |
  |   Qwen3VLDecoderLayer[i]  (i = 0..27)                              |
  |                                                                    |
  |   +----------------------------------------------------------+     |
  |   |  1. Input RMSNorm (in_ln) + residual                     |     |
  |   |     weight: (2048,)                                      |     |
  |   |     (N, 2048) --> (N, 2048)                               |     |
  |   +-----------------------------+----------------------------+     |
  |                                 |                                  |
  |                                 v                                  |
  |   +----------------------------------------------------------+     |
  |   |  2. Qwen3VLAttention (attn)                              |     |
  |   |     QKVColumnParallelLinear + Q/K Norm + MRoPE          |     |
  |   |     + Flash/Paged Attention + RowParallelLinear           |     |
  |   |     (N, 2048) --> (N, 2048)                               |     |
  |   +-----------------------------+----------------------------+     |
  |                                 |                                  |
  |   ------ (+) ------  <-- 残差: x = attn_out + residual             |
  |                                 |                                  |
  |                                 v                                  |
  |   +----------------------------------------------------------+     |
  |   |  3. Post-Attention RMSNorm (post_ln) + residual           |     |
  |   |     (N, 2048) --> (N, 2048)                               |     |
  |   +-----------------------------+----------------------------+     |
  |                                 |                                  |
  |                                 v                                  |
  |   +----------------------------------------------------------+     |
  |   |  4. Qwen3VLMLP (mlp)  SwiGLU                             |     |
  |   |     gate_up --> SiLU*Mul --> down                         |     |
  |   |     (N, 2048) --> (N, 2048)                               |     |
  |   +-----------------------------+----------------------------+     |
  |                                 |                                  |
  |   +----------------------------------------------------------+     |
  |   |  DeepStack 注入 (如果 i < 3 且有图像)                    |     |
  |   |     Layer 0: x[img_idx] += deepstack[0]                   |     |
  |   |     Layer 1: x[img_idx] += deepstack[1]                   |     |
  |   |     Layer 2: x[img_idx] += deepstack[2]                   |     |
  |   +----------------------------------------------------------+     |
  |                                                                    |
  |   return (x, residual)                                             |
  +-------------------------------+------------------------------------+
                                  |
                                  v
  +---------------------------------------------------------------------+
  |  Final RMSNorm (norm)                                               |
  |  x = x + residual; x = RMSNorm(x)                                  |
  |  weight: (2048,)                                                    |
  |                                                                     |
  |  (N, 2048) --> (N, 2048)    [Prefill]                              |
  |  (B, 2048) --> (B, 2048)    [Decode]                               |
  +-------------------------------+-------------------------------------+
                                  |
                                  v
  +---------------------------------------------------------------------+
  |  compute_logits()                                                   |
  |                                                                     |
  |  +---------------------------------------------------------------+  |
  |  |  ParallelLMHead (lm_head)                                     |  |
  |  |  weight: (num_embeddings_per_partition, 2048)                 |  |
  |  |                                                               |  |
  |  |  Prefill:                                                     |  |
  |  |    cu_seqlens_q = [0, n0, n0+n1, ...]                        |  |
  |  |    last_token_indices = [n0-1, n0+n1-1, ...]                 |  |
  |  |    x = x[last_token_indices]                                  |  |
  |  |    (N, 2048) --> (num_seqs, 2048)                             |  |
  |  |                                                               |  |
  |  |  Decode:                                                      |  |
  |  |    x 直接就是 (B, 2048)                                       |  |
  |  |                                                               |  |
  |  |  logits = F.linear(x, weight)                                 |  |
  |  |    --> (num_seqs, vocab_per_partition)    [Prefill]           |  |
  |  |    --> (B, vocab_per_partition)           [Decode]            |  |
  |  |                                                               |  |
  |  |  TP>1: gather to rank0 + cat + trim                           |  |
  |  |    --> (num_seqs, 151936)    [Prefill]                        |  |
  |  |    --> (B, 151936)           [Decode]                         |  |
  |  +---------------------------------------------------------------+  |
  |                                                                     |
  |  return logits                                                      |
  +-------------------------------+-------------------------------------+
                                  |
                                  v
  +---------------------------------------------------------------------+
  |  SamplerLayer (sampler)  (仅 rank 0 执行)                           |
  |                                                                     |
  |  logits: (num_seqs, 151936) or (B, 151936)                         |
  |  temperature: (num_seqs,) or (B,)                                   |
  |                                                                     |
  |  1. logits /= temperature.unsqueeze(-1)                             |
  |  2. probs = softmax(logits, dim=-1)                                 |
  |  3. Gumbel-max: tokens = (probs / exp_noise).argmax(dim=-1)        |
  |     --> (num_seqs,) or (B,)    dtype: int64                         |
  |                                                                     |
  |  return token_ids                                                   |
  +---------------------------------------------------------------------+
```

---

## 六、Vision Tower 完整结构 (与 Embedding 版相同)

```
VisionTower.forward()  qwen3_vl.py:183

  pixel_values: (total_patches, 1536)
  grid_thw: (num_images, 3)
  |
  v
  PatchEmbed3D: Conv3d(3, 1024, k=(2,16,16), s=(2,16,16))
    (total_patches, 1536) --> (total_patches, 1024)
  |
  v
  + 2D 位置嵌入 (48x48 双线性插值到实际 h,w)
    (total_patches, 1024)
  |
  v
  24 x VisionBlock:
    LayerNorm --> VisionAttention(双向,2D RoPE) --> 残差
    LayerNorm --> MLP(GELU) --> 残差
    DeepStack 抽取 @ 层 [5, 11, 17]
  |
  v
  _merge2x2: 2x2 patch 折叠
    (total_patches, 1024) --> (merged_patches, 4096)
  |
  v
  主 Merger: LN(1024) + fc1(4096,4096) + GELU + fc2(4096,2048)
    --> visual_emb: (merged_patches, 2048)
  |
  return visual_emb, deepstack (list of 3 tensors)
```

---

## 七、KV Cache 结构

```
model_runner.py:244  allocate_kv_cache()

全局 KV Cache 池:

  allocated_kv_cache = torch.zeros(
      2,                    # [0]=K, [1]=V
      num_layers,           # 28
      max_cached_blocks,    # 由 GPU 可用显存计算
      block_size,           # 256
      num_kv_heads,         # 8  (per-GPU, = total/tp)
      head_dim,             # 128
  )

  shape: (2, 28, max_cached_blocks, 256, 8, 128)

  每层 Attention 模块持有:
    k_cache = allocated_kv_cache[0, layer_id]
              shape: (max_cached_blocks, 256, 8, 128)
    v_cache = allocated_kv_cache[1, layer_id]
              shape: (max_cached_blocks, 256, 8, 128)

  Prefill 时: 图像 token 的 K/V 也写入 cache
    --> Decode 时可以直接 attend 到图像特征 (无需重跑 Vision Tower)
```

---

## 八、CUDA Graph 加速 (Decode 阶段)

```
model_runner.py:572  capture_cudagraph()

预分配固定大小缓冲区:
  input_ids:    (max_bs,)           dtype: int64
  slot_mapping: (max_bs,)           dtype: int64
  context_lens: (max_bs,)           dtype: int64
  block_tables: (max_bs, max_num_blocks)  dtype: int32
  outputs:      (max_bs, 2048)      dtype: float32

捕获 batch_sizes = [1, 2, 4, 8, 16, 32, ...]

执行时:
  1. 找到 >= 实际 bs 的最小捕获图
  2. 将实际数据 copy 进预分配缓冲区
  3. graph.replay()  --> 重放捕获的 CUDA kernel 序列
  4. 从 outputs[:bs] 取结果

  --> 避免每次 decode 的 kernel launch 开销
```

---

## 九、完整 Shape 汇总表 (Prefill, N tokens, tp=1)

| 模块 | 操作 | 输入 Shape | 输出 Shape |
|------|------|-----------|-----------|
| **Vision Tower** | | | |
| PatchEmbed3D | Conv3d | (P, 1536) | (P, 1024) |
| pos_embed | Embedding+插值 | (h*w,) | (h*w, 1024) |
| VisionBlock x24 | LN+Attn+MLP | (P, 1024) | (P, 1024) |
| _merge2x2 | 2x2 折叠 | (P, 1024) | (P/4, 4096) |
| PatchMerger (主) | LN+MLP | (P/4, 4096) | (P/4, 2048) |
| PatchMerger (DS) x3 | MLP+LN | (P/4, 4096) | (P/4, 2048) |
| **文本 Decoder** | | | |
| VocabParallelEmbedding | 查表 | (N,) | (N, 2048) |
| scatter visual_emb | 覆盖 | (N, 2048) | (N, 2048) |
| RMSNorm (in_ln) | 归一化 | (N, 2048) | (N, 2048) |
| QKVColumnParallelLinear | qkv投影 | (N, 2048) | (N, 4096) |
| Split+Reshape | 拆分 | (N, 4096) | q:(N,16,128) 等 |
| Q/K Norm | RMSNorm | (N, H, 128) | (N, H, 128) |
| MRoPE | 3D旋转 | q,k | q,k |
| Flash Attention | 因果attn | q,k,v | (N, 16, 128) |
| store_kvcache | 写入KV | k,v | KV cache |
| RowParallelLinear (o) | 输出投影 | (N, 2048) | (N, 2048) |
| RMSNorm (post_ln) | 归一化 | (N, 2048) | (N, 2048) |
| MergedColumnParallelLinear | gate_up | (N, 2048) | (N, 12288) |
| SiluAndMul | SwiGLU | (N, 12288) | (N, 6144) |
| RowParallelLinear (down) | down投影 | (N, 6144) | (N, 2048) |
| DeepStack 注入 x3 | 加法 | (N, 2048) | (N, 2048) |
| x 28 layers | 循环 | (N, 2048) | (N, 2048) |
| Final RMSNorm | 归一化 | (N, 2048) | (N, 2048) |
| ParallelLMHead | gather last + linear | (N, 2048)->(S, 2048) | (S, 151936) |
| SamplerLayer | temp+softmax+gumbel | (S, 151936) | (S,) |

> P = total_patches, N = total_tokens, S = num_seqs

---

## 十、完整 Shape 汇总表 (Decode, B batch, tp=1)

| 模块 | 操作 | 输入 Shape | 输出 Shape |
|------|------|-----------|-----------|
| prepare_decode | gather last token | seqs | (B,) |
| VocabParallelEmbedding | 查表 | (B,) | (B, 2048) |
| RMSNorm (in_ln) | 归一化 | (B, 2048) | (B, 2048) |
| QKVColumnParallelLinear | qkv投影 | (B, 2048) | (B, 4096) |
| Split+Reshape | 拆分 | (B, 4096) | q:(B,16,128) 等 |
| Q/K Norm | RMSNorm | (B, H, 128) | (B, H, 128) |
| MRoPE | 3D旋转 | q,k | q,k |
| store_kvcache | 写入KV | k,v | KV cache |
| Paged Attention | decode kernel | q + KV cache | (B, 16, 128) |
| RowParallelLinear (o) | 输出投影 | (B, 2048) | (B, 2048) |
| RMSNorm (post_ln) | 归一化 | (B, 2048) | (B, 2048) |
| MergedColumnParallelLinear | gate_up | (B, 2048) | (B, 12288) |
| SiluAndMul | SwiGLU | (B, 12288) | (B, 6144) |
| RowParallelLinear (down) | down投影 | (B, 6144) | (B, 2048) |
| x 28 layers | 循环 | (B, 2048) | (B, 2048) |
| Final RMSNorm | 归一化 | (B, 2048) | (B, 2048) |
| ParallelLMHead | linear | (B, 2048) | (B, 151936) |
| SamplerLayer | temp+softmax+gumbel | (B, 151936) | (B,) |

注意: Decode 阶段不跑 Vision Tower, 图像特征已在 Prefill 时通过
      store_kvcache 写入 KV cache, Paged Attention 直接读取。

---

## 十一、关键源码文件索引

| 文件 | 核心类/函数 | 职责 |
|------|------------|------|
| models/qwen3_vl.py | Qwen3VLForEmbedding, VisionTower, Qwen3VLDecoderLayer | 模型定义 (需适配为 CausalLM) |
| layers/attention.py | Attention, flash_attention_prefill, paged_attention_decode | Triton attention kernels |
| layers/linear.py | QKVColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear | TP 线性层 |
| layers/layernorm.py | LayerNorm (RMSNorm) | 归一化 |
| layers/rotary_embedding.py | MRotaryEmbedding, VisionRotaryEmbedding | 3D/2D RoPE |
| layers/activation.py | SiluAndMul | SwiGLU 激活 |
| layers/embedding_head.py | VocabParallelEmbedding, ParallelLMHead | 嵌入层 + LM head |
| layers/sampler.py | SamplerLayer | Gumbel-max 采样 |
| engine/llm_engine.py | LLMEngine | 引擎入口, 调度循环 |
| engine/model_runner.py | ModelRunner | GPU 执行, KV cache, CUDA graph |
| engine/scheduler.py | Scheduler | 连续批处理调度 |
| engine/block_manager.py | BlockManager | Paged KV 地址空间管理 |
| utils/context.py | Context, set_context, get_context | Attention 元数据单例 |
