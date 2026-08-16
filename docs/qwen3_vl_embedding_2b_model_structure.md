# Qwen3-VL-Embedding-2B 模型结构全流程 (HermesInfer 实现)

> 基于 `src/myvllm/models/qwen3_vl.py` 及相关 layers/engine 源码梳理。
> 涵盖从多模态输入到 embedding 输出的完整数据流，每个模块均标注 shape 信息。

## 模型配置参数 (main_embedding.py)

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
num_position_embeddings = 2304  (48×48 学习位置网格)
deepstack_visual_indexes = [5, 11, 17]

# ---- Pooling ----
pooling_mode       = "last_token"
normalize          = True
mrl_dim            = None      (可选 Matryoshka 截断)
```

---

## 一、端到端总览 (Engine Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLMEngine.encode()                                  │
│                         llm_engine.py:143                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  inputs: list[dict]                                                         │
│    每条: {"text": str} | {"image": path} | {"text":str, "image":path}       │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────┐                               │
│  │  AutoProcessor.apply_chat_template()     │                               │
│  │  + process_vision_info()                  │                               │
│  │  → input_ids, mm_token_type_ids,         │                               │
│  │    pixel_values, image_grid_thw,         │                               │
│  │    image_token_spans                     │                               │
│  └──────────────┬───────────────────────────┘                               │
│                 ▼                                                           │
│  ┌──────────────────────────────────────────┐                               │
│  │  Scheduler.add_sequence()                │                               │
│  │  Sequence(token_ids, mm_data)            │                               │
│  │  入队 waiting: deque[Sequence]           │                               │
│  └──────────────┬───────────────────────────┘                               │
│                 ▼                                                           │
│  ┌─────────── while not is_finished() ──────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────┐                              │   │
│  │  │  Scheduler._schedule_pooling()     │                              │   │
│  │  │  纯 prefill, 无 decode/preempt     │                              │   │
│  │  │  按 token/image-patch 预算分批     │                              │   │
│  │  │  return (seqs, is_prefill=True)    │                              │   │
│  │  └──────────────┬─────────────────────┘                              │   │
│  │                 ▼                                                    │   │
│  │  ┌────────────────────────────────────┐                              │   │
│  │  │  ModelRunner._run_pooling()        │                              │   │
│  │  │  ① _prepare_prefill_vl()           │                              │   │
│  │  │    - 打包 input_ids + cu_seqlens   │                              │   │
│  │  │    - 计算 MRoPE 3D 位置            │                              │   │
│  │  │    - 收集 pixel_values/grid_thw    │                              │   │
│  │  │  ② model.forward()                 │                              │   │
│  │  │    - VisionTower → visual_emb      │                              │   │
│  │  │    - scatter 到 image token 位置   │                              │   │
│  │  │    - 28 层文本 decoder             │                              │   │
│  │  │    - DeepStack 注入 @ [0,1,2]      │                              │   │
│  │  │    - EmbeddingHead → embedding     │                              │   │
│  │  │  return embedding: (num_seqs, dim) │                              │   │
│  │  └──────────────┬─────────────────────┘                              │   │
│  │                 ▼                                                    │   │
│  │  ┌────────────────────────────────────┐                              │   │
│  │  │  Scheduler.postprocess()           │                              │   │
│  │  │  标记 FINISHED, 移出 running       │                              │   │
│  │  └────────────────────────────────────┘                              │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                 ▼                                                           │
│  output: list[torch.Tensor]                                                 │
│    每条: (embed_dim,) 归一化向量                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Vision Tower 完整结构

```
qwen3_vl.py:141  VisionTower

输入:
  pixel_values: (total_patches, C * T_patch * P * P)
               = (total_patches, 3 * 2 * 16 * 16)
               = (total_patches, 1536)
  grid_thw: (num_images, 3)   每图 (t, h, w) patch 数

═══════════════════════════════════════════════════════════════════════════
                        VisionTower.forward()
═══════════════════════════════════════════════════════════════════════════

  pixel_values: (total_patches, 1536)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PatchEmbed3D (patch_embed)                                          │
│  qwen3_vl.py:45                                                      │
│                                                                      │
│  Conv3d: in_channels=3, out_channels=1024                            │
│        kernel=(2, 16, 16), stride=(2, 16, 16), bias=True             │
│                                                                      │
│  reshape: (total_patches, 1536)                                      │
│        → (total_patches, 3, 2, 16, 16)                               │
│  Conv3d → (total_patches, 1024, 1, 1, 1)                             │
│  flatten → (total_patches, 1024)                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2D 位置嵌入 (pos_embed) + 双线性插值                                │
│  qwen3_vl.py:164                                                     │
│                                                                      │
│  pos_embed: nn.Embedding(2304, 1024)                                 │
│           = 48×48 学习位置网格                                       │
│                                                                      │
│  对每张图 (t, h, w):                                                 │
│    从 48×48 双线性插值到 (h, w)                                      │
│    pos: (h*w, 1024) → repeat t 帧 → (t*h*w, 1024)                   │
│                                                                      │
│  h = h + pos                                                         │
│  (total_patches, 1024) ──→ (total_patches, 1024)                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  VisionRotaryEmbedding 位置准备                                      │
│  rotary_embedding.py:183                                             │
│                                                                      │
│  对每张图: meshgrid(arange(h), arange(w)) → (h*w, 2) = (hpos, wpos) │
│  按 t 帧 repeat → (t*h*w, 2)                                        │
│  拼接所有图 → position_ids: (total_patches, 2)                       │
│                                                                      │
│  cu_seqlens: (num_images+1,)  各图 patch 边界                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌─────────── × 24 blocks ─────────────────────────────────────────────┐
│                                                                      │
│   VisionBlock[i]  (i = 0..23)                                        │
│   qwen3_vl.py:123                                                    │
│                                                                      │
│   输入: h: (total_patches, 1024)                                     │
│                                                                      │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │  ① LayerNorm (n1)                                         │     │
│   │  nn.LayerNorm(1024, eps=1e-6)                              │     │
│   │  (total_patches, 1024) → (total_patches, 1024)             │     │
│   └──────────────────────────┬─────────────────────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │  ② VisionAttention (attn)  双向, 无因果 mask              │     │
│   │  qwen3_vl.py:92                                            │     │
│   │                                                            │     │
│   │  (详见下方 第三节 ViT Attention 展开)                       │     │
│   │                                                            │     │
│   │  (total_patches, 1024) → (total_patches, 1024)             │     │
│   └──────────────────────────┬─────────────────────────────────┘     │
│                              │                                       │
│   ────── ⊕ ──────  ← 残差: h = h + attn_out                        │
│                              │                                       │
│                              ▼                                       │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │  ③ LayerNorm (n2)                                         │     │
│   │  nn.LayerNorm(1024, eps=1e-6)                              │     │
│   │  (total_patches, 1024) → (total_patches, 1024)             │     │
│   └──────────────────────────┬─────────────────────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │  ④ ViT MLP                                                │     │
│   │  mlp_fc1: Linear(1024, 4096, bias=True)                    │     │
│   │  act: GELU(approximate="tanh")                              │     │
│   │  mlp_fc2: Linear(4096, 1024, bias=True)                    │     │
│   │                                                            │     │
│   │  (total_patches, 1024) → (total_patches, 4096)             │     │
│   │                    → (total_patches, 4096)                 │     │
│   │                    → (total_patches, 1024)                 │     │
│   └──────────────────────────┬─────────────────────────────────┘     │
│                              │                                       │
│   ────── ⊕ ──────  ← 残差: h = h + mlp_out                         │
│                              │                                       │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │  DeepStack 特征抽取 (如果 i ∈ [5, 11, 17])                 │     │
│   │                                                            │     │
│   │  _merge2x2(h, shapes):                                     │     │
│   │    每张图 (t, h, w, 1024) → 2×2 折叠                       │     │
│   │    → (t, h/2, w/2, 4*1024) → flatten                       │     │
│   │    → (t*(h/2)*(w/2), 4096)                                 │     │
│   │                                                            │     │
│   │  dsm[idx]: PatchMerger(use_postshuffle_norm=True)           │     │
│   │    LayerNorm(4096) → fc1: Linear(4096, 4096)               │     │
│   │                    → GELU → fc2: Linear(4096, 2048)        │     │
│   │    → (merged_patches, 2048)                                 │     │
│   │                                                            │     │
│   │  merged_patches = sum(t*(h/2)*(w/2)) 对所有图              │     │
│   └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  主 Merger (merger)                                                  │
│  qwen3_vl.py:65  PatchMerger(use_postshuffle_norm=False)             │
│                                                                      │
│  _merge2x2(h, shapes):                                               │
│    每张图 (t, h, w, 1024) → 2×2 折叠                                 │
│    → (t*(h/2)*(w/2), 4096)                                           │
│                                                                      │
│  LayerNorm(1024)  ← use_postshuffle_norm=False, norm 在合并前       │
│  reshape → (merged_patches, 4096)                                    │
│  fc1: Linear(4096, 4096, bias=True)                                  │
│  GELU                                                                │
│  fc2: Linear(4096, 2048, bias=True)                                  │
│                                                                      │
│  visual_emb: (merged_patches, 2048)                                  │
│                                                                      │
│  其中 merged_patches = total_patches / 4                             │
│  (因为 2×2 = 4 个 patch 合并为 1 个 token)                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
  return:
    visual_emb: (merged_patches, 2048)      ← 主视觉 embedding
    deepstack: list of 3 tensors            ← ViT 层 [5,11,17] 的特征
      每个: (merged_patches, 2048)
```

---

## 三、ViT Attention 展开 (VisionAttention)

```
qwen3_vl.py:92  VisionAttention

输入: x: (total_patches, 1024)
      position_ids: (total_patches, 2)   ← (hpos, wpos)
      cu_seqlens: (num_images+1,)

═══════════════════════════════════════════════════════════════════════════
                      VisionAttention.forward()
═══════════════════════════════════════════════════════════════════════════

  x: (S, 1024)    S = total_patches
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  QKV Linear (qkv)                                                    │
│  nn.Linear(1024, 3072, bias=True)                                    │
│                                                                      │
│  qkv = x @ W^T + b                                                   │
│  (S, 1024) → (S, 3072)                                               │
│                                                                      │
│  reshape → (S, 3, 16, 64) → permute → (3, S, 16, 64)                │
│  unbind → q, k, v 各 (S, 16, 64)                                    │
│                                                                      │
│  注意: ViT 的 head_dim = 1024/16 = 64                                │
│  注意: 无 GQA, Q/K/V 头数相同 = 16                                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  VisionRotaryEmbedding + rotate_half                                 │
│  rotary_embedding.py:183                                             │
│                                                                      │
│  dim = head_dim // 2 = 32                                            │
│  theta = 10000                                                       │
│                                                                      │
│  inv_freq = 1 / (10000 ^ (arange(0, 32, 2) / 32))                   │
│           shape: (16,)                                               │
│                                                                      │
│  freqs = (position_ids.unsqueeze(-1) * inv_freq).flatten(1)          │
│        shape: (S, 32)                                                │
│                                                                      │
│  freqs = cat([freqs, freqs], dim=-1)                                 │
│        shape: (S, 64) = (S, head_dim)                                │
│                                                                      │
│  cos = freqs.cos()    sin = freqs.sin()                              │
│    各 (S, 64) → unsqueeze(1) → (S, 1, 64)                           │
│                                                                      │
│  rotate_half:                                                        │
│    a, b = q.chunk(2, dim=-1)     各 (S, 16, 32)                     │
│    rotate_half(q) = cat(-b, a)   (S, 16, 64)                        │
│                                                                      │
│  q = q * cos + rotate_half(q) * sin                                  │
│  k = k * cos + rotate_half(k) * sin                                  │
│    各 (S, 16, 64) → (S, 16, 64)                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Flash Attention (双向, is_causal=False)                              │
│  attention.py:468  Attention(num_heads=16, head_dim=64,              │
│                              num_kv_heads=16, is_causal=False)        │
│                                                                      │
│  q: (S, 16, 64)                                                      │
│  k: (S, 16, 64)                                                      │
│  v: (S, 16, 64)                                                      │
│  cu_seqlens: (num_images+1,)                                         │
│                                                                      │
│  双向注意力: 每个 patch 可以 attend 到同一张图的所有 patch             │
│  (无因果 mask, IS_CAUSAL=False)                                      │
│                                                                      │
│  output: (S, 16, 64) → reshape → (S, 1024)                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Output Projection (proj)                                            │
│  nn.Linear(1024, 1024, bias=True)                                    │
│                                                                      │
│  (S, 1024) → (S, 1024)                                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 四、文本 Decoder 完整前向 (Qwen3VLForEmbedding)

```
qwen3_vl.py:431  Qwen3VLForEmbedding

═══════════════════════════════════════════════════════════════════════════
                  Qwen3VLForEmbedding.forward()
═══════════════════════════════════════════════════════════════════════════

  input_ids: (N,)           N = total_tokens (varlen packed)
  pixel_values: (total_patches, 1536) | None
  grid_thw: (num_images, 3) | None
  image_token_spans: list[(start, end)] | None
  cu_seqlens_q: (num_seqs+1,)
  pos3d: (3, N)             ← 从 context 读取, MRoPE 3D 位置

  ┌─────────────────────────────────────────────────────────────────────┐
  │  VocabParallelEmbedding (embed_tokens)                              │
  │  embedding_head.py:12                                               │
  │  weight: (num_embeddings_per_partition, 2048)                       │
  │                                                                     │
  │  (N,) ──→ (N, 2048)                                                │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 │  x: (N, 2048)
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Vision Tower + Scatter (如果有图像)                                │
  │                                                                     │
  │  if pixel_values is not None and image_token_spans:                 │
  │                                                                     │
  │    visual_emb, deepstack = VisionTower(pixel_values, grid_thw)      │
  │      visual_emb: (merged_patches, 2048)                             │
  │      deepstack: list of 3 × (merged_patches, 2048)                 │
  │                                                                     │
  │    img_idx = [s, s+1, ..., e-1 for (s,e) in image_token_spans]      │
  │      shape: (merged_patches,)                                      │
  │                                                                     │
  │    x[img_idx] = visual_emb   ← 覆盖 <|image_pad|> 的 embedding     │
  │                                                                     │
  │  x: (N, 2048)  (文本 token 保持原 embedding, 图像 token 被替换)    │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 │  residual = None
                                 ▼
  ┌─────────── × 28 layers ───────────────────────────────────────────┐
  │                                                                    │
  │   Qwen3VLDecoderLayer[i]  (i = 0..27)                              │
  │   qwen3_vl.py:405                                                  │
  │                                                                    │
  │   输入: x: (N, 2048), residual: (N, 2048) | None, pos3d: (3, N)   │
  │                                                                    │
  │   ┌──────────────────────────────────────────────────────────┐     │
  │   │  ① Input RMSNorm (in_ln)                                │     │
  │   │  + residual 融合                                         │     │
  │   │  weight: (2048,)                                         │     │
  │   │                                                          │     │
  │   │  首次: residual = x; x = RMSNorm(x)                      │     │
  │   │  后续: x = x + residual; residual = x; x = RMSNorm(x)   │     │
  │   │                                                          │     │
  │   │  (N, 2048) → (N, 2048)                                   │     │
  │   └──────────────────────┬───────────────────────────────────┘     │
  │                          │                                         │
  │                          ▼                                         │
  │   ┌──────────────────────────────────────────────────────────┐     │
  │   │  ② Qwen3VLAttention (attn)                              │     │
  │   │  qwen3_vl.py:359                                         │     │
  │   │                                                          │     │
  │   │  (详见下方 第五节 展开图)                                 │     │
  │   │                                                          │     │
  │   │  (N, 2048) → (N, 2048)                                   │     │
  │   └──────────────────────┬───────────────────────────────────┘     │
  │                          │                                         │
  │                          │  x: (N, 2048), residual: (N, 2048)     │
  │                          ▼                                         │
  │   ┌──────────────────────────────────────────────────────────┐     │
  │   │  ③ Post-Attention RMSNorm (post_ln) + residual           │     │
  │   │  x = x + residual; residual = x; x = RMSNorm(x)         │     │
  │   │  (N, 2048) → (N, 2048)                                   │     │
  │   └──────────────────────┬───────────────────────────────────┘     │
  │                          │                                         │
  │                          ▼                                         │
  │   ┌──────────────────────────────────────────────────────────┐     │
  │   │  ④ Qwen3VLMLP (mlp)                                     │     │
  │   │  qwen3_vl.py:392                                         │     │
  │   │                                                          │     │
  │   │  (详见下方 第六节 展开图)                                 │     │
  │   │                                                          │     │
  │   │  (N, 2048) → (N, 2048)                                   │     │
  │   └──────────────────────┬───────────────────────────────────┘     │
  │                          │                                         │
  │   ┌──────────────────────────────────────────────────────────┐     │
  │   │  DeepStack 注入 (如果 i < len(deepstack) 且有图像)       │     │
  │   │                                                          │     │
  │   │  层 0 ← deepstack[0] (来自 ViT 层 5)                     │     │
  │   │  层 1 ← deepstack[1] (来自 ViT 层 11)                    │     │
  │   │  层 2 ← deepstack[2] (来自 ViT 层 17)                    │     │
  │   │                                                          │     │
  │   │  x[img_idx] = x[img_idx] + deepstack[li]                 │     │
  │   │  (仅在图像 token 位置加, 文本 token 不受影响)             │     │
  │   └──────────────────────────────────────────────────────────┘     │
  │                                                                    │
  │   return (x, residual)                                             │
  └────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Final RMSNorm (norm)                                               │
  │  x = x + residual; x = RMSNorm(x)                                  │
  │  weight: (2048,)                                                    │
  │                                                                     │
  │  (N, 2048) → (N, 2048)                                             │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  EmbeddingHead (head)                                               │
  │  qwen3_vl.py:332                                                    │
  │                                                                     │
  │  ① Last-token gather:                                              │
  │     cu_seqlens_q = [0, n0, n0+n1, ...]                             │
  │     last_indices = [n0-1, n0+n1-1, ...]                            │
  │     pooled = x[last_indices]                                       │
  │     (N, 2048) → (num_seqs, 2048)                                   │
  │                                                                     │
  │  ② L2 归一化:                                                      │
  │     pooled = pooled / ||pooled||_2                                  │
  │     (num_seqs, 2048) → (num_seqs, 2048)                            │
  │                                                                     │
  │  ③ 可选 MRL 截断:                                                  │
  │     if mrl_dim is not None:                                        │
  │       pooled = pooled[:, :mrl_dim]                                 │
  │       pooled = pooled / ||pooled||_2                                │
  │       (num_seqs, 2048) → (num_seqs, mrl_dim)                       │
  │                                                                     │
  │  return: (num_seqs, embed_dim)                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 五、Qwen3VLAttention 展开图

```
qwen3_vl.py:359  Qwen3VLAttention

输入: x: (N, 2048)    pos3d: (3, N)   ← T/H/W 3D 位置

═══════════════════════════════════════════════════════════════════════════
                     Qwen3VLAttention.forward()
═══════════════════════════════════════════════════════════════════════════

  x: (N, 2048)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  QKVColumnParallelLinear (qkv)                                       │
│  linear.py:152                                                       │
│                                                                      │
│  weight: (head_dim*(num_heads+2*num_kv_heads), hidden_size)          │
│        = (128*(16+8+8), 2048) = (4096, 2048)    [full, before TP]    │
│  bias: None  (bias=False)                                            │
│                                                                      │
│  qkv = F.linear(x, weight)                                           │
│  (N, 2048) → (N, 4096)        [tp=1]                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Split QKV                                                           │
│  q_size = 128 * 16 = 2048    kv_size = 128 * 8 = 1024   [tp=1]      │
│                                                                      │
│  q, k, v = qkv.split([2048, 1024, 1024], dim=-1)                     │
│  q: (N, 2048)    k: (N, 1024)    v: (N, 1024)                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Reshape (varlen 模式)                                               │
│  q: (N, 2048) → (N, 16, 128)                                        │
│  k: (N, 1024) → (N, 8, 128)                                         │
│  v: (N, 1024) → (N, 8, 128)                                         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Q Norm + K Norm (RMSNorm per-head)                                  │
│  q_norm: LayerNorm  weight: (128,)                                   │
│  k_norm: LayerNorm  weight: (128,)                                   │
│                                                                      │
│  q: (N, 16, 128) → (N, 16, 128)                                     │
│  k: (N, 8, 128)  → (N, 8, 128)                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MRotaryEmbedding (rotary)  ← 3D 位置 T/H/W                         │
│  rotary_embedding.py:118                                             │
│                                                                      │
│  base = 5,000,000                                                    │
│  head_dim = 128                                                      │
│  mrope_section = [24, 20, 20]                                       │
│                                                                      │
│  (详见下方 MRoPE 展开)                                               │
│                                                                      │
│  q: (N, 16, 128) → (N, 16, 128)                                     │
│  k: (N, 8, 128)  → (N, 8, 128)                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Flash Attention (因果, is_causal=True)                               │
│  attention.py:468                                                    │
│                                                                      │
│  q: (N, 16, 128)   k: (N, 8, 128)   v: (N, 8, 128)                 │
│  cu_seqlens: (num_seqs+1,)                                           │
│  scale = 1.0 / sqrt(128) ≈ 0.0884                                   │
│                                                                      │
│  GQA: 每 2 个 Q head 共享 1 个 KV head                               │
│  因果 mask: 序列内 token i 只能 attend 到 ≤ i                        │
│                                                                      │
│  output: (N, 16, 128) → reshape → (N, 2048)                          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RowParallelLinear (o)                                               │
│  weight: (2048, 2048/tp)    bias: None                               │
│                                                                      │
│  (N, 2048) → (N, 2048)                                               │
│  TP>1: all_reduce(SUM)                                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 六、MRoPE 展开 (3D 旋转位置编码)

```
rotary_embedding.py:118  MRotaryEmbedding

输入: positions_3d: (3, N)   ← [T, H, W] 三条位置轴
      query: (N, num_heads, 128)
      key:   (N, num_kv_heads, 128)

═══════════════════════════════════════════════════════════════════════════

  mrope_section = [24, 20, 20]
  head_dim = 128, head_dim/2 = 64 个频率

  频率布局 (交错式):
  ┌──────────────────────────────────────────────────────────────┐
  │  前 min(section)=20 组三轴交错:                              │
  │    [T0, H0, W0, T1, H1, W1, ..., T19, H19, W19]            │
  │    = 60 个频率槽                                             │
  │                                                              │
  │  尾部 T 轴独占:                                              │
  │    [T20, T21, T22, T23]                                     │
  │    = 4 个频率槽                                              │
  │                                                              │
  │  合计: 60 + 4 = 64 = head_dim / 2                           │
  └──────────────────────────────────────────────────────────────┘

  inv_freq: (64,) = 1 / (base ^ (arange(0, 128, 2) / 128))

  mrope_perm: (64,)  ← 每个输出槽取哪个 inv_freq 源索引
  mrope_section_id: (64,)  ← 每个输出槽属于 T(0)/H(1)/W(2)

  forward:
    inv_freq_perm = inv_freq[mrope_perm]          # (64,)
    pos = positions_3d[mrope_section_id]           # (64, N) 按段选 T/H/W
    freqs = inv_freq_perm[:, None] * pos           # (64, N) 角度
    freqs = freqs.t()                              # (N, 64)
    cos = freqs.cos()    sin = freqs.sin()          # (N, 64)

    apply_rotary_pos_emb(query, cos, sin):
      q1, q2 = query.chunk(2, dim=-1)              # 各 (N, heads, 64)
      out1 = q1 * cos - q2 * sin
      out2 = q1 * sin + q2 * cos
      q_rot = cat([out1, out2], dim=-1)            # (N, heads, 128)

    k 同理
```

---

## 七、Qwen3VLMLP 展开图

```
qwen3_vl.py:392  Qwen3VLMLP

输入: x: (N, 2048)

═══════════════════════════════════════════════════════════════════════════

  x: (N, 2048)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MergedColumnParallelLinear (gate_up)                                │
│  output_sizes = [6144, 6144]                                         │
│  weight: (12288/tp, 2048)    bias: (12288/tp,)    [bias=True]        │
│                                                                      │
│  (N, 2048) → (N, 12288)        [tp=1]                                │
│  输出布局: [gate_out | up_out]                                        │
│              ← 6144 → ← 6144 →                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SiluAndMul (SwiGLU)                                                 │
│  gate_out, up_out = chunk(2, dim=-1)                                  │
│  output = SiLU(gate_out) × up_out                                    │
│                                                                      │
│  (N, 12288) → (N, 6144)                                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RowParallelLinear (down)                                            │
│  weight: (2048, 6144/tp)    bias: (2048,)    [bias=True]             │
│                                                                      │
│  (N, 6144) → (N, 2048)                                               │
│  TP>1: all_reduce(SUM)                                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 八、DeepStack 机制详解

```
DeepStack: 从 ViT 中间层抽取特征, 注入到文本 decoder 的早期层

═══════════════════════════════════════════════════════════════════════════

Vision Tower (24 层):
  Layer 0  ─┐
  Layer 1   │
  Layer 2   │
  Layer 3   │
  Layer 4   │
  Layer 5  ─┼──→ deepstack[0]: _merge2x2 + dsm[0] → (merged_patches, 2048)
  Layer 6   │
  ...       │
  Layer 11 ─┼──→ deepstack[1]: _merge2x2 + dsm[1] → (merged_patches, 2048)
  ...       │
  Layer 17 ─┼──→ deepstack[2]: _merge2x2 + dsm[2] → (merged_patches, 2048)
  ...       │
  Layer 23 ─┘──→ 主 merger → visual_emb (merged_patches, 2048)

文本 Decoder (28 层):
  Layer 0  ←── x[img_idx] += deepstack[0]   (ViT 层 5 的特征)
  Layer 1  ←── x[img_idx] += deepstack[1]   (ViT 层 11 的特征)
  Layer 2  ←── x[img_idx] += deepstack[2]   (ViT 层 17 的特征)
  Layer 3  ─┐
  ...        │  无 deepstack 注入
  Layer 27 ─┘

关键点:
  - ViT 层 [5, 11, 17] 的特征被抽取
  - 注入到文本 decoder 层 [0, 1, 2] (不是 5, 11, 17!)
  - 只在图像 token 位置 (img_idx) 加, 文本 token 不受影响
  - 每个 deepstack merger 是独立的 PatchMerger(use_postshuffle_norm=True)
```

---

## 九、PatchMerger 详解 (2×2 合并)

```
qwen3_vl.py:65  PatchMerger

作用: 将 2×2 相邻 patch 合并为 1 个 token, 减少序列长度

═══════════════════════════════════════════════════════════════════════════

_merge2x2(h, shapes):
  对每张图 (t, h, w, hidden):
    view(t, h/2, 2, w/2, 2, hidden)
    permute(0, 1, 3, 2, 4, 5)   ← 把 2×2 移到最后两维
    reshape(t*(h/2)*(w/2), 4*hidden)

  例: 一张图 t=1, h=32, w=32, hidden=1024
    (1, 32, 32, 1024) → (1, 16, 2, 16, 2, 1024)
                      → (1, 16, 16, 2, 2, 1024)
                      → (256, 4096)

  合并后 token 数 = 原 patch 数 / 4

PatchMerger (use_postshuffle_norm=False, 主 merger):
  LayerNorm(hidden_size=1024)    ← norm 在合并前
  reshape → (merged, 4096)
  fc1: Linear(4096, 4096)
  GELU
  fc2: Linear(4096, 2048)

PatchMerger (use_postshuffle_norm=True, deepstack merger):
  reshape → (merged, 4096)
  LayerNorm(4096)                ← norm 在合并后
  fc1: Linear(4096, 4096)
  GELU
  fc2: Linear(4096, 2048)
```

---

## 十、完整 Shape 汇总表

| 模块 | 操作 | 输入 Shape | 输出 Shape | 权重 Shape |
|------|------|-----------|-----------|-----------|
| **Vision Tower** | | | | |
| PatchEmbed3D | Conv3d | (P, 1536) | (P, 1024) | Conv3d(3,1024,k=(2,16,16)) |
| pos_embed | Embedding+插值 | (h*w,) | (h*w, 1024) | (2304, 1024) |
| VisionBlock ×24 | | (P, 1024) | (P, 1024) | |
|  ├ LayerNorm (n1) | 归一化 | (P, 1024) | (P, 1024) | (1024,) ×2 |
|  ├ VisionAttention | QKV+RoPE+Attn | (P, 1024) | (P, 1024) | (3072,1024)+(1024,1024) |
|  ├ LayerNorm (n2) | 归一化 | (P, 1024) | (P, 1024) | (1024,) ×2 |
|  └ ViT MLP | fc1+GELU+fc2 | (P, 1024) | (P, 1024) | (4096,1024)+(1024,4096) |
| _merge2x2 | 2×2 折叠 | (P, 1024) | (P/4, 4096) | — |
| PatchMerger (主) | LN+MLP | (P/4, 4096) | (P/4, 2048) | LN(1024)+fc1+fc2 |
| PatchMerger (DS) ×3 | MLP+LN | (P/4, 4096) | (P/4, 2048) | LN(4096)+fc1+fc2 |
| **文本 Decoder** | | | | |
| VocabParallelEmbedding | 查表 | (N,) | (N, 2048) | (151936, 2048) |
| scatter visual_emb | 覆盖 | (N, 2048) | (N, 2048) | — |
| RMSNorm (in_ln) | 归一化 | (N, 2048) | (N, 2048) | (2048,) |
| QKVColumnParallelLinear | qkv投影 | (N, 2048) | (N, 4096) | (4096, 2048) |
| Split+Reshape | 拆分 | (N, 4096) | q:(N,16,128) 等 | — |
| Q/K Norm | RMSNorm | (N, H, 128) | (N, H, 128) | (128,) |
| MRoPE | 3D旋转 | q,k | q,k | — |
| Flash Attention | 因果attn | q,k,v | (N, 16, 128) | — |
| RowParallelLinear (o) | 输出投影 | (N, 2048) | (N, 2048) | (2048, 2048) |
| RMSNorm (post_ln) | 归一化 | (N, 2048) | (N, 2048) | (2048,) |
| MergedColumnParallelLinear | gate_up | (N, 2048) | (N, 12288) | (12288, 2048) |
| SiluAndMul | SwiGLU | (N, 12288) | (N, 6144) | — |
| RowParallelLinear (down) | down投影 | (N, 6144) | (N, 2048) | (2048, 6144) |
| DeepStack 注入 ×3 | 加法 | (N, 2048) | (N, 2048) | — |
| × 28 layers | 循环 | (N, 2048) | (N, 2048) | |
| Final RMSNorm | 归一化 | (N, 2048) | (N, 2048) | (2048,) |
| **Pooling** | | | | |
| EmbeddingHead | last-token | (N, 2048) | (S, 2048) | — |
| L2 归一化 | 归一化 | (S, 2048) | (S, 2048) | — |
| MRL 截断 (可选) | 截断+归一化 | (S, 2048) | (S, dim) | — |

> P = total_patches, N = total_tokens, S = num_seqs

---

## 十一、与 Qwen3-0.6B 的关键差异

| 特性 | Qwen3-0.6B | Qwen3-VL-Embedding-2B |
|------|-----------|----------------------|
| 任务 | 文本生成 (Causal LM) | 多模态 Embedding |
| 推理模式 | Prefill + Decode (自回归) | 纯 Prefill (单次前向) |
| 输入 | 纯文本 | 文本 + 图像 + 混合 |
| Vision Tower | 无 | 24 层 ViT |
| RoPE | 1D (标准 RoPE) | 3D (MRoPE, T/H/W) |
| hidden_size | 1024 | 2048 |
| intermediate_size | 3072 | 6144 |
| rope_theta | 1,000,000 | 5,000,000 |
| DeepStack | 无 | ViT [5,11,17] → Decoder [0,1,2] |
| 输出 | logits → 采样 token | embedding 向量 |
| LM Head | ParallelLMHead | 无 (用 EmbeddingHead) |
| KV Cache | 有 (PagedAttention) | 无 (纯 prefill) |
| CUDA Graph | 有 (decode 加速) | 无 |
| MLP bias | False | True |
| Weight Tying | True | True (但无 LM Head) |
