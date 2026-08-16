# Qwen3-VL-Embedding-2B 纯模型结构解析

> 剥离推理框架，仅关注模型本身的计算结构与数据流。
> 所有 shape 基于 Qwen3-VL-Embedding-2B 实际配置。
> 与 Qwen3-0.6B 的关键差异：多模态输入、Vision Tower、MRoPE、DeepStack、Embedding 输出。

---

## 一、模型超参数

```
# ---- 文本 Decoder ----
vocab_size        = 151936
hidden_size       = 2048
num_layers        = 28
num_attention_heads (Q) = 16
num_kv_heads (KV)       = 8        ← GQA: 2个Q头共享1个KV头
head_dim          = 128
intermediate_size = 6144
rope_theta        = 5,000,000
rms_norm_eps      = 1e-6
mrope_section     = [24, 20, 20]   ← T/H/W 三段频率分配
mlp_bias          = True

# ---- Vision Tower (ViT) ----
vision_depth      = 24
vision_hidden_size = 1024
vision_intermediate_size = 4096
vision_num_heads  = 16
patch_size        = 16
temporal_patch_size = 2
spatial_merge_size = 2             ← 2×2 patch 合并
out_hidden_size   = 2048
deepstack_visual_indexes = [5, 11, 17]

# ---- Embedding 输出 ----
embed_dim         = 2048
pooling_mode      = "last_token"
normalize         = True           ← L2 归一化
```

---

## 二、端到端总览

```
输入:
  input_ids: (seq_len,) 或 (batch, seq_len)
  pixel_values: (total_patches, 3, 2, 16, 16)  [可选, 有图像时]
  grid_thw: (num_images, 3)                    [可选, 每图 t,h,w]

                    ┌──────────────────────┐
                    │   Embedding Layer     │
                    │   (151936, 2048)      │
                    └──────────┬───────────┘
                               │  (*, 2048)
                               │
                    ┌──────────┴───────────┐
                    │   Vision Tower       │  ← 如果有图像
                    │   (24 层 ViT)        │
                    │   → visual_emb       │
                    │   → deepstack ×3     │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │   Scatter visual_emb │  ← 覆盖 image token 位置
                    └──────────┬───────────┘
                               │  (*, 2048)
                               ▼
                    ┌──────────────────────┐
              ┌────→│   Decoder Layer 0    │ ← + deepstack[0]
              │     └──────────┬───────────┘
              │                │  (*, 2048)
              │                ▼
              │     ┌──────────────────────┐
              │     │   Decoder Layer 1    │ ← + deepstack[1]
              │     └──────────┬───────────┘
              │                │
              │     ┌──────────────────────┐
              │     │   Decoder Layer 2    │ ← + deepstack[2]
              │     └──────────┬───────────┘
              │                │
              │          ... × 28 ...
              │                │
              │                ▼
              │     ┌──────────────────────┐
              │     │   Decoder Layer 27   │
              │     └──────────┬───────────┘
              │                │  (*, 2048)
              │                ▼
              │     ┌──────────────────────┐
              │     │   Final RMSNorm      │
              │     └──────────┬───────────┘
              │                │  (*, 2048)
              │                ▼
              │     ┌──────────────────────┐
              │     │   EmbeddingHead      │
              │     │   last-token + L2    │
              │     └──────────┬───────────┘
              │                │  (batch, 2048)
              │                ▼
              │           embedding 向量
              │
              └─── 每层都有残差连接 (residual stream)
```

---

## 三、Vision Tower (视觉塔)

### 3.1 PatchEmbed3D — 把图像切成 patch 并嵌入

```
输入: pixel_values: (total_patches, C * T_patch * P * P)
                  = (total_patches, 3 * 2 * 16 * 16)
                  = (total_patches, 1536)

操作: Conv3d(in_channels=3, out_channels=1024,
             kernel=(2, 16, 16), stride=(2, 16, 16), bias=True)

输出: (total_patches, 1024)

通俗解释:
  把一张图像切成小块 (patch)，每个小块变成一个向量。
  类似把一张照片剪成拼图，每块拼图用一个 1024 维向量表示。
  temporal_patch_size=2 表示视频时把 2 帧叠在一起处理。
```

### 3.2 2D 位置嵌入 — 告诉模型每个 patch 在图像的哪个位置

```
输入: 每张图的 (h, w) patch 网格

操作:
  pos_embed: nn.Embedding(2304, 1024)  ← 48×48 学习位置网格
  对每张图: 从 48×48 双线性插值到实际 (h, w)
  h = h + pos

输出: (total_patches, 1024)

通俗解释:
  切完拼图后，模型不知道每块原来在哪里。
  位置嵌入给每块拼图贴上"坐标标签"，让模型知道空间关系。
  48×48 是学习到的"标准网格"，对不同尺寸的图像做插值适配。
```

### 3.3 VisionBlock — ViT 的基本单元 (×24)

```
每个 VisionBlock 的结构:

  输入: h: (total_patches, 1024)

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   ┌────────────────────┐                                    │
  │   │  LayerNorm (n1)    │  weight: (1024,)                   │
  │   └─────────┬──────────┘                                    │
  │             │ (total_patches, 1024)                         │
  │             ▼                                               │
  │   ┌────────────────────┐                                    │
  │   │  VisionAttention   │  双向, 无因果 mask                 │
  │   │  (详见 3.4)        │                                    │
  │   └─────────┬──────────┘                                    │
  │             │ (total_patches, 1024)                         │
  │             │                                               │
  │   ────── ⊕ ──────  ← 残差连接                              │
  │             │                                               │
  │             ▼                                               │
  │   ┌────────────────────┐                                    │
  │   │  LayerNorm (n2)    │  weight: (1024,)                   │
  │   └─────────┬──────────┘                                    │
  │             │ (total_patches, 1024)                         │
  │             ▼                                               │
  │   ┌────────────────────┐                                    │
  │   │  ViT MLP           │                                    │
  │   │  fc1: (1024, 4096) │                                    │
  │   │  GELU              │                                    │
  │   │  fc2: (4096, 1024) │                                    │
  │   └─────────┬──────────┘                                    │
  │             │ (total_patches, 1024)                         │
  │             │                                               │
  │   ────── ⊕ ──────  ← 残差连接                              │
  │             │                                               │
  │             ▼                                               │
  │   输出: h: (total_patches, 1024)                            │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

与文本 Decoder 的关键差异:
  - 用 LayerNorm (减均值) 而非 RMSNorm (不减均值)
  - 用 GELU 激活而非 SwiGLU
  - 注意力是双向的 (每个 patch 可以看到同一张图的所有 patch)
  - 无 GQA, Q/K/V 头数相同
```

### 3.4 VisionAttention — ViT 的自注意力

```
输入: x: (total_patches, 1024)
      position_ids: (total_patches, 2)  ← (hpos, wpos)

操作:
  ① QKV 投影:
     qkv = Linear(1024, 3072, bias=True)(x)
     → (total_patches, 3072)
     reshape → (3, total_patches, 16, 64)
     q, k, v 各 (total_patches, 16, 64)

  ② VisionRotaryEmbedding (2D RoPE):
     对每个 patch 的 (hpos, wpos) 计算旋转角度
     freqs = cat([h_freqs, w_freqs], dim=-1)  ← 前 32 维用 h, 后 32 维用 w
     q = q * cos + rotate_half(q) * sin
     k = k * cos + rotate_half(k) * sin

  ③ Flash Attention (双向):
     scores = (q @ k^T) / sqrt(64)
     weights = softmax(scores)  ← 无因果 mask, 全双向
     output = weights @ v
     → (total_patches, 16, 64) → reshape → (total_patches, 1024)

  ④ 输出投影:
     output = Linear(1024, 1024)(output)

输出: (total_patches, 1024)

通俗解释:
  每个 patch 通过注意力机制"看看"同一张图的其他 patch。
  比如"猫头"的 patch 会关注"猫耳"的 patch，建立局部与全局的关系。
  双向注意力意味着 patch A 可以看 B，B 也可以看 A (不像文本生成只能看前面的)。
```

### 3.5 PatchMerger — 2×2 patch 合并

```
作用: 把 2×2 相邻 patch 合并为 1 个 token, 减少序列长度

操作:
  _merge2x2:
    每张图 (t, h, w, 1024) → 2×2 折叠 → (t, h/2, w/2, 4096)
    flatten → (t*(h/2)*(w/2), 4096)

  主 Merger (use_postshuffle_norm=False):
    LayerNorm(1024)  ← norm 在合并前
    reshape → (merged, 4096)
    fc1: Linear(4096, 4096)
    GELU
    fc2: Linear(4096, 2048)

  DeepStack Merger (use_postshuffle_norm=True):
    reshape → (merged, 4096)
    LayerNorm(4096)  ← norm 在合并后
    fc1: Linear(4096, 4096)
    GELU
    fc2: Linear(4096, 2048)

输出: (merged_patches, 2048)
      其中 merged_patches = total_patches / 4

通俗解释:
  2×2 合并类似"降采样"：把 4 个相邻的小 patch 拼成 1 个大 patch。
  这样序列长度变成原来的 1/4，减少后续文本 decoder 的计算量。
  合并后用 MLP 把 4096 维压缩到 2048 维，与文本 hidden_size 对齐。
```

### 3.6 DeepStack — 从 ViT 中间层抽取特征

```
Vision Tower 24 层中，在层 [5, 11, 17] 抽取中间特征:

  Layer 5  → deepstack[0]: _merge2x2 + dsm[0] → (merged_patches, 2048)
  Layer 11 → deepstack[1]: _merge2x2 + dsm[1] → (merged_patches, 2048)
  Layer 17 → deepstack[2]: _merge2x2 + dsm[2] → (merged_patches, 2048)

这些特征稍后会被注入到文本 decoder 的层 [0, 1, 2]。

通俗解释:
  ViT 的不同层学到不同抽象级别的视觉特征：
    - 浅层 (5): 边缘、纹理等低级特征
    - 中层 (11): 物体部件等中级特征
    - 深层 (17): 整体语义等高级特征
  DeepStack 把这三个级别的特征都抽取出来，稍后"喂"给文本模型，
  让文本模型同时获得细粒度细节和粗粒度语义。
```

---

## 四、文本 Decoder

### 4.1 Embedding — 把 token ID 变成向量

```
输入: input_ids: (seq_len,)

操作:
  weight: (151936, 2048)
  x = weight[input_ids]  ← 查表

输出: (seq_len, 2048)
```

### 4.2 Scatter visual_emb — 把视觉特征插入到图像 token 位置

```
输入:
  x: (seq_len, 2048)  ← 文本 embedding
  visual_emb: (merged_patches, 2048)  ← Vision Tower 输出
  image_token_spans: [(start, end), ...]  ← 图像 token 的位置范围

操作:
  img_idx = [s, s+1, ..., e-1 for (s,e) in image_token_spans]
  x[img_idx] = visual_emb  ← 覆盖原 embedding

输出: (seq_len, 2048)
      文本 token 保持原 embedding
      图像 token 被替换为 visual_emb

通俗解释:
  input_ids 中图像位置原本是占位符 (<|image_pad|>)，
  对应的 embedding 是随机初始化的，没有意义。
  这一步用 Vision Tower 提取的真实视觉特征替换这些占位符，
  让文本模型能"看到"图像内容。
```

### 4.3 Decoder Layer — 文本解码器的基本单元 (×28)

```
每个 Decoder Layer 的结构:

  输入: x: (seq_len, 2048), residual: (seq_len, 2048) | None

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   ┌────────────────────┐                                    │
  │   │  RMSNorm (in_ln)   │  weight: (2048,)                   │
  │   └─────────┬──────────┘                                    │
  │             │ (seq_len, 2048)                               │
  │             ▼                                               │
  │   ┌────────────────────┐                                    │
  │   │  Qwen3VLAttention  │  因果, GQA, MRoPE                  │
  │   │  (详见 4.4)        │                                    │
  │   └─────────┬──────────┘                                    │
  │             │ (seq_len, 2048)                               │
  │             │                                               │
  │   ────── ⊕ ──────  ← 残差连接                              │
  │             │                                               │
  │             ▼                                               │
  │   ┌────────────────────┐                                    │
  │   │  RMSNorm (post_ln) │  weight: (2048,)                   │
  │   └─────────┬──────────┘                                    │
  │             │ (seq_len, 2048)                               │
  │             ▼                                               │
  │   ┌────────────────────┐                                    │
  │   │  Qwen3VLMLP        │  SwiGLU                            │
  │   │  (详见 4.5)        │                                    │
  │   └─────────┬──────────┘                                    │
  │             │ (seq_len, 2048)                               │
  │             │                                               │
  │   ────── ⊕ ──────  ← 残差连接                              │
  │             │                                               │
  │   ┌────────────────────────────────────────────────────┐    │
  │   │  DeepStack 注入 (如果 layer_idx < 3 且有图像)      │    │
  │   │                                                    │    │
  │   │  Layer 0: x[img_idx] += deepstack[0]               │    │
  │   │  Layer 1: x[img_idx] += deepstack[1]               │    │
  │   │  Layer 2: x[img_idx] += deepstack[2]               │    │
  │   │                                                    │    │
  │   │  只在图像 token 位置加, 文本 token 不受影响         │    │
  │   └────────────────────────────────────────────────────┘    │
  │             │                                               │
  │             ▼                                               │
  │   输出: x: (seq_len, 2048)                                  │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

与 Qwen3-0.6B 的差异:
  - hidden_size = 2048 (而非 1024)
  - intermediate_size = 6144 (而非 3072)
  - MLP 有 bias (Qwen3-0.6B 无 bias)
  - 用 MRoPE (3D 位置) 而非标准 RoPE (1D 位置)
  - 前 3 层有 DeepStack 视觉特征注入
```

### 4.4 Qwen3VLAttention — 文本注意力 (带 MRoPE)

```
输入: x: (seq_len, 2048), pos3d: (3, seq_len)  ← T/H/W 3D 位置

操作:
  ① QKV 投影:
     W_qkv: (4096, 2048)  ← [Q(2048) + K(1024) + V(1024)]
     qkv = x @ W_qkv^T
     → (seq_len, 4096)
     split → q: (seq_len, 2048), k: (seq_len, 1024), v: (seq_len, 1024)
     reshape → q: (seq_len, 16, 128), k: (seq_len, 8, 128), v: (seq_len, 8, 128)

  ② Q/K Norm:
     q = RMSNorm(q)  weight: (128,)
     k = RMSNorm(k)  weight: (128,)

  ③ MRoPE (3D 旋转位置编码, 详见 4.6):
     q, k = MRotaryEmbedding(pos3d, q, k)

  ④ Flash Attention (因果, GQA):
     scores = (q @ k^T) / sqrt(128)
     因果 mask: j > i 时 scores[i,j] = -inf
     weights = softmax(scores)
     output = weights @ v
     → (seq_len, 16, 128) → reshape → (seq_len, 2048)

  ⑤ 输出投影:
     W_o: (2048, 2048)
     output = output @ W_o^T
     → (seq_len, 2048)

输出: (seq_len, 2048)
```

### 4.5 Qwen3VLMLP — 文本 MLP (SwiGLU)

```
输入: x: (seq_len, 2048)

操作:
  ① Gate + Up 合并投影:
     W_gate_up: (12288, 2048)  ← [gate(6144) + up(6144)]
     gate_up = x @ W_gate_up^T + bias
     → (seq_len, 12288)

  ② SiLU × Mul (SwiGLU):
     gate, up = chunk(2, dim=-1)  ← 各 (seq_len, 6144)
     activated = SiLU(gate) × up
     → (seq_len, 6144)

  ③ Down 投影:
     W_down: (2048, 6144)
     output = activated @ W_down^T + bias
     → (seq_len, 2048)

输出: (seq_len, 2048)
```

### 4.6 MRoPE — 3D 旋转位置编码

```
输入: positions_3d: (3, seq_len)  ← [T, H, W] 三条位置轴
      query: (seq_len, num_heads, 128)
      key: (seq_len, num_kv_heads, 128)

mrope_section = [24, 20, 20]
head_dim = 128, head_dim/2 = 64 个频率

频率布局 (交错式):
  前 20 组三轴交错: [T0, H0, W0, T1, H1, W1, ..., T19, H19, W19]  ← 60 个
  尾部 T 轴独占:    [T20, T21, T22, T23]                           ← 4 个
  合计: 64 = head_dim / 2

操作:
  inv_freq = 1 / (5,000,000 ^ (arange(0, 128, 2) / 128))  ← (64,)

  对每个频率槽 i:
    从 mrope_section_id[i] 知道它属于 T(0)/H(1)/W(2)
    从 positions_3d[mrope_section_id[i]] 取对应位置
    freq[i] = inv_freq[mrope_perm[i]] × pos

  freqs: (seq_len, 64)
  cos = freqs.cos()    sin = freqs.sin()

  apply_rotary_pos_emb:
    q1, q2 = query.chunk(2, dim=-1)  ← 各 (seq_len, heads, 64)
    out1 = q1 × cos - q2 × sin
    out2 = q1 × sin + q2 × cos
    q_rot = cat([out1, out2], dim=-1)

输出: q_rot, k_rot 各 (seq_len, heads, 128)

通俗解释:
  标准 RoPE 用 1D 位置 (token 在序列中的序号)。
  MRoPE 用 3D 位置 (T, H, W)，分别编码时间、高度、宽度。

  对于文本 token: T=H=W=序号，三轴同步前进。
  对于图像 token: T=帧号，H/W=patch 在图像中的行列坐标。

  这样模型能同时理解：
    - 文本的线性顺序 (T 轴)
    - 图像的空间布局 (H/W 轴)
    - 视频的时序关系 (T 轴在多帧时)

  交错布局 [T0,H0,W0,...] 让三个轴的信息在每个频率维度上都混合，
  避免模型只关注某一个轴。
```

### 4.7 DeepStack 注入 — 把视觉特征"喂"给文本模型

```
在文本 decoder 的前 3 层 (0, 1, 2)，注入对应的 deepstack 特征:

  Layer 0: x[img_idx] += deepstack[0]  ← 来自 ViT 层 5
  Layer 1: x[img_idx] += deepstack[1]  ← 来自 ViT 层 11
  Layer 2: x[img_idx] += deepstack[2]  ← 来自 ViT 层 17

通俗解释:
  文本模型在处理图像 token 时，需要"看到"不同抽象级别的视觉特征。
  DeepStack 在文本模型的早期层逐步注入视觉特征：
    - 第 0 层: 注入低级特征 (边缘、纹理)
    - 第 1 层: 注入中级特征 (物体部件)
    - 第 2 层: 注入高级特征 (整体语义)

  这样文本模型从第 3 层开始就已经"消化"了多层次的视觉信息，
  后续层可以更好地融合文本和图像进行推理。

  只在图像 token 位置加，文本 token 不受影响，
  避免视觉信息"污染"纯文本的表示。
```

---

## 五、EmbeddingHead — 从隐藏状态提取 embedding 向量

```
输入: hidden: (seq_len, 2048), cu_seqlens_q: (num_seqs+1,)

操作:
  ① Last-token gather:
     last_indices = cu_seqlens_q[1:] - 1  ← 每条序列的最后一个 token
     pooled = hidden[last_indices]
     → (num_seqs, 2048)

  ② L2 归一化:
     pooled = pooled / ||pooled||_2
     → (num_seqs, 2048)

  ③ 可选 MRL 截断:
     if mrl_dim is not None:
       pooled = pooled[:, :mrl_dim]
       pooled = pooled / ||pooled||_2
       → (num_seqs, mrl_dim)

输出: (num_seqs, embed_dim)

通俗解释:
  经过 28 层 decoder 后，每个 token 的隐藏状态都包含了上下文信息。
  对于 embedding 任务，我们只需要一个固定长度的向量表示整条输入。

  Last-token pooling: 取最后一个 token 的隐藏状态。
  因为因果注意力，最后一个 token 能"看到"前面所有 token，
  所以它的隐藏状态是对整条输入的"总结"。

  L2 归一化: 把向量缩放到单位球面上，使得余弦相似度等于点积。
  这样下游检索时可以直接用矩阵乘法计算相似度。

  MRL (Matryoshka Representation Learning): 可选的维度截断。
  取前 dim 维并重新归一化，得到更短但仍有用的 embedding。
  用于在精度和存储/计算成本之间做权衡。
```

---

## 六、完整数据流示例

```
假设输入:
  文本: "A cat"  → token_ids: [32, 5834]
  图像: 1 张 32×32 patch 的图 (t=1, h=32, w=32)
    → pixel_values: (1024, 1536)  ← 1024 个 patch
    → grid_thw: [[1, 32, 32]]

═══════════════════════════════════════════════════════════════════════

① Vision Tower:
   pixel_values: (1024, 1536)
   → PatchEmbed3D → (1024, 1024)
   → + pos_embed → (1024, 1024)
   → 24 × VisionBlock → (1024, 1024)
      层 5 → deepstack[0]: _merge2x2 + dsm[0] → (256, 2048)
      层 11 → deepstack[1]: _merge2x2 + dsm[1] → (256, 2048)
      层 17 → deepstack[2]: _merge2x2 + dsm[2] → (256, 2048)
   → 主 merger: _merge2x2 + merger → visual_emb: (256, 2048)

② 文本 Embedding:
   token_ids: [32, 5834, 151655×256]  ← 2 个文本 + 256 个图像占位符
   → Embedding → (258, 2048)

③ Scatter visual_emb:
   x[2:258] = visual_emb  ← 覆盖图像占位符位置
   → (258, 2048)

④ 28 层 Decoder:
   Layer 0: x[2:258] += deepstack[0]  ← 注入 ViT 层 5 特征
   Layer 1: x[2:258] += deepstack[1]  ← 注入 ViT 层 11 特征
   Layer 2: x[2:258] += deepstack[2]  ← 注入 ViT 层 17 特征
   Layer 3~27: 正常 decoder 层
   → (258, 2048)

⑤ Final RMSNorm:
   → (258, 2048)

⑥ EmbeddingHead:
   last_token = x[257]  ← 最后一个 token
   → (2048,)
   → L2 归一化 → (2048,)

输出: 2048 维 embedding 向量
```

---

## 七、参数量统计

```
模块                          参数量
─────────────────────────────────────────────────────
Vision Tower:
  PatchEmbed3D                Conv3d: 3×2×16×16×1024 + 1024 = 1,572,864 + 1024
  pos_embed                   2304 × 1024 = 2,359,296
  VisionBlock ×24:
    n1, n2                    1024 × 2 × 2 = 4,096
    attn.qkv                  3072 × 1024 + 3072 = 3,148,800
    attn.proj                 1024 × 1024 + 1024 = 1,049,600
    mlp_fc1                   4096 × 1024 + 4096 = 4,198,400
    mlp_fc2                   1024 × 4096 + 1024 = 4,195,328
    每层合计                  12,596,224
  24 层合计                   302,309,376
  merger (主):
    norm                      1024 × 2 = 2,048
    fc1                       4096 × 4096 + 4096 = 16,781,312
    fc2                       2048 × 4096 + 2048 = 8,390,656
    合计                      25,174,016
  dsm ×3 (deepstack merger):
    每个: norm(4096×2) + fc1 + fc2 = 25,174,016
    3 个合计                  75,522,048
  Vision Tower 总计           ≈ 406M

文本 Decoder:
  embed_tokens                151936 × 2048 = 311,164,928
  Decoder Layer ×28:
    in_ln, post_ln            2048 × 2 × 2 = 8,192
    attn.qkv                  4096 × 2048 = 8,388,608
    attn.q_norm, k_norm       128 × 2 = 256
    attn.o                    2048 × 2048 = 4,194,304
    mlp.gate_up               12288 × 2048 + 12288 = 25,178,112
    mlp.down                  2048 × 6144 + 2048 = 12,584,960
    每层合计                  50,356,432
  28 层合计                   1,409,980,096
  Final RMSNorm               2048 × 2 = 4,096
  文本 Decoder 总计           ≈ 1.72B

总计                          ≈ 2.13B
```

---

## 八、与 Qwen3-0.6B 的架构对比

| 特性 | Qwen3-0.6B | Qwen3-VL-Embedding-2B |
|------|-----------|----------------------|
| **任务** | 文本生成 | 多模态 Embedding |
| **推理模式** | Prefill + Decode | 纯 Prefill |
| **输入模态** | 纯文本 | 文本 + 图像 + 混合 |
| **Vision Tower** | 无 | 24 层 ViT |
| **RoPE** | 1D (标准) | 3D (MRoPE, T/H/W) |
| **hidden_size** | 1024 | 2048 |
| **intermediate_size** | 3072 | 6144 |
| **rope_theta** | 1,000,000 | 5,000,000 |
| **DeepStack** | 无 | ViT [5,11,17] → Decoder [0,1,2] |
| **输出** | logits → token | embedding 向量 |
| **LM Head** | 有 | 无 (用 EmbeddingHead) |
| **MLP bias** | False | True |
| **注意力类型** | 因果 (文本) | 因果 (文本) + 双向 (ViT) |
| **Norm 类型** | RMSNorm (全文本) | RMSNorm (文本) + LayerNorm (ViT) |
| **激活函数** | SwiGLU (文本) | SwiGLU (文本) + GELU (ViT) |

---

## 九、关键设计特点

```
1. Vision Tower + 文本 Decoder 双塔架构
   ┌──────────────────────────────────────────────────┐
   │  图像 → ViT (24 层) → visual_emb + deepstack    │
   │                                                  │
   │  文本 + visual_emb → 文本 Decoder (28 层)        │
   │                                                  │
   │  ViT 独立运行，输出"喂"给文本模型                │
   │  文本模型负责融合多模态信息并产出最终表示        │
   └──────────────────────────────────────────────────┘

2. DeepStack 多层次视觉特征注入
   ┌──────────────────────────────────────────────────┐
   │  ViT 层 5  → 文本层 0  (低级特征: 边缘、纹理)   │
   │  ViT 层 11 → 文本层 1  (中级特征: 物体部件)     │
   │  ViT 层 17 → 文本层 2  (高级特征: 整体语义)     │
   │                                                  │
   │  让文本模型从第 3 层开始就有多层次视觉信息      │
   └──────────────────────────────────────────────────┘

3. MRoPE (3D 旋转位置编码)
   ┌──────────────────────────────────────────────────┐
   │  T 轴: 时间/序列顺序                             │
   │  H 轴: 图像高度方向                              │
   │  W 轴: 图像宽度方向                              │
   │                                                  │
   │  文本: T=H=W=序号                                │
   │  图像: T=帧号, H/W=patch 坐标                    │
   │                                                  │
   │  交错布局 [T0,H0,W0,...] 让三轴信息混合          │
   └──────────────────────────────────────────────────┘

4. 2×2 Patch Merger
   ┌──────────────────────────────────────────────────┐
   │  把 4 个相邻 patch 合并为 1 个 token             │
   │  序列长度变为 1/4，减少计算量                    │
   │                                                  │
   │  1024 patch → 256 token                          │
   │  保持空间局部性，同时降采样                      │
   └──────────────────────────────────────────────────┘

5. Last-token Pooling + L2 归一化
   ┌──────────────────────────────────────────────────┐
   │  取最后一个 token 的隐藏状态作为整条输入的表示   │
   │  因果注意力保证最后 token 看到了全部上下文       │
   │                                                  │
   │  L2 归一化到单位球面，余弦相似度 = 点积          │
   │  适合下游检索任务                                │
   └──────────────────────────────────────────────────┘

6. ViT 与文本 Decoder 的差异
   ┌──────────────────────────────────────────────────┐
   │  ViT:                                            │
   │    - LayerNorm (减均值)                          │
   │    - GELU 激活                                   │
   │    - 双向注意力                                  │
   │    - 无 GQA                                      │
   │    - 2D RoPE (H/W)                               │
   │                                                  │
   │  文本 Decoder:                                   │
   │    - RMSNorm (不减均值)                          │
   │    - SwiGLU 激活                                 │
   │    - 因果注意力                                  │
   │    - GQA (16 Q : 8 KV)                           │
   │    - 3D MRoPE (T/H/W)                            │
   └──────────────────────────────────────────────────┘
```
