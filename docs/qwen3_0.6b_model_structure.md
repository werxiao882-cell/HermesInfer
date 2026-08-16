# Qwen3-0.6B 模型结构全流程 (HermesInfer 实现)

> 基于 `src/myvllm/models/qwen3.py` 及相关 layers/engine 源码梳理。
> 涵盖从提示词输入到 token 输出的完整数据流，每个模块均标注 shape 信息。

## 模型配置参数 (main.py)

```
vocab_size       = 151936
hidden_size      = 1024
num_heads        = 16        (Q/O heads)
head_dim         = 128
num_kv_heads     = 8         (KV heads, GQA)
intermediate_size= 3072
num_layers       = 28
tie_word_embeddings = True
rope_theta (base) = 1000000
rms_norm_epsilon = 1e-6
qkv_bias         = False
scale            = 1.0
max_position     = 32768
ffn_bias         = False
block_size       = 256       (KV cache block)
```

---

## 一、端到端总览 (Engine Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLMEngine.generate()                                │
│                         llm_engine.py:105                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  prompts: list[str]                                                         │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────┐                               │
│  │  Tokenizer (AutoTokenizer)               │                               │
│  │  apply_chat_template + encode            │                               │
│  │  prompt str → token_ids: list[int]       │                               │
│  └──────────────┬───────────────────────────┘                               │
│                 ▼                                                           │
│  ┌──────────────────────────────────────────┐                               │
│  │  Scheduler.add_sequence()                │                               │
│  │  token_ids → Sequence 对象               │                               │
│  │  入队 waiting: deque[Sequence]           │                               │
│  └──────────────┬───────────────────────────┘                               │
│                 ▼                                                           │
│  ┌─────────── while not is_finished() ──────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────┐                              │   │
│  │  │  Scheduler.schedule()              │                              │   │
│  │  │  waiting → prefill (is_prefill=T)  │                              │   │
│  │  │  running → decode  (is_prefill=F)  │                              │   │
│  │  │  BlockManager.allocate/append      │                              │   │
│  │  │  return (seqs, is_prefill)         │                              │   │
│  │  └──────────────┬─────────────────────┘                              │   │
│  │                 ▼                                                    │   │
│  │  ┌────────────────────────────────────┐                              │   │
│  │  │  ModelRunner.run()                 │                              │   │
│  │  │  ① prepare_prefill / prepare_decode                              │   │
│  │  │  ② set_context (attention metadata)                              │   │
│  │  │  ③ run_model → model forward                                     │   │
│  │  │  ④ compute_logits                                               │   │
│  │  │  ⑤ sampler (rank 0 only)                                        │   │
│  │  │  return token_ids: list[int]                                     │   │
│  │  └──────────────┬─────────────────────┘                              │   │
│  │                 ▼                                                    │   │
│  │  ┌────────────────────────────────────┐                              │   │
│  │  │  Scheduler.postprocess()           │                              │   │
│  │  │  append_token → check EOS/max_len  │                              │   │
│  │  │  FINISHED → deallocate blocks      │                              │   │
│  │  └────────────────────────────────────┘                              │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                 ▼                                                           │
│  output: {'text': list[str], 'token_ids': dict}                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Prefill 阶段 — 数据准备 (prepare_prefill)

```
model_runner.py:312

输入: seqs: list[Sequence]
  每条 seq.token_ids = [t0, t1, ..., t_{n-1}]

多条序列打包为 varlen 格式:

  input_ids:     [seq0_tokens... | seq1_tokens... | seq2_tokens... ]
                  ←── 1D tensor, 所有 token 拼接 ──→

  cu_seqlens_q:  [0, len(seq0), len(seq0)+len(seq1), ...]
                  shape: (num_seqs + 1,)  dtype: int32

  slot_mapping:  [slot_0, slot_1, ..., slot_{N-1}]
                  shape: (N,)  dtype: int64
                  每个 slot = block_id * block_size + offset_in_block

  Context 写入:
    is_prefill    = True
    cu_seqlens_q  = (num_seqs + 1,)
    cu_seqlens_k  = (num_seqs + 1,)
    slot_mapping  = (N,)
    max_seqlen_q  = max(seqlens_q)

其中 N = sum(len(seq) for seq in seqs) = 总 token 数
```

---

## 三、Decode 阶段 — 数据准备 (prepare_decode)

```
model_runner.py:365

输入: seqs: list[Sequence]  (每条只取 last_token)

  input_ids:     [seq0_last, seq1_last, ..., seqB_last]
                  shape: (B,)  dtype: int64

  context_lens:  [len(seq0), len(seq1), ..., len(seqB)]
                  shape: (B,)  dtype: int64

  slot_mapping:  [slot_0, slot_1, ..., slot_{B-1}]
                  shape: (B,)  dtype: int64

  block_tables:  [[b0_0, b0_1, ..., -1, -1],
                  [b1_0, b1_1, b1_2, -1, -1],
                  ...                        ]
                  shape: (B, max_num_blocks)  dtype: int32

  Context 写入:
    is_prefill   = False
    slot_mapping = (B,)
    context_lens = (B,)
    block_tables = (B, max_num_blocks)

其中 B = len(seqs) = batch_size
```

---

## 四、Qwen3ForCausalLM 模型前向 (完整结构)

```
qwen3.py:285  Qwen3ForCausalLM

═══════════════════════════════════════════════════════════════════════════
                    Qwen3ForCausalLM.forward()
═══════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │  input_ids                                                          │
  │  Prefill: shape (N,)        N = total tokens (varlen packed)        │
  │  Decode:  shape (B,)        B = batch_size                          │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Qwen3Model.forward()                                               │
  │  qwen3.py:273                                                       │
  │                                                                     │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  VocabParallelEmbedding (embed_tokens)                        │  │
  │  │  embedding_head.py:12                                         │  │
  │  │  weight: (num_embeddings_per_partition, 1024)                 │  │
  │  │  num_embeddings = 151936, 按 vocab 维度 TP 分片               │  │
  │  │                                                               │  │
  │  │  Prefill: (N,) ──→ (N, 1024)                                 │  │
  │  │  Decode:  (B,) ──→ (B, 1024)                                 │  │
  │  │                                                               │  │
  │  │  TP>1: mask + F.embedding + all_reduce(SUM)                  │  │
  │  └───────────────────────────┬───────────────────────────────────┘  │
  │                              │                                      │
  │                              │  x: (N, 1024) or (B, 1024)          │
  │                              │  residual = None                      │
  │                              ▼                                      │
  │  ┌─────────── × 28 layers ───────────────────────────────────────┐  │
  │  │                                                                │  │
  │  │   Qwen3DecoderLayer[i]  (i = 0..27)                           │  │
  │  │   qwen3.py:159                                                │  │
  │  │                                                                │  │
  │  │   输入: x: (*, 1024), residual: (*, 1024) | None              │  │
  │  │                                                                │  │
  │  │   ┌────────────────────────────────────────────────────────┐   │  │
  │  │   │  ① Input LayerNorm (RMSNorm)                          │   │  │
  │  │   │  layernorm.py:4                                        │   │  │
  │  │   │  weight(gamma): (1024,)                                │   │  │
  │  │   │                                                        │   │  │
  │  │   │  首次 (residual=None):                                 │   │  │
  │  │   │    residual = x                                        │   │  │
  │  │   │    x = RMSNorm(x)                                      │   │  │
  │  │   │                                                        │   │  │
  │  │   │  后续 (residual!=None):                                │   │  │
  │  │   │    x = x + residual   →  new residual = x              │   │  │
  │  │   │    x = RMSNorm(x)                                      │   │  │
  │  │   │                                                        │   │  │
  │  │   │  RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ            │   │  │
  │  │   │                                                        │   │  │
  │  │   │  Prefill: (N, 1024) ──→ (N, 1024)                     │   │  │
  │  │   │  Decode:  (B, 1024) ──→ (B, 1024)                     │   │  │
  │  │   └──────────────────────┬─────────────────────────────────┘   │  │
  │  │                          │                                     │  │
  │  │                          ▼                                     │  │
  │  │   ┌────────────────────────────────────────────────────────┐   │  │
  │  │   │  ② Qwen3Attention (self_attn)                         │   │  │
  │  │   │  qwen3.py:11                                          │   │  │
  │  │   │                                                        │   │  │
  │  │   │  (详见下方 第五节 展开图)                               │   │  │
  │  │   │                                                        │   │  │
  │  │   │  Prefill: (N, 1024) ──→ (N, 1024)                     │   │  │
  │  │   │  Decode:  (B, 1024) ──→ (B, 1024)                     │   │  │
  │  │   └──────────────────────┬─────────────────────────────────┘   │  │
  │  │                          │                                     │  │
  │  │                          │  x: (*, 1024)                       │  │
  │  │                          │  residual: (*, 1024)                │  │
  │  │                          ▼                                     │  │
  │  │   ┌────────────────────────────────────────────────────────┐   │  │
  │  │   │  ③ Post-Attention LayerNorm (RMSNorm + residual)      │   │  │
  │  │   │  layernorm.py:4                                        │   │  │
  │  │   │                                                        │   │  │
  │  │   │  x, residual = residual_rms_forward(x, residual)       │   │  │
  │  │   │    x = x + residual                                    │   │  │
  │  │   │    residual_new = x                                    │   │  │
  │  │   │    x = RMSNorm(x)                                      │   │  │
  │  │   │                                                        │   │  │
  │  │   │  Prefill: (N, 1024) ──→ (N, 1024)                     │   │  │
  │  │   │  Decode:  (B, 1024) ──→ (B, 1024)                     │   │  │
  │  │   └──────────────────────┬─────────────────────────────────┘   │  │
  │  │                          │                                     │  │
  │  │                          ▼                                     │  │
  │  │   ┌────────────────────────────────────────────────────────┐   │  │
  │  │   │  ④ Qwen3MLP (mlp)                                     │   │  │
  │  │   │  qwen3.py:129                                         │   │  │
  │  │   │                                                        │   │  │
  │  │   │  (详见下方 第六节 展开图)                               │   │  │
  │  │   │                                                        │   │  │
  │  │   │  Prefill: (N, 1024) ──→ (N, 1024)                     │   │  │
  │  │   │  Decode:  (B, 1024) ──→ (B, 1024)                     │   │  │
  │  │   └──────────────────────┬─────────────────────────────────┘   │  │
  │  │                          │                                     │  │
  │  │                          │  return (x, residual)               │  │
  │  │                          │  下一层继续                          │  │
  │  └──────────────────────────┼─────────────────────────────────────┘  │
  │                              │                                      │
  │                              ▼                                      │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  Final LayerNorm (norm)                                       │  │
  │  │  layernorm.py:4                                               │  │
  │  │  weight(gamma): (1024,)                                       │  │
  │  │                                                               │  │
  │  │  x, _ = residual_rms_forward(x, residual)                     │  │
  │  │    x = x + residual                                           │  │
  │  │    x = RMSNorm(x)                                             │  │
  │  │                                                               │  │
  │  │  Prefill: (N, 1024) ──→ (N, 1024)                            │  │
  │  │  Decode:  (B, 1024) ──→ (B, 1024)                            │  │
  │  └───────────────────────────┬───────────────────────────────────┘  │
  │                              │                                      │
  └──────────────────────────────┼──────────────────────────────────────┘
                                 │
                                 │  hidden_states: (N, 1024) or (B, 1024)
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  compute_logits()                                                   │
  │  qwen3.py:340                                                       │
  │                                                                     │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  ParallelLMHead (lm_head)                                     │  │
  │  │  embedding_head.py:64                                         │  │
  │  │  weight: (num_embeddings_per_partition, 1024)                 │  │
  │  │                                                               │  │
  │  │  Prefill 路径:                                                │  │
  │  │    cu_seqlens_q = [0, n0, n0+n1, ...]                        │  │
  │  │    last_token_indices = [n0-1, n0+n1-1, ...]                 │  │
  │  │    x = x[last_token_indices]                                  │  │
  │  │    shape: (N, 1024) ──gather──→ (num_seqs, 1024)             │  │
  │  │                                                               │  │
  │  │  Decode 路径:                                                 │  │
  │  │    x 直接就是 (B, 1024)                                       │  │
  │  │                                                               │  │
  │  │  logits = F.linear(x, weight)                                 │  │
  │  │    (num_seqs, 1024) × (1024, vocab_per_part)                  │  │
  │  │    → (num_seqs, vocab_per_partition)                          │  │
  │  │                                                               │  │
  │  │  TP>1: gather to rank0 + cat + trim                           │  │
  │  │    → (num_seqs, 151936)    [Prefill]                          │  │
  │  │    → (B, 151936)           [Decode]                           │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                                                                     │
  │  return logits                                                      │
  │  Prefill: (num_seqs, 151936)                                        │
  │  Decode:  (B, 151936)                                               │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SamplerLayer (sampler)                                             │
  │  sampler.py:5                                                       │
  │  (仅 rank 0 执行)                                                   │
  │                                                                     │
  │  logits: (num_seqs, 151936) or (B, 151936)                         │
  │  temperature: (num_seqs,) or (B,)                                   │
  │                                                                     │
  │  ① logits /= temperature.unsqueeze(-1)                              │
  │     → (num_seqs, 151936)                                            │
  │                                                                     │
  │  ② probs = softmax(logits, dim=-1)                                  │
  │     → (num_seqs, 151936)                                            │
  │                                                                     │
  │  ③ Gumbel-max 采样:                                                 │
  │     tokens = (probs / exponential_noise).argmax(dim=-1)             │
  │     → (num_seqs,) or (B,)    dtype: int64                           │
  │                                                                     │
  │  return token_ids                                                   │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 五、Qwen3Attention 展开图

```
qwen3.py:11  Qwen3Attention

输入: x: (*, 1024)    positions: (*,)
      * = N (prefill) 或 B (decode)

═══════════════════════════════════════════════════════════════════════════
                       Qwen3Attention.forward()
═══════════════════════════════════════════════════════════════════════════

  x: (*, 1024)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  QKVColumnParallelLinear (qkv_projection)                            │
│  linear.py:152                                                       │
│                                                                      │
│  weight: (head_dim*(num_heads+2*num_kv_heads), hidden_size)          │
│        = (128*(16+8+8), 1024) = (4096, 1024)    [full, before TP]    │
│  bias: None  (qkv_bias=False)                                        │
│                                                                      │
│  TP 分片后 per-GPU weight:                                           │
│    (128*(16/tp + 8/tp + 8/tp), 1024)                                 │
│    tp=1: (4096, 1024)                                                │
│                                                                      │
│  qkv = F.linear(x, weight)                                           │
│                                                                      │
│  Prefill: (N, 1024) ──→ (N, 4096)       [tp=1]                      │
│  Decode:  (B, 1024) ──→ (B, 4096)       [tp=1]                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Split QKV                                                           │
│  qwen3.py:89                                                         │
│                                                                      │
│  q_size = head_dim * num_heads     = 128 * 16 = 2048    [tp=1]      │
│  kv_size = head_dim * num_kv_heads = 128 * 8  = 1024    [tp=1]      │
│                                                                      │
│  q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)            │
│                                                                      │
│  q: (*, 2048)                                                        │
│  k: (*, 1024)                                                        │
│  v: (*, 1024)                                                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Reshape                                                             │
│  qwen3.py:94-104                                                     │
│                                                                      │
│  Prefill (varlen 2D mode):                                           │
│    q: (N, 2048) → (N, 16, 128)     [num_heads, head_dim]            │
│    k: (N, 1024) → (N, 8, 128)      [num_kv_heads, head_dim]         │
│    v: (N, 1024) → (N, 8, 128)      [num_kv_heads, head_dim]         │
│                                                                      │
│  Decode (batched 2D mode):                                           │
│    q: (B, 2048) → (B, 16, 128)                                       │
│    k: (B, 1024) → (B, 8, 128)                                        │
│    v: (B, 1024) → (B, 8, 128)                                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Q Norm + K Norm  (仅当 qkv_bias=False 时执行)                      │
│  qwen3.py:109-111                                                    │
│                                                                      │
│  q_norm: LayerNorm   weight: (head_dim,) = (128,)                    │
│  k_norm: LayerNorm   weight: (head_dim,) = (128,)                    │
│                                                                      │
│  q = RMSNorm(q)   per-head 独立归一化                                │
│    Prefill: (N, 16, 128) ──→ (N, 16, 128)                           │
│    Decode:  (B, 16, 128) ──→ (B, 16, 128)                           │
│                                                                      │
│  k = RMSNorm(k)   per-head 独立归一化                                │
│    Prefill: (N, 8, 128) ──→ (N, 8, 128)                             │
│    Decode:  (B, 8, 128) ──→ (B, 8, 128)                             │
│                                                                      │
│  v 不做 norm                                                         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RotaryEmbedding (rotary_emb)                                        │
│  rotary_embedding.py:48                                              │
│                                                                      │
│  base = 1000000                                                      │
│  rotary_embedding = head_dim = 128                                   │
│  max_position = 32768                                                │
│                                                                      │
│  inv_freq = 1 / (base ^ (arange(0, 128, 2) / 128))                  │
│           shape: (64,)                                               │
│                                                                      │
│  cos_sin_cache: (max_position, rotary_embedding)                     │
│               = (32768, 128)                                         │
│    前 64 列 = cos, 后 64 列 = sin                                    │
│                                                                      │
│  forward(positions, q, k):                                           │
│    cos_sin = cos_sin_cache[positions]                                │
│      positions: (*,)                                                 │
│      cos_sin: (*, 128)                                               │
│    cos, sin = chunk(2, dim=-1)                                       │
│      cos: (*, 64)    sin: (*, 64)                                    │
│                                                                      │
│    apply_rotary_pos_emb(q, cos, sin):                                │
│      q1, q2 = q.chunk(2, dim=-1)                                     │
│        q1: (*, num_heads, 64)    q2: (*, num_heads, 64)             │
│      out1 = q1 * cos - q2 * sin                                      │
│      out2 = q1 * sin + q2 * cos                                      │
│      q_rot = cat([out1, out2], dim=-1)                               │
│                                                                      │
│    Prefill:                                                          │
│      q: (N, 16, 128) ──→ (N, 16, 128)                               │
│      k: (N, 8, 128)  ──→ (N, 8, 128)                                │
│                                                                      │
│    Decode:                                                           │
│      q: (B, 16, 128) ──→ (B, 16, 128)                               │
│      k: (B, 8, 128)  ──→ (B, 8, 128)                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Attention (attention)                                               │
│  attention.py:468                                                    │
│                                                                      │
│  num_heads=16, head_dim=128, num_kv_heads=8, scale=1.0              │
│  effective_scale = scale / sqrt(head_dim) = 1.0 / sqrt(128)         │
│                  ≈ 0.0884                                            │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Step 1: Store KV to Cache (如果 cache 已分配)                 │  │
│  │  store_kvcache_kernel (Triton)                                 │  │
│  │  attention.py:8                                                │  │
│  │                                                                │  │
│  │  k: (*, 8, 128) → reshape → (num_tokens, 8, 128)              │  │
│  │  v: (*, 8, 128) → reshape → (num_tokens, 8, 128)              │  │
│  │                                                                │  │
│  │  k_cache: (num_blocks, 256, 8, 128)                           │  │
│  │  v_cache: (num_blocks, 256, 8, 128)                           │  │
│  │  slot_mapping: (num_tokens,)                                   │  │
│  │                                                                │  │
│  │  写入: cache[block_idx][block_offset][head][:] = k/v           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────── Prefill 路径 ──────────────────────────────────┐    │
│  │  Flash Attention (flash_attention_varlen_kernel, Triton)      │    │
│  │  attention.py:112                                             │    │
│  │                                                               │    │
│  │  q: (N, 16, 128)                                              │    │
│  │  k: (N, 8, 128)    (GQA: 每 2 个 Q head 共享 1 个 KV head)   │    │
│  │  v: (N, 8, 128)                                               │    │
│  │  cu_seqlens: (num_seqs+1,)                                    │    │
│  │                                                               │    │
│  │  Grid: (ceil(max_seq_len/BLOCK_M), 16, num_seqs)             │    │
│  │  BLOCK_M=64, BLOCK_N=64  (head_dim<=64 时)                   │    │
│  │             =32          (head_dim<=128 时, 本模型)           │    │
│  │                                                               │    │
│  │  每个 program 处理:                                            │    │
│  │    某序列 × 某 head × 一块 query (BLOCK_M 行)                 │    │
│  │                                                               │    │
│  │  在线 softmax:                                                 │    │
│  │    m_i, l_i 递推, 逐 KV block 累加                            │    │
│  │    qk = dot(q_block, k_block^T) * scale                       │    │
│  │      (BLOCK_M, head_dim) × (head_dim, BLOCK_N)                │    │
│  │      → (BLOCK_M, BLOCK_N)                                     │    │
│  │    因果 mask: offs_m >= offs_n (IS_CAUSAL=True)              │    │
│  │    p = exp(qk - m_i_new)                                      │    │
│  │    acc += dot(p, v_block)                                     │    │
│  │                                                               │    │
│  │  output: (N, 16, 128)                                         │    │
│  │  reshape → (N, 2048)     [= num_heads * head_dim]             │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────── Decode 路径 ───────────────────────────────────┐    │
│  │  Paged Attention (paged_attention_decode_kernel, Triton)      │    │
│  │  attention.py:297                                             │    │
│  │                                                               │    │
│  │  q: (B, 16, 128)                                              │    │
│  │  k_cache: (num_blocks, 256, 8, 128)                           │    │
│  │  v_cache: (num_blocks, 256, 8, 128)                           │    │
│  │  block_tables: (B, max_num_blocks)                            │    │
│  │  context_lens: (B,)                                           │    │
│  │                                                               │    │
│  │  Grid: (B, 16)                                                │    │
│  │  BLOCK_N = 64  (head_dim<=128)                                │    │
│  │                                                               │    │
│  │  每个 program 处理:                                            │    │
│  │    某 batch × 某 head 对全部历史 token 的 attention            │    │
│  │                                                               │    │
│  │  逐 chunk (BLOCK_N) 遍历历史 KV:                              │    │
│  │    logical_block = token_idx // block_size                     │    │
│  │    physical_block = block_tables[batch][logical_block]         │    │
│  │    k = k_cache[physical_block][offset_in_block][kv_head][:]    │    │
│  │    score = sum(q * k) * scale                                 │    │
│  │    在线 softmax 累加                                           │    │
│  │    v = v_cache[physical_block][offset_in_block][kv_head][:]    │    │
│  │    acc += weight * v                                          │    │
│  │                                                               │    │
│  │  output: (B, 16, 128)                                         │    │
│  │  reshape → (B, 2048)                                          │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  return o: (*, 2048)     [= num_heads * head_dim]                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RowParallelLinear (o_proj)                                          │
│  linear.py:199                                                       │
│                                                                      │
│  weight: (hidden_size, head_dim * num_heads / tp)                    │
│        = (1024, 2048/tp)                                             │
│  bias: None  (bias=False)                                            │
│                                                                      │
│  o = F.linear(o, weight)                                             │
│    Prefill: (N, 2048) ──→ (N, 1024)                                 │
│    Decode:  (B, 2048) ──→ (B, 1024)                                 │
│                                                                      │
│  TP>1: dist.all_reduce(SUM) → 所有 GPU 得到相同结果                  │
│                                                                      │
│  return o: (*, 1024)                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 六、Qwen3MLP 展开图

```
qwen3.py:129  Qwen3MLP

输入: x: (*, 1024)
      * = N (prefill) 或 B (decode)

═══════════════════════════════════════════════════════════════════════════
                         Qwen3MLP.forward()
═══════════════════════════════════════════════════════════════════════════

  x: (*, 1024)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MergedColumnParallelLinear (gate_up)                                │
│  linear.py:113                                                       │
│                                                                      │
│  output_sizes = [intermediate_size, intermediate_size]               │
│               = [3072, 3072]                                         │
│  total_output = 6144                                                 │
│                                                                      │
│  weight: (6144/tp, 1024)     [TP 按输出列分片]                       │
│  bias:   (6144/tp,)          [ffn_bias=False → None]                 │
│                                                                      │
│  gate_up_out = F.linear(x, weight)                                   │
│                                                                      │
│  Prefill: (N, 1024) ──→ (N, 6144)       [tp=1]                      │
│  Decode:  (B, 1024) ──→ (B, 6144)       [tp=1]                      │
│                                                                      │
│  输出布局: [gate_proj_out | up_proj_out]                              │
│           ←── 3072 ──→←── 3072 ──→                                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SiluAndMul (activation)                                             │
│  activation.py:6                                                     │
│                                                                      │
│  gate_up_out: (*, 6144)                                              │
│                                                                      │
│  x_gate, y_up = chunk(2, dim=-1)                                     │
│    x_gate: (*, 3072)     y_up: (*, 3072)                             │
│                                                                      │
│  output = SiLU(x_gate) * y_up                                        │
│         = (x_gate * sigmoid(x_gate)) * y_up                          │
│                                                                      │
│  Prefill: (N, 6144) ──→ (N, 3072)                                   │
│  Decode:  (B, 6144) ──→ (B, 3072)                                   │
│                                                                      │
│  这就是 SwiGLU 激活函数                                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RowParallelLinear (down_proj)                                       │
│  linear.py:199                                                       │
│                                                                      │
│  weight: (hidden_size, intermediate_size/tp)                         │
│        = (1024, 3072/tp)                                             │
│  bias:   None  (ffn_bias=False)                                      │
│                                                                      │
│  down_out = F.linear(activated, weight)                               │
│                                                                      │
│  Prefill: (N, 3072) ──→ (N, 1024)                                   │
│  Decode:  (B, 3072) ──→ (B, 1024)                                   │
│                                                                      │
│  TP>1: dist.all_reduce(SUM) → 所有 GPU 得到相同结果                  │
│                                                                      │
│  return down_out: (*, 1024)                                          │
└──────────────────────────────────────────────────────────────────────┘
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

  逻辑结构:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Block 0          Block 1          Block 2        ...           │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
  │  │ token 0~255 │  │ token 256~  │  │ token 512~  │             │
  │  │             │  │      511    │  │      767    │             │
  │  │ 256 slots   │  │ 256 slots   │  │ 256 slots   │             │
  │  │ × 8 heads   │  │ × 8 heads   │  │ × 8 heads   │             │
  │  │ × 128 dim   │  │ × 128 dim   │  │ × 128 dim   │             │
  │  └─────────────┘  └─────────────┘  └─────────────┘             │
  └─────────────────────────────────────────────────────────────────┘

  slot_mapping 映射:
    slot_id = block_id * block_size + offset_in_block
    如 block_id=3, offset=10 → slot_id = 3*256+10 = 778
```

---

## 八、单层 DecoderLayer 完整数据流 (Prefill, tp=1)

```
Qwen3DecoderLayer.forward()    qwen3.py:197

以 Prefill 为例, N = total_tokens, tp=1:

  x: (N, 1024)          residual: (N, 1024) | None
  │                              │
  │  ┌───────────────────────────┘
  │  │
  ▼  ▼
┌────────────────────────────────────────┐
│  input_layernorm (RMSNorm)             │
│  + residual 融合                       │
│                                        │
│  if residual is not None:              │
│    x = x + residual                    │  ← 上一层 MLP 输出 + 残差
│    residual = x                        │
│    x = RMSNorm(x)                      │
│  else:                                 │
│    residual = x                        │  ← 第一层, embedding 输出
│    x = RMSNorm(x)                      │
│                                        │
│  x: (N, 1024) → (N, 1024)             │
│  residual: (N, 1024)                   │
└───────────────┬────────────────────────┘
                │ x: (N, 1024)
                ▼
┌────────────────────────────────────────┐
│  Qwen3Attention                        │
│                                        │
│  qkv_proj:  (N,1024)→(N,4096)        │
│  split:     q(N,2048) k(N,1024)       │
│                   v(N,1024)            │
│  reshape:   q(N,16,128) k(N,8,128)    │
│                   v(N,8,128)           │
│  q_norm:    q(N,16,128)→(N,16,128)    │
│  k_norm:    k(N,8,128)→(N,8,128)      │
│  RoPE:      q(N,16,128) k(N,8,128)    │
│  store_kv:  k,v → KV cache            │
│  Flash Attn: → (N,16,128)             │
│  reshape:    → (N,2048)               │
│  o_proj:    (N,2048)→(N,1024)         │
│                                        │
│  output: (N, 1024)                     │
└───────────────┬────────────────────────┘
                │ x: (N, 1024)
                │ residual: (N, 1024)
                ▼
┌────────────────────────────────────────┐
│  post_attention_layernorm (RMSNorm)    │
│  + residual 融合                       │
│                                        │
│  x = x + residual                      │  ← attention 输出 + 残差
│  residual = x                          │
│  x = RMSNorm(x)                        │
│                                        │
│  x: (N, 1024) → (N, 1024)             │
│  residual: (N, 1024)                   │
└───────────────┬────────────────────────┘
                │ x: (N, 1024)
                ▼
┌────────────────────────────────────────┐
│  Qwen3MLP                              │
│                                        │
│  gate_up:    (N,1024)→(N,6144)        │
│  SiLU*Mul:  (N,6144)→(N,3072)         │
│  down_proj:  (N,3072)→(N,1024)        │
│                                        │
│  output: (N, 1024)                     │
└───────────────┬────────────────────────┘
                │
                ▼
  return x: (N, 1024), residual: (N, 1024)
  → 传入下一层 DecoderLayer
```

---

## 九、Tensor Parallelism 数据流 (tp>1)

```
以 tp=2 为例, Qwen3Attention 内的 TP 分片:

═══════════════════════════════════════════════════════════════════

  x: (*, 1024)     ← REPLICATED (所有 GPU 相同)
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  QKVColumnParallelLinear (Column Parallel)                   │
│                                                              │
│  Full weight: (4096, 1024)                                   │
│  GPU0 weight: (2048, 1024)   ← 前 2048 行                   │
│  GPU1 weight: (2048, 1024)   ← 后 2048 行                   │
│                                                              │
│  GPU0 output: (*, 2048)      GPU1 output: (*, 2048)         │
│  (各自独立计算, 无通信)                                       │
└──────────────┬───────────────────────────┬───────────────────┘
               │                           │
               ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  GPU0:                   │  │  GPU1:                   │
│  split → q,k,v per GPU   │  │  split → q,k,v per GPU   │
│  q: (*,8,128)  [8 heads] │  │  q: (*,8,128)  [8 heads] │
│  k: (*,4,128)  [4 kv_h]  │  │  k: (*,4,128)  [4 kv_h]  │
│  v: (*,4,128)  [4 kv_h]  │  │  v: (*,4,128)  [4 kv_h]  │
│                          │  │                          │
│  Attention → (*,8,128)   │  │  Attention → (*,8,128)   │
│  reshape → (*,1024)      │  │  reshape → (*,1024)      │
└──────────────┬───────────┘  └──────────┬───────────────┘
               │                         │
               ▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│  RowParallelLinear (o_proj, Row Parallel)                    │
│                                                              │
│  Full weight: (1024, 2048)                                   │
│  GPU0 weight: (1024, 1024)   ← 前 1024 列                   │
│  GPU1 weight: (1024, 1024)   ← 后 1024 列                   │
│                                                              │
│  GPU0 partial: (*, 1024)     GPU1 partial: (*, 1024)        │
│               │                              │               │
│               └──────── all_reduce(SUM) ─────┘               │
│                              │                               │
│                              ▼                               │
│                    output: (*, 1024)                         │
│                    REPLICATED (所有 GPU 相同)                 │
└──────────────────────────────────────────────────────────────┘


MLP 的 TP 模式相同:
  gate_up (ColumnParallel): 按输出列分片
  down_proj (RowParallel):  按输入行分片 + all_reduce
```

---

## 十、CUDA Graph 加速 (Decode 阶段)

```
model_runner.py:572  capture_cudagraph()

预分配固定大小缓冲区:
  input_ids:    (max_bs,)           dtype: int64
  slot_mapping: (max_bs,)           dtype: int64
  context_lens: (max_bs,)           dtype: int64
  block_tables: (max_bs, max_num_blocks)  dtype: int32
  outputs:      (max_bs, 1024)      dtype: float32

捕获 batch_sizes = [1, 2, 4, 8, 16, 32, ...]

执行时:
  ① 找到 >= 实际 bs 的最小捕获图
  ② 将实际数据 copy 进预分配缓冲区
  ③ graph.replay()  → 重放捕获的 CUDA kernel 序列
  ④ 从 outputs[:bs] 取结果

  → 避免每次 decode 的 kernel launch 开销
```

---

## 十一、完整 Shape 汇总表 (Prefill, N tokens, tp=1)

| 模块 | 操作 | 输入 Shape | 输出 Shape |
|------|------|-----------|-----------|
| Tokenizer | encode | str | list[int] |
| VocabParallelEmbedding | embedding lookup | (N,) | (N, 1024) |
| LayerNorm (input) | RMSNorm | (N, 1024) | (N, 1024) |
| QKVColumnParallelLinear | qkv projection | (N, 1024) | (N, 4096) |
| Split | q,k,v split | (N, 4096) | q:(N,2048) k:(N,1024) v:(N,1024) |
| Reshape | view | q:(N,2048) | (N, 16, 128) |
| Reshape | view | k:(N,1024) | (N, 8, 128) |
| Reshape | view | v:(N,1024) | (N, 8, 128) |
| LayerNorm (q_norm) | RMSNorm per-head | (N, 16, 128) | (N, 16, 128) |
| LayerNorm (k_norm) | RMSNorm per-head | (N, 8, 128) | (N, 8, 128) |
| RotaryEmbedding | RoPE | q:(N,16,128) k:(N,8,128) | q:(N,16,128) k:(N,8,128) |
| store_kvcache | Triton kernel | k:(N,8,128) v:(N,8,128) | 写入 KV cache |
| Flash Attention | varlen kernel | q:(N,16,128) k:(N,8,128) v:(N,8,128) | (N, 16, 128) |
| Reshape | flatten heads | (N, 16, 128) | (N, 2048) |
| RowParallelLinear (o_proj) | output projection | (N, 2048) | (N, 1024) |
| LayerNorm (post_attn) | RMSNorm+residual | (N, 1024) | (N, 1024) |
| MergedColumnParallelLinear | gate_up | (N, 1024) | (N, 6144) |
| SiluAndMul | SwiGLU | (N, 6144) | (N, 3072) |
| RowParallelLinear (down) | down projection | (N, 3072) | (N, 1024) |
| × 28 layers | 循环 | (N, 1024) | (N, 1024) |
| LayerNorm (final) | RMSNorm+residual | (N, 1024) | (N, 1024) |
| ParallelLMHead | gather last + linear | (N, 1024) → (S, 1024) | (S, 151936) |
| SamplerLayer | temp + softmax + gumbel | (S, 151936) | (S,) |

> S = num_seqs (batch 中的序列数)

---

## 十二、完整 Shape 汇总表 (Decode, B batch, tp=1)

| 模块 | 操作 | 输入 Shape | 输出 Shape |
|------|------|-----------|-----------|
| prepare_decode | gather last token | seqs | (B,) |
| VocabParallelEmbedding | embedding lookup | (B,) | (B, 1024) |
| LayerNorm (input) | RMSNorm | (B, 1024) | (B, 1024) |
| QKVColumnParallelLinear | qkv projection | (B, 1024) | (B, 4096) |
| Split | q,k,v split | (B, 4096) | q:(B,2048) k:(B,1024) v:(B,1024) |
| Reshape | view | q:(B,2048) | (B, 16, 128) |
| Reshape | view | k:(B,1024) | (B, 8, 128) |
| Reshape | view | v:(B,1024) | (B, 8, 128) |
| LayerNorm (q_norm) | RMSNorm per-head | (B, 16, 128) | (B, 16, 128) |
| LayerNorm (k_norm) | RMSNorm per-head | (B, 8, 128) | (B, 8, 128) |
| RotaryEmbedding | RoPE | q:(B,16,128) k:(B,8,128) | q:(B,16,128) k:(B,8,128) |
| store_kvcache | Triton kernel | k:(B,8,128) v:(B,8,128) | 写入 KV cache |
| Paged Attention | decode kernel | q:(B,16,128) + KV cache | (B, 16, 128) |
| Reshape | flatten heads | (B, 16, 128) | (B, 2048) |
| RowParallelLinear (o_proj) | output projection | (B, 2048) | (B, 1024) |
| LayerNorm (post_attn) | RMSNorm+residual | (B, 1024) | (B, 1024) |
| MergedColumnParallelLinear | gate_up | (B, 1024) | (B, 6144) |
| SiluAndMul | SwiGLU | (B, 6144) | (B, 3072) |
| RowParallelLinear (down) | down projection | (B, 3072) | (B, 1024) |
| × 28 layers | 循环 | (B, 1024) | (B, 1024) |
| LayerNorm (final) | RMSNorm+residual | (B, 1024) | (B, 1024) |
| ParallelLMHead | linear | (B, 1024) | (B, 151936) |
| SamplerLayer | temp + softmax + gumbel | (B, 151936) | (B,) |

---

## 十三、权重加载映射 (loader.py)

```
HF Checkpoint 参数名                 →  HermesInfer 参数名
─────────────────────────────────────────────────────────────────

model.layers.{i}.self_attn.q_proj.weight  ┐
model.layers.{i}.self_attn.k_proj.weight  ├─cat→ model.layers.{i}.self_attn.qkv_projection.weight
model.layers.{i}.self_attn.v_proj.weight  ┘       shape: (4096, 1024)

model.layers.{i}.self_attn.o_proj.weight         → model.layers.{i}.self_attn.o_proj.weight
                                                    shape: (1024, 2048)

model.layers.{i}.self_attn.q_norm.weight          → model.layers.{i}.self_attn.q_norm.weight
                                                    shape: (128,)
model.layers.{i}.self_attn.k_norm.weight          → model.layers.{i}.self_attn.k_norm.weight
                                                    shape: (128,)

model.layers.{i}.mlp.gate_proj.weight     ┐
model.layers.{i}.mlp.up_proj.weight       ├─cat→ model.layers.{i}.mlp.gate_up.weight
                                           ┘       shape: (6144, 1024)

model.layers.{i}.mlp.down_proj.weight            → model.layers.{i}.mlp.down_proj.weight
                                                    shape: (1024, 3072)

model.layers.{i}.input_layernorm.weight          → model.layers.{i}.input_layernorm.weight
                                                    shape: (1024,)
model.layers.{i}.post_attention_layernorm.weight → model.layers.{i}.post_attention_layernorm.weight
                                                    shape: (1024,)

model.embed_tokens.weight                        → model.embed_tokens.weight
                                                    shape: (151936, 1024)
model.norm.weight                                → model.norm.weight
                                                    shape: (1024,)

tie_word_embeddings=True → lm_head.weight = embed_tokens.weight (共享)
```

---

## 十四、关键源码文件索引

| 文件 | 核心类/函数 | 职责 |
|------|------------|------|
| `models/qwen3.py` | Qwen3ForCausalLM, Qwen3Model, Qwen3DecoderLayer, Qwen3Attention, Qwen3MLP | 模型定义 |
| `layers/attention.py` | Attention, flash_attention_prefill, paged_attention_decode, store_kvcache | Triton attention kernels |
| `layers/linear.py` | QKVColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear | TP 线性层 |
| `layers/layernorm.py` | LayerNorm (RMSNorm) | 归一化 |
| `layers/rotary_embedding.py` | RotaryEmbedding, apply_rotary_pos_emb | RoPE 旋转位置编码 |
| `layers/activation.py` | SiluAndMul | SwiGLU 激活 |
| `layers/embedding_head.py` | VocabParallelEmbedding, ParallelLMHead | 嵌入层 + LM head |
| `layers/sampler.py` | SamplerLayer | Gumbel-max 采样 |
| `engine/llm_engine.py` | LLMEngine | 引擎入口, 调度循环 |
| `engine/model_runner.py` | ModelRunner | GPU 执行, KV cache, CUDA graph |
| `engine/scheduler.py` | Scheduler | 连续批处理调度 |
| `engine/block_manager.py` | BlockManager | Paged KV 地址空间管理 |
| `engine/sequence.py` | Sequence | 单条请求状态 |
| `utils/context.py` | Context, set_context, get_context | Attention 元数据单例 |
| `utils/loader.py` | load_weights_from_checkpoint | HF 权重加载 (QKV/gate_up 合并) |
