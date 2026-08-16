<h1 align="center">vLLM 技术路线</h1>
<p align="center">
| <a href="./HowToApproachvLLM.md"><b>English</b></a> 
| <a href="./HowToApproachvLLM_zh.md"><b>简体中文</b></a> |
</p>

本文档提供了理解和复现一个最小vLLM的分步指南。通过该文档的小结顺序以获得最佳学习体验。

**原始开发环境及测试基于A6000 GPU。**

[配套视频链接](https://www.bilibili.com/video/BV1Vjz1B2EQu)

---

## Step 1: Layers

首先构建基本的神经网络层，`\layers`目录中存放了模型的基础结构块。

### 1.1 激活函数 ✅

具体实现：[activation.py](src/myvllm/layers/activation.py)

首先实现激活函数。本项目使用的是 **SwiGLU** 变体——`SiluAndMul`，而非简单的 SiLU 或 GELU。

**源码解析（`activation.py:6-18`）：**

```python
class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)      # 沿最后一维对半切分
        return F.silu(x) * y        # 前半做 SiLU，再与后半逐元素相乘
```

**为什么叫 SwiGLU？** 输入张量在最后一维被 `chunk(2, -1)` 等分为两半 `x` 和 `y`。前半部分经过 SiLU 激活（即 Swish：`x * sigmoid(x)`），然后与后半部分逐元素相乘。这正是 GLU（Gated Linear Unit）家族的 SwiGLU 变体。

**数据流示意：**
```
输入: (batch, seq_len, 2 * intermediate_size)
       ↓ chunk(2, -1)
x: (batch, seq_len, intermediate_size)    y: (batch, seq_len, intermediate_size)
       ↓ F.silu()
silu_x: (batch, seq_len, intermediate_size)
       ↓ 逐元素相乘
输出: (batch, seq_len, intermediate_size)
```

**`@torch.compile` 的作用：** 装饰器会将 `chunk` → `silu` → `mul` 这三个操作融合为一个 CUDA kernel，减少显存读写和 kernel 启动开销。

**关键学习: `torch.compile` 优化**
- 基准测试:
	```python
	for _ in range(10): # 预热循环
		_ = layer(input_tensor)
	
	times = []
	for _ in range(100): # 计算循环
		torch.cuda.synchronize()
		start_time = time.time()
		output_tensor = layer(input_tensor)
		torch.cuda.synchronize()
		end_time = time.time()
		times.append(end_time - start_time)
	```

**测试结果:**
| tensor shape         | torch.compile | time (ms) |
| ---------------      | ------------- | --------- |
| (400, 800)           | on            |  0.2044   |
| (400, 800)           | off           |  0.0823   |
| (4000, 8000)         | on            |  0.4494   |
| (4000, 8000)         | off           |  0.5290   |
| (8, 4000, 8000)      | on            |  2.3865   |
| (8, 4000, 8000)      | off           |  3.7650   |

**要点:** `torch.compile` 由于编译成本，有助于加速大型tensor的计算，对于小型tensor的计算反而会因为编译时间过长降低效率。

---

### 1.2 RMS LayerNorm ✅

具体实现：[layernorm.py](src/myvllm/layers/layernorm.py)

实现RMS层归一化，帮助稳定训练。

**源码解析（`layernorm.py:4-34`）：**

```python
class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        self.eps = eps

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        x_norm = (x / sqrt_variance * self.weight)
        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual                    # 先加 residual
        return self.rms_forward(x), x       # 返回 (归一化结果, 加过 residual 的原始值)

    def forward(self, x, residual=None):
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)
```

**公式推导：**
```
标准 LayerNorm:  x_norm = (x - mean(x)) / sqrt(var(x) + ε) × γ + β
RMS LayerNorm:   x_norm = x / sqrt(mean(x²) + ε) × γ
```
RMSNorm 省略了中心化步骤（不减均值），只保留缩放。这在大模型中效果等价但计算更快。

**残差连接的双返回值设计：**
`residual_rms_forward` 返回两个值：
1. **归一化后的 x**：送入下一层（attention 或 MLP）
2. **加过 residual 的原始 x**：作为下一层的 residual

这实现了经典的 Pre-Norm 残差模式：
```
residual = x                    # 保存原始输入
x = LayerNorm(x)               # 归一化
x = Attention(x)               # 子层计算
x = x + residual               # 残差连接
# 下一轮：residual = x, 再 LayerNorm...
```

**关键知识:**
- 对激活进行归一化，但不做均值中心化（只使用 RMS 均方根）
- 对大模型而言比 LayerNorm 更高效
- 对训练稳定性至关重要
- 基准测试:
	```python
    for _ in range(10): # 预热循环
        _ = layer(x)
    
    # 不使用残差的情况
    times = [] 
    for _ in range(100): # 计算循环
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
	```

**基准测试:**
| tensor shape    | torch.compile | residuals | time (ms) |
| --------------- | ------------- | --------- | --------: |
| (400, 800)      | off           | off       |  0.1630   |
| (400, 800)      | off           | on        |  0.1703   |
| (400, 800)      | on            | off       |  0.2024   |
| (400, 800)      | on            | on        |  0.3470   |
| (4000, 8000)    | off           | off       |  1.3725   |
| (4000, 8000)    | off           | on        |  1.9269   |
| (4000, 8000)    | on            | off       |  0.6029   |
| (4000, 8000)    | on            | on        |  1.1786   |
| (8, 4000, 8000) | off           | off       | 10.4689   |
| (8, 4000, 8000) | off           | on        | 15.3257   |
| (8, 4000, 8000) | on            | off       |  3.6483   |
| (8, 4000, 8000) | on            | on        |  8.1566   |

**要点:** 类似于激活函数的基准测试，`torch.compile` 在计算量较大的场景下更有帮助，但对于小规模算子会带来额外开销。

---

### 1.3 线性层 （支持张量并行） ✅

具体实现：[linear.py](src/myvllm/layers/linear.py)

线性层是最复杂的一层，因为需要支持分布式训练，所以需要实现张量并行。

**源码架构（`linear.py`）：**

```
LinearBase (抽象基类)
├── ReplicatedLinear      — 不做切分，每张 GPU 存完整权重
├── ColumnParallelLinear  — 沿输出维度切分
│   ├── MergedColumnParallelLinear  — 合并多个列并行层
│   └── QKVColumnParallelLinear     — QKV 专用列并行
└── RowParallelLinear     — 沿输入维度切分
```

**基类 `LinearBase`（`linear.py:6-39`）：**
```python
class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=True, tp_dim=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()          # 当前 GPU 编号
        self.tp_size = dist.get_world_size()    # GPU 总数

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # 绑定自定义加载器
```

**`weight_loader` 机制：** 每个参数对象上挂载了一个 `weight_loader` 方法。当从 checkpoint 加载权重时，加载器会检查参数是否有自定义 `weight_loader`，如果有就调用它来自动切分出当前 GPU 对应的分片。

**核心概念：分布式模型中的权重加载**
```python
# 将 checkpoint 加载到分片（sharded）模型时：
for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # 完整模型参数 (4096, 4096)
        
        # 检查该参数是否有自定义的 weight_loader
        if hasattr(param, 'weight_loader'):
            # 调用自定义 weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader 会自动完成：
            # 1. 取出与当前 GPU 对应的分片（shard）
            # 2. 将其拷贝到 param.data
        else:
            # 默认行为：直接拷贝
            param.data.copy_(loaded_weight)

```

**并行线性层的类型：**

1. **ColumnParallelLinear** ✅（`linear.py:84-110`）
    - 沿输出维度在多张 GPU 上切分
    - 每张 GPU 计算输出特征的一部分
    - 前向传播过程中不需要通信

    ```python
    # 初始化：output_size 除以 tp_size
    super().__init__(input_size, output_size // tp_size, bias, tp_dim=0)

    # weight_loader：从完整权重中切出当前 GPU 的列分片
    def weight_loader(self, param, loaded_weights):
        shard_size = full_data_output_size // self.tp_size
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    # forward：与普通线性层一样，无需通信
    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)
    ```

    **切分示意（2 GPU，output_size=8）：**
    ```
    完整权重 (8, input):  [row0, row1, row2, row3, row4, row5, row6, row7]
    GPU 0 权重 (4, input): [row0, row1, row2, row3]
    GPU 1 权重 (4, input): [row4, row5, row6, row7]
    ```

2. **RowParallelLinear** ✅（`linear.py:199-226`）
    - 沿输入维度在多张 GPU 上切分
    - 需要用 `dist.all_reduce` 对部分结果求和
    - 通常接在 `ColumnParallel` 层之后使用

    ```python
    # 初始化：input_size 除以 tp_size
    super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    # weight_loader：从完整权重中切出当前 GPU 的行分片
    def weight_loader(self, param, loaded_weights):
        shard_size = full_data_input_size // self.tp_size
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    # forward：先做本地线性计算，再 all_reduce 求和
    def forward(self, x):
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result
    ```

    **为什么 Column + Row 是标准搭配？**
    ```
    ColumnParallel: 输入 replicated → 输出 sharded
    RowParallel:    输入 sharded    → 输出 replicated (after all_reduce)
    组合:           输入 replicated → 输出 replicated
    ```

3. **MergedColumnParallelLinear** ✅（`linear.py:113-149`）
    - 将多个列并行层合并（例如 gate + up 两个投影）
    - 必须同时对 `param_data` 和 `loaded_weights` 进行切分，以匹配对应的矩阵
    - 对 MLP 层更高效

    ```python
    def __init__(self, input_size, output_sizes, bias=True):
        self.output_sizes = output_sizes  # 例如 [intermediate_size, intermediate_size]
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param, loaded_weights, loaded_weight_id):
        # 计算当前矩阵在合并参数中的偏移
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # 在 param_data 中找到正确位置
        param_data = param_data.narrow(0, offset, shard_size)
        # 从完整权重中切出当前 GPU 的分片
        shard_weights = loaded_weights.narrow(0, self.tp_rank * shard_size, shard_size)
        param_data.copy_(shard_weights)
    ```

    **合并示意（gate + up，2 GPU）：**
    ```
    合并后的参数布局 (每 GPU):
    [gate_shard_0 | up_shard_0]   ← GPU 0
    [gate_shard_1 | up_shard_1]   ← GPU 1
    ```

4. **QKVColumnParallel** ✅（`linear.py:152-196`）
    - Attention 中 Q/K/V 投影的特殊情况
    - 每张 GPU 存完整的 heads（不对 `head_size` 维度做切分）
    - 使每张 GPU 可以独立完成注意力计算

    ```python
    def __init__(self, input_size, head_size, num_heads, num_kv_heads=None, bias=False):
        self.num_heads = num_heads // self.tp_size       # 每 GPU 的 Q head 数
        self.num_kv_heads = num_kv_heads // self.tp_size  # 每 GPU 的 KV head 数
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)

    def weight_loader(self, param, loaded_weights, load_weight_id):
        # load_weight_id: 'q', 'k', 'v'
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
    ```

    **GQA 布局示意（num_heads=16, num_kv_heads=8, 2 GPU）：**
    ```
    GPU 0: [Q_head_0..7 | KV_head_0..3 | KV_head_0..3]
    GPU 1: [Q_head_8..15 | KV_head_4..7 | KV_head_4..7]
    每张 GPU 独立做 attention，无需跨卡通信
    ```

**基准测试：**
使用如下指令，进入`src/myvllm/layers`目录，在分布式环境测试运行结果是否正确
```
cd src/myvllm/layers
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 linear.py
```
输出结果`allclose=True`则代表多机并行结果与单机一致
```
[ColumnParallel] allclose=True, max_abs_err=0.000107
[MergedColumnParallel] allclose=True, max_abs_err=0.000103
[QKVColumnParallel] allclose=True, max_abs_err=0.000061
[RowParallel] allclose=True, max_abs_err=0.000011
```
**MLP 层的常见模式:**
    - 一个 `ColumnParallel` → 一个 RowParallel → `dist.all_reduce`
    - 第一层的输出切分方式 = 第二层的输入切分方式



---

### 1.4 词表嵌入（Vocab Embedding）与 LM Head ✅

具体实现：[embedding_head.py](src/myvllm/layers/embedding_head.py)


**词表嵌入（Vocab Embedding）：**
- 将词表按 GPU 进行切分（分片）
- 每张 GPU 只存储词表的一部分

**源码解析（`embedding_head.py:12-61`）：**

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.padded_num_embeddings = (num_embeddings + tp_size - 1) // tp_size * tp_size
        self.num_embeddings_per_partition = self.padded_num_embeddings // tp_size
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, x):
        # 1. 构造 mask：哪些 token 属于当前 GPU 的词表范围
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        # 2. 将 token id 转换为本地偏移
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        output = F.embedding(x, self.weight)
        # 3. 对不属于本 GPU 的 token，将 embedding 置零
        output = mask.unsqueeze(1) * output
        # 4. all_reduce 求和：每张 GPU 只对自己的 token 有非零 embedding
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
```

**词表切分示意（vocab_size=10000, 2 GPU）：**
```
GPU 0: 词表 [0, 5000)     → weight shape: (5000, hidden_size)
GPU 1: 词表 [5000, 10000)  → weight shape: (5000, hidden_size)

token_id=3000 → GPU 0 有值, GPU 1 mask=0 → all_reduce 后得到正确 embedding
token_id=7000 → GPU 0 mask=0, GPU 1 有值 → all_reduce 后得到正确 embedding
```

**LM Head（`embedding_head.py:64-93`）：**
- 可以与词表嵌入共享权重（tied embeddings，权重绑定）
- `F.linear` 会自动对权重做转置以完成线性计算
- 最终 logits 可使用 `dist.gather` 或 `dist.all_gather` 汇总

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            # prefill 时只取每个序列的最后一个 token 计算 logits
            last_token = context.cu_seqlens_q[1:] - 1
            x = x[last_token].contiguous()

        logits = F.linear(x, self.weight)  # (batch, vocab_per_partition)
        if self.tp_size > 1:
            # 只在 rank 0 收集全部 logits
            all_logits = [torch.empty(...) for _ in range(tp_size)] if tp_rank == 0 else None
            dist.gather(logits, gather_list=all_logits, dst=0)
            if tp_rank == 0:
                logits = torch.cat(all_logits, dim=-1)
                logits = logits[..., :self.num_embeddings]  # 去掉 padding
```

**关键区别（Key Differences）：**
- `dist.gather(tensor, gather_list, dst)`：只有 `dst` 这张 GPU 会收到全部数据
- `dist.all_gather(tensor_list, tensor)`：所有 GPU 都会收到全部数据（没有 `dst` 参数）

**为什么 LM Head 用 `gather` 而不是 `all_gather`？** 因为只有 rank 0 需要做采样（sampling），其他 GPU 不需要完整 logits。使用 `gather` 可以节省显存和通信带宽。

**内存布局（Memory Layout）- contiguous()：**
```python
# 连续内存
x = [1, 2, 3, 4, 5, 6]  # 物理存储: [1][2][3][4][5][6]

# 非连续内存
y = x.reshape(2, 3).T   # 逻辑视图: [[1,4],[2,5],[3,6]]
                        # 物理存储: [1][2][3][4][5][6] ← 仍是旧顺序！
                        # 通过 stride() 来访问元素
```
- `contiguous()` 会让内存块保持相邻 → 访问更快，不需要 `stride()`

---

### 1.5 注意力层（Attention Layer）✅

具体实现：[attention.py](src/myvllm/layers/attention.py)
性能测试：[benchmark_decoding.py](benchmark_decoding.py)

**`Attention` 类总览（`attention.py:468-535`）：**

`Attention` 是统一的入口类，内部根据 `context.is_prefill` 决定走 prefill 还是 decode 路径：

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale=1.0, num_kv_heads=None,
                 block_size=16, is_causal=True):
        self.num_heads = num_heads          # 每 GPU 的 Q head 数
        self.num_kv_heads = num_kv_heads    # 每 GPU 的 KV head 数
        self.k_cache = self.v_cache = torch.tensor([])  # 稍后由 ModelRunner 注入

    def forward(self, q, k, v):
        context = get_context()
        # 1. 将当前 K/V 写入 paged cache（仅 decode 路径）
        if k_cache.numel() > 0 and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # 2a. Prefill：Flash Attention（变长，online softmax）
            o = flash_attention_prefill(q, k, v, context.cu_seqlens_q, scale, ...)
        else:
            # 2b. Decode：Paged Attention（遍历 paged cache）
            o = paged_attention_decode(q, k_cache, v_cache,
                                       context.block_tables, context.context_lens, ...)
        return o.reshape(o.shape[0], self.num_heads * self.head_dim)
```

**Context 单例模式（`utils/context.py`）：**

`Attention` 不通过参数传递元数据，而是通过模块级单例 `Context` 获取：

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    ...

_context = Context()
def get_context() -> Context: return _context
def set_context(...): global _context; _context = Context(...)
def reset_context(): global _context; _context = Context()
```

`ModelRunner` 在每次 `run()` 前调用 `set_context()`，`Attention` 和各层通过 `get_context()` 读取。执行完毕后 `reset_context()` 清理状态。

**关键张量概念（Key Tensor Concepts）：**
- **`stride()`**：当一个张量存储在内存中时，本质上是一个连续的一维数组。stride 用来描述：沿着某个维度移动到“下一个元素”时，需要在底层内存中跳过多少个元素。
	```
	Memory layout: [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
	                  ↑                    ↑                   ↑
	             row 0                  row 1               row 2
	```
- **`numel()`**: 参数总数量

**GPU 架构（A100）：**
- 每个 3D grid 有 4 个 WARP
- 每个 WARP 有 32 个线程
- 每个 grid 会同时处理 128 个线程

**Triton Kernel 备注：**
- 当将 PyTorch 张量传给 Triton kernel 时，**Triton 会自动从张量中提取指针**（内存地址）

---

#### FlashAttention 与 PagedAttention

这两者不是二选一的关系，而是**正交的两个维度**，decode kernel 两者都用：

- **FlashAttention** 是**计算**技术：把序列分块，用 online softmax 逐块合并，从而不需要 materialize `(seq_len, seq_len)` 的分数矩阵。解决的是访存开销。
- **PagedAttention** 是**存储**技术：KV cache 按固定大小的块存放，通过每个序列各自的 block table 间接寻址。解决的是显存碎片化。

|  | Prefill | Decode |
|---|---|---|
| Query 长度 | 整个 prompt（数百到数千 token） | 每个序列恰好 **1** 个 token |
| 计算方式 | 分块 + online softmax（**flash**） | 分块 + online softmax（**flash**） |
| K/V 来源 | 刚算出的 `k`、`v`，varlen 打包 | **paged** KV cache |
| 瓶颈 | 计算（大矩阵乘） | 访存（为产出 1 个 token 要遍历整个 cache） |
| Kernel | `flash_attention_prefill` | `paged_attention_decode` |

所以 decode 是 *flash **加** paged*；本仓库的 prefill 只用了 flash，因为它直接读传进来的张量，不读 cache。

推理引擎的绝大部分时间花在 decode 上，因此本节剩余部分逐步讲解 paged decode kernel。

---

#### Step 1：确定 KV cache 的内存布局

```python
k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
```

cache 是一个由固定大小**块（block）**组成的扁平池。一个序列并不占据其中连续的一段，而是持有一个**块列表** —— `BlockManager` 从空闲块里任意分配。[main.py](main.py) 中 `block_size` 为 256。

展平之后，某个 (block, slot, kv_head, dim) 元素的偏移量为：

```python
offset = (physical_block * block_size * num_kv_heads * head_dim   # 跳过前面整块
          + slot          * num_kv_heads * head_dim               # 跳过块内前面的槽位
          + kv_head       * head_dim                              # 跳过前面的头
          + dim)
```

记住这个公式 —— Step 7 里的每个 bug 都是这个公式中某一项写错了。

#### Step 2：把新的 K/V 写入 cache

`store_kvcache` 启动 `(num_tokens, num_kv_heads)` 的 grid。每个 program 把一个 token 在一个头上的 K 和 V 拷贝到 `slot_mapping[token]` 指定的槽位，该映射由 ModelRunner 预先算好。`slot_mapping` 为 `-1` 表示跳过该 token。

**源码解析（`attention.py:7-108`）：**

```python
@triton.jit
def store_kvcache_kernel(key_ptr, value_ptr, k_cache_ptr, v_cache_ptr,
                         slot_mapping_ptr, num_kv_heads, head_dim, block_size):
    token_idx = tl.program_id(0)       # 第几个 token
    head_idx = tl.program_id(1)        # 第几个 KV head

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return                          # 跳过不需要写入的 token

    # 从 slot_idx 反算出 block 和 block 内偏移
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # 输入偏移：(num_tokens, num_kv_heads, head_dim) 的展平索引
    input_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim + head_offsets

    # Cache 偏移：(num_blocks, block_size, num_kv_heads, head_dim) 的展平索引
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim
                    + block_offset * num_kv_heads * head_dim
                    + head_idx * head_dim + head_offsets)

    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)
```

**`slot_mapping` 的计算逻辑（`model_runner.py:336-340`）：**

在 `prepare_prefill` 中，只为**未缓存**的 blocks 生成 slot：
```python
for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
    if seq.num_cached_blocks + i != seq.num_blocks - 1:
        # 非最后一块：整块都是新 token
        slot_mappings.extend(range(block_id * block_size, (block_id+1) * block_size))
    else:
        # 最后一块：可能不满
        slot_mappings.extend(range(block_id * block_size,
                                   block_id * block_size + seq.last_block_num_tokens))
```

在 `prepare_decode` 中，每个序列只有一个新 token：
```python
slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
```

#### Step 3：Prefill —— 变长注意力

同一批次内的序列长度不同，因此它们被首尾拼接，用 `cu_seqlens`（累积长度）划分边界：

```
cu_seqlens = [0, 5, 12, 20]   ->   seq 0 = tokens 0..4, seq 1 = 5..11, seq 2 = 12..19
```

grid 为 `(cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)`：每个 program 负责一个序列、一个头、一个 query 分块。program 把自己的 Q 块常驻寄存器，让 K/V 块流经它，并不断更新 running softmax。

#### Step 4：Online softmax 递推

两个 kernel 都依赖它。第 `j` 块的注意力结果可以合并进已有结果，全程不需要持有所有分数：

```python
m_new = max(m_old, max(qk_j))          # running max，保证数值稳定
alpha = exp(m_old - m_new)             # 已累积部分需要缩放的比例
acc   = acc * alpha + sum(exp(qk_j - m_new) * V_j)
l     = l   * alpha + sum(exp(qk_j - m_new))
# 最后一块处理完之后：
out   = acc / l
```

`m` 的唯一作用是防止 `exp()` 溢出；`l` 是到目前为止累积的 softmax 分母。

#### Step 5：Decode —— 遍历 paged cache

grid 为 `(batch_size, num_heads)`：每个 program 负责一个（序列，query 头）组合。program 载入它那一个 query 向量，然后以 `BLOCK_N` 个 token 为一块遍历该序列的 KV cache，套用上面的递推。

难点在于：一块**逻辑上**连续的 token，在**物理上**散落在不同的块里。解析一个逻辑 token 需要**两次除法**：

```python
logical_block  = t // block_size                       # 属于本序列的第几块
slot           = t %  block_size                       # 在该块内的第几个位置
physical_block = block_tables[seq, logical_block]      # 间接寻址
```

对 GQA 而言，还需要把 query 头映射到对应的 KV 头：

```python
kv_head_idx = head_idx // (num_heads // num_kv_heads)
```

#### Step 6：如何遍历一个 chunk

为一块 `BLOCK_N` 个 token 填出 `physical_block` 和 `slot`，有三种写法：

**(a) 逐 token 处理** —— 正确但慢。`for i in range(BLOCK_N)` 会展开成 `BLOCK_N` 次标量加载再加 `BLOCK_N` 次 `tl.where` 合并，整个 SIMD 宽度都在闲置。

**(b) 每个 chunk 查一次 block table** —— 快，但仅当一个 chunk 不会跨越两个块时才成立（`block_size % BLOCK_N == 0`），而且仍然需要 `slot = t % block_size`。

**(c) 逐 token gather block table** —— 既快又通用。每条 lane 解析自己的 token，因此一个 chunk 可以跨越任意多个块：

```python
offs_n        = token_start + tl.arange(0, BLOCK_N)
logical_block = offs_n // block_size
offs_in_block = offs_n %  block_size

in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

# 每个 token 一个 block table 条目，而不是每个 chunk 一个
physical_block = tl.load(block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                         mask=in_range, other=-1)
valid = in_range & (physical_block != -1)

# 被 mask 掉的 lane 仍会参与地址运算：把它们指向第 0 块
physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
             + offs_in_block[None, :] * (num_kv_heads * head_dim)
             + kv_head_idx * head_dim
             + offs_d[:, None])
```

K 和 V 共用 `kv_offset`，因此每个 chunk 只 gather 一次 block table，而不是两次。

#### Step 7：容易踩的坑

1. **把全局 token 下标当成块内偏移。** `physical_block * block_size` 已经把指针移到了该块的起始位置，所以下一项必须是 `t % block_size` 而不是 `t`。用 `t` 会越过块尾读到后续块所属的其他序列的数据 —— 越界得足够远时会直接变成非法内存访问。只要测试用的序列都能装进单个块，这个错误就看不出来。
2. **每个 chunk 只查一次 block table。** 只有当 chunk 能装进一个块时才成立。`block_size=16` 配 `BLOCK_N=64` 时，一个 chunk 横跨 4 个块，其中 3 个从未被查过。
3. **被 mask 掉的 lane 仍然会计算地址。** `tl.load(..., mask=...)` 抑制的是**加载**，不是地址运算。`physical_block` 中残留的 `-1` 会算出负偏移，因此要把无效 lane 钳到第 0 块。
4. **int32 溢出。** 每块 256×8×128 = 262144 个元素，`physical_block * block_size * num_kv_heads * head_dim` 在块数超过约 8192 时会溢出 int32。需要把块下标转成 `int64`。
5. **测试里用恒等的 block table。** 如果测试用 `block_tables = torch.arange(...)` 构造，逻辑顺序就等于物理顺序，上面**所有** bug 都会消失。应当使用打乱的映射 —— 真实的 `BlockManager` 产出的就是这样。
6. **丢掉边界检查。** `logical_block < max_num_blocks` 用于防止读越 block table 的尾部。

---

#### 性能测试结果

[benchmark_decoding.py](benchmark_decoding.py) 在完全相同的输入上对比四种实现。每种实现在计时**之前**都会先和 float32 参考结果比对 —— 一个因为读错内存而“变快”的 kernel，会在 correct 列显示 `NO`，而不是显示成加速。

Qwen3-0.6B 的注意力形状（`num_heads=16, num_kv_heads=8, head_dim=128, block_size=256`），fp16，单张 A6000。单次调用耗时（ms），加速比相对 Naive PyTorch：

| batch | seq_len | Naive PyTorch | Fast PyTorch | Triton 逐 token | Triton 分块 |
|------:|--------:|--------------:|-------------:|----------------:|------------:|
| 1 | 128 | 0.385 (1.0x) | 0.778 (0.5x) | 0.088 (4.4x) | **0.028 (14.0x)** |
| 1 | 512 | 0.445 (1.0x) | 0.977 (0.5x) | 0.340 (1.3x) | **0.026 (16.8x)** |
| 1 | 2048 | 0.735 (1.0x) | 0.818 (0.9x) | 1.939 (0.4x) | **0.098 (7.5x)** |
| 8 | 128 | 0.833 (1.0x) | 3.780 (0.2x) | 0.084 (10.0x) | **0.026 (31.8x)** |
| 8 | 512 | 1.232 (1.0x) | 4.029 (0.3x) | 0.465 (2.7x) | **0.040 (31.1x)** |
| 8 | 2048 | 4.297 (1.0x) | 5.042 (0.9x) | 1.848 (2.3x) | **0.127 (33.9x)** |
| 32 | 128 | 2.272 (1.0x) | 14.410 (0.2x) | 0.127 (17.9x) | **0.034 (67.5x)** |
| 32 | 512 | 4.574 (1.0x) | 15.147 (0.3x) | 0.518 (8.8x) | **0.130 (35.1x)** |
| 32 | 2048 | 16.849 (1.0x) | 19.681 (0.9x) | 1.975 (8.5x) | **0.493 (34.2x)** |

如何解读这张表：

- **分块比逐 token 快 3.1x–21.3x**，且上下文越长差距越大。`seq_len=2048` 时逐 token 版本**比 PyTorch 还慢** —— 它的开销取决于 token 数量，而不是访存量。
- **“Fast PyTorch”往往是最慢的。** 它的向量化 gather 会 materialize 一份稠密的 `(batch, max_ctx, kv_heads, head_dim)` cache 副本，batch 32 时这次分配就成了主要开销。而“填充成稠密张量”恰恰是 paged attention 要避免的事情。
- **Triton kernel 在大 batch 下优势最明显**，此时 `(batch, num_heads)` 的 grid 才有足够多的 program 填满 GPU。
- **batch 1 是短板**：只有 16 个 program，GPU 大部分处于空闲。把 KV 序列拆分到多个 program（flash-decoding）是下一步优化 —— grid 变成 `(batch, heads, kv_splits)`，再用第二个 kernel 合并各 split 的 `(m, l, acc)`。

通用的逐 token gather 相比块对齐版本没有可测量的开销（kernel 是访存受限的），而且对任意 `block_size` 都成立：

| block_size | Triton 逐 token | Triton 分块 | 加速比 |
|-----------:|----------------:|------------:|-------:|
| 16 | 0.978 ms | 0.056 ms | 17.5x |
| 32 | 0.979 ms | 0.056 ms | 17.5x |
| 100 | 1.014 ms | 0.057 ms | 17.8x |
| 256 | 0.955 ms | 0.053 ms | 18.0x |

（batch 4，seq_len 1000。`block_size=100` 是刻意选的：既不是 2 的幂，也不是 `BLOCK_N` 的整数倍。）

运行方式：

```bash
uv run python benchmark_decoding.py                                  # 默认扫描
uv run python benchmark_decoding.py --block-size 16 --seq-lens 1000  # 单个配置
```

---

### 1.6 旋转位置编码（RoPE）✅

具体实现：[rotary_embedding.py](src/myvllm/layers/rotary_embedding.py)

为具备位置信息的注意力实现旋转位置嵌入（rotary position embeddings）。

**源码解析（`rotary_embedding.py:48-109`）：**

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, base, rotary_embedding, max_position=2048, is_llama3=False, ...):
        # 计算逆频率：θ_i = 1 / (base^(2i/d))
        self.inv_freq = 1 / (base ** (torch.arange(0, rotary_embedding, 2) / rotary_embedding))

        if is_llama3:
            # Llama 3.2 的频率缩放（详见下文）
            ...

        # 预计算 cos/sin 缓存表
        positions = torch.arange(max_position).float()
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)  # (max_pos, dim/2)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)                 # (max_pos, dim)
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]   # 按位置索引查表
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (apply_rotary_pos_emb(query, cos, sin),
                apply_rotary_pos_emb(key, cos, sin))
```

**旋转操作 `apply_rotary_pos_emb`（`rotary_embedding.py:4-45`）：**

```python
def apply_rotary_pos_emb(x, cos, sin):
    # 将 x 沿 head_dim 对半切分
    x1, x2 = x.chunk(2, dim=-1)
    # 旋转：[x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)
```

**数学原理：**

RoPE 将位置信息编码为旋转矩阵，使得两个位置的注意力分数只取决于它们的**相对距离**：

```
R(θ, m) = diag(cos(mθ₁), cos(mθ₁), cos(mθ₂), cos(mθ₂), ...)
        + diag(-sin(mθ₁), sin(mθ₁), -sin(mθ₂), sin(mθ₂), ...) · P

其中 θᵢ = base^(-2i/d)，m 为位置索引，P 为置换矩阵
```

**Llama 3.2 的频率缩放（`rotary_embedding.py:69-87`）：**

```python
if is_llama3:
    wave_len = 2 * math.pi / inv_freq
    # 计算平滑因子
    smooth = (original_max_pos / wave_len - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth = torch.clamp(smooth, 0, 1)
    factor = (1 - smooth) / rope_factor + smooth
    inv_freq = factor * inv_freq
```

- **高频分量**（`wave_len` 短）：`smooth ≈ 1`，频率几乎不变 → 模型已见过足够多周期
- **低频分量**（`wave_len` 长）：`smooth ≈ 0`，频率除以 `rope_factor` → 压缩位置以适应训练范围

**理解 base 参数（Understanding Base Parameter）：**

1. **base 越大 → 频率越低：**
   - 对远距离位置具有更独特的编码
   - 局部平滑性更弱
   - 不太能很好地区分相邻位置

2. **base 越小 → 频率越高：**
   - 在远距离位置会出现周期性碰撞（重复）
   - 更适合短序列

3. **不同维度会在不同位置发生碰撞：**

	```
	Dim 0 (freq=1.0):   Good for positions 0-10 (then repeats) 
	Dim 2 (freq=0.1):   Good for positions 0-100 (then repeats) 
	Dim 4 (freq=0.01):  Good for positions 0-1000 (then repeats) 
	Dim 6 (freq=0.001): Good for positions 0-10000 (then repeats)
	```

**长上下文策略（当推理时上下文长度超过训练长度）：**

1. 直接使用 RoPE（可能会性能退化）
2. 修改 base：base 越大 = 频率越低 + 平滑性更好
3. 缩放位置：0, 1, 2, 3 → 0, 0.1, 0.2, 0.3
4. **YARN** ✅
   - 高频部分：模型在训练中见过很多周期 → 具备外推能力
   - 低频部分：模型从未见过完整周期 → 通过压缩位置让分布保持在训练范围内
5. **NTK** ✅
   - 针对更长上下文动态增大 base

---

## Step 2: 模型构建 ✅

具体实现：[qwen3.py](src/myvllm/models/qwen3.py)

组合所有层，构建完整的 Qwen 模型。

**源码架构（`qwen3.py`）：**

```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding          — 词表嵌入（TP 切分）
│   ├── Qwen3DecoderLayer × num_layers  — 解码器层堆叠
│   │   ├── LayerNorm (input)           — 输入归一化
│   │   ├── Qwen3Attention              — 自注意力
│   │   │   ├── QKVColumnParallelLinear — QKV 投影（TP 列切分）
│   │   │   ├── LayerNorm (q_norm/k_norm) — Q/K 归一化
│   │   │   ├── RotaryEmbedding         — 旋转位置编码
│   │   │   ├── Attention               — Flash/Paged Attention
│   │   │   └── RowParallelLinear       — 输出投影（TP 行切分 + all_reduce）
│   │   ├── LayerNorm (post_attn)       — 注意力后归一化
│   │   └── Qwen3MLP                    — 前馈网络
│   │       ├── MergedColumnParallelLinear (gate_up) — gate+up 合并投影
│   │       ├── SiluAndMul              — SwiGLU 激活
│   │       └── RowParallelLinear (down) — down 投影
│   └── LayerNorm (final)               — 最终归一化
└── ParallelLMHead                      — LM Head（TP 词表切分 + gather）
```

**`Qwen3DecoderLayer.forward` 完整流程（`qwen3.py:197-225`）：**

```python
def forward(self, x, residual=None):
    # 1. 输入归一化 + 残差连接
    if residual is not None:
        x, residual = self.input_layernorm(x, residual)
    else:
        residual = x
        x = self.input_layernorm(x)

    # 2. 计算位置编码（prefill 时按序列边界重置，decode 时用 context_lens - 1）
    context = get_context()
    if context.is_prefill and context.cu_seqlens_q is not None:
        positions = []
        for i in range(len(cu_seqlens) - 1):
            seq_len = cu_seqlens[i+1] - cu_seqlens[i]
            positions.extend(range(seq_len))
    else:
        positions = context.context_lens - 1

    # 3. 自注意力
    x = self.self_attn(x, positions=positions)

    # 4. 后注意力归一化 + 残差
    x, residual = self.post_attention_layernorm(x, residual)

    # 5. MLP
    x = self.mlp(x)
    return x, residual
```

**`Qwen3Model.forward`（`qwen3.py:273-279`）：**

```python
def forward(self, input_ids):
    x = self.embed_tokens(input_ids)    # (total_tokens, hidden_size)
    residual = None
    for layer in self.layers:
        x, residual = layer(x, residual)
    x, _ = self.norm(x, residual)       # 最终归一化 + 最后一次残差
    return x
```

**关键架构决策（Key Architecture Decisions）：**

**为什么在 Attention 中 `self.num_heads` 是按 GPU（per-GPU）来设置的？**
- 在注意力计算过程中不需要通信
- 每张 GPU 可以独立处理不同的 head
- 完整流程：
  1. 输入在所有 GPU 上复制（replicated）
  2. QKV 投影（ColumnParallel）按输出维度切分
  3. 通过 `.view()` 将本地的 Q、K、V 重新 reshape
  4. 在本地参数上运行 attention
  5. 在本地应用 RMS 和 rotary embedding
  6. 输出投影（RowParallel）使用 `dist.all_reduce` 求和聚合

**为什么 RMS 只作用在 Q 和 K 上？**
- Q 和 K 参与注意力权重（attention score / weight）的计算
- 去除会导致 softmax 不稳定的大数值
- V 不需要归一化（不会影响 score 的计算）

**为什么 gate_up 使用 MergedColumnParallelLinear？**
- 为了与模型 checkpoint 兼容！
- checkpoint 结构：
	```python
	checkpoint = {
	    'mlp.gate_proj.weight': torch.randn(intermediate_size, hidden_size),
	    'mlp.up_proj.weight': torch.randn(intermediate_size, hidden_size),
	    'mlp.down_proj.weight': torch.randn(hidden_size, intermediate_size),
	}
	```
- 不能直接用普通 ColumnParallel，把维度简单写成 `intermediate_size * 2`

**残差连接（Residual Connections）：**
- 始终在 attention 输出的 layernorm 之后加上 residual
- 始终在最后一层的 normalization 之后加上 residual

**验证正确运行！** ✅


---

## Step 3：序列管理

现在模型已经能跑起来了，接下来实现调度（scheduling）与内存管理（memory management）系统。

### 3.1 序列类（Sequence Class）

具体实现：[sequence.py](src/myvllm/engine/sequence.py)

**目的：** 存储一个序列的全部信息（prompt + 生成的 tokens）。

**源码解析（`sequence.py:14-89`）：**

```python
class Sequence:
    counter = count()  # 类级别的全局自增 ID 生成器

    def __init__(self, token_ids, block_size, sampling_params=SamplingParams()):
        self.block_size = block_size
        self.seq_id = next(Sequence.counter)       # 唯一序列 ID
        self.status = SequenceStatus.WAITING       # 初始状态
        self.token_ids = copy(token_ids)           # 必须 copy！
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0                 # 前缀缓存命中数
        self.block_table = []                      # 物理 block ID 列表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens

    # 关键属性
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
```

**关键实现细节：**

```python
# In __init__:
self.token_ids = copy(token_ids)  # MUST copy! Creates new list
```

**为什么要使用`copy()`？** 如果不使用 `copy()`，`self.token_ids` 会引用外部传入的 list，并且会受到外部修改的影响。使用 `copy()` 可以保证内部数据独立。

**序列状态跟踪：**
- `WAITING`：等待 prefill
- `RUNNING`：正在生成
- `FINISHED`：已完成（EOS / max_tokens / max_length）

**重要属性：**
- `token_ids`：所有 token（prompt + 生成）
- `num_tokens`：当前长度
- `block_table`：该序列的 KV cache 存储在哪些内存块中
- `status`：该序列在系统中的当前状态
- `num_cached_tokens`：前缀缓存命中的 token 数（用于跳过已缓存的 KV）

**跨进程序列化（`sequence.py:91-127`）：**

`__getstate__` / `__setstate__` 用于 TP 多卡时 rank 0 → worker 的共享内存 RPC 传输：
```python
def __getstate__(self):
    return (
        self.num_tokens,
        self.num_prompt_tokens,
        self.num_cached_tokens,
        self.block_table,
        # prefill 传完整 token_ids，decode 只传 last_token 以省带宽
        self.token_ids if self.num_completion_tokens == 0 else self.last_token,
    )
```


---

### 3.2 内存块类（Block Class）

具体实现：[block_manager.py](src/myvllm/engine/block_manager.py)


**目的：** 表示一个固定大小的内存块，用于存储 KV cache。

**源码解析（`block_manager.py:7-27`）：**

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id    # 物理块编号（在 cache 池中的位置）
        self.hash = -1              # 内容哈希（-1 表示未计算/部分块）
        self.ref_count = 0          # 引用计数
        self.token_ids = []         # 该块存储的 token 列表

    def update(self, h, token_ids):
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.ref_count = 1          # 分配即引用，从 1 开始
        self.token_ids = []
```

**关键概念：**

**引用计数（`ref_count`）：**
- 用于跟踪有多少个序列正在使用该 block
- 对 **前缀缓存** 至关重要 —— 当多个序列共享前缀时复用 KV cache
- 释放一个序列时，需要检查 `ref_count` 来决定该 block 是否应该被清空

**为什么要做哈希？**
- 目的：通过按内容查找 block 来启用 **前缀缓存**
- 不做哈希：无法知道 tokens `[1,2,3,...,256]` 是否已经被缓存
- 做哈希：`hash_value = compute_hash([1,2,3,...,256])` → `block_id = hash_to_block_id.get(hash_value)`
- 只有当 block 被填满（256 个 token 全部就位）时才计算 hash

**哈希计算源码（`block_manager.py:43-48`）：**
```python
def compute_hash(self, token_ids, prefix_hash_value):
    h = xxhash.xxh64()
    if prefix_hash_value != -1:
        h.update(prefix_hash_value.to_bytes(8, 'little'))  # 链式哈希
    h.update(np.array(token_ids, dtype=np.int32).tobytes())
    return h.intdigest()
```

**为什么哈希函数的参数要包含 prefix？**
- 即使当前 block 的 tokens 相同，也能在不同上下文中保持唯一性
- 例子：`[prefix_hash_1][1,2,3]` 与 `[prefix_hash_2][1,2,3]` 是不同的
- 链式哈希确保：`hash(block_i) = f(hash(block_{i-1}), tokens_i)`

**为什么在 reset() 里设置 `ref_count = 1`？**
- 当一个 block 被分配时（`_allocate_block` 会调用 `reset()`），它会立刻被某个序列使用
- 从 1（而不是 0）开始，反映了这种"立即被使用"的状态
- 如果从 0 开始，`deallocate()` 会把 `ref_count` 减到 -1，导致 block 永远不会被释放

**缓存未命中检测（`block_manager.py:91`）：**
```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    no_cache_found = True
```

**为什么要同时检查这两个条件？**
- `block_id == -1`：哈希表中未找到对应项
- `token_ids != ...`：避免哈希碰撞！不同的 tokens 可能会产生相同的哈希值

---

### 3.3 内存块管理器类（BlockManager Class）

具体实现：[block_manager.py](src/myvllm/engine/block_manager.py)

**目的：** 管理所有序列的 KV cache 显存分配/释放。

**源码架构（`block_manager.py:28-160`）：**

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]  # 所有物理块
        self.hash_to_block_id = {}                            # 哈希 → 块 ID（前缀缓存）
        self.free_block_ids = deque(range(num_blocks))        # 空闲块队列
        self.used_block_ids = set()                           # 已用块集合
```

**关键方法：**

**`allocate(seq)`（`block_manager.py:80-113`）—— 为序列分配 blocks（含前缀缓存）：**

```python
def allocate(self, seq):
    h = -1  # 初始前缀哈希为 -1
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 只有满块才计算哈希（部分块不缓存）
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)

        # 缓存未命中或哈希碰撞
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            block = self._allocate_block(self.free_block_ids[0])
            block.update(h=h, token_ids=token_ids)
            if h != -1:
                self.hash_to_block_id[h] = block.block_id
        else:
            # 缓存命中！复用已有块
            seq.num_cached_tokens += self.block_size
            block = self.blocks[block_id]
            block.ref_count += 1

        seq.block_table.append(block.block_id)
```

**`can_append(seq)`（`block_manager.py:128-136`）：**
```python
def can_append(self, seq):
    # 只有当新 token 是某块的第 1 个 token 时才需要新块
    # （num_tokens % block_size == 1 意味着上一块已满，需要新块）
    if seq.num_tokens % self.block_size == 1:
        return len(self.free_block_ids) > 0
    return True  # 当前块还有空间
```

**`append(seq)`（`block_manager.py:141-160`）：**
```python
def append(self, seq):
    last_block_id = seq.block_table[-1]

    if seq.num_tokens % self.block_size == 0:
        # 当前块刚满 → 计算哈希并注册到缓存
        h = self.compute_hash(seq.block(seq.num_blocks - 1), prefix_hash)
        block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
        self.hash_to_block_id[h] = block.block_id

    elif seq.num_tokens % self.block_size == 1:
        # 需要新块
        block = self._allocate_block(self.free_block_ids[0])
        seq.block_table.append(block.block_id)

    else:
        # 当前块还有空间，什么都不做
        pass
```

**`deallocate(seq)`（`block_manager.py:115-124`）：**
```python
def deallocate(self, seq):
    for block_id in seq.block_table:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)  # 归还空闲队列
    seq.block_table = []
    seq.num_cached_tokens = 0
```

**`_deallocate_block` 的缓存清理策略（`block_manager.py:59-73`）：**

释放时**清空 `token_ids`**，这使得该块在后续 `allocate` 中无法被匹配到（即使哈希表中仍有记录）。这是一个刻意的设计——当前版本的 prefill kernel 不读 `block_tables`，RoPE 位置也直接从 `cu_seqlens_q` 推导，所以跨序列的缓存复用会导致位置和注意力计算错误。清空 `token_ids` 是保持正确性的守卫。


---

## Step 4：模型运行器（Model Runner）✅

具体实现：[model_runner.py](src/myvllm/engine/model_runner.py)

**目的：** 作为序列与模型执行之间的桥梁。负责数据准备、CUDA Graph 优化以及采样。

**源码架构（`model_runner.py`）：**

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        # 1. 初始化分布式进程组
        dist.init_process_group('nccl', "tcp://localhost:12345", ...)
        # 2. 根据模型名创建模型实例
        match model_name:
            case 'Qwen3-0.6B': self.model = Qwen3ForCausalLM(...)
            case 'Llama-3.2-1B-Instruct': self.model = LlamaForCausalLM(...)
        # 3. 移到 GPU 并加载权重
        self.model = self.model.cuda(rank)
        load_weights_from_checkpoint(self.model, config['model_name_or_path'])
        # 4. warmup → 分配 KV cache → 捕获 CUDA graph
        self.warmup_model()
        self.allocate_kv_cache()
        self.capture_cudagraph()
```

### 4.1 权重加载

可以在CPU或GPU中加载权重，不同设备中进行模型的权重加载可能会导致权重出问题。具体可以查看 [Issues #36](https://github.com/Wenyueh/MinivLLM/issues/36)。

```python
# Load weights in GPU (model moved to GPU before loading weights)
self.model = self.model.cuda(rank)

# Load pretrained weights if model_name_or_path is provided
if config.get('model_name_or_path'):
    from myvllm.utils.loader import load_weights_from_checkpoint
    load_weights_from_checkpoint(self.model, config['model_name_or_path'])

# Load weights in CPU (move the model to GPU after loading weights)
# self.model = self.model.cuda(rank)
```

**权重加载器源码解析（`utils/loader.py:16-214`）：**

`load_weights_from_checkpoint` 的核心逻辑是处理 HF checkpoint 与自定义模型之间的名称映射：

```python
for hf_name, hf_weight in hf_weights.items():
    # 1. QKV 合并：q_proj + k_proj + v_proj → qkv_projection
    if '.self_attn.q_proj.weight' in hf_name:
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        param = model.get_parameter(f"model.layers.{i}.self_attn.qkv_projection.weight")
        param.data.copy_(qkv_weight)

    # 2. gate_up 合并：gate_proj + up_proj → gate_up
    elif '.mlp.gate_proj.weight' in hf_name:
        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
        param = model.get_parameter(f"model.layers.{i}.mlp.gate_up.weight")
        param.data.copy_(gate_up_weight)

    # 3. 其他参数直接按名称匹配
    else:
        param = model.get_parameter(hf_name)
        param.data.copy_(hf_weight)
```

**注意：** 此加载器直接 `copy_` 到 `param.data`，绕过了参数上的 `weight_loader` 方法。这意味着在 TP > 1 时，每个 rank 都会加载完整权重再各自 `copy_`——这对单卡没问题，但多卡场景需要走 `loader_vl.py` 中的 TP-aware 路径。

### 4.2 核心函数概览

```python
class ModelRunner:
    def __init__(self): pass
    
def read_shm(self): pass          # 从共享内存读取（worker 进程）
def write_shm(self): pass         # 写入共享内存（master 进程）

def warmup_model(self): pass      # 测量峰值显存占用
def allocate_kv_cache(self): pass # 分配 KV cache 显存

def prepare_prefill(self): pass   # 为 prefill 前向推理准备数据
def prepare_decode(self): pass    # 为 decode 前向推理准备数据  
def prepare_sample(self): pass    # 为采样准备温度（temperature）

def run_model(self): pass         # 执行模型（decode 阶段使用 CUDA graph）
def run(self): pass               # 主入口：prepare → run → sample

def capture_cudagraph(self): pass # 捕获 CUDA graphs 用于优化

```

---

### 4.3 共享内存通信

**源码解析（`model_runner.py:172-227`）：**

**`read_shm()`：**（Worker 进程从 master 进程读取）

```python
def read_shm(self):
    self.event.wait()                              # 阻塞等待信号
    n = int.from_bytes(self.shm.buf[:4], 'little') # 前 4 字节是数据长度
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
    self.event.clear()                             # 清除信号
    return method_name, args
```

**为什么长度用 4 字节？** 写入端无论 `n` 的值是多少，都固定用 4 字节来写：`n.to_bytes(4, "little")`。

**共享内存布局：**
```
┌──────────┬──────────────────────────────────────┐
│ 4 bytes  │ n bytes (pickle 数据)                 │
│ 长度 n   │ (method_name, arg1, arg2, ...)       │
└──────────┴──────────────────────────────────────┘
```

**同步机制（Synchronization）：**
- `self.event.wait()`：阻塞等待，直到 master 调用 `event.set()` 发出"消息已就绪"的信号
- `self.event.clear()`：清除信号，为下一条消息重置状态（回到"未就绪"）

**`write_shm()`：**（Master 进程写入给 workers）

```python
def write_shm(self, method_name, args):
    data = pickle.dumps((method_name, *args))      # 序列化
    n = len(data)
    self.shm.buf[:4] = n.to_bytes(4, 'little')    # 写长度
    self.shm.buf[4:n+4] = data                     # 写数据
    for event in self.event:                       # 通知所有 worker
        event.set()
```

**为什么使用循环?** 每个worker对应一个event - master 将信号分别发送给每个worker.

**`call()` 方法——master 和 worker 共用（`model_runner.py:221-227`）：**

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, args)          # master 写共享内存
    method = getattr(self, method_name, None)
    if method:
        return method(*args)                       # 所有 rank 都执行同一方法
```

**Worker 主循环（`model_runner.py:209-216`）：**

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()        # 等待并读取
        self.call(method_name, *args)              # 执行
        if method_name == 'exit':
            self.exit()
            break
```

**关于 `self.event` vs `self.events` 的说明：**
- Master（rank 0）：`self.event = [Event(), Event(), ...]`（列表，每个 worker 一个）
- Worker（rank ≠ 0）：`self.event = Event()`（单个，只监听自己的信号）

---

### 4.4 内存管理

**`warmup_model()`（`model_runner.py:233-241`）：**

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_tokens = self.config['max_num_batch_tokens']
    max_model_length = self.config['max_model_length']
    batch_size = max_tokens // max_model_length
    # 构造最大 batch 的虚拟序列
    seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.block_size)
            for _ in range(batch_size)]
    self.run(seqs, is_prefill=True)       # 跑一遍 prefill
    torch.cuda.empty_cache()
```

**为什么在处理请求前先 warmup？**
- 用于测量显存：跑一遍最大 batch 来估计峰值显存占用
- 测量的是模型显存（权重 + 激活），**不包含** KV cache
- 使用 `torch.cuda.memory_stats()['allocated_bytes.all.peak']`
- 结果会在 `allocate_kv_cache()` 中用于计算可用显存

**`allocate_kv_cache()`（`model_runner.py:244-300`）：**

```python
def allocate_kv_cache(self):
    free_mem, total_mem = torch.cuda.mem_get_info()
    total_free_mem = free_mem * gpu_memory_utilization
    peak_mem = torch.cuda.memory_stats()['allocated_bytes.all.peak']
    current_mem = torch.cuda.memory_stats()['allocated_bytes.all.current']
    available_mem = total_free_mem - (peak_mem - current_mem)

    # 每块所需字节数
    block_bytes = block_size * 2 * num_layers * num_kv_heads * head_dim * itemsize
    num_blocks = int(available_mem // block_bytes)

    # 跨 rank 同步：取所有 rank 的最小值
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)

    # 一次性分配整个 KV cache 池
    allocated_kv_cache = torch.zeros(
        2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
        device=f'cuda:{rank}')

    # 将 cache 切片注入每个 Attention 层
    for module in self.model.modules():
        if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
            module.k_cache = allocated_kv_cache[0, layer_id]
            module.v_cache = allocated_kv_cache[1, layer_id]
            layer_id += 1
```

**目的：** 基于 block_size，确定能够分配多少个 KV cache block。

**KV cache 内存布局：**
```
shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        │    │           │            │            │            │
        │    │           │            │            │            └─ 每个 head 的维度
        │    │           │            │            └─ 每 GPU 的 KV head 数
        │    │           │            └─ 每块存 block_size 个 token
        │    │           └─ 总块数（所有序列共享的扁平池）
        │    └─ 模型层数
        └─ 0=K, 1=V
```

**关键设计：**
- 为峰值占用预留显存（即使并非全部在用）
- 预留的是**模型级别**的显存，而不是每个序列各自预留
- 使用 `slot_mapping` 跟踪"哪个序列的哪个 token"写到哪个位置
- 这是实现 **PagedAttention** 的关键
- 跨 rank 同步取 MIN：确保所有 GPU 使用相同的 block 数量，避免某个 rank OOM

---

### 4.5 数据准备

**`prepare_prefill(seqs)`（`model_runner.py:312-361`）：**

**目的：** 为 prefill 前向计算准备数据，并支持前缀缓存（prefix caching）。

**源码解析：**

```python
def prepare_prefill(self, seqs):
    input_ids = []
    slot_mappings = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]

    for seq in seqs:
        num_cached = seq.num_cached_tokens
        # 只取未缓存的 token（前缀缓存命中的部分跳过）
        input_ids.extend(seq.token_ids[num_cached:])
        # Q 长度 = 未缓存 token 数
        seqlens_q.append(len(seq) - num_cached)
        # K 长度 = 全部 token 数（attention 需要看所有历史）
        seqlens_k.append(len(seq))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])

        # slot_mapping：只为未缓存的 blocks 生成写入位置
        for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
            if seq.num_cached_blocks + i != seq.num_blocks - 1:
                slot_mappings.extend(range(block_id * block_size, (block_id+1) * block_size))
            else:
                slot_mappings.extend(range(block_id * block_size,
                                           block_id * block_size + seq.last_block_num_tokens))

    # 设置 Context 单例
    set_context(is_prefill=True, cu_seqlens_q=..., slot_mapping=..., ...)
    return input_ids
```

**输出：**
- `input_ids`：所有序列的未缓存 tokens 合并成一个 1D tensor
- `cu_seqlens_q/k`：累计序列长度（用于标记边界）
- `slot_mapping`：新 KV 应写入的位置
- `block_tables`：KV 应从哪里读取（仅在有前缀缓存时需要）

**为什么把 input_ids 展平成一个 list？**
- FlashAttention 的要求：单次 kernel launch
- `cu_seqlens_q` 用于标记边界：`[0, 3, 5, 9]`
  ```
  │ │ │ │
  │ │ │ └─ end of seq3 (position 9)
  │ │ └──── end of seq2 (position 5)
  │ └─────── end of seq1 (position 3)
  └────────── start (position 0)
  ```

**为什么 `cu_seqlens_q` 和 `cu_seqlens_k` 可能不同？**
- 当有前缀缓存时，`seqlens_q` = 未缓存 token 数，`seqlens_k` = 全部 token 数
- 例如：序列有 512 个 token，前 256 个命中缓存 → `seqlens_q=256, seqlens_k=512`
- 当前版本的 flash attention kernel 只用 `cu_seqlens_q`，所以跨序列缓存复用尚未完全启用

**为什么 `pin_memory=True`?**
- **Pinned memory** = 物理内存页锁定（不能被 swap 到磁盘）
- 支持通过 DMA（Direct Memory Access）直接进行 CPU→GPU 传输
- 更快:
  ```
  普通情况:    pageable → pinned buffer → GPU (2次拷贝)
  Pinned:    pinned → GPU (1次拷贝, DMA)
  ```

**为什么 `non_blocking=True`?**
- 控制 CPU 是否等待拷贝完成
- `non_blocking=False`: CPU 阻塞直到 GPU 拿到数据
- `non_blocking=True`: CPU 立即继续（异步传输）
- 支持并行拷贝！

**为什么 `slot_mapping` 只包含未缓存的 blocks？**
- 只为**新token** 写入 KV，不重复写已缓存的 KV
- 已缓存的 KV 已经存在于显存中

---

**`prepare_decode(seqs)`（`model_runner.py:365-390`）：**

**目的:** 为解码阶段准备数据（每个序列一个 token）。

**源码解析：**

```python
def prepare_decode(self, seqs):
    input_ids = []
    context_lens = []
    slot_mappings = []
    block_tables = []

    for seq in seqs:
        input_ids.append(seq.last_token)           # 只取最后一个 token
        context_lens.append(len(seq))              # 序列总长度
        # 新 token 的 slot = 最后一块的起始 + 块内已有 token 数 - 1
        slot_mappings.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

    # block_tables 需要 padding 到相同长度
    max_num_blocks = max(len(seq.block_table) for seq in seqs)
    for seq in seqs:
        block_table = seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
        block_tables.append(block_table)

    set_context(is_prefill=False, slot_mapping=..., context_lens=..., block_tables=...)
    return input_ids
```

**新的 slot 映射:**
```python
new_slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
```

**为什么不用担心 slot 重叠？**
- BlockManager 的 `append()` 保证不会重叠:
  ```python
  # Seq has 256 tokens (block full)
  seq.num_tokens = 256
  256 % 256 = 0  → Block full, finalize it
  
  # Next token appended → num_tokens = 257
  257 % 256 = 1  → Need new block!
  block = self._allocate_block(self.free_block_ids[0])
  seq.block_table.append(block.block_id)
  ```

---

**`prepare_sample(seqs)`（`model_runner.py:393-394`）：**

```python
def prepare_sample(self, seqs):
    return torch.tensor([seq.temperature for seq in seqs],
                        dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
```

**目的:** 准备温度（temperature）数值，用于采样时的 logits 缩放。

---

### 4.6 模型执行

**`run_model()`（`model_runner.py:400-425`）：**

**用于 Prefill：** 直接计算前向传播。

**用于 Decode：** 使用 CUDA Graph 来提升速度！

```python
@torch.inference_mode()
def run_model(self, input_ids, is_prefill):
    if is_prefill or self.enforce_eager:
        hidden_states = self.model(input_ids)
        logits = self.model.compute_logits(hidden_states)
    else:
        bs = input_ids.size(0)
        # 找到能容纳当前 batch 的最小已捕获 graph
        graph = self.graphs[next(bs_ for bs_ in self.graphs if bs_ >= bs)]
        vars = self.graph_vars

        # 将实际数据拷贝进 graph 的预分配 buffer
        vars['input_ids'][:bs].copy_(input_ids)
        vars['slot_mapping'][:bs].fill_(-1)          # 哨兵值
        vars['slot_mapping'][:bs].copy_(context.slot_mapping)
        vars['context_lens'].zero_()
        vars['context_lens'][:bs].copy_(context.context_lens)
        vars['block_tables'][:bs, :n] = context.block_tables

        graph.replay()                               # 回放 CUDA graph
        logits = self.model.compute_logits(vars['outputs'][:bs])
    return logits
```

**为什么要找到能容纳的最小图？**
- 并不是每个 batch size 都一定有已捕获的图
- 通过 padding 复用更大的图
- 例如：捕获了 `[1, 2, 4, 8, 16]`，当前 bs=3 → 使用 bs=4 的图

**为什么要用哨兵值填充 `slot_mapping` 和 `context_lens`？**
- 使用的图比实际需求更大 → 用虚拟值填充未使用的槽位
- `slot_mapping` 填 `-1`：`store_kvcache_kernel` 遇到 `-1` 会直接 `return`，跳过写入
- `context_lens` 填 `0`：`paged_attention_decode_kernel` 中 `token_start < context_len` 为 False，跳过计算

---

**`run()`（`model_runner.py:433-447`）：**

```python
def run(self, seqs, is_prefill):
    if is_prefill:
        input_ids = self.prepare_prefill(seqs)
    else:
        input_ids = self.prepare_decode(seqs)

    logits = self.run_model(input_ids, is_prefill)

    token_ids = None
    if self.rank == 0:
        token_ids = self.sampler(logits, self.prepare_sample(seqs))

    reset_context()                                  # 清理 Context 单例
    return token_ids
```

**主入口：**
1. 组合 `prepare_prefill/decode` + `run_model` + `sample`
2. 调用 `reset_context()` 清除缓存数据

**为什么只有 rank 0 进行采样？**
- `ParallelLMHead` 使用 `dist.gather` 将 logits 汇总到 rank 0
- 其他 rank 的 `compute_logits` 返回的是局部 logits（只有自己的词表分片）
- 只需要 **采样一次** 即可得到 token ID
- 避免重复采样或采样结果不一致

---

### 4.7 CUDA Graph 优化

**`capture_cudagraph()`（`model_runner.py:572-625`）：**

**目的：** 记录 CUDA kernel 的执行序列以便快速回放（消除 kernel 启动开销）。

**为什么只用于 decoding？**
- Decode 的输入模式固定（每个序列 1 个 token，batch_size 是唯一变量）
- Prefill 的输入长度可变，无法预捕获

**源码解析：**

```python
@torch.inference_mode()
def capture_cudagraph(self):
    max_bs = self.config['max_num_seqs']
    max_num_blocks = math.ceil(max_len / self.block_size)

    # 1. 预分配最大尺寸的 buffer（所有 graph 共享同一组内存）
    input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{rank}')
    slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{rank}')
    context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{rank}')
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{rank}')
    outputs = torch.zeros(max_bs, vocab_size, device=f'cuda:{rank}')

    # 2. 按 batch size 从大到小捕获
    batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    graph_pool = None

    for batch_size in reversed(batch_sizes):        # 从最大 batch 开始
        graph = torch.cuda.CUDAGraph()
        set_context(slot_mapping=slot_mapping[:batch_size], ...)

        # warmup：触发惰性内存分配
        outputs[:batch_size] = self.model(input_ids[:batch_size])

        # 正式捕获
        with torch.cuda.graph(graph, graph_pool):
            outputs[:batch_size] = self.model(input_ids[:batch_size])
            if graph_pool is None:
                graph_pool = graph.pool()           # 共享内存池

        self.graphs[batch_size] = graph
        torch.cuda.synchronize()                    # 确保捕获完成
        reset_context()
```

**捕获策略：**
- 在最大尺寸上预分配 buffer
- 针对常见 batch size 进行捕获：`[1, 2, 4, 8] + list(range(16, max_bs + 1, 16))`
- 先捕获最大 batch（内存池按最大场景进行尺寸规划）

**为什么从大到小捕获？** 第一个 graph 创建 `graph_pool`，后续 graph 共享同一内存池。从最大 batch 开始确保池的尺寸足够大，后续较小的 graph 可以复用已分配的内存。

**Graph 回放（`model_runner.py:407-423`）：**

```python
def run_model(self, input_ids, is_prefill):
    if is_prefill or self.enforce_eager:
        hidden_states = self.model(input_ids)
        logits = self.model.compute_logits(hidden_states)
    else:
        bs = input_ids.size(0)
        # 找到能容纳当前 batch 的最小已捕获 graph
        graph = self.graphs[next(bs_ for bs_ in self.graphs if bs_ >= bs)]

        # 将实际数据拷贝进 graph 的输入 buffer
        vars['input_ids'][:bs].copy_(input_ids)
        vars['slot_mapping'][:bs].fill_(-1)          # 哨兵值：跳过 padding 位置
        vars['slot_mapping'][:bs].copy_(context.slot_mapping)
        vars['context_lens'][:bs].copy_(context.context_lens)
        vars['block_tables'][:bs, :n] = context.block_tables

        graph.replay()                               # 回放捕获的 kernel 序列
        logits = self.model.compute_logits(vars['outputs'][:bs])
```

**为什么在 capture 前要 warmup？**
- CUDA graph 要求在 capture 之前完成所有内存分配
- Warmup 会触发惰性分配 → 确保 capture 期间内存分配稳定

**为什么在 `reset_context()` 前要 `torch.cuda.synchronize()`？**
- 确保当前 capture 完成后，再为下一次 capture 重置状态

**`@torch.inference_mode()`:**
- 禁用梯度跟踪的装饰器
- 优化推理性能

---

### 4.8 辅助方法

**`loop()`:**
- worker 进程的主循环
- 等待事件并调用被请求的方法

**`call()`:**
- 同时被 master 和 workers 调用
- master 写入共享内存
- workers 从共享内存读取

---

### 4.9 关系：torch.compile vs CUDA Graph

**torch.compile：**
- 将多个操作融合成一个 kernel
- 节省 kernel 执行时间
- 在本项目中的使用位置：
  - `SiluAndMul.forward`（`activation.py:15`）
  - `LayerNorm.rms_forward`（`layernorm.py:16`）
  - `RotaryEmbedding.forward`（`rotary_embedding.py:100`）
  - `SamplerLayer.forward`（`sampler.py:14`）
- 示例：
  ```python
  @torch.compile
  def rms_forward(self, x):
      variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps  # ┐
      sqrt_variance = variance.sqrt()                             # ├─ Fused
      x_norm = (x / sqrt_variance * self.weight)                 # ┘
      return x_norm
  ```

**CUDA Graph：**
- 记录 kernel 执行序列以便回放
- 节省 kernel 启动开销（无需 CPU 参与）
- 在本项目中的使用位置：`ModelRunner.capture_cudagraph`（`model_runner.py:572`）
- 仅用于 decode 阶段（输入模式固定）

**对比：**

| 维度 | torch.compile | CUDA Graph |
|------|--------------|------------|
| 优化目标 | kernel 融合 | kernel 启动开销 |
| 适用场景 | 小算子（activation, norm） | 固定形状的完整 forward |
| 编译时机 | 首次调用时 JIT 编译 | 初始化时预捕获 |
| 输入约束 | 形状可变 | 形状必须与捕获时一致 |

**组合使用：** `torch.compile` 减少 kernel 数量，CUDA graph 消除启动开销。在 decode 阶段，模型 forward 内部的 `rms_forward`、`rotary_emb` 等已经被 `torch.compile` 融合过，整个 forward 又被 CUDA graph 捕获，两者叠加效果。



---

## Step 5：调度器（Scheduler） ✅

具体实现：[scheduler.py](src/myvllm/engine/scheduler.py)

**目的：** 决定每次迭代运行哪些序列，并管理 waiting/running 队列。

### 5.1 核心设计

**源码架构（`scheduler.py:6-17`）：**

```python
class Scheduler:
    def __init__(self, max_num_sequences, max_num_batched_tokens,
                 max_cached_blocks, block_size, eos, ...):
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.waiting: deque[Sequence] = deque()   # 等待 prefill 的新序列
        self.running: deque[Sequence] = deque()   # 正在 decode 的序列
        self.eos = eos
```

**两类队列：**
1. **Waiting 队列**：尚未开始的新序列
2. **Running 队列**：正在运行的序列

---

### 5.2 调度逻辑

**优先级：Prefill > Decode**

调度器 **总是先尝试 prefill**，即使 running 队列不为空！

**源码解析（`scheduler.py:51-112`）：**

```python
def schedule(self):
    scheduled_sequences = []
    current_scheduled_tokens = 0
    preempted = False

    # ===== 阶段 1：尝试 prefill =====
    while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
        seq = self.waiting[0]
        if self.block_manager.can_allocate(seq) and \
           len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)          # 分配 blocks（含前缀缓存）
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_sequences.append(seq)
            current_scheduled_tokens += len(seq)
        else:
            break

    if scheduled_sequences:
        return scheduled_sequences, True              # is_prefill=True

    # ===== 阶段 2：decode =====
    while self.running:
        seq = self.running.popleft()
        if not self.block_manager.can_append(seq):
            # 显存不足 → 抢占
            preempted = True
            if self.running:
                self.running.appendleft(seq)
                self.preempt(self.running.pop())      # 抢占队尾序列
            else:
                self.preempt(seq)
                break
        else:
            self.block_manager.append(seq)
            scheduled_sequences.append(seq)
            current_scheduled_tokens += 1             # decode 每序列只加 1 token

    # 将调度好的序列按原顺序放回 running 队列
    if scheduled_sequences:
        self.running.extendleft(reversed(scheduled_sequences))

    return scheduled_sequences, False                 # is_prefill=False
```

**调度流程：**
1. **尝试加入 prefill 序列：**
   - 检查 waiting 队列里的新序列能否放得下（block 容量 + token 预算）
   - 没有空间继续 prefill 时停止

2. **如果没有新增 prefill，则调度 decode：**
   - 继续运行现有的 running 序列
   - 若没有空间容纳更多，则 **抢占** 优先级最低的序列

**无进度守卫（`scheduler.py:99-110`）：**

```python
elif not preempted and (self.waiting or self.running):
    raise RuntimeError(
        "Scheduler made no progress: ..."
    )
```

如果既没有调度任何序列，也没有发生抢占，说明引擎陷入了死循环。此时主动抛出异常，而不是让 `LLMEngine.generate()` 无限空转。

---

### 5.3 抢占（Preemption）

**源码解析（`scheduler.py:144-150`）：**

```python
def preempt(self, seq):
    self.block_manager.deallocate(seq)       # 释放所有 KV cache blocks
    seq.status = SequenceStatus.WAITING      # 状态回退为 WAITING
    self.waiting.appendleft(seq)             # 重新入队等待 prefill
```

当显存不足以容纳所有 running 序列时，调度器会抢占队尾的序列：释放其 KV cache，将其放回 waiting 队列。下次被调度时会重新 prefill（可能命中前缀缓存，减少重复计算）。

---

### 5.4 后处理

**源码解析（`scheduler.py:155-179`）：**

```python
def postprocess(self, seqs, outputs):
    for seq, token_id in zip(seqs, outputs):
        seq.append_token(token_id)

        # 三种停止条件
        stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
        stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
        stop_due_to_max_length = seq.max_model_length is not None and \
                                 seq.num_tokens >= seq.max_model_length

        if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)   # 释放 KV cache
            self.running.remove(seq)             # 移出 running 队列
```

**生成之后：**
- 检查序列是否结束（EOS token / max_tokens / max_model_length）
- 若结束：通过 BlockManager 释放 block
- 将已完成序列从 running 队列移出

---

## Step 6: LLM Engine ✅

具体实现：[llm_engine.py](src/myvllm/engine/llm_engine.py)

**目的：** 顶层 API，用于编排 scheduler、model runner 和请求处理。

**源码架构（`llm_engine.py:25-63`）：**

```python
class LLMEngine:
    def __init__(self, config):
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")

        # 1. 启动 worker 进程（rank 1, 2, ...）
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            process.start()

        # 2. 初始化 rank 0 的 ModelRunner
        self.model_runner = ModelRunner(config, rank=0, event=self.events)

        # 3. 初始化 Scheduler（必须在 ModelRunner 之后！）
        self.scheduler = Scheduler(...)

        atexit.register(self.exit)
```

**多进程架构示意：**
```
┌─────────────────────────────────────────────────┐
│  LLMEngine (rank 0, 主进程)                      │
│  ├── ModelRunner (rank 0)                        │
│  │   ├── model (GPU 0)                           │
│  │   ├── KV cache (GPU 0)                        │
│  │   └── CUDA graphs                             │
│  ├── Scheduler                                   │
│  │   ├── waiting queue                           │
│  │   └── running queue                           │
│  └── SharedMemory ──write──→ Event.set()          │
└─────────────────────────────────────────────────┘
         │ SharedMemory + Event
         ▼
┌─────────────────────────────────────────────────┐
│  Worker Process (rank 1)                         │
│  └── ModelRunner (rank 1)                        │
│      ├── model (GPU 1)                           │
│      ├── KV cache (GPU 1)                        │
│      └── CUDA graphs                             │
│  loop(): Event.wait() → read_shm() → call()      │
└─────────────────────────────────────────────────┘
```

### 6.1 核心方法

**`add_prompt(prompt_str)`（`llm_engine.py:98-99`）：**
```python
def add_prompt(self, prompt, sampling_params):
    self.scheduler.add_sequence(
        Sequence(token_ids=self.tokenizer.encode(prompt),
                 block_size=self.config['block_size'],
                 sampling_params=sampling_params)
    )
```

**`step()`（`llm_engine.py:78-94`）：**
```python
def step(self):
    # 1. 调度
    scheduled_sequences, is_prefill = self.scheduler.schedule()
    if not scheduled_sequences:
        return [], 0, is_prefill

    # 2. 执行（rank 0 写共享内存 → 所有 rank 同时执行 model forward）
    outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)

    # 3. 后处理（追加 token、检查停止条件、释放 block）
    self.scheduler.postprocess(scheduled_sequences, outputs)

    # 4. 收集已完成的序列
    outputs = [(seq.seq_id, seq.completion_token_ids)
               for seq in scheduled_sequences if seq.is_finished]
    return outputs, num_processed_tokens, is_prefill
```

**`generate(prompts)`（`llm_engine.py:105-122`）：**
```python
def generate(self, prompts, sampling_params):
    for prompt in prompts:
        self.add_prompt(prompt, sampling_params)

    generated_tokens = {}
    while not self.scheduler.is_finished():
        outputs, num_processed_tokens, is_prefill = self.step()
        generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

    # 按 seq_id 排序，恢复输入顺序
    generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(...)]
    return {'text': [self.tokenizer.decode(t) for t in generated_tokens],
            'token_ids': generated_tokens}
```

**推理主循环流程：**
```
generate()
  ├── add_prompt() × N          # 所有 prompt 入 waiting 队列
  └── while not is_finished():
        └── step()
              ├── scheduler.schedule()     # 决定本步跑哪些序列
              ├── model_runner.call("run") # 所有 rank 执行 forward
              └── scheduler.postprocess()  # 追加 token / 检查停止 / 释放 block
```

---

### 6.2 初始化顺序

**为什么 Scheduler 要在 ModelRunner 之后初始化？**

当 `world_size > 1` 时，`ModelRunner.__init__` 会调用 `dist.init_process_group('nccl', ...)`，这是一个**集合屏障（collective barrier）**——rank-0 会阻塞，直到所有 worker 进程也完成该调用后才继续执行。只有在所有 rank 都完成汇合后，`ModelRunner.__init__` 才会返回。Scheduler 在此之后创建，确保分布式环境完全就绪后引擎才进入可用状态。

**初始化时序图（`llm_engine.py:26-63` + `model_runner.py:16-168`）：**

```
Rank 0 (主进程)                    Rank 1 (Worker)
───────────────                    ───────────────
spawn worker process ──────────→   worker_process() 启动
                                     │
ModelRunner.__init__()               ModelRunner.__init__()
  dist.init_process_group() ────────── dist.init_process_group()
  [barrier: 双方在此汇合]              [barrier: 双方在此汇合]
  model.cuda(0)                      model.cuda(1)
  load_weights()                     load_weights()
  warmup_model()                     warmup_model()
  allocate_kv_cache()                allocate_kv_cache()
  all_reduce(MIN, max_blocks) ──────── all_reduce(MIN, max_blocks)
  capture_cudagraph()                capture_cudagraph()
  dist.barrier() ──────────────────── dist.barrier()
  创建 SharedMemory                  等待 SharedMemory
                                     │
Scheduler.__init__()                 loop() 开始等待 Event
```

当 `world_size == 1` 时，不会启动任何 worker 进程，也不存在屏障，因此此时初始化顺序没有实际影响。

---

### 6.3 清理

**源码解析（`llm_engine.py:68-72`）：**

```python
def exit(self):
    self.model_runner.call("exit")     # 通知所有 rank 执行 exit
    del self.model_runner
    for process in self.processes:
        process.join()                 # 等待 worker 进程退出
```

**ModelRunner.exit（`model_runner.py:193-204`）：**
```python
def exit(self):
    if self.world_size > 1:
        self.shm.close()
        if self.rank == 0:
            self.shm.unlink()          # 删除共享内存
    if not self.enforce_eager:
        del self.graphs                # 释放 CUDA graphs
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.destroy_process_group()   # 销毁进程组
```

**为什么要 `exit()` 以及 `atexit.register(self.exit)`？**

```python
atexit.register(self.exit)
```

**目的：** 当 Python 程序停止时，自动：
1. 调用 `engine.exit()` 清理资源
2. 等待 worker 进程优雅退出
3. 防止出现僵尸进程或状态损坏

---

### 6.4 采样层（Sampler）

具体实现：[sampler.py](src/myvllm/layers/sampler.py)

**源码解析（`sampler.py:5-19`）：**

```python
class SamplerLayer(nn.Module):
    @torch.compile
    def forward(self, logits, temperature):
        logits /= temperature.unsqueeze(-1)           # 温度缩放
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max 采样：等价于 multinomial 但更高效
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

**Gumbel-max 技巧：** `probs / Exponential(1)` 再取 `argmax`，数学上等价于按 `probs` 分布做多项式采样，但避免了显式调用 `torch.multinomial`（后者在某些 CUDA 版本上有性能问题）。

**为什么只有 rank 0 采样？** 在 `model_runner.py:444` 中：
```python
if self.rank == 0:
    token_ids = self.sampler(logits, self.prepare_sample(seqs))
```
因为 `ParallelLMHead` 使用 `dist.gather` 将 logits 汇总到 rank 0，其他 rank 没有完整 logits，也不需要采样。

---

## 总结：学习顺序

1. **层组件**（activation → layernorm → linear → vocab/lmhead → attention → rotary）
2. **模型**（组装各层，并验证可运行）
3. **序列管理**（Sequence → Block → BlockManager）
4. **Model Runner**（数据准备、CUDA graphs、采样）
5. **调度器**（队列管理、prefill/decode 调度）
6. **LLM Engine**（顶层编排）

每一步都建立在前一步之上，逐步构建一个完整的推理系统，并加入诸如 PagedAttention、CUDA graphs 与 prefix caching 等高级优化。

---

## 附录：完整数据流

**一次 `step()` 的完整数据流（以 decode 为例）：**

```
LLMEngine.step()
  │
  ├── Scheduler.schedule()
  │     ├── 从 running 队列取出序列
  │     ├── BlockManager.can_append() → 检查显存
  │     └── BlockManager.append() → 分配新 block（如需要）
  │
  ├── ModelRunner.call("run", seqs, is_prefill=False)
  │     │
  │     ├── [rank 0] write_shm() → 通知 worker
  │     │
  │     ├── [所有 rank] prepare_decode()
  │     │     ├── input_ids = [last_token_0, last_token_1, ...]
  │     │     ├── slot_mapping = [block*bs+offset, ...]
  │     │     ├── context_lens = [len(seq_0), len(seq_1), ...]
  │     │     ├── block_tables = [[b0, b1, ...], [-1, -1, ...], ...]
  │     │     └── set_context(is_prefill=False, ...)
  │     │
  │     ├── [所有 rank] run_model()
  │     │     ├── 拷贝数据到 graph buffer
  │     │     ├── graph.replay()
  │     │     │     ├── embed_tokens(input_ids)
  │     │     │     ├── for layer in layers:
  │     │     │     │     ├── LayerNorm(x, residual)
  │     │     │     │     ├── QKVColumnParallel(x) → q, k, v
  │     │     │     │     ├── q_norm(q), k_norm(k)
  │     │     │     │     ├── RotaryEmbedding(positions, q, k)
  │     │     │     │     ├── Attention(q, k, v)
  │     │     │     │     │     ├── store_kvcache(k, v, slot_mapping)
  │     │     │     │     │     └── paged_attention_decode(q, k_cache, v_cache, block_tables)
  │     │     │     │     ├── RowParallel(o) → all_reduce
  │     │     │     │     ├── LayerNorm(x, residual)
  │     │     │     │     └── MLP(x) → gate_up → SiluAndMul → down_proj → all_reduce
  │     │     │     └── compute_logits(hidden_states)
  │     │     │           └── ParallelLMHead → gather to rank 0
  │     │     └── return logits
  │     │
  │     ├── [rank 0] SamplerLayer(logits, temperature)
  │     │     └── Gumbel-max → token_ids
  │     │
  │     └── reset_context()
  │
  └── Scheduler.postprocess(seqs, token_ids)
        ├── seq.append_token(token_id)
        ├── 检查停止条件（EOS / max_tokens / max_length）
        └── 若完成：BlockManager.deallocate(seq)
```

## 课程练习

感兴趣的读者可以在本地尝试向 MinivLLM 添加 `meta-llama/Llama-3.2-1B-Instruct` 作为练习。

`meta-llama/Llama-3.2-1B-Instruct`（以下简称 Llama3.2） 和 `Qwen/Qwen3-0.6B` 有着相似的结构，模型组件上仅有 Rotary Embedding 的实现略有不同，在保持字段名相同的前提下，现有的权重加载代码 `loader.py` 不需要修改就能直接用在 Llama3.2 上。

参考资料：
- Llama3.2 的实现可以参考 [mini-sglang 中的 Llama3.2](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/models/llama.py)
- Rotary Embedding 实现的不同可以在 [mini-sglang 中的 Rotary Embedding](https://github.com/sgl-project/mini-sglang/blob/dae78f6bb97d5c5aaadbc0772fc964d48a8ee726/python/minisgl/layers/rotary.py#L72-L86) 中找到。
- 各种模型参数可以在 [Hugging Face 中的 Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) 的 `config.json` 文件中找到。

为了完成练习，可以先把仓库克隆到本地，然后删除仓库中的 Llama3.2 实现：`rm src/myvllm/models/llama.py`，再自己创建一个 `src/myvllm/models/llama.py` 文件，通过参考链接中的 Llama3.2，自己基于 MinivLLM 实现 Llama3.2。

添加 Llama3.2 只涉及以下文件的修改：
- `src/myvllm/models/llama.py`: 模型实现。需要你动手实现
- `src/myvllm/layers/rotary_embedding.py`: 需要添加 Llama3.2 的不同实现。
- `src/myvllm/engine/model_runner.py`: ModelRunner 需要能够调用实现的 Llama3.2。
- `main_llama32.py`: 负责测试 Llama3.2 的实现效果。

运行 `main_llama32.py`，效果如下：

![llama32-effect](assets/llama32-effect.png)

由于后三个文件 `rotary_embedding.py`、`model_runner.py`、`main_llama32.py` 中要修改的地方不多，MinivLLM 已经实现好了，你所要做的就只是删除 `src/myvllm/models/llama.py` 文件，然后反复对照 [mini-sglang 中的 Llama3.2](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/models/llama.py) 和 `src/myvllm/models/qwen3.py`，在 `src/myvllm/models/llama.py` 中实现你自己的 Llama3.2。实现好后，运行 `uv run main_llama32.py` 进行测试。如果实现无误，你应该可以看到和上面相似的效果。如果实在不会，请及时参考仓库中的原始 `src/myvllm/models/llama.py`。
