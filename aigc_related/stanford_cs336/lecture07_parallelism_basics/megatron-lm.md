# Megatron-LM 张量并行：MLP 与 Attention 的实现原理

Megatron-LM 是 NVIDIA 提出的大模型训练框架，其核心贡献之一就是将张量并行（Tensor Parallelism）优雅地应用到了 Transformer 的 MLP 和 Attention 模块中。它的设计哲学极其统一：通过"列并行 + 行并行"的组合，将每个模块的通信次数压缩到最低，整个过程只在模块末尾进行一次 All-Reduce 操作。

## 目录

- [基本假设](#基本假设)
- [MLP 的张量并行](#mlp-的张量并行)
- [Attention 的张量并行](#attention-的张量并行)
- [Attention 与 MLP 的对称美](#attention-与-mlp-的对称美)

## 基本假设

在深入分析之前，先统一符号和假设：

- 输入张量 X 的形状为：\[B, L, D\]（Batch size, Sequence length, Hidden dimension）
- MLP 第一层权重 A（通常是把维度从 D 扩增到 4D）：形状为 \[D, 4D\]
- MLP 第二层权重 B（把维度从 4D 还原回 D）：形状为 \[4D, D\]
- 并行 GPU 数量：N 台（为了方便计算，假设 N = 2）

## MLP 的张量并行

MLP 模块的张量并行分为两个阶段，巧妙地利用了矩阵分块乘法的代数性质，将整个过程的通信压缩到最后一步。

### 第一阶段：列并行（Column Parallelism）—— 处理权重 A

在这一步，我们的目标是将巨大的 4D 维度拆分到不同的 GPU 上。

拆分权重 A：我们将权重 A 按列切成两半。

- GPU 1 持有 A1，形状为 \[D, 2D\]
- GPU 2 持有 A2，形状为 \[D, 2D\]

本地计算：

- 每个 GPU 都会拿到一份完全相同的输入 X（\[B, L, D\]）。
- GPU 1 计算：Y1 = X × A1，输出形状为 \[B, L, 2D\]。
- GPU 2 计算：Y2 = X × A2，输出形状为 \[B, L, 2D\]。

激活函数：由于 GeLU 是逐元素计算的，每个 GPU 可以在本地直接对自己那部分输出做激活，无需通信。

- 输出：Z1 = GeLU(Y1)，形状为 \[B, L, 2D\]。
- 输出：Z2 = GeLU(Y2)，形状为 \[B, L, 2D\]。

### 第二阶段：行并行（Row Parallelism）—— 处理权重 B

这是 Megatron-LM 最精妙的一步。它直接利用上一步已经在不同 GPU 上的 Z1 和 Z2，对权重 B 进行行拆分。

拆分权重 B：我们将权重 B 按行切开。

- GPU 1 持有 B1，形状为 \[2D, D\]
- GPU 2 持有 B2，形状为 \[2D, D\]

本地计算：

- GPU 1 计算：Out1 = Z1 × B1。注意这里的矩阵乘法：\[B, L, 2D\] × \[2D, D\] = \[B, L, D\]。
- GPU 2 计算：Out2 = Z2 × B2。同样地：\[B, L, 2D\] × \[2D, D\] = \[B, L, D\]。

通信合并（All-Reduce）：

- 根据矩阵乘法的性质：Z × B = \[Z1, Z2\] × \[B1; B2\] = Z1B1 + Z2B2。
- 此时，每个 GPU 手里的结果都是最终形状 \[B, L, D\] 的一部分。为了得到最终结果，两台 GPU 进行一次 All-Reduce 操作，将 Out1 和 Out2 相加。
- 最终输出：Out = Out1 + Out2，形状为 \[B, L, D\]。

### 核心亮点：为什么高效？

如果我们观察整个 MLP 的流动，你会发现张量的变化非常有趣：

1. 输入 X \[B, L, D\]
2. 列并行计算后，中间结果被拆分成了 \[B, L, 2D\] 和 \[B, L, 2D\]。
3. 行并行计算时，直接把这两个中间结果当做输入。
4. 只有在最后，为了拿到完整的 \[B, L, D\]，才做了一次 All-Reduce。

整个 MLP 模块，无论包含多少计算，只需要一次 All-Reduce 通信，通信效率极高。

## Attention 的张量并行

理解了 MLP 的张量并行逻辑后，看 Attention（多头注意力机制）的张量并行就会非常轻松，因为它们遵循的是完全相同的"列并行 + 行并行"设计模式。

在 Megatron-LM 中，Attention 层的张量并行主要分为三个阶段。我们同样假设输入 X 的形状是 \[B, L, D\]，有 2 个 GPU。

### 第一阶段：QKV 线性投影的列并行（Column Parallelism）

在 Transformer 的注意力机制中，第一步是将输入 X 映射到 Q（查询）、K（键）、V（值）。

权重拆分：Megatron-LM 将 Q、K、V 的权重矩阵合并在一起看作一个大矩阵 W_QKV。它的总形状通常是 \[D, 3 × D\]（假设 3 × D 是为了简化，实际是 3 × heads × head_dim）。我们将这个大矩阵按列切分：

- GPU 1 持有 W_QKV1，负责前一半的"头"（Heads）。
- GPU 2 持有 W_QKV2，负责后一半的"头"。

本地计算：每个 GPU 拿到相同的输入 X \[B, L, D\]，分别计算属于自己的 Q、K、V 部分。

- GPU 1 得到 Q1，K1，V1，对应第 1 到 h/2 个头。
- GPU 2 得到 Q2，K2，V2，对应第 h/2 + 1 到 h 个头。

### 第二阶段：核心注意力计算（Local Attention）

这是张量并行在 Attention 中最高效的地方：多头注意力的每一个头（Head）在数学上是完全独立的。

独立计算：

- GPU 1 利用手里的 Q1，K1，V1 独立计算这 h/2 个头的注意力得分和加权结果。
- GPU 2 同理，计算另外 h/2 个头的结果。

无需通信：因为计算第 1 个头的信息不需要第 10 个头的数据。所以在这一步，GPU 之间不需要交换任何数据。

输出结果：每个 GPU 得到一个局部输出 Zi，形状为 \[B, L, D/2\]。这实际上就是把所有头的结果拼接（Concatenate）后的其中一部分。

### 第三阶段：输出投影的行并行（Row Parallelism）

计算完注意力后，Transformer 会有一个 Linear 层（通常称为 Output Projection 或 Dense 层）把多头的结果映射回原始维度 D。其权重矩阵 W_O 的形状是 \[D, D\]。

权重拆分：我们将 W_O 按行切开：

- GPU 1 持有 W_O1，形状为 \[D/2, D\]。
- GPU 2 持有 W_O2，形状为 \[D/2, D\]。

本地计算：

- GPU 1 计算：Out1 = Z1 × W_O1。由于 \[B, L, D/2\] × \[D/2, D\]，得到 \[B, L, D\]。
- GPU 2 计算：Out2 = Z2 × W_O2。同样得到 \[B, L, D\]。

通信合并（All-Reduce）：就像 MLP 第二层一样，这里运用了矩阵乘法的性质：

$$Final\ Output = Z1 \times W_{O1} + Z2 \times W_{O2}$$

两台 GPU 进行一次 All-Reduce 累加操作。

## Attention 与 MLP 的对称美

你会发现 Megatron-LM 的设计极其统一：

| 模块 | 第一阶段（Column） | 中间层 | 第二阶段（Row） | 通信点 |
|---|---|---|---|---|
| MLP | 切分第一个 Linear（A） | GeLU（逐元素独立） | 切分第二个 Linear（B） | 仅在末尾 All-Reduce |
| Attention | 切分 QKV 投影矩阵 | 多头注意力（Head 独立） | 切分输出投影矩阵（W_O） | 仅在末尾 All-Reduce |

两个模块都遵循同样的三步走策略：

1. 列并行切分第一个投影层，让各 GPU 计算独立的输出分片。
2. 中间的核心计算（GeLU 或 Multi-head Attention）天然可以在本地独立完成，无需任何通信。
3. 行并行切分第二个投影层，利用矩阵分块乘法的性质，让各 GPU 计算部分和，最后通过一次 All-Reduce 得到完整结果。

这种设计让整个 Transformer Block（Attention + MLP）只需要两次 All-Reduce 通信（各模块末尾各一次），极大地降低了通信开销，是 Megatron-LM 能高效扩展到数千 GPU 的核心原因。
