# Transformer

## 目录

- [一、Encoder 与 Decoder 如何交互？](#一encoder-与-decoder-如何交互)
- [二、并行化体现在哪里？Decoder 能并行吗？](#二并行化体现在哪里decoder-能并行吗)
- [三、为什么 Attention 要除以根号d_k，而不是d_k？](#三为什么-attention-要除以-sqrtd_k而不是-d_k)
- [四、Transformer 的非线性体现在哪里？](#四transformer-的非线性体现在哪里)

## 一、Encoder 与 Decoder 如何交互？

在 Transformer 架构中，Encoder（编码器）与 Decoder（解码器）之间的交互是模型完成 序列到序列（Seq2Seq） 任务（如机器翻译）的核心。这种交互主要通过 交叉注意力（Cross-Attention），也称 Encoder-Decoder Attention 实现。

### 1. 交互的核心：交叉注意力

Decoder 的每一层中通常包含三个子模块：

1. Masked Self-Attention（掩码自注意力）：让 Decoder 只关注已生成（或当前位置左侧）的 token。
2. Encoder-Decoder Attention（交叉注意力）：让 Decoder 查询 Encoder 对源序列的表示，即 Encoder 与 Decoder 的「接口层」。
3. Feed-Forward Network（前馈网络）。

### 2. Q、K、V 的来源

在交叉注意力中，Q、K、V 的来源与自注意力不同，这是理解交互的关键：

| 角色 | 来源 | 含义 |
| :--- | :--- | :--- |
| Query（Q） | Decoder 上一层（常为 Masked Self-Attention）的输出经线性投影 | 「当前解码状态」在问：源序列里哪些位置与我最相关？ |
| Key / Value（K、V） | Encoder 栈的最终输出（各层堆叠后，最后一层再作为 memory）经线性投影 | 源句的上下文表示，供 Decoder 检索与加权读出 |

### 3. 交互流程

对 Decoder 某一位置的表示，交叉注意力大致经历：

1. 线性映射：Decoder 状态 $\rightarrow Q$；Encoder 输出 $\rightarrow K$、V。
2. 相似度：计算 $Q$ 与各个位置 $K$ 的点积（再缩放），得到与源序列各位置的对齐强度。
3. Softmax：得到在源序列长度上的权重分布。
4. 加权求和：用权重对 $V$ 加权求和，得到 上下文向量（Context），作为「从源句读出的信息」，再送入 FFN 等后续模块。

### 4. 数学表达

$$
\mathrm{Attention}(Q_{\mathrm{dec}}, K_{\mathrm{enc}}, V_{\mathrm{enc}})
= \mathrm{softmax}\left(\frac{Q_{\mathrm{dec}} K_{\mathrm{enc}}^{\mathsf{T}}}{\sqrt{d_k}}\right) V_{\mathrm{enc}}
$$

$K$、V 带下标 $\mathrm{enc}$，表示来自 Encoder；Q 带 $\mathrm{dec}$，表示来自 Decoder。

---

## 二、并行化体现在哪里？Decoder 能并行吗？

Transformer 相对 RNN 的一大优势是 易于并行计算，但需要区分 训练 与 推理，以及 Encoder / Decoder 的差异。

### 1. Transformer 的并行化主要体现在哪里

RNN 在第 $t$ 步依赖第 $t-1$ 步的隐状态，时间步之间存在串行依赖，难以对整条序列并行。

Transformer 中，自注意力与随后的矩阵运算把「序列长度」主要放进 矩阵的某一维，从而可用一次（或少数几次）大矩阵乘法完成所有位置两两之间的注意力相关计算：
**通过矩阵乘法 $Q K^T$，模型可以一次性计算出文中所有词两两之间的相互关系。这种计算不再是“一个接一个”，而是大规模的矩阵运算，能够极大地利用 GPU 的并行计算能力。**

- 将输入排成矩阵 $X$，一次性得到 $Q = XW_Q$  $K = XW_K$  $V = XW_V$。
- 计算 $\mathrm{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right) V$ 时，Encoder 侧所有位置可 同时 参与运算，无需按时间步循环等待。

因此，并行化主要体现在：用矩阵乘法替代沿时间步的串行循环，尤其在 Encoder 的自注意力中非常明显。

### 2. Decoder 端能否并行

训练阶段：可以（在 Masked Self-Attention 的前提下）。
- 使用 Teacher Forcing：一次性把完整目标序列喂给 Decoder。
- Look-ahead Mask（上三角掩码）：保证位置 $i$ 只能看见 $\le i$ 的信息，不会「偷看」未来词。
- 在此条件下，Decoder 各层仍可按 大矩阵 形式并行计算，与 Encoder 类似。

推理阶段：自回归生成通常是串行的。
- 生成第 $t$ 个 token 往往依赖已生成的前缀；每步要跑一次 Decoder（或至少更新 KV cache 并算新一步），无法像训练那样对目标长度整段一次性算完自回归链。
- 因此 推理时 Decoder 常成为瓶颈，总代价随生成长度大致线性增长（工程上会用 KV Cache、投机解码等缓解，但本质仍是逐步生成）。

小结：并行化主要体现在 注意力与投影的批矩阵运算；Decoder 在训练时可并行，在典型自回归推理时则按步串行。

---

## 三、为什么 Attention 要除以 $\sqrt{d_k}$，而不是 $d_k$？

在 Transformer 的 Scaled Dot-Product Attention 中，除以 $\sqrt{d_k}$ 是一个非常精妙的设计，其核心目的是为了防止梯度消失，并保持训练过程中的数值稳定性。以下是为什么要除以 $\sqrt{d_k}$ 而不是 $d_k$ 的深度解析：

### 1. 方差控制（Mathematical Variance）

假设 Query和 Key的各个分量是相互独立的随机变量，且满足均值为 0、方差为 1 的标准正态分布：

$$
E[q_i] = E[k_i] = 0, \quad \mathrm{Var}(q_i) = \mathrm{Var}(k_i) = 1
$$

当我们计算点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 时：

- 每个乘积项 $q_i k_i$ 的均值为 0；
- 每个乘积项 $q_i k_i$ 的方差为 $\mathrm{Var}(q_i)\mathrm{Var}(k_i) = 1 \times 1 = 1$；
- 根据方差的和性质，d_k 个独立变量之和的方差为 $d_k$。

即：

$$
\mathrm{Var}(q \cdot k) = d_k
$$

**这意味着随着维度 $d_k$ 的增大，点积结果的幅度会剧烈波动。为了让点积结果的方差重新回到 1，我们需要将其除以其标准差，即 $\sqrt{d_k}$。**

### 2. Softmax 的饱和区与梯度消失

Softmax 函数对输入的数值非常敏感。如果点积的值非常大，Softmax 后的概率分布会变得极其“尖锐”（极化）：

- 最大值所在的概率会逼近 1；
- 其余位置的概率会逼近 0。

当 Softmax 进入这种极化状态时，其梯度会变得非常小（接近于 0）。在反向传播过程中，这会导致严重的梯度消失问题，使得模型难以学习。通过除以 $\sqrt{d_k}$，我们将输入控制在 Softmax 梯度较大的区域，确保了训练的稳定性。

### 3. 为什么不是除以 $d_k$？

如果我们将点积除以 $d_k$，会出现“过度平滑”的问题：

1. 数学层面：方差会变得过小

根据之前的推导，点积 $q \cdot k$ 的方差是 $d_k$。如果我们除以 $d_k$，那么缩放后的方差为：

$$
\mathrm{Var}\left(\frac{q \cdot k}{d_k}\right)=\frac{1}{d_k^2}\mathrm{Var}(q \cdot k)=\frac{d_k}{d_k^2}=\frac{1}{d_k}
$$

随着维度 $d_k$ 的增大，这个值会变得非常小。

- 例如，当 $d_k = 64$ 时，方差仅为 0.015。
- 这意味着绝大多数的点积结果都会被压缩在 0 附近一个极小的范围内。

2. 对 Softmax 的影响：趋于均匀分布

当输入 Softmax 的数值差异极小时，Softmax 的输出会发生什么？

- 极低方差（除以 $d_k$）：所有的输入值都挤在一起，Softmax 会把它们看作几乎相等的数。结果是输出的概率分布非常接近均匀分布（Uniform Distribution）。
- 结果：每一个 Token 获得的权重都差不多（比如都是 $1/n$）。

3. “不敏感”带来的后果

你说的“不敏感”在深度学习中意味着区分度丧失：

- **失去重点：注意力机制的核心在于“选择性”。如果除以 $d_k$，模型会给所有 Token 分配几乎相同的注意力。它无法从一堆背景噪声中准确锁定那个最重要的关键信息。**
- 模型退化：这种状态下的注意力层实际上退化成了一个简单的“平均池化”层。模型失去了建复杂特征依赖的能力，原本想让它“聚焦”，结果它变成了“散焦”。

---

## 四、Transformer 的非线性体现在哪里？

Transformer 模型之所以能够处理极其复杂的语言模式，是因为它在架构中巧妙地引入了**非线性 (Non-linearity)**。如果没有这些非线性部分，无论模型堆叠多少层，最终都只会退化为一个简单的线性变换，无法捕捉深层的语义逻辑。

Transformer 的非线性主要体现在以下四个核心环节：

### 1. 前馈神经网络 (Feed-Forward Network, FFN)

这是 Transformer 中最主要、最直接的非线性来源。在每个 Transformer Block 中，自注意力层之后都会跟一个全连接的前馈网络。

其标准结构如下：

$$
\mathrm{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

- **核心逻辑**：它通常由两个线性层组成，但在中间插入了一个**非线性激活函数 $\sigma$**。
- **作用**：这个激活函数（如 ReLU、GELU 或 Swish）负责对特征进行非线性映射，允许模型在不同维度上进行复杂的特征提取 and 变换。

### 2. 自注意力机制中的 Softmax 函数

虽然自注意力 (Self-Attention) 的核心是矩阵乘法（线性操作），但其权重的计算过程是非线性的。

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- **核心逻辑**：Softmax 函数是一个指数级非线性函数。
- **作用**：它将注意力分数 (Scores) 归一化为概率分布。这种“胜者为王”机制通过非线性手段放大了相关性高的特征，抑制了相关性低的特征，从而实现对上下文的选择性关注。

### 3. 激活函数 (Activation Functions)

在现代 Transformer（如 GPT 系列、BERT、Llama）中，激活函数的选择至关重要：

- **ReLU**：在原始 Transformer 中使用，通过将负值设为零来引入稀疏非线性。
- **GELU (Gaussian Error Linear Unit)**：在 BERT 和 GPT 中广泛使用，提供了更平滑的非线性过渡。
- **SwiGLU**：在 Llama 等最新模型中使用，通过 GLU（门控线性单元）变体进一步增强了模型的非线性表达能力。

### 4. 层归一化 (Layer Normalization)

层归一化虽然主要目的是为了训练稳定，但它本身也包含非线性操作。

- **核心逻辑**：计算输入的均值 $\mu$ 和方差 $\sigma^2$，并进行归一化：

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

- **作用**：方差的计算（平方根倒数）是非线性变换。这种对特征权重的重新缩放和平移，使得每一层的数据分布都经过了一次非线性的“整形”。

