# InternVL3

## 目录
- [一、Native Multimodal Pre-Training](#一native-multimodal-pre-training)
  - [1. 核心理念：从“后验对齐”转向“原生学习”](#1-核心理念从后验对齐转向原生学习)
  - [2. 联合参数优化 (Joint Parameter Optimization)](#2-联合参数优化-joint-parameter-optimization)
  - [3. 训练目标与平方平均策略 (Square Averaging)](#3-训练目标与平方平均策略-square-averaging)
  - [4. 数据组成与采样策略](#4-数据组成与采样策略)
- [二、Variable Visual Position Encoding (V2PE)](#二variable-visual-position-encoding-v2pe)
  - [1. 核心背景：突破位置窗口限制](#1-核心背景突破位置窗口限制)
  - [2. V2PE 的工作原理](#2-v2pe-的工作原理)
  - [3. 训练与推理策略](#3-训练与推理策略)
- [三、Mixed Preference Optimization](#三mixed-preference-optimization)
  - [1. MPO 的三个核心组成部分](#1-mpo-的三个核心组成部分)
  - [2. 为什么要使用 MPO？](#2-为什么要使用-mpo)
- [四、Test-Time Scaling](#四test-time-scaling)

## 一、Native Multimodal Pre-Training

InternVL3 引入了一种原生多模态预训练 (Native Multimodal Pre-Training) 范式，这是其区别于前代及传统多模态模型 (MLLMs) 的核心改进。

### 1. 核心理念：从“后验对齐”转向“原生学习”

传统的多模态模型通常采用“两步走”或“多步走”的“后验 (Post-hoc)”流程：先训练一个纯文本 LLM，再通过视觉-文本对齐将其改造为多模态模型。这种方式往往面临模态间的对齐挑战，且可能牺牲模型原有的语言能力。

InternVL3 则更进一步，在预训练阶段就将语言预训练与多模态对齐整合到单一阶段中。

| 特性 | 传统 MLLM (Post-hoc) | InternVL3 (Native) | 优势 |
| :--- | :--- | :--- | :--- |
| 训练阶段 | 多阶段（语言预训练 -> 模态对齐） | 单阶段（语言 + 多模态同步） | 简化流程，降低对齐复杂度 |
| 参数更新 | 冻结部分组件（如冻结 LLM 或 ViT） | 全参数更新（ViT, MLP, LLM 联合优化） | 模态表征同步进化，深度对齐 |
| 数据配比 | 阶段性切换数据源 | 混合交织（纯文本 + 图文 + 视频） | 同时习得语言与多模态能力 |

### 2. 联合参数优化 (Joint Parameter Optimization)

在 InternVL3 的原生预训练中，模型的所有参数（ViT 编码器、MLP 投影层和 LLM 基座）都会根据大规模多模态语料库进行联合更新。

- 打破冻结限制：不同于常规做法中为了保留语言能力而冻结 LLM 的做法，InternVL3 通过混合约 25% 的纯文本数据，在全参数更新的同时成功保留并强化了语言能力。
- 架构一致性：这种训练方式确保了视觉特征和文本表示能够在同一优化目标下演进，消除了后期引入“桥接模块”的必要性。

### 3. 训练目标与平方平均策略 (Square Averaging)

InternVL3 遵循标准的左到右自回归 (Autoregressive) 训练目标。

- Selective Loss (选择性损失)：虽然梯度会通过所有模态的 Token 传播，但模型仅对文本 Token 计算预测损失 (Loss)。这意味着视觉 Token 主要作为预测后续文本的上下文条件。
- Square Averaging (平方平均)：为了解决传统“Token 平均”（偏向长文本）或“样本平均”（偏向短文本）导致的梯度偏差，InternVL3 采用了平方平均策略。

$$
w_i = \frac{1}{\sqrt{N}}
$$

其中 $N$ 表示训练样本中需要计算 Loss 的 Token 数量。这种权重分配方案能更好地平衡不同长度的训练样本。

### 4. 数据组成与采样策略

InternVL3 的预训练语料由多模态数据和纯语言数据组成。

- 数据配比：研究表明，1:3 的语言数据与多模态数据比例能实现最优的性能平衡。
- 总规模：训练总量的 Token 数约为 2000 亿 (200B)。
  - 语言数据：500 亿 (50B) Tokens。
  - 多模态数据：1500 亿 (150B) Tokens。

> 💡 思考：为什么原生预训练能提升语言能力？
> 
> 根据论文测试结果，InternVL3 的语言能力甚至超过了其初始化时所用的原始 Qwen2.5 Chat 模型。这归功于：
> 1. 高质量语料集成：集成了 InternLM2.5 的预训练数据及其他开源高质量文本。
> 2. 联合优化效应：视觉信息的引入在某些任务（如空间推理或带有图表的数学题）中不仅没干扰，反而辅助了模型对复杂逻辑的理解。

---

## 二、Variable Visual Position Encoding (V2PE)

为了解决多模态大模型在处理长上下文 (Long-Context) 场景下的性能瓶颈，InternVL3 引入了可变视觉位置编码 (V2PE) 机制。

### 1. 核心背景：突破位置窗口限制

在传统的多模态模型中，无论是文本 Token 还是视觉 Token，其位置索引 (Position Index) 通常都是统一按 1 递增的。

- 痛点：对于高分辨率图像或视频，视觉 Token 数量巨大，按 1 递增会迅速消耗掉模型有效的上下文窗口 (Position Window)，导致模型无法处理更长的多模态序列。
- V2PE 的方案：为视觉 Token 引入更小、更灵活的位置增量，从而在不扩展位置窗口的前提下，支持更长的多模态上下文。

### 2. V2PE 的工作原理

V2PE 采用了一种模态特定的递归函数来计算位置索引 $p_i$：

1. 递归公式：对于序列中的第 $i$ 个 Token $x_i$，其位置索引计算如下：

$$
p_i = \begin{cases} 0, & \text{if } i = 1 \\ f_{pos}(p_{i-1}, x_i), & \text{for } i = 2, 3, \dots, N \end{cases}
$$

2. 模态特定的增量策略：
   - 文本 Token：保留标准增量 1，以维持文本内部的顺序辨别力。
   - 视觉 Token：采用一个较小的分数增量 $\delta$ ($\delta < 1$)。

| Token 类型 | 位置计算公式 | 增量大小 |
| :--- | :--- | :--- |
| 文本 Token | $p_i = p_{i-1} + 1$ | 1 (固定) |
| 视觉 Token | $p_i = p_{i-1} + \delta$ | $\delta < 1$ (可变) |

### 3. 训练与推理策略

为了使模型具备处理不同长度上下文的鲁棒性，InternVL3 采用了动态的采样与选择策略：

- 训练阶段 (Training)：对于每一张图像，从预定义的集合 $\Delta$ 中随机选择一个分数作为增量 $\delta$：

$$
\Delta = \{1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}\}
$$

  在同一个图像块内部，$\delta$ 保持不变，以维护相对位置关系。

- 推理阶段 (Inference)：
  可以根据输入序列的实际长度灵活选择 $\delta$。当序列非常长时，选择较小的 $\delta$ 可以确保位置索引保持在模型的有效范围内。

> 注意：当 $\delta = 1$ 时，V2PE 退化为 InternVL2.5 所使用的常规位置编码。

---

## 三、Mixed Preference Optimization

### 1. MPO 的三个核心组成部分

MPO 的总损失函数 $\mathcal{L}$ 是由三个不同的损失项加权组合而成的：

$$
\mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g
$$

① 偏好损失 $\mathcal{L}_p$ (Preference Loss)

- 具体实现：采用了 DPO (Direct Preference Optimization) 损失。

$$
\mathcal{L}_p = -\log \sigma \left( \beta \log \frac{\pi_\theta (y_c | x)}{\pi_0 (y_c | x)} - \beta \log \frac{\pi_\theta (y_r | x)}{\pi_0 (y_r | x)} \right)
$$

其中 $\beta$ 为 KL 惩罚系数，x 为用户 Query，y_c 和 $y_r$ 分别为选择回答 (Chosen) 和拒绝回答 (Rejected)。

- 代表意义：让模型学习如何“二选一”。通过对比“被选中 (Chosen)”的高质量回答和“被拒绝 (Rejected)”的回答，让模型学会识别并生成更符合人类偏好的结果。

② 质量损失 $\mathcal{L}_q$ (Quality Loss)

- 具体实现：采用了 BCO (Binary Classifier Optimization) 损失。

$$
\mathcal{L}_q = \mathcal{L}_q^+ + \mathcal{L}_q^-
$$

  其中正负样本的损失项分别独立计算：

$$
\mathcal{L}_q^+ = -\log \sigma \left( \beta \log \frac{\pi_\theta (y_c | x)}{\pi_0 (y_c | x)} - \delta \right)
$$

$$
\mathcal{L}_q^- = -\log \sigma \left( - \left( \beta \log \frac{\pi_\theta (y_r | x)}{\pi_0 (y_r | x)} - \delta \right) \right)
$$

  这里 $\delta$ 表示奖励偏移 (Reward Shift)，通常为之前奖励的移动平均，用于稳定训练。

- 代表意义：让模型理解回答的“绝对质量”。与 DPO 只看相对好坏不同，BCO 独立计算正向和负向样本的损失，帮助模型掌握单个响应本身的质量水平。

③ 生成损失 $\mathcal{L}_g$ (Generation Loss)

- 具体实现：采用标准的 LM (Language Modeling) 损失。
- 代表意义：确保模型不忘“本”，即基础的文本生成能力。它引导模型去学习如何正确地生成那些被选中的优选回答。

### 2. 为什么要使用 MPO？

- 解决分布偏移：在预训练和 SFT 阶段，模型是根据标准答案预测下一个字；但在实际使用中，它是根据自己之前生成的字来预测。这种差异会积累误差，影响推理逻辑。MPO 通过引入正负样本对齐，缓解了这一问题。
- 提升推理性能：实验证明，MPO 对大参数模型提升尤为明显。例如，InternVL3-78B 在经过 MPO 优化后，其多模态推理能力的综合得分提升了 4.1 分。
- 算法效率高：MPO 使用的数据集（约 30 万条）其实是 SFT 数据集的子集。这意味着性能的巨大提升主要归功于算法设计的优化，而非单纯堆砌数据量。

---

## 四、Test-Time Scaling

在 InternVL3 论文中，Test-Time Scaling (测试时缩放) 是一种通过增加推理阶段的计算资源来显著提升模型（尤其是推理和数学任务）性能的方法。

它的核心逻辑不再是训练更大的模型，而是在模型回答问题时，让它多思考、多尝试，并从中选出最优解。

### 1. 核心机制：Best-of-N 采样

InternVL3 采用了 Best-of-N 评估策略：

1. 生成多个候选答案：对于同一个问题，模型会生成 $N$ 个不同的预测结果 (Rollouts)。
2. 引入“评论员”模型：使用一个专门训练的 VisualPRM-8B (视觉过程奖励模型) 作为评委。
3. 筛选最优解：VisualPRM 会对每一个候选答案的推理步骤进行打分，最终选出得分最高的那个作为正式回答。

### 2. VisualPRM 的工作原理

为了让“评委”更专业，论文详细介绍了 VisualPRM 的训练和推理过程：

- 分步打分：它不是只给最终答案打分，而是为解题过程中的每一个步骤 (Step) 分配质量得分，然后取平均值。
- 多轮对话形式：将图像、问题和推理步骤构建成多轮对话。在训练时，模型需要预测每一个给定步骤的正确性 ($c_i \in \{+, -\}$)。
- 概率评估：在推理时，某一步骤的得分被定义为模型生成“+”(正确) 符号的概率。