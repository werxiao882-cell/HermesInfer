# Mixed Precision Training

## 目录
- [纯FP16训练面临的痛点](#纯fp16训练面临的痛点)
- [AMP的三大核心机制](#amp的三大核心机制)
  - [机制一：FP32主权重(Master Weights)](#机制一fp32主权重master-weights)
  - [机制二：自动精度转换(AutoCasting)](#机制二自动精度转换autocasting)
  - [机制三：损失缩放(Loss Scaling)](#机制三损失缩放loss-scaling)
- [为什么Softmax需要使用FP32？](#为什么softmax需要使用fp32)
- [PyTorch代码实现](#pytorch代码实现)

AMP全称为Automatic Mixed Precision，即自动混合精度。它的核心目的在于：**在不牺牲模型最终精度的前提下，通过混合使用FP16(或BF16)和FP32两种数值格式，大幅降低显存占用，并利用GPU的TensorCore加速前向与反向传播的计算过程。**

### 纯FP16训练面临的痛点
在了解AMP的机制之前，我们需要明白为什么不能直接把整个模型全部转成FP16来训练。纯FP16训练会遇到两个致命问题：

* **梯度下溢(Underflow)**：FP16的动态范围非常有限。在反向传播时，大量梯度值往往极小，直接超出了FP16能表示的最小正数范围，导致梯度在计算中直接变成0，网络停止学习。
* **舍入误差(Rounding Error)**：当学习率乘以梯度得到一个极小的参数更新量时，如果直接将这个微小的更新量加到FP16格式的权重上，由于尾数精度不够，这个变化会被直接抹零，导致权重实际上没有更新。

### AMP的三大核心机制
为了解决上述痛点，AMP在底层引入了以下三大巧妙的设计：

#### 机制一：FP32主权重(Master Weights)
模型在显存中会始终保留一份FP32格式的完整权重(即主权重)，这份权重专门用于确保参数更新的高精度。核心策略：计算用 FP16，更新用 FP32。
* **前向传播(Forward)**：将FP32主权重按需转换为FP16参与计算，得到FP16的激活值(Activation)。
* **反向传播(Backward)**：利用FP16的权重和激活值，快速计算出FP16的梯度。
* **参数更新(Optimizer Step)**：将计算出的FP16梯度转换为FP32格式，然后加到那份高精度的FP32主权重上。这样就完美避开了舍入误差导致权重不更新的问题。

#### 机制二：自动精度转换(AutoCasting)
在网络进行前向计算时，框架(如PyTorch)会根据每种算子的数学特性，自动在后台为其分配最安全的计算精度：
* **采用FP16计算的算子**：主要为计算密集型且对精度相对不敏感的操作，例如Linear、Conv2d、矩阵乘法等。这类操作在FP16下能最大化触发TensorCore的矩阵运算加速。
* **采用FP32计算的算子**：主要为对数值精度极其敏感的操作，例如Softmax、LayerNorm、CrossEntropyLoss等。如果在这些算子上强制使用FP16，极易引发数值溢出(NaN)或严重的精度损失。

#### 机制三：损失缩放(Loss Scaling)
这一机制是专门用来应对梯度下溢问题的。
* **缩放(Scale)**：在反向传播计算梯度之前，将前向传播得到的FP32的Loss值乘以一个缩放因子(Scale Factor，通常初始值为一个很大的常数，如65536)。Loss放大后，反向传播链条上计算出的FP16梯度也会等比例放大，从而成功进入FP16的安全表示区间，避免了下溢变0。
* **还原(Unscale)**：在梯度累加完毕、准备更新FP32主权重之前，将高精度的梯度除以之前那个缩放因子，将其还原回真实的数量级。
* **动态调整**：如果训练中途某个阶段的梯度过大，导致放大后在FP16下溢出(出现NaN或Inf)，优化器会直接跳过当前步的参数更新，并将缩放因子减半；反之，如果连续几百个Step都没有发生溢出，说明当前处于安全区间，缩放因子就会自动翻倍。

### 为什么Softmax需要使用FP32？
在混合精度训练中，Softmax、Loss计算等操作通常被列入黑名单，强制回退到FP32执行，原因主要有两点：
1. 指数爆炸(Overflow)：Softmax包含 $e^x$ 运算。FP16的最大值仅为65504，一旦输入超过11，$e^{11}$ 左右就会导致溢出(变成inf)，导致训练崩溃。FP32则可以容忍更大的输入范围。
2. 累加误差：Softmax的分母需要对大量词表(如50000个词)的概率进行求和。在FP16低精度下，大量小数值的累加会产生严重的舍入误差，导致概率分布总和不为1，影响梯度计算。

### PyTorch代码实现
PyTorch提供了 `torch.cuda.amp` 工具包来实现自动混合精度。对于Float16，我们需要使用 `GradScaler` 来实现上述的Loss Scaling机制。

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 1. 定义模型和优化器
model = MyModel().cuda()
# 优化器中维护的是FP32的Master Weights
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. 初始化GradScaler
# 它的作用是管理Loss Scaling：
# 自动监测梯度是否溢出，动态调整缩放因子(Scaling Factor)。
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # 3. 开启autocast上下文
    # 在这个范围内，PyTorch会自动选择精度：
    # 卷积、矩阵乘法 -> FP16 (速度快)
    # Softmax、Loss -> FP32 (稳定)
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(data)
        loss = loss_fn(output, target)
    
    # 4. 反向传播 (Loss Scaling 核心步骤)
    # scaler.scale(loss): 
    # 将loss乘以缩放因子(如65536)，防止梯度下溢，然后进行backward
    scaler.scale(loss).backward()
    
    # 5. 参数更新 (Unscale & Update)
    # scaler.step(optimizer) 内部流程：
    # 1. 把梯度除回去(Unscale)，还原成真实大小
    # 2. 检查是否有Inf/NaN(溢出)
    #    - 如果有：跳过即使更新，并减小缩放因子
    #    - 如果无：用FP32梯度更新Master Weights
    scaler.step(optimizer)
    
    # 6. 更新Scaler的缩放因子
    # 如果连续多次迭代没有溢出，下一次可能会尝试更大的缩放因子
    scaler.update()
```
