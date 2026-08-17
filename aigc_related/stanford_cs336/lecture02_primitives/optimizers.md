# Optimizers: SGD to AdamW

## 目录
- [SGD (随机梯度下降) & Momentum](#sgd-随机梯度下降--momentum)
  - [1. SGD (随机梯度下降)](#1-sgd-随机梯度下降)
  - [2. Momentum (动量法)](#2-momentum-动量法)
- [Adaptive Methods: AdaGrad & RMSProp](#adaptive-methods-adagrad--rmsprop)
- [Adam & AdamW](#adam--adamw)

## SGD (随机梯度下降) & Momentum

### 1. SGD (随机梯度下降)

SGD 是深度学习中最基础的优化算法。在每次迭代中，它从训练集中随机抽取一小批样本 (mini-batch)，计算损失函数关于当前参数的梯度，并沿着梯度的反方向更新参数。

公式如下：

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

其中：
- $w_t$ 表示第 $t$ 步的参数
- $\eta$ 表示学习率 (learning rate)
- $\nabla L(w_t)$ 表示损失函数 $L$ 在参数 $w_t$ 处的梯度

**特点：**
- **优点**：每次只用一小批数据计算梯度，计算速度快，内存占用小。随机性有助于跳出局部最优解。
- **缺点**：在遇到“峡谷”地形（某一方向的梯度很大，另一方向的梯度很小）时，SGD 会在峡谷两壁之间来回震荡，导致向最优解的收敛速度非常缓慢。

### 2. Momentum (动量法)

为了解决 SGD 在峡谷地形中震荡且收敛缓慢的问题，引入了 Momentum。它借鉴了物理学中的“动量”概念：想象一个小球从山坡上滚下，它不仅会受当前坡度（梯度）的影响，还会保留之前的运动状态（惯性）。

在优化算法中，Momentum 通过引入一个速度变量 (velocity) 来累加历史梯度。如果当前梯度与历史梯度方向一致，参数更新的步长会变大（加速收敛）；如果方向不一致，参数更新的步长会因为动量的抵消而变小（减缓震荡）。

公式如下：

$$v_{t+1} = \gamma v_t + \eta \nabla L(w_t)$$
$$w_{t+1} = w_t - v_{t+1}$$

其中：
- $v_{t+1}$ 表示当前时刻的速度（即累加的动量）
- $\gamma$ 表示动量系数 (momentum coefficient)，通常取值为 0.9

**特点：**
- **减弱震荡**：在梯度方向不断改变的维度上，历史梯度和当前梯度相互抵消，从而抑制了震荡现象。
- **加速收敛**：在梯度方向一致的维度上，历史梯度不断累加，使得更新步长越来越大，从而加速了向最优解的收敛过程。

## Adaptive Methods: AdaGrad & RMSProp

## Adam & AdamW
