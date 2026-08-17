# Score-Based Generative Models

本仓库包含了一个使用随机微分方程 (SDE) 的分数生成模型 (Score-Based Generative Models) 的极简 PyTorch 实现。

## 项目结构

本项目主要包含两个部分：理论理解和代码实现。

### 1. 理论与推导
关于分数生成模型和 SDE 的详细数学推导及理论背景，请参考随附的 PDF 笔记：

- 📄 **[sde_note.pdf](./sde_note.pdf)**: 包含理论基础，目录结构如下：
  1. **Score and Naive Score-Based Models**
      - 1.1 Score
      - 1.2 Sampling with Langevin Dynamics
      - 1.3 Score Matching
      - 1.4 Denoising Score Matching
      - 1.5 Challenges of Denoising Score Matching
  2. **Noise Conditional Score Network (NCSN)**
      - 2.1 Definition of NCSN
      - 2.2 Learning NCSN via Score Matching
      - 2.3 NCSN inference via annealed Langevin dynamics
          - 2.3.1 步长选择 (Step size selection)
  3. **Score-based generative modeling with stochastic differential equations (SDEs)**
      - 3.1 Perturbing Data with an SDE
      - 3.2 Reversing the SDE for sample generation
      - 3.3 Estimating the reverse SDE with denoising score matching

### 2. 代码实现
代码库主要分为两个用于训练和推理的脚本：

- **`train.py`**: 在 MNIST 数据集上训练 ScoreNet 模型的脚本。
- **`sample.py`**: 推理脚本，使用训练好的模型通过欧拉-丸山 (Euler-Maruyama) 求解器生成样本。
- **`model.py`**: 定义了时间依赖的 ScoreNet (基于 U-Net 架构)。

## 使用方法

### 训练 (Training)

要从头开始训练模型，请运行 `train.py`。你可以通过命令行参数配置超参数。

```bash
# 使用默认设置在 GPU 上训练
python train.py --device cuda

# 使用自定义超参数训练
python train.py --device cuda --sigma 25.0 --n_epochs 50 --batch_size 64 --lr 1e-4
```

**参数说明:**
- `--device`: 使用的设备 (`cuda` 或 `cpu`)。默认值: `cpu`。
- `--sigma`: SDE 扰动的 sigma 值。默认值: `25.0`。
- `--n_epochs`: 训练轮数 (Epochs)。默认值: `50`。
- `--batch_size`: 训练的批次大小。默认值: `32`。
- `--lr`: 学习率。默认值: `1e-4`。

### 采样/推理 (Sampling)

要使用训练好的检查点生成样本，请运行 `sample.py`。这将求解逆 SDE 从噪声中生成图像。

```bash
# 使用训练好的检查点生成样本
python sample.py --device cuda --ckpt ckpt.pth --output samples.png

# 自定义采样步数和批次大小
python sample.py --device cuda --ckpt ckpt.pth --num_steps 1000 --batch_size 64
```

**参数说明:**
- `--device`: 使用的设备 (`cuda` 或 `cpu`)。默认值: `cpu`。
- `--ckpt`: 模型检查点文件的路径。默认值: `ckpt.pth`。
- `--output`: 保存生成的图像网格的路径。默认值: `samples.png`。
- `--num_steps`: 欧拉-丸山采样步数。默认值: `500`。
- `--batch_size`: 生成图像的数量。默认值: `64`。
- `--sigma`: 必须与训练时使用的 sigma 匹配。默认值: `25.0`。

## 结果

训练和采样完成后，你应该会在 `samples.png` 中看到生成的类似 MNIST 数据集的数字图像。
