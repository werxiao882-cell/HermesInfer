# Qwen2-VL 图像前处理详解

Qwen2-VL 的图像前处理分为两个层次：**图像处理器**（`Qwen2VLImageProcessor`）负责像素级变换和 Patch 化；**整体处理器**（`Qwen2VLProcessor`）负责将图像信息与文本 token 序列对齐。

## 目录

- [一、关键参数](#一关键参数)
- [二、图像处理器 Qwen2VLImageProcessor](#二图像处理器-qwen2vlimageprocessor)
  - [1. 像素级标准处理](#1-像素级标准处理)
  - [2. smart_resize：动态分辨率的核心](#2-smart_resize动态分辨率的核心)
  - [3. Patch 化：将图像转换为序列](#3-patch-化将图像转换为序列)
- [三、整体处理器 Qwen2VLProcessor](#三整体处理器-qwen2vlprocessor)
- [四、完整处理流程总结](#四完整处理流程总结)

## 一、关键参数

理解前处理流程，先需要了解三个核心参数，它们贯穿整个 Patch 化逻辑：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `patch_size` | 14 | 空间切块大小，ViT 在 2D 空间维度上切分的基本单元（14×14 像素为一个 patch） |
| `temporal_patch_size` | 2 | 时间切块大小，在时间维度上合并的帧数（2 帧为一个时间切片，单张图像会复制补齐） |
| `merge_size` | 2 | 融合池化大小，视觉编码器输出给 LLM 前，在空间维度上压缩的倍数（2×2 融合，序列长度缩小 4 倍） |

此外，还有控制图像分辨率范围的参数：

- `min_pixels`：图像总像素数下限，默认 `56 × 56 = 3136`
- `max_pixels`：图像总像素数上限，默认 `28 × 28 × 1280 ≈ 100 万`

## 二、图像处理器 Qwen2VLImageProcessor

### 1. 像素级标准处理

图像首先经历三步标准的像素级变换：

**① 格式转换**：将图像统一转为 RGB 格式，并转成 numpy array 方便后续计算。

**② 缩放像素值**（Rescale）：将像素值从 `[0, 255]` 归一化到 `[0.0, 1.0]`。

```python
image = self.rescale(image, scale=1/255)
```

**③ 标准化**（Normalize）：使用 OpenAI CLIP 的 `mean` 和 `std` 对图像进行标准化，使像素值分布与 ViT 预训练数据对齐。

```python
# OpenAI CLIP 的均值和标准差
image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std  = [0.26862954, 0.26130258, 0.27577711]

image = self.normalize(image=image, mean=image_mean, std=image_std)
```

### 2. smart_resize：动态分辨率的核心

在像素级处理之前，`smart_resize` 会先对图像做智能缩放。这是 Qwen2-VL "原生动态分辨率"的核心实现，目标是在**保持宽高比**的前提下，将图像调整到一个满足以下约束的尺寸：

1. 高度和宽度都能被 `factor = patch_size × merge_size = 14 × 2 = 28` 整除
2. 总像素数落在 `[min_pixels, max_pixels]` 范围内

```python
def smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    # 极端宽高比保护，超过 200:1 直接报错
    if max(height, width) / min(height, width) > 200:
        raise ValueError(...)
    
    # 先将高宽四舍五入对齐到 factor 的倍数
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        # 图片太大 → 等比缩小，用 floor 确保不超上限
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        # 图片太小 → 等比放大，用 ceil 确保不低于下限
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    return h_bar, w_bar
```

**三种情况说明：**

- **像素数在范围内**：只做最小调整，将高宽四舍五入对齐到 28 的倍数，基本保持原始尺寸。
- **像素数超过 `max_pixels`（图片太大）**：计算缩小比例 `beta = sqrt(原始像素 / max_pixels)`，等比缩小到上限附近。例如一张 4000×3000 的图（1200 万像素）会被缩放到约 1148×868。
- **像素数小于 `min_pixels`（图片太小）**：计算放大比例 `beta = sqrt(min_pixels / 原始像素)`，等比放大到下限附近。例如一张 40×40 的图（1600 像素）会被放大到 56×56。

两个方向取整方式不同（`floor` vs `ceil`）是有意为之：超出上限时用 `floor` 确保不超过 `max_pixels`，低于下限时用 `ceil` 确保不低于 `min_pixels`。

### 3. Patch 化：将图像转换为序列

像素级处理完成后，核心的 Patch 化流程将图像从 `(T, C, H, W)` 的张量转换成 ViT 可处理的一维 patch 序列。

**步骤一：时间维度补齐（Temporal Padding）**

单张图像只有 1 帧（T=1），无法被 `temporal_patch_size=2` 整除。因此会将最后一帧复制一次，补齐为 2 帧，使得图像和视频在底层格式上保持一致。

```python
if patches.shape[0] % temporal_patch_size != 0:
    repeats = np.repeat(
        patches[-1][np.newaxis],
        temporal_patch_size - (patches.shape[0] % temporal_patch_size),
        axis=0
    )
    patches = np.concatenate([patches, repeats], axis=0)
```

**步骤二：计算网格尺寸**

```python
channel = patches.shape[1]
grid_t = patches.shape[0] // temporal_patch_size  # 时间维度的切块数
grid_h = resized_height // patch_size              # 高度方向的切块数
grid_w = resized_width  // patch_size              # 宽度方向的切块数
```

以一张缩放后 `448×448` 的图像为例：

- `grid_t = 2 // 2 = 1`（2 帧 / temporal_patch_size）
- `grid_h = 448 // 14 = 32`
- `grid_w = 448 // 14 = 32`

**步骤三：高维 Reshape（解构空间结构）**

这是最关键的一步，将 `(T, C, H, W)` 张量按照时间、空间的多层次结构逐层解构。高度被拆成三层：`[大网格数 = grid_h // merge_size] × [merge_size] × [patch内像素数 = patch_size]`，宽度同理，时间维度也被拆成两层。

```python
patches = patches.reshape(
    grid_t,                  # 时间大网格数
    temporal_patch_size,     # 每个时间块内的帧数
    channel,                 # 通道数（3）
    grid_h // merge_size,    # 高度方向的"大网格"数量
    merge_size,              # 2×2 合并块中的高度位置（0 或 1）
    patch_size,              # 单个 patch 内的像素高度
    grid_w // merge_size,    # 宽度方向的"大网格"数量
    merge_size,              # 2×2 合并块中的宽度位置（0 或 1）
    patch_size,              # 单个 patch 内的像素宽度
)
```

**步骤四：维度转置（将相邻 patch 排在一起）**

Reshape 后，将"宏观网格坐标"提到前面，"块内像素细节"放到后面。这样空间上相邻的 4 个 patch（一个 2×2 方块）在内存上也会连续排布，为后续 `PatchMerger` 的 MLP 压缩做准备。

```python
# 转置前维度顺序：(grid_t, temporal, C, grid_h//m, m, patch, grid_w//m, m, patch)
# 转置后维度顺序：(grid_t, grid_h//m, grid_w//m, m, m, C, temporal, patch, patch)
patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
```

**步骤五：展平为最终序列**

```python
flatten_patches = patches.reshape(
    grid_t * grid_h * grid_w,                         # 总 patch 数量
    channel * temporal_patch_size * patch_size * patch_size  # 每个 patch 展平后的特征长度
)
```

每个 patch 的特征长度 = `3 × 2 × 14 × 14 = 1176`。

最终 `_preprocess` 函数的输出是：
- `flatten_patches`：形状 `(grid_t × grid_h × grid_w, 1176)` 的 patch 序列
- `(grid_t, grid_h, grid_w)`：用于后续位置编码计算的网格尺寸信息

## 三、整体处理器 Qwen2VLProcessor

`Qwen2VLProcessor` 在图像处理器的基础上，负责将**图像 patch 数量**与**文本 token 序列**对齐。

原始文本中的图像占位符（如 `<|image_pad|>`）只有 1 个，但实际上一张图片可能对应几十到几百个视觉 token。处理器会根据 `image_grid_thw`（即 `grid_t × grid_h × grid_w`）计算出实际的 token 数量，并将占位符展开：

```python
# merge_size=2，所以每 4 个 patch（2×2）合并成 1 个 LLM token
merge_length = self.image_processor.merge_size ** 2  # = 4

index = 0
for i in range(len(text)):
    while self.image_token in text[i]:
        # 计算当前图片对应的 LLM token 数
        num_image_tokens = image_grid_thw[index].prod() // merge_length
        # 将单个占位符展开为对应数量的 token
        text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
        index += 1
    # 将临时占位符还原为真实的 image token
    text[i] = text[i].replace("<|placeholder|>", self.image_token)
```

此外，处理器还会生成 `mm_token_type_ids`，用来区分序列中的不同 token 类型：

```python
mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
mm_token_type_ids[array_ids == self.image_token_id] = 1  # 图像 token 标记为 1
mm_token_type_ids[array_ids == self.video_token_id] = 2  # 视频 token 标记为 2
# 文本 token 默认为 0
```

这个类型 ID 在后续 M-RoPE 的位置编码计算中会用来区分不同模态，给不同模态分配对应的 3D 位置 ID。

## 四、完整处理流程总结

```
原始图片 (任意 H × W)
        ↓
① 格式转换（转 RGB + 转 numpy）
        ↓
② smart_resize（动态分辨率：保持宽高比，对齐到28的倍数，约束总像素数）
        ↓
③ rescale（÷255，归一化到 [0, 1]）
        ↓
④ normalize（CLIP mean/std 标准化）
        ↓
⑤ 时间维度补齐（单张图像复制为 2 帧，与视频格式统一）
        ↓
⑥ 9D Reshape（按时间×空间多层次结构解构张量）
        ↓
⑦ Transpose（将空间相邻的4个 patch 排列连续，为 MLP 压缩做准备）
        ↓
⑧ Flatten → 输出 pixel_values: shape (grid_t × grid_h × grid_w, 1176)
                  image_grid_thw: (grid_t, grid_h, grid_w)
        ↓
⑨ Processor 将文本中的图像占位符展开为实际 token 数量，完成图文序列对齐
        ↓
最终输出: input_ids + attention_mask + pixel_values + image_grid_thw + mm_token_type_ids
```

> **设计亮点**：Qwen2-VL 前处理最核心的创新在于 **动态分辨率**（`smart_resize`）与 **3D 时空 Patch 化**（统一处理图片和视频）的结合。不同于传统 VLM 将所有图片强制缩放到同一固定分辨率，动态分辨率方案在保留图片原始宽高比的同时，也为后续的 M-RoPE 提供了真实的空间网格信息（`grid_thw`），使模型能够感知图像的实际空间结构。
