# M-RoPE 时间维度（T）的演进：从 Qwen2-VL 到 Qwen3-VL

## 目录

- [背景：M-RoPE 的 T 维度是什么](#背景m-rope-的-t-维度是什么)
- [1. Qwen2-VL：逻辑帧序号（0, 1, 2, 3...）](#1-qwen2-vl逻辑帧序号-0-1-2-3)
- [2. Qwen2.5-VL：物理绝对时间对齐](#2-qwen25-vl物理绝对时间对齐)
- [3. Qwen3-VL：时间交还给文本 Token](#3-qwen3-vl时间交还给文本-token)
- [三代对比总结](#三代对比总结)

---

## 背景：M-RoPE 的 T 维度是什么

在 Qwen2-VL 引入的 M-RoPE（Multimodal Rotary Position Embedding）中，视觉 token 的位置信息被分解成三个独立的维度：

```
position_ids 形状: [3, batch_size, seq_len]
                    ^
                    ├── 维度 0: T（Temporal，时间）
                    ├── 维度 1: H（Height，高度）
                    └── 维度 2: W（Width，宽度）
```

对于文本 token，T=H=W（退化为标准 1D RoPE）。
对于视频 token，每一帧的空间 patch 都有独立的 H/W 坐标；而 **T 维度则决定了模型如何感知时间**。

三代模型在这个 T 维度上走了截然不同的路：

---

## 1. Qwen2-VL：逻辑帧序号（0, 1, 2, 3...）

### 设计

Qwen2-VL 给视频的每个 temporal patch（每两帧合并为一个 patch）分配一个**纯逻辑序号**，从 0 开始递增，步长固定为 1。

**代码位置：** `vlm_architecture/qwen2_vl/modeling_qwen2_vl.py`

```python
# get_rope_index 中处理视频 token（简化）
def get_rope_index(self, input_ids, mm_token_type_ids, video_grid_thw, ...):
    current_pos = 0
    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:  # 文本
            text_len = end_idx - start_idx
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + current_pos
            )
            current_pos += text_len
        else:                   # 视频
            grid_thw = next(grid_iters[modality_type])
            vision_position_ids = self.get_vision_position_ids(
                current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                # ↑ 注意：没有 time_interval 参数，默认 time_interval=1
            )
            llm_pos_ids_list.append(vision_position_ids)
            current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
```

```python
# get_vision_position_ids 中 T 维度的计算
def get_vision_position_ids(self, start_position, grid_thw, ..., time_interval=1, ...):
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0] // temp_merge_size,
        grid_thw[1] // spatial_merge_size,
        grid_thw[2] // spatial_merge_size,
    )
    # T 维度：对整段视频序列，统一使用 start_position 打底，步长为 time_interval（=1）
    position_temporal = torch.arange(llm_grid_t, device=device) * time_interval + start_position
    # H 和 W：在每个时间帧内按空间坐标递增
    position_height = torch.arange(llm_grid_h).repeat_interleave(llm_grid_w)
    position_width  = torch.arange(llm_grid_w).repeat(llm_grid_h)
    # 将 T×H×W 广播展开成序列
    # 最终 position_temporal 形如: [0,0,0,0, 1,1,1,1, 2,2,2,2] （T=3，H=2，W=2）
```

### 结果（以 T=3, H=2, W=2 为例）

```
# 12 个视觉 token 的 position IDs (省略 H/W，只看 T)：
T: [ 0,  0,  0,  0,   # 第 0 个 temporal patch 的 4 个空间 token
     1,  1,  1,  1,   # 第 1 个 temporal patch
     2,  2,  2,  2 ]  # 第 2 个 temporal patch

步长固定为 1，与真实物理时间无关。
```

### 致命缺陷

```
30 FPS 视频（总时长 1 秒）: T IDs → [0, 1, 2, ..., 29]
10 FPS 视频（总时长 3 秒）: T IDs → [0, 1, 2, ..., 29]
```

两段视频帧数相同，**Temporal IDs 完全一样**，但实际时长相差 3 倍。模型无法区分"这段视频有多长"，也无法回答"这个动作发生在第几秒"。

---

## 2. Qwen2.5-VL：物理绝对时间对齐

### 设计

Qwen2.5-VL 在 Processor 阶段引入了 `second_per_grid_ts`（每个 temporal patch 占据的真实秒数），在 Model 阶段将其与 `tokens_per_second`（每秒对应的基准步长）相乘，得到 `time_interval`，从而让 T 维度的步长**与真实物理时间成正比**。

**Processor 侧（计算物理时间间隔）：**

`vlm_architecture/qwen2_5_vl/processing_qwen2_5_vl.py`

```python
# 根据视频真实 FPS，计算每个 temporal patch 占多少秒
# temporal_patch_size = 2（每 2 帧合并为 1 个 temporal patch）
second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})
```

**Model 侧（物理时间 → T ID 步长）：**

`vlm_architecture/qwen2_5_vl/modeling_qwen2_5_vl.py`

```python
# get_rope_index 中处理视频 token
def get_rope_index(self, ..., second_per_grid_ts, ...):
    tokens_per_second = self.config.vision_config.tokens_per_second  # = 25

    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:  # 文本
            ...
        else:                   # 视频
            grid_thw = next(grid_iters[modality_type])

            # ★ 核心变化：物理时间 → T ID 步长
            time_interval = tokens_per_second * int(next(second_per_grid_ts))
            #               ↑ 基准标尺(25)    ↑ 这帧占多少秒（如 2/fps）

            vision_position_ids = self.get_vision_position_ids(
                current_pos, grid_thw, 1, spatial_merge_size,
                time_interval,          # ← 传入步长
                device=input_ids.device
            )
```

### 结果

```
# 假设 tokens_per_second=25，temporal_patch_size=2：

1 FPS 视频: second_per_grid = 2/1 = 2s/patch → interval = 25 × 2 = 50
T IDs: [ 0,   0,   0,   0,      # patch 0 → 物理时刻 0s
         50,  50,  50,  50,     # patch 1 → 物理时刻 2s
         100, 100, 100, 100 ]   # patch 2 → 物理时刻 4s

2 FPS 视频: second_per_grid = 2/2 = 1s/patch → interval = 25 × 1 = 25
T IDs: [ 0,   0,   0,   0,      # patch 0 → 物理时刻 0s
         25,  25,  25,  25,     # patch 1 → 物理时刻 1s
         50,  50,  50,  50 ]    # patch 2 → 物理时刻 2s
```

> 💡 **物理对齐的意义**
>
> 无论视频是 1 FPS、2 FPS 还是 30 FPS，**相隔 1 秒的事件，在 M-RoPE 的 T 维度上 ID 差值永远是固定的 25**。
>
> 这样模型能感知"这两帧之间过了多少秒"，不再被帧率混淆，能够准确回答时间相关问题（"第 3 秒发生了什么？"）。

### Qwen2-VL vs Qwen2.5-VL 时间感知对比

| 场景 | Qwen2-VL（逻辑帧号） | Qwen2.5-VL（物理时间） |
|---|---|---|
| 30帧 @ 30FPS（1秒） | T: 0, 1, 2, ... 29 | T: 0, 25, 50, ... 725 |
| 30帧 @ 10FPS（3秒） | T: 0, 1, 2, ... 29（**完全相同！**） | T: 0, 75, 150, ... 2175（**明显不同**） |
| 模型能否区分？ | ❌ 无法区分 | ✅ T ID 间距反映真实时长 |

---

## 3. Qwen3-VL：时间交还给文本 Token

### 设计

Qwen3-VL 做出了一个逆转：**放弃在 Position ID 中编码物理时间**，回到类似 Qwen2-VL 的简单逻辑步长（`time_interval=1`）。但与此同时，在 Processor 阶段为每一个 temporal patch **前置插入一个文本格式的时间戳 token**，如 `<0.0 seconds>`、`<1.0 seconds>`，将时间信息显式嵌入到文本序列中。

**Model 侧（T ID：回归简单步长）：**

`vlm_architecture/qwen3_vl/modeling_qwen3_vl.py`

```python
# get_rope_index 中处理视频 token
def get_rope_index(self, ...):
    # ★ 注意：没有 second_per_grid_ts 参数，不再做物理时间对齐
    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:  # 文本
            ...
        else:                   # 视频
            grid_thw = next(grid_iters[modality_type])
            vision_position_ids = self.get_vision_position_ids(
                current_pos, grid_thw, 1, spatial_merge_size,
                device=input_ids.device
                # ↑ 没有 time_interval，与 Qwen2-VL 一样，步长 = 1
            )
```

**Processor 侧（显式文本时间戳）：**

`vlm_architecture/qwen3_vl/processing_qwen3_vl.py`

```python
# 对视频的每一个 temporal patch，在视觉 token 前插入时间戳文本
video_placeholder = ""
frame_seqlen = video_grid_thw[index][1:].prod() // merge_length

for frame_idx in range(video_grid_thw[index][0]):   # 遍历每个 temporal patch
    curr_time = curr_timestamp[frame_idx]            # 计算当前帧的物理时刻（秒）
    video_placeholder += f"<{curr_time:.1f} seconds>"  # ← 插入文本时间戳
    video_placeholder += (
        self.vision_start_token
        + "<|placeholder|>" * frame_seqlen   # 视觉 patch tokens
        + self.vision_end_token
    )
```

### 结果：输入序列结构对比

**Qwen2.5-VL 的序列（时间隐藏在 Position ID 里）：**

```
输入 token 序列:
[TEXT] [<|vision_start|>][视觉token×N][<|vision_end|>]
                                    ...
[<|vision_start|>][视觉token×N][<|vision_end|>] [TEXT]

时间信息：不在文本里，藏在 T Position ID 的步长中
        T ID: [0,0,...,0, 50,50,...,50, 100,100,...,100]
```

**Qwen3-VL 的序列（时间显式在文本里）：**

```
输入 token 序列:
[TEXT] <0.0 seconds>[<|vision_start|>][视觉token×N][<|vision_end|>]
       <2.0 seconds>[<|vision_start|>][视觉token×N][<|vision_end|>]
       <4.0 seconds>[<|vision_start|>][视觉token×N][<|vision_end|>] [TEXT]

时间信息：明确写在文本 token 里，模型直接读
T ID: [0,0,...,0, 1,1,...,1, 2,2,...,2]  ← 简单步长=1
```

### 时间戳计算逻辑

`vlm_architecture/qwen3_vl/processing_qwen3_vl.py`

```python
def _calculate_timestamps(
    self,
    frames_indices: list[int],
    fps: float,
    temporal_patch_size: int,
) -> list[float]:
    """
    将采样帧的原始帧号列表转换为对应的物理时间戳（秒）。
    每 temporal_patch_size 帧合并为一个 temporal patch，
    取其中间帧的时刻作为该 patch 的代表时间。
    """
    timestamps = []
    for i in range(0, len(frames_indices), temporal_patch_size):
        patch_frames = frames_indices[i : i + temporal_patch_size]
        center_frame = patch_frames[len(patch_frames) // 2]
        timestamps.append(center_frame / fps)
    return timestamps
```

> 💡 **为什么 Qwen3-VL 这样设计？**
>
> **Qwen2.5-VL 的 Position ID 方案有一个潜在限制**：时间信息被编码到 RoPE 的旋转角度里。对于超长视频（例如 2 小时），T 维度的 Position ID 会非常大（如 `7200 × 25 = 180000`），模型需要从数值上感知时间差，这依赖于旋转频率的精度，在极端情况下会有精度退化的风险。
>
> Qwen3-VL 将时间改为**显式文本 token**（`<1.5 seconds>`），有以下优势：
> - **可读性**：模型直接读到"这帧在第 1.5 秒"，无需从位置编码反推
> - **精度无关长度**：无论视频多长，时间戳都是精确的数值文本，不依赖 ID 步长的精度
> - **推理透明**：模型的注意力可以直接对准时间戳文本，便于做显式的时间推理（"第 3 秒之前发生了什么？"）

---

## 三代对比总结

| 维度 | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| **T ID 步长** | 1（固定逻辑步长） | `tokens_per_second × second_per_grid_ts`（物理时间） | 1（回归固定逻辑步长） |
| **时间信息载体** | T Position ID（无物理意义） | T Position ID（物理时间编码） | 显式文本 token（`<X.X seconds>`） |
| **时间信息位置** | 隐式，在 RoPE 旋转角度里 | 隐式，在 RoPE 旋转角度里 | **显式，在 token 序列里** |
| **核心参数** | 无 | `second_per_grid_ts`、`tokens_per_second` | 无（Processor 计算文本时间戳） |
| **关键代码** | `get_vision_position_ids(..., time_interval=1)` | `time_interval = tokens_per_second * second_per_grid_ts` | `f"<{curr_time:.1f} seconds>"` |
| **超长视频精度** | ❌ 帧率敏感，无时间感知 | ⚠️ 依赖 ID 数值精度，极端情况有退化风险 | ✅ 文本表示，精度恒定 |
| **跨帧率一致性** | ❌ 不同帧率同帧数 = 相同 T ID | ✅ 相同物理时刻 = 相同 T ID 间距 | ✅ 直接由文本时间戳保证 |

### 演进路线图

```
Qwen2-VL                Qwen2.5-VL                  Qwen3-VL
   │                         │                           │
   │  T = 0, 1, 2, 3...      │  T = 0, 50, 100, 150...  │  T = 0, 1, 2, 3...（同 V2）
   │  （纯逻辑帧序号）          │  （物理时间 × 基准标尺）    │  + 文本: <0.0s><2.0s><4.0s>
   │                         │                           │
   │  缺陷：                  │  改进：                   │  改进：
   │  不感知物理时间            │  跨帧率时间一致              │  时间显式可读
   │                         │  模型能"数秒"              │  精度不依赖 ID 数值
   └────────────────────────►└──────────────────────────►└
```
