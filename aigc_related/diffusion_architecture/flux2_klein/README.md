# FLUX.2 Klein 模型架构深度解析

## 摘要

FLUX.2 Klein 依旧属于 Latent Diffusion + Flow Matching + Diffusion Transformer 的路线：像素图像先由 VAE 压缩到 latent，文本由大语言模型式的 Text Encoder 编成条件 token，DiT 在噪声 latent 上预测从噪声到图像的速度场，最后由 VAE Decoder 还原成 RGB 图像。

相较 FLUX.1，FLUX.2 Klein 在 DiffSynth-Studio 的实现里有几个非常适合面试回答的变化：

* **VAE latent 更宽**：最终送入 DiT 的 latent 是 `128` 通道，并且空间分辨率是 `H/16 × W/16`，不是 SD1.5 / SDXL 的 `4` 通道 `H/8 × W/8`，也不是 FLUX.1 常见的 `16` 通道再 pack 成 `64` 维 token。
* **文本编码器更像 LLM**：Klein 示例加载 `black-forest-labs/FLUX.2-klein-*` 的 `text_encoder`，在 DiffSynth 的模型映射里对应 `z_image_text_encoder`，也就是 Qwen3 系列文本编码器，而不是 FLUX.2 dev 使用的 Mistral3 路径。
* **DiT 主干仍是先双流、后单流**：Klein 4B 是 `5` 层双流 + `20` 层单流，Klein 9B 是 `8` 层双流 + `24` 层单流；两者都先保持文本流和图像流分开，再把 token 拼接成单流统一处理。
* **Flow Matching 采样带经验 shift**：调度器使用 `FlowMatchScheduler("FLUX.2")`，根据图像 token 数和采样步数计算经验 `mu`，调整 sigma/timestep 分布。

下文主要参考 `DiffSynth-Studio/diffsynth/pipelines/flux2_image.py`、`models/flux2_dit.py`、`models/flux2_vae.py`、`models/z_image_text_encoder.py`、`configs/model_configs.py` 和 `diffusion/flow_match.py`。

## 目录

- [1. 整体生成流程](#1-整体生成流程)
- [2. VAE：从像素到 128 通道 Latent Token](#2-vae从像素到-128-通道-latent-token)
- [3. 文本编码器：Qwen3 中间层拼接](#3-文本编码器qwen3-中间层拼接)
- [4. DiT 主干：Klein 4B 与 9B 配置](#4-dit-主干klein-4b-与-9b-配置)
- [5. 位置编码与条件注入](#5-位置编码与条件注入)
- [6. Flow Matching 调度与 CFG](#6-flow-matching-调度与-cfg)
- [7. 和 SD / FLUX.1 的核心区别](#7-和-sd--flux1-的核心区别)
- [8. 面试速答](#8-面试速答)

---

## 1. 整体生成流程

FLUX.2 Klein 的推理链路可以按五步理解：

1. **准备文本条件**：Prompt 进入 Text Encoder，取多层 hidden states 拼接，得到 `prompt_embeds`；同时生成文本 token 的四维位置 ID。
2. **准备图像 latent**：如果是文生图，直接采样高斯噪声；如果是图生图或编辑图，先用 VAE Encoder 把输入图像编码成 latent，再按 `denoising_strength` 加噪。
3. **准备位置 ID**：图像 latent 被视为 `H/16 × W/16` 的 token 网格，每个 token 都带 `(t, h, w, l)` 四维坐标。
4. **DiT 迭代去噪**：每个 timestep 中，DiT 接收 noisy latent、文本 token、位置 ID、时间步和 embedded guidance，预测 Flow Matching 的速度场。
5. **VAE 解码**：迭代结束后，将 `[B, H/16*W/16, 128]` reshape 回 `[B, 128, H/16, W/16]`，再由 VAE Decoder 还原成图像。

DiffSynth-Studio 中的主入口是：

```python
# flux2_image.py
self.scheduler = FlowMatchScheduler("FLUX.2")
self.units = [
    Flux2Unit_ShapeChecker(),
    Flux2Unit_PromptEmbedder(),
    Flux2Unit_Qwen3PromptEmbedder(),
    Flux2Unit_NoiseInitializer(),
    Flux2Unit_InputImageEmbedder(),
    Flux2Unit_EditImageEmbedder(),
    Flux2Unit_ImageIDs(),
    Flux2Unit_Inpaint(),
]
```

这里的 `PipelineUnit` 设计很清晰：文本、噪声、图像 latent、编辑图、位置 ID、inpaint mask 都先被拆成独立预处理单元，最后统一送入 `model_fn_flux2` 调 DiT。

---

## 2. VAE：从像素到 128 通道 Latent Token

| 维度 | SD1.5 / SDXL | FLUX.1 | FLUX.2 Klein |
|------|--------------|--------|--------------|
| VAE 基础 latent 通道 | 4 | 16 | 32 |
| 送入 Transformer 的 token 通道 | 不适用，U-Net 直接吃 4D latent | `16 × 2 × 2 = 64` | `32 × 2 × 2 = 128` |
| token 空间尺寸 | `H/8 × W/8` | `H/16 × W/16` after pack | `H/16 × W/16` after rearrange |
| 典型输入给 DiT 的形状 | 无 | `[B, L, 64]` | `[B, L, 128]` |

FLUX.2 Klein 的 VAE 默认 `latent_channels=32`。Encoder 先得到 `32` 通道的 latent，随后把空间上 `2 × 2` 的邻域折叠到通道维，所以最终 DiT 看到的是 `128` 通道 token：

```python
# flux2_vae.py
h = rearrange(h, "B C (H P) (W Q) -> B (C P Q) H W", P=2, Q=2)
h = h[:, :128]
latents_bn_mean = self.bn.running_mean.view(1, -1, 1, 1).to(h.device, h.dtype)
latents_bn_std = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + 0.0001).to(h.device, h.dtype)
h = (h - latents_bn_mean) / latents_bn_std
```

这一段有三个关键信息：

* **下采样等效到 16 倍**：VAE 主体完成常规空间压缩后，又通过 `2 × 2` folding 把空间尺寸再压一半，所以 pipeline 要求 `height_division_factor=16`。
* **通道变宽**：`32` 个 VAE latent 通道乘上 `2 × 2` patch，得到 `128` 维 token。这样每个 token 承载更多局部视觉信息。
* **用 BatchNorm 统计量归一化 latent**：这里没有直接使用 SD1.5 的固定 `0.18215`，而是用 `bn.running_mean / running_var` 对 latent 做标准化；解码时再反标准化。

解码过程正好反过来：

```python
# flux2_vae.py
z = z * latents_bn_std + latents_bn_mean
z = rearrange(z, "B (C P Q) H W -> B C (H P) (W Q)", P=2, Q=2)
dec = self.decoder(z)
```

> 面试回答重点：FLUX.2 Klein 的 VAE 不只是“压缩图像”，它还在进入 DiT 前完成了一次 patch 化，把 `32` 通道 latent 变成 `128` 通道 token，降低序列长度、提高单 token 信息密度。

---

## 3. 文本编码器：Qwen3 中间层拼接

FLUX.2 Klein 的文本侧不再是 SD1.5 的单 CLIP，也不是 SD3 / FLUX.1 常见的 T5 + CLIP 组合。在 DiffSynth 的模型配置里，Klein 的 `text_encoder/*.safetensors` 会被映射成 `z_image_text_encoder`，模型类是 `ZImageTextEncoder`，底层使用 Qwen3：

```python
# model_configs.py
"model_name": "z_image_text_encoder",
"model_class": "diffsynth.models.z_image_text_encoder.ZImageTextEncoder",
"extra_kwargs": {"model_size": "8B"},
```

Pipeline 里也能看到：如果 `text_encoder_qwen3` 存在，就跳过普通 `Flux2TextEncoder`，改由 `Flux2Unit_Qwen3PromptEmbedder` 处理 prompt。

```python
# flux2_image.py
pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")

# Skip if Qwen3 text encoder is available
if pipe.text_encoder_qwen3 is not None:
    return {}
```

Qwen3 分支真正送给 DiT 的不是最后一层 hidden state，而是中间多层拼接：

```python
# flux2_image.py
self.hidden_states_layers = (9, 18, 27)  # Qwen3 layers
out = torch.stack([output.hidden_states[k] for k in self.hidden_states_layers], dim=1)
prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
```

所以文本条件维度由 Qwen3 的 hidden size 决定：

| 版本 | Qwen3 hidden size | 取层 | 拼接后维度 | 对应 DiT `joint_attention_dim` |
|------|-------------------|------|------------|-------------------------------|
| Klein 4B | `2560` | `9 / 18 / 27` | `2560 × 3 = 7680` | `7680` |
| Klein 9B | `4096` | `9 / 18 / 27` | `4096 × 3 = 12288` | `12288` |

为什么要取中间层而不是只取最后一层？

* 中低层通常保留更多词法、实体和局部关系信息。
* 高层更偏全局语义和指令抽象。
* 多层拼接相当于把“细节 + 关系 + 语义”同时交给 DiT，对复杂 prompt、物体归属、动作关系会更友好。

> 面试回答重点：FLUX.2 Klein 的文本编码器有 Qwen。准确说，在 DiffSynth 的 Klein 路径里，它用 `ZImageTextEncoder` 包装 Qwen3，取第 `9/18/27` 层 hidden states 拼接，再送入 DiT 的 `context_embedder`。

---

## 4. DiT 主干：Klein 4B 与 9B 配置

FLUX.2 Klein 的核心去噪网络是 `Flux2DiT`，但 Klein 4B / 9B 并不使用 `flux2_dit.py` 里的 FLUX.2 dev 默认参数，而是在 `model_configs.py` 中覆盖了层数、头数、文本条件维度等参数：

```python
# model_configs.py：FLUX.2-klein-4B transformer
"extra_kwargs": {
    "guidance_embeds": False,
    "joint_attention_dim": 7680,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 20,
}
```

```python
# model_configs.py：FLUX.2-klein-9B transformer
"extra_kwargs": {
    "guidance_embeds": False,
    "joint_attention_dim": 12288,
    "num_attention_heads": 32,
    "num_layers": 8,
    "num_single_layers": 24,
}
```

由于 `attention_head_dim=128`，内部 hidden dim 分别是：

| 版本 | attention heads | hidden dim | 双流层数 | 单流层数 |
|------|-----------------|------------|----------|----------|
| Klein 4B | `24` | `24 × 128 = 3072` | `5` | `20` |
| Klein 9B | `32` | `32 × 128 = 4096` | `8` | `24` |

整体结构是：

```text
image latent [B, H/16*W/16, 128] -> x_embedder       -> DiT hidden dim
text embeds  [B, 512, D_text]    -> context_embedder -> DiT hidden dim

Double Stream Blocks:
    text stream 和 image stream 分开保留表示
    但在同一个 joint attention 矩阵中互相可见

concat(text, image)

Single Stream Blocks:
    文本 token 和图像 token 合成一个序列
    使用统一 self-attention + parallel MLP

remove text tokens
norm_out + proj_out -> [B, H/16*W/16, 128]
```

### 4.1 双流模块：分开表示，共同注意力

双流模块里，图像流和文本流各自有 LayerNorm、AdaLN 调制、FFN，但注意力时会把两边的 Q/K/V 拼起来：

```python
# flux2_dit.py
attention_outputs = self.attn(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=norm_encoder_hidden_states,
    image_rotary_emb=image_rotary_emb,
)

attn_output, context_attn_output = attention_outputs
hidden_states = hidden_states + gate_msa * attn_output
encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
```

这和传统 U-Net Cross-Attention 不一样。SD 系列里通常是图像特征做 Query，文本做 Key/Value，文本本身不被更新；而这里文本 token 和图像 token 都参与 joint attention，并且两边都会被更新。

### 4.2 单流模块：文本图像 token 统一建模

经过双流阶段后，模型直接把文本和图像序列拼接：

```python
# flux2_dit.py
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

for block in self.single_transformer_blocks:
    hidden_states = block(hidden_states=hidden_states, ...)

hidden_states = hidden_states[:, num_txt_tokens:, ...]
```

单流模块使用 parallel transformer block：QKV 投影和 MLP 输入投影融合在一个线性层里，注意力输出和 MLP 输出再拼接后一次性投影出去：

```python
# flux2_dit.py
self.to_qkv_mlp_proj = torch.nn.Linear(
    self.query_dim,
    self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
    bias=bias,
)
self.to_out = torch.nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)
```

这种设计的好处是减少模块内部串行依赖：attention 分支和 MLP 分支可以并行计算，再统一融合，适合大模型推理优化。

> 面试回答重点：FLUX.2 Klein 沿用了 FLUX 系列“先双流对齐，再单流融合”的思想，但 Klein 4B 是 `5 + 20`，Klein 9B 是 `8 + 24`。双流阶段保护模态差异，单流阶段让文本和图像 token 在统一空间里深度绑定。

---

## 5. 位置编码与条件注入

### 5.1 四维 RoPE 坐标

FLUX.2 Klein 的 token ID 不是简单的二维 `(h, w)`，而是四维：

```python
# flux2_image.py
latent_ids = torch.cartesian_prod(t, h, w, l)
```

对于普通图像 latent：

* `t = 0`：可以理解为图像组或时间/参考图维度。
* `h, w`：latent token 在二维网格中的位置。
* `l = 0`：层或局部 token 维度。

文本 token 也构造为四维坐标，只是 `t/h/w` 基本固定，最后一维 `l` 表示 token 序号：

```python
# flux2_image.py
coords = torch.cartesian_prod(t, h, w, l)
```

随后 `Flux2PosEmbed` 会对每个坐标轴分别生成 RoPE：

```python
# flux2_dit.py
axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32)
rope_theta: int = 2000
self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
```

四个轴的维度加起来是 `128`，正好对应每个 attention head 的 `head_dim=128`。也就是说，每个 head 内部的 RoPE 维度被拆成四段，分别编码不同坐标轴。

### 5.2 时间步与 embedded guidance

FLUX.2 Klein 不只把 timestep 送入 DiT，还把 `embedded_guidance` 也编码进去：

```python
# flux2_dit.py
temb = self.time_guidance_embed(timestep, guidance)
double_stream_mod_img = self.double_stream_modulation_img(temb)
double_stream_mod_txt = self.double_stream_modulation_txt(temb)
single_stream_mod = self.single_stream_modulation(temb)[0]
```

这些调制参数会产生 `shift / scale / gate`，注入到 attention 和 FFN 子层：

```python
norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
hidden_states = hidden_states + gate_msa * attn_output
```

这和 SD1.5 的 ResBlock 时间嵌入很不一样：SD1.5 主要把 timestep 加到卷积 ResBlock；FLUX.2 Klein 则用 AdaLN 风格的 `shift/scale/gate` 控制 Transformer 每个子层的更新强度。

---

## 6. Flow Matching 调度与 CFG

FLUX.2 Klein 的训练和采样目标不是 DDPM 的噪声预测，而是 Flow Matching 的速度场。Scheduler 中的加噪公式是：

```python
# flow_match.py
sample = (1 - sigma) * original_samples + sigma * noise
```

训练目标是：

```python
# flow_match.py
target = noise - sample
```

采样时，DiT 输出 `model_output` 后，用 Euler 形式沿 sigma 轨迹更新 latent：

```python
# flow_match.py
prev_sample = sample + model_output * (sigma_ - sigma)
```

直观理解：

* `sigma=1` 附近更像纯噪声。
* `sigma=0` 附近更像干净图像 latent。
* 模型学习的是“从当前点往目标分布走的方向”，而不是显式预测某一步加进去的噪声。

### 6.1 FLUX.2 的经验 shift

FLUX.2 的 timestep 设置不是简单线性。它会先生成线性 sigmas，再根据图像 token 数和步数计算 `mu`：

```python
# flow_match.py
mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
timesteps = sigmas * num_train_timesteps
```

其中 `dynamic_shift_len` 来自：

```python
# flux2_image.py
self.scheduler.set_timesteps(
    num_inference_steps,
    denoising_strength=denoising_strength,
    dynamic_shift_len=height//16*width//16,
)
```

所以分辨率越高，latent token 越多，调度器就会调整去噪节奏。这和 FLUX.1 / SD3 中的动态 shift 思想类似，但 FLUX.2 这里用的是经验公式 `compute_empirical_mu(image_seq_len, num_steps)`。

### 6.2 两种 guidance：CFG 与 embedded guidance

FLUX.2 Klein 的 pipeline 里同时存在两个容易混淆的参数：

| 参数 | 默认值 | 作用位置 | 含义 |
|------|--------|----------|------|
| `cfg_scale` | `1.0` | pipeline 外部组合正负 prompt 输出 | 标准 Classifier-Free Guidance |
| `embedded_guidance` | `4.0` | DiT 内部 timestep/guidance embedding | 作为模型条件注入每层 AdaLN 调制 |

标准 CFG 的公式仍然是：

```python
# base_pipeline.py
noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
```

但当 `cfg_scale=1.0` 时，pipeline 只跑正向 prompt，一次 DiT forward 就够了；而 `embedded_guidance` 无论是否做负向 prompt CFG，都会作为一个连续条件进入 DiT。

> 面试回答重点：`cfg_scale` 是推理时正负 prompt 输出的外部线性组合；`embedded_guidance` 是模型内部条件，和 timestep 一起产生调制参数。两者不是一回事。

---

## 7. 和 SD / FLUX.1 的核心区别

| 维度 | SD1.5 / SDXL | FLUX.1 | FLUX.2 Klein |
|------|--------------|--------|--------------|
| 去噪骨干 | U-Net | DiT / MMDiT | DiT / MMDiT |
| 生成范式 | DDPM 噪声预测 | Flow Matching | Flow Matching |
| 文本编码 | CLIP / 双 CLIP | CLIP + T5 | Qwen3 hidden states 拼接 |
| 图像 token | U-Net 处理 4D feature map | 16 通道 latent pack 成 64 维 token | 32 通道 latent fold 成 128 维 token |
| 主干融合 | Cross-Attention，文本通常不更新 | 双流 + 单流 | 4B: `5+20`；9B: `8+24` |
| 位置编码 | 卷积先验 / 2D 位置条件较弱 | RoPE | 四维坐标 RoPE |
| guidance | 主要是 CFG | CFG / guidance embedding | CFG + embedded guidance |

如果面试官问“FLUX.2 Klein 相比 FLUX.1 最大变化是什么”，可以从三个角度回答：

1. **文本侧**：从 CLIP + T5 的双文本编码路线，变成以 Qwen3 这类 LLM hidden states 为核心的文本条件。
2. **VAE / latent 侧**：送入 DiT 的 token 通道更宽，FLUX.2 Klein 是 `128` 维 latent token。
3. **主干细节**：仍保留双流到单流的 FLUX 思想，但具体层数、hidden dim、RoPE 坐标、guidance 注入和调度方式都有更新。

---

## 8. 面试速答

### Q1：FLUX.2 Klein 的整体架构是什么？

它是一个 Latent Diffusion 模型：VAE 把图像压到 latent，Text Encoder 把 prompt 编成条件 token，DiT 在 latent token 上做 Flow Matching 去噪，最后 VAE Decoder 还原图像。核心主干是 `8` 层双流 Transformer 加 `48` 层单流 Transformer。

### Q2：为什么说它不是传统 SD 的 U-Net 架构？

SD1.5 / SDXL 的去噪网络是卷积 U-Net，通过 Cross-Attention 注入文本；FLUX.2 Klein 是 Transformer，把图像 latent 和文本都表示成 token，通过 joint attention 做跨模态建模，后半段甚至把文本和图像 token 拼成一个序列统一处理。

### Q3：FLUX.2 Klein 的 VAE 有什么特殊？

它的 VAE 基础 latent 是 `32` 通道，进入 DiT 前会把空间上 `2 × 2` 的 patch 折叠到通道维，变成 `128` 通道、`H/16 × W/16` 的 latent token。这样序列更短、单 token 信息更多。

### Q4：文本编码器输出维度是多少？

Klein 走 Qwen3 分支，取第 `9/18/27` 层 hidden states 拼接。4B 版本 hidden size 是 `2560`，所以文本条件是 `7680` 维；9B 版本使用 Qwen3 8B 配置，hidden size 是 `4096`，所以文本条件是 `12288` 维。随后通过 `context_embedder` 映射到 DiT 内部维度。

### Q5：双流和单流分别解决什么问题？

双流阶段让文本和图像保持各自表示空间，同时通过 joint attention 互相看见，适合早期跨模态对齐；单流阶段把文本和图像 token 拼接起来，用统一 self-attention 深度融合，适合把 prompt 细节绑定到图像结构里。

### Q6：Flow Matching 和 DDPM 噪声预测有什么区别？

DDPM 学的是每个时间步的噪声或 score；Flow Matching 学的是从噪声分布到数据分布的速度场。FLUX.2 Klein 采样时使用 Euler 更新：`sample + model_output * (sigma_next - sigma)`，沿着 sigma 轨迹逐步从噪声走向图像。

### Q7：`cfg_scale` 和 `embedded_guidance` 有什么区别？

`cfg_scale` 是外部 CFG，把正向 prompt 和负向 prompt 的 DiT 输出按公式线性组合；`embedded_guidance` 是内部条件，它和 timestep 一起编码成 AdaLN 的 shift/scale/gate，用来调制每一层 Transformer。简单说，一个在模型外组合输出，一个在模型内控制生成强度。

### Q8：如果只用一句话概括 FLUX.2 Klein？

FLUX.2 Klein 是一个用 Qwen3 文本隐藏层作为条件、用 128 通道 VAE latent token 作为图像表示、通过双流到单流多模态 DiT 做 Flow Matching 去噪的文生图/图像编辑模型。
