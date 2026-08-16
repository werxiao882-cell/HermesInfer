# 增加 Qwen3-VL 生成式推理(文/图 → 文本自回归)+ VL 专项优化

@ myvllm/engine
@ myvllm/models
@ myvllm/layers
@ myvllm/multimodal

## 为什么做

`myvllm` 已有:文本因果 LM 生成(Qwen3/Llama),以及 Qwen3-VL-**Embedding** 的
pooling/纯 prefill 路径(已实现:vision tower、MRoPE、multimodal 预处理、IS_CAUSAL
flash、TP-aware 加载)。本变更新增 **Qwen3-VL 生成式推理**——`Qwen3-VL-2B-Instruct` /
`Qwen3-VL-8B-Instruct`(`Qwen3VLForConditionalGeneration`),输入 文本+图像,自回归
生成文本(带 chat)。

与已实现的 Embedding 路径的区别:Embedding 是**单次 prefill + pooling、无 decode**;本变更是
**完整 decode 链路**(KV cache、连续批、paged decode、CUDA graph、采样)。两者共用同一套
vision tower / MRoPE / multimodal 输入处理(复用已实现代码),区别只在引擎任务层:

- Embedding:`runner_type="pooling"`,跳过 KV/cudagraph/sampler,输出向量。
- 生成式:`runner_type="generation"`(现有),但 prefill 带多模态、decode 走 paged attention,
  输出 token 序列 → 文本。

"并增加优化":VL 推理有几项专项优化机会,本变更一并设计(见 design §4):

1. **视觉 prefix 缓存**:同一图被多请求复用时,只编码一次视觉特征。
2. **分离式视觉编码**(disaggregated):重 vision 计算不阻塞 decode 批。
3. **VL 连续批处理**:prefill(带图)与 decode 交错。
4. **decode CUDA graph** 扩展到 VL(图像 token 的 K/V 已在 cache)。
5. **TP**:vision 复制、text 分片(与 Embedding 同)。

## 改动内容

### 复用已实现(Embedding 路径已建)
- `models/qwen3_vl.py` 的 VisionTower / PatchEmbed3D / MRoPE 位置 / multimodal 预处理 /
  IS_CAUSAL flash / `loader_vl.py` —— 这些**直接复用**,本变更不重写。

### 新增/扩展
- `src/myvllm/models/qwen3_vl_gen.py`(或在 `qwen3_vl.py` 增类)—— `Qwen3VLForCausalLM`:
  在 `Qwen3VLForEmbedding` 基础上加 `lm_head`(`ParallelLMHead`,权重 tying),forward 返回
  logits 而非 embedding;decode 一次出一个 token。
- 引擎:`runner_type="generation"` 路径扩展多模态——
  - `Scheduler`:连续批处理容纳带图序列(图像 token 计入 token 预算、KV block 预算)。
  - `ModelRunner.prepare_prefill`:复用 Embedding 的多模态构造(vision tower + MRoPE 位置 +
    image_token_spans + scatter),但**写 KV cache**(图像 token 的 K/V 进 paged cache,供 decode)。
  - `prepare_decode`:图像 token 不再出现(decode 只喂 last_token),沿用现有 paged decode。
  - `capture_cudagraph`:VL decode batch 形状与纯文本一致,直接复用。
- `utils/loader.py`:生成路径加载器也支持 VL 权重名 remapping(复用 `loader_vl` 的 remap,但
  绕过 weight_loader 的现状不动 → 生成路径 TP>1 仍受 R-7 影响,见风险)。

### 优化(见 design §4)
- `engine/visual_cache.py`:视觉 prefix 缓存(按 pixel_values 哈希缓存 visual_emb)。
- `engine/scheduler.py`:分离式视觉编码调度选项(prefill 前先排 image-encode 批)。

## 影响

- **受影响**:`engine/`(scheduler/model_runner 连续批 + 多模态 prefill 写 KV)、`models/`(新增
  生成式 VL 模型)、`utils/loader.py`(VL 名 remap 用于生成路径)。
- **向后兼容**:纯文本生成路径(Qwen3/Llama)与 Embedding 路径行为不变;VL 生成式经
  `model_name` 分发 `Qwen3-VL-*-Instruct` 触发。
- **依赖**:复用 Embedding 已加的 `transformers>=4.57.0`、`qwen-vl-utils`、`Pillow`。
- **硬件**:CUDA-only;2B 单卡可跑,8B 建议 TP≥2。
- **文档**:`AGENTS.md` 增"VL 生成式"小节;`README` 增 chat quickstart。
