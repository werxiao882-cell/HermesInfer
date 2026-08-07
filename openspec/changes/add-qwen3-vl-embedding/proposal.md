# 增加 Qwen3-VL-Embedding-2B 推理支持

@ myvllm/engine
@ myvllm/models
@ myvllm/layers
@ myvllm/multimodal
@ myvllm/vision

## 为什么做

`myvllm` 目前只支持文本因果语言模型生成(Qwen3、Llama-3.2)。我们希望将其扩展到
`Qwen/Qwen3-VL-Embedding-2B` 的多模态 **embedding** 推理——这是一个基于
`Qwen3VLForConditionalGeneration` 架构、2B 参数的视觉-语言 embedding 模型。使用场景是
离线批量编码 文本 / 图像 / 图像+文本 输入为稠密向量,用于检索与相似度计算。

支持该模型有两层叠加动机:

1. **产品价值**:为多模态检索提供一个自托管、依赖轻量的 vLLM / SGLang
   `runner="pooling"` 路径替代方案,契合本项目"从零实现 vLLM"的教学定位。
2. **架构演练**:该模型是 *纯 prefill* 的——它会暴露出引擎里大量 decode 期机制
   (PagedAttention decode kernel、CUDA graph 捕获、连续批处理 preemption)在本工作负载下
   全是死代码,从而强制把 **prefill 引擎核心** 与 **生成/嵌入任务层** 干净地切开。这正是
   vLLM 用 `RunnerType` 枚举形式化的同一条接缝。

模型卡里明确给出了 vLLM 与 SGLang 的用法(`EngineArgs(runner="pooling")` 与
`is_embedding=True`)。我们目标是用 `LLMEngine.encode()` 对齐这个接口面。

## 改动内容

### 新增子包
- `src/myvllm/vision/` —— 3D patch embedding、ViT 编码器(24 层、QK-norm)、
  spatial merger(spatial_merge_size=2)、在 LLM decoder 层 `[5, 11, 17]` 注入的
  deep-stack 特征。
- `src/myvllm/multimodal/` —— 图像预处理(resize/normalize/patchify)、MRoPE 位置 id 计算
  (T/H/W,`mrope_section=[24,20,20]`,interleaved)、每请求多模态数据注册表。
- `src/myvllm/pooling/` —— `EmbeddingHead`(last-token / mean / CLS 池化)、L2 归一化、
  Matryoshka(MRL)维度截断。

### 新增/扩展层
- 扩展 `layers/rotary_embedding.py`,在现有 Llama-3 scaling 分支旁新增 **MRoPE** 模式。
- 扩展 `layers/attention.py` 的 `Attention.forward`,在 `is_prefill` 且模型为 VL 时从
  `Context` 单例读取多模态位置字段。
- ViT 块的注意力层复用现有 `Attention`(ViT 在全 patch 序列上做普通双向注意力,varlen
  flash kernel 直接可用;无 causal mask、无 paged decode)。

### 新增模型
- `src/myvllm/models/qwen3_vl.py` —— `Qwen3VLForEmbedding`,封装
  `Qwen3VLForConditionalGeneration` 拓扑(vision tower + merger + 带 deep-stack 注入的
  28 层文本 decoder + embedding head)。无 LM head、无采样。

### 引擎改动
- 给引擎配置加 `runner_type` 字段:`"generation"`(默认,行为不变)或 `"pooling"`。
- `LLMEngine.encode(inputs)` 新公开 API(对标 `generate`),返回 `(embedding_dim,)`
  的 float 张量列表。
- `Scheduler` 增加 **纯 prefill 模式**:跳过 decode 分支(`scheduler.py:57-93`)、跳过
  `preempt()`、绕过 no-progress guard 对 decode 分支的假设。一个请求恰好在一次 `step()` 完成。
- `ModelRunner.run()` 在 pooling 模式下:跑 prefill,返回池化+归一化后的 embedding,
  而非 logits→采样。跳过 `capture_cudagraph()` 与 `run_model()` 的图重放分支
  (`model_runner.py:354-378,406`)。
- `ModelRunner.__init__` 的模型分发(`model_runner.py:32-69`):新增
  `case "Qwen3-VL-Embedding-2B"` 分支。
- `prepare_prefill`(`model_runner.py:265-314`):构造多模态输入——跑 vision tower、
  merger、把视觉 embedding 散播到 `<|image_pad|>` token 位置、计算每 token 的 MRoPE 位置。

### 张量并行(本变更必须真正可用)
- 拓扑:文本 decoder 沿用现有 TP 线性层(QKV/gate_up 列分片、o/down 行分片、词表分片);
  **ViT 整塔复制**;`DeepstackProj` 与 `EmbeddingHead` 复制;MRoPE 位置不分片;KV cache
  pooling 模式整体跳过。
- 加载器关键修正:VL 路径改走 per-param `weight_loader`(`linear.py` 挂在 param 上的
  callable)而非 `loader.py` 现状的 `torch.cat` 后 `copy_(full)`,使 `world_size>1` 下每 rank
  拿到正确切片。**生成路径的 TP>1 加载现状不动**(留作后续变更,R-7)。
- 多 GPU 执行复用 `LLMEngine` 的 worker spawn + NCCL over `tcp://localhost:12345`;
  `Sequence` 跨进程序列化需附带 `mm_data`。

### 权重加载器改动
- `utils/loader.py` 增加:vision tower 参数(`visual.*`)、merger(`merger.*`)、
  deep-stack 投影权重、embedding head 的加载。VL 文本 decoder 的 QKV/gate_up 改走
  `_load_param` + `weight_loader`;ViT 用标准 `qkv` + `o` + `mlp` 布局,复制全量。

### 入口脚本与 processor
- 新增 `main_embedding.py` 演示(文本 / 图像 / 图像+文本 编码 + 余弦相似度)。
- VL 路径从 `AutoTokenizer` 切到 `AutoProcessor`(负责 chat template + `<|image_pad|>`
  插入 + 像素值抽取)。

## 影响

- **受影响模块**:`engine/`(配置、scheduler 纯 prefill 模式、runner pooling 分支)、
  `layers/rotary_embedding.py`、`layers/attention.py`(读 context)、`utils/loader.py`、
  `utils/context.py`、`models/`,以及上面所有新子包。
- **向后兼容**:现有文本生成路径(`main.py`、`main_llama32.py`、Qwen3/Llama 生成)行为必须
  逐字节不变。Pooling 模式通过 `runner_type` opt-in;默认配置不变。
- **依赖**:新增 `Pillow`、`qwen-vl-utils>=0.0.14`,并把 `transformers` 锁到
  `>=4.57.0`(当前 `pyproject.toml:7-12` 是无上界 `transformers`)。不引入任何 `vllm`
  运行时依赖——`vllm>=0.15.0` 仍只服务于 `benchmark_tps.py`。
- **测试**:新增 `tests/test_pooling_scheduler.py` 与 `tests/test_mrope.py`;现有
  `tests/test_scheduler.py` 必须仍通过(纯 prefill 模式是超集,不是替代)。
- **硬件**:仍是 CUDA-only;vision tower 带来可观的 prefill 内存占用(单图最高可达 1000+
  patch),会与 `allocate_kv_cache` 的容量计算交互——见 design。
- **文档**:`AGENTS.md` 增补 pooling 模式小节;`README` 增补多模态 quickstart。
