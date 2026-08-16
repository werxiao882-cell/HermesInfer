# 增加 Wan2.1 T2V/I2V DiT 扩散推理支持(USP 并行 + 长序列)

@ myvllm/diffusion
@ myvllm/usp
@ myvllm/layers
@ myvllm/models

## 为什么做

`myvllm` 现在是文本因果 LM / 多模态 embedding 的 prefill 引擎。本变更新增**视频扩散**
推理能力,支持 `Wan-AI/Wan2.1` 系列的 **T2V(文生视频)** 与 **I2V(图生视频)** DiT 模型,
并要求**真正可用**的两项硬约束:

1. **USP(Ulysses Sequence Parallelism)并行** —— Wan2.1 官方训练即用 USP
   (`usp` / `ring-flash-attn`)把超长 spatiotemporal token 序列切分到多卡。USP 在训练/推理
   是对称的(同样的 all-to-all 转置),所以"训练并行"与"推理并行"可共用同一套注意力层。
   我们在推理侧实现 USP,使其与权重训练时的切分方式一致。
2. **超长序列** —— 720P × 81 帧经 VAE 后 latent token 数已达 ~1.9 万,自注意力 O(N²)
   单卡显存/算力不可行;更长视频(数百帧)更需要 USP + 可选 ring/chunked 注意力扩展。

动机:

- **产品价值**:为 `myvllm` 增加视频生成能力,对标 diffusers `WanPipeline` 但用项目自研的
  Triton 内核与 TP/USP 并行栈,而非直接依赖 diffusers。
- **架构演练**:扩散 DiT 与因果 LM 在 prefill 引擎上有同构点(变长 flash 注意力、3D RoPE),
  但多了流匹配调度、VAE、跨模态条件注入、USP 并行这些新部件,把 `prefill 核心` 与 `任务层`
  的接缝进一步泛化。

## 改动内容

### 新增子包
- `src/myvllm/diffusion/` —— Wan2.1 DiT 模型、3D VAE、流匹配调度器、T2V/I2V pipeline。
- `src/myvllm/usp/` —— Ulysses 序列并行:基于 `dist.all_to_all` 的注意力转置层
  (sequence↔heads),供 DiT 自注意力使用;FSDP 兼容的权重分片辅助。

### 新增/扩展层
- 复用 `layers/attention.py` 的 flash varlen kernel(非因果分支 `IS_CAUSAL=False`,
  Qwen3-VL 已加)做 DiT 全 spatiotemporal 自注意力。
- 复用 `layers/rotary_embedding.py` 的思路新增 **3D 视频 RoPE**(freq_dim=256,时间/高/宽
  三轴频率)。
- 新增 `layers/usp_attention.py`:`USPAttention`,在自注意力前后做 all-to-all 把
  `[seq//P, heads, dim]` ↔ `[heads//P, seq, dim]` 转置,使每 rank 持有**全序列**但只负责
  **部分头**,从而把 O(N²) 摊到 P 卡。

### 新增模型
- `src/myvllm/models/wan.py` —— `WanDiT`(对应 diffusers `WanModel`):
  patch embed(1,2,2 patchify)、timestep emb、文本/图像 cross-attn、N 层 DiT block、
  unpatchify → out_dim=16。I2V 用 in_dim=36(16 latent + 16 图像 VAE + 4 mask)。

### 引擎与 pipeline
- 新增 `DiffusionEngine`(与 `LLMEngine` 并列),不沿用连续批处理/scheduler/decode 那套
  (扩散是固定步数迭代采样,非 token 自回归)。提供 `t2v(prompt, ...)` 与 `i2v(image, prompt, ...)`
  两个入口。
- 流匹配调度器:rectified flow,t: 1→0,shifted(sigma),默认 ~50 步;支持自定义步数与
  蒸馏少步(如 4 步 Lightning 风格,若社区出蒸馏权重)。

### 权重与多卡
- 新增 `utils/loader.py` 的 Wan 加载分支:DiT/VAE/T5/CLIP 分别路由;USP 下 DiT 权重按
  head 切分加载(`num_heads % world_size == 0`)。
- USP 并行组复用现有 NCCL(`tcp://localhost:12345`)或新建组;USP 的 all-to-all 与现有
  TP 的 all_reduce 不冲突(USP 切序列/头,TP 切权重列,二者择一,本项目 DiT 走 USP)。

## 影响

- **受影响模块**:`layers/`(新增 usp_attention、3D RoPE)、新增 `diffusion/`、`usp/`、
  `models/wan.py`、`utils/loader.py`、`utils/context.py`(扩散 context 字段)。
- **向后兼容**:文本生成 / embedding 路径行为不变;`runner_type` 新增 `"diffusion"`。
- **依赖**:新增 `diffusers`(仅借用其 VAE/调度器参考实现,可改为自研)、`transformers`
  (T5/CLIP 文本与图像编码器)、`imageio`/`torchvision`(视频 IO)、`einops`。`flash-attn`
  可选(USP 路径用自研 Triton)。
- **硬件**:CUDA-only;USP 要求 `world_size ≤ num_heads`(14B:40 头,实际 ≤8;
  1.3B:12 头,≤12)。长序列显存随 `seq² / world_size` 缩放。
- **文档**:`AGENTS.md` 增"Diffusion / USP"小节;`README` 增 T2V/I2V quickstart。
