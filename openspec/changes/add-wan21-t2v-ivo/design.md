# 设计:Wan2.1 T2V/I2V DiT 扩散推理(USP 并行 + 长序列)

针对 `proposal.md` 的技术设计。架构参数来自已确认的 HF `config.json`;DiT block 内部
的精确调制方式在实现前需从 Wan2.1 官方 `wan/modules` 源码核对(见风险 R-1)。

## 1. 背景

### 1.1 模型(Wan2.1)

`Wan-AI/Wan2.1` 系列是阿里开源的视频生成 DiT,用 rectified flow(流匹配)训练。
diffusers 中类名 `WanModel`,`model_type` 区分 `t2v`/`i2v`。已确认配置:

| 模型 | dim | ffn_dim | freq_dim | in_dim | out_dim | num_heads | num_layers | text_len | eps |
|---|---|---|---|---|---|---|---|---|---|
| T2V-1.3B | 1536 | 8960 | 256 | 16 | 16 | 12 | 30 | 512 | 1e-6 |
| T2V-14B | 5120 | 13824 | 256 | 16 | 16 | 40 | 40 | 512 | 1e-6 |
| I2V-14B-480P | 5120 | 13824 | 256 | **36** | 16 | 40 | 40 | 512 | 1e-6 |
| I2V-14B-720P | 5120 | 13824 | 256 | **36** | 16 | 40 | 40 | 512 | 1e-6 |

要点:
- `in_dim=16`(T2V)= VAE latent 通道数;`in_dim=36`(I2V)= 16 latent + 16 图像 VAE 编码 +
  4 mask 通道,沿通道拼接注入。
- `freq_dim=256`:timestep/3D RoPE 的频率嵌入维度。
- `out_dim=16`:DiT 输出回到 VAE latent 空间。
- `text_len=512`:文本 token 上限(umt5-xxl,多语)。
- VAE:Wan2.1 3D 因果 VAE,16 通道,**空间下采样 8×、时间下采样 4×**(causal 3D conv)。
- 文本编码器:Google `umt5-xxl`(多语);I2V 另用 CLIP 视觉编码器提供图像 pooled 特征
  做 cross-attn(R-1 待核对是否 CLIP 还是 VAE-pooled)。

### 1.2 DiT 结构(待 R-1 核对精确调制)

Wan2.1 DiT 一次去噪步的大致数据流:

```
z_t (latent: T', H', W', 16)  ──patchify(1,2,2)──►  tokens (N, dim)
                                                     + timestep emb (freq_dim→dim)
                                                     + 3D RoPE pos (T,H,W)
for block in num_layers:
    h = block(h, t_emb, text_emb, [image_emb])
      ├ adaLN-modulate(scale/shift)  (R-1: 是否 adaLN 还是 inject)
      ├ USP self-attn (full, non-causal)  ──► all_to_all x2
      ├ adaLN-modulate
      ├ cross-attn to text_emb (replicated, text_len=512 短)
      └ MLP (SiLU, USP sequence-sharded)
unpatchify → eps/v 预测 (T', H', W', out_dim=16)
flow-matching update:  z_{t-dt} = z_t + dt * (v_pred)
```

patchify 用 `(1,2,2)`:时间不切、空间 2×2。720P×81f 经 VAE 后 latent
`(21, 60, 120, 16)`(时间 81→21@4×,空间 480→60@8×、x 720→120),patchify 后 token 数
`21 × 30 × 60 = 37800`。自注意力 O(N²)≈1.4e9 per head —— 单卡不可行,必须 USP。

### 1.3 USP(Ulysses Sequence Parallelism)

USP 把序列维度切到 P 卡、把头维度留在卡内,通过两次 `all_to_all` 在"序列分片×全头"
与"全序列×头分片"间转置:

```
输入:  [seq/P,  heads,    dim]   (每 rank 持有 1/P 序列、全头)
all_to_all ─────────────────────►  [heads/P, seq, dim]   (每 rank 持有 全序列、1/P 头)
                                            │
                                   标准 flash attention(本 rank 的头子集,全序列)
                                            │
all_to_all ◄─────────────────────────────  回到 [seq/P, heads, dim]
```

要点:
- 要求 `num_heads % P == 0`:14B(40 头)→ P ∈ {1,2,4,5,8,10,20,40};1.3B(12 头)→
  P ∈ {1,2,3,4,6,12}。本项目 DiT 走 USP(不走 TP 列分片)。
- 跨模态 cross-attn(text_len=512,短)**不走 USP**,每 rank 复制全文本;MLP 在 USP 下
  是序列分片的逐 token MLP(无需通信)。
- USP 与训练对称:权重在训练时即按此布局,加载时按 `num_heads//P` 把每头权重分发到
  对应 rank(R-2)。
- 通信量 ≈ `2 × bsz × seq × dim × 4B`(两次 all-to-all),随 seq 线性(非平方),这是
  USP 对长序列的关键优势。

### 1.4 长序列

| 场景 | latent | patch token N | 单头 attn 成本 | 单卡可行性 |
|---|---|---|---|---|
| 480P × 81f | (21,30,53,16) | ~8.5k | O(7e7) | 单卡可行 |
| 720P × 81f | (21,60,120,16) | ~19k-38k | O(1.4e9) | 需 USP P≥4 |
| 720P × 480f(超长) | (120,60,120,16) | ~108k | O(1e10) | USP + ring/chunk |

长序列策略分层:
- **USP 主路径**:P 卡摊平 O(N²/P),覆盖官方默认分辨率(480P/720P×81f)。
- **ring attention(可选,v2)**:对超长(数百帧),用 zigzag 序列切分 + ring 通信的
  flash attention,使每卡 O(N²/P) 且总序列长度无界(P 越大越长)。
- **context chunking(可选,v2)**:对极长视频用滑窗上下文注意力,牺牲少量质量换显存。

## 2. 目标与非目标

### 目标
- T2V 与 I2V 两条 pipeline,输入 prompt(+ 图像)、输出 mp4 tensor,数值与 diffusers
  `WanPipeline` 在 FP 容差内一致。
- USP 并行:world_size>1 时每 rank 持有序列分片×全头 → all-to-all → 全序列×头分片做
  flash attention;单卡与多卡结果一致(`1e-4`)。
- 长序列:720P×81f 在 8×4090/8×A100 可跑;USP 线性扩展到更长。
- 复用现有 flash varlen kernel(非因果)、NCCL 组、`@torch.compile` 热路径惯例。
- 支持自定义采样步数与 shift 超参。

### 非目标(v1)
- **训练**:仅推理;USP 路径与训练对称(故"训练并行"可复用),但不实现反向/优化器。
- **蒸馏少步权重**:v1 用官方 ~50 步;4 步 Lightning 等留待社区权重出现。
- **ring attention**:v2;v1 用 USP 覆盖官方默认分辨率。
- **音频**:Wan2.1-Audio 不在范围。
- **CPU/MPS**:CUDA-only。

## 3. 提议架构

### 3.1 数据流(T2V 一次采样步)

```
prompt ─► umt5(text_len=512) ─► text_emb (512, dim_text)   [replicated]
t (scalar timestep) ─► timestep_emb (freq_dim→dim)         [replicated]
z_t (latent: T',H',W',16) ──patchify(1,2,2)──► tokens (seq//P, dim)   [USP shard]
for block:
  ├ adaLN-modulate(tokens, t_emb)
  ├ USPAttention(tokens, 3D-RoPE) ─ all_to_all×2 ─ flash(non-causal)
  ├ adaLN-modulate
  ├ cross-attn(tokens, text_emb)            [replicated, text 短]
  └ MLP(tokens)                              [USP shard, 无通信]
unpatchify → v_pred (seq//P, out_dim=16)
flow update: z_{t-dt} = z_t + dt * v_pred      [per-rank shard 更新本地 latent 分片]
循环 ~50 步 → z_0 → VAE.decode → (T, 3, H, W) → mp4
```

I2V 区别:`in_dim=36`,patchify 前把参考图的 VAE 编码(latent 16ch)+ mask(4ch)与噪声
latent(16ch)沿通道拼成 36ch;额外 CLIP 图像 pooled 特征做一次 cross-attn(R-1 核对)。

### 3.2 新增子包

#### `src/myvllm/diffusion/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 重导出 `WanDiT`、`WanVAE`、`FlowScheduler`、`T2VPipeline`、`I2VPipeline` |
| `dit.py` | `WanDiT`:patch embed、timestep emb、N 层 `WanDiTBlock`、unpatchify;按 config 构 1.3B/14B |
| `block.py` | `WanDiTBlock`:adaLN 调制 + USP self-attn + cross-attn + MLP(R-1 核对调制) |
| `vae.py` | `WanVAE`:3D 因果 VAE encode/decode,空间 8×、时间 4× 下采样,16 通道 |
| `scheduler.py` | `FlowScheduler`:rectified flow,shifted,1→0,可配步数;`step(z_t, v, t)` |
| `pipeline.py` | `T2VPipeline`/`I2VPipeline`:文本/图像编码 → 采样循环 → VAE decode → 视频 |
| `patch_embed.py` | `PatchEmbed3D`:`(1,2,2)` patchify + unpatchify |

#### `src/myvllm/usp/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 重导出 `USPAttention`、`usp_group()`、`all_to_all_seq2head`/`head2seq` |
| `ulysses.py` | `all_to_all_seq2head(x)` / `head2seq(x)`:基于 `dist.all_to_all` 的转置;校验 `num_heads % P == 0` |
| `usp_attention.py` | `USPAttention`:forward 做 seq2head → flash(non-causal, varlen)→ head2seq;支持 3D RoPE |
| `group.py` | `init_usp_group(world_size, rank)`:新建 NCCL 组(不与 TP 组共用,避免 all_reduce 干扰) |

### 3.3 层扩展

#### 3D 视频 RoPE(`layers/` 新增 `video_rope.py`)

Wan2.1 用 `freq_dim=256` 的 3D 频率:时间轴与 H/W 轴各自一组频率,与 Qwen3-VL 的 MRoPE
思路类似但**非交错、非按文本段**——直接对每个 token 的 (T,H,W) 坐标做
`freq(t)⊕freq(h)⊕freq(w)` 拼接到 head_dim。R-1 核对确切布局(可能 Wan 用 sin/cos 时间嵌入
而非 RoPE;`freq_dim` 既给 timestep 又给位置,需分清)。

#### `flash_attention_prefill` 复用

DiT 自注意力是**全序列非因果**,直接用 `flash_attention_prefill(..., is_causal=False)`
(Qwen3-VL 已加该开关)。cu_seqlens 在 USP 下:序列分片仍是变长(多视频 batched),用
`cu_seqlens` 划分。

### 3.4 USP 注意力(`usp/usp_attention.py`)

```python
class USPAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        P = usp_world_size()
        assert num_heads % P == 0
        self.local_heads = num_heads // P
        # 每张卡只持有 local_heads 个头的 q/k/v/o 投影权重(加载时按 head 切)
        self.qkv = nn.Linear(dim, 3 * self.local_heads * head_dim)
        self.o = nn.Linear(self.local_heads * head_dim, dim)

    def forward(self, x, pos):
        # x: (seq//P, heads, dim)  实际进入时已是 (seq//P, dim)
        q, k, v = split(self.qkv(x))                       # (seq//P, local_heads, head_dim)
        q = apply_3d_rope(q, pos)
        k = apply_3d_rope(k, pos)
        # seq2head: 全头分到各 rank -> 各 rank 全序列、本 rank 头子集
        q = all_to_all_seq2head(q)   # -> (local_heads, seq_full, head_dim) on this rank
        k = all_to_all_seq2head(k)
        v = all_to_all_seq2head(v)
        o = flash_attention_prefill(q, k, v, cu_seqlens, scale,
                                    num_heads=local_heads, num_kv_heads=local_heads,
                                    head_dim=head_dim, is_causal=False)
        o = all_to_all_head2seq(o)   # -> (seq//P, local_heads, head_dim)
        return self.o(o.flatten(-2))
```

权重加载:DiT 的 q/k/v/o 在 USP 下按 `head` 切——每 rank 只装 `num_heads//P` 个头的权重
(R-2)。这与 TP 列分片不同(USP 切的是头,不是输出列),需加载器新增 USP 分发逻辑。

### 3.5 引擎:`DiffusionEngine`

不沿用 `LLMEngine` 的 scheduler/decode(扩散非自回归)。结构:

```python
class DiffusionEngine:
    def __init__(self, config):
        init_usp_group(world_size, rank)
        self.dit = WanDiT(config).cuda(rank)
        self.vae = WanVAE(...).cuda(rank)
        self.text_encoder = AutoModel.from_pretrained(umt5).cuda(rank)  # 复用 transformers
        self.flow = FlowScheduler(steps=config["steps"], shift=config["shift"])

    def t2v(self, prompt, *, size, num_frames, ...) -> Tensor: ...
    def i2v(self, image, prompt, *, size, num_frames, ...) -> Tensor: ...
```

每次采样:文本/图像编码(一次)→ 初始化 `z_1 ~ N(0,I)` 并按 USP 切片 → 循环 `flow` 步数
跑 DiT → VAE decode(每 rank 解码本 rank 的时空分片后 all-gather,或仅 rank0 解码)。

### 3.6 权重加载(`utils/loader.py`)

- DiT:USP 分发——按 `head` 把 q/k/v/o 权重切到对应 rank(每 rank `num_heads//P` 头)。
- VAE / T5 / CLIP:**复制**(不分片,体量小或短序列)。
- 新增 Wan 加载分支,保留现有 LLM/embedding 路径不动。

## 4. USP 并行拓扑

| 部件 | 策略 | 理由 |
|---|---|---|
| DiT 自注意力 q/k/v/o | **USP 头分片**(每 rank `num_heads//P` 头,全序列) | O(N²/P),all-to-all 通信 O(N) |
| DiT 自注意力激活 | 序列分片(`seq//P`)进入、全序列×头子集计算 | 见 3.4 |
| DiT cross-attn | **复制**(全文本,文本 512 短) | 通信不划算 |
| DiT MLP / adaLN / norm | **USP 序列分片**(逐 token,无通信) | MLP 逐 token,序列分片即逐 token 分片 |
| timestep emb / text emb / image pooled | **复制** | 标量/短序列 |
| VAE encode/decode | rank0 解码后 all-gather(或每 rank 解码本地时空分片) | VAE 非 DiT,不复用 USP |
| 文本/图像编码器 | rank0 跑 → broadcast | 一次性,小开销 |

## 5. 长序列策略(分层)

1. **USP(v1,必做)**:P 卡摊平 attn 到 O(N²/P),覆盖 480P/720P×81f。
2. **ring attention(v2)**:zigzag 切序列 + ring 通信 flash,使总长度随 P 无界增长
   (数百帧);复用现有 flash kernel 的 online softmax,ring 方向做 KV 块传递。
3. **context chunking(v2,可选)**:超长视频滑窗上下文注意力,牺牲少量质量换显存。
4. **序列打包**:batched 多视频用 `cu_seqlens` 变长 flash(复用现有),USP 下每 rank
   仍是变长。

## 6. 测试计划

| 测试 | 文件 | 守护 |
|---|---|---|
| `all_to_all_seq2head`/`head2seq` 往返保形 + 保数值 | `tests/test_usp.py` | §3.4 转置 |
| USP 与单卡注意力等价(P=2,随机权重,固定 seed) | `tests/test_usp.py` | USP 正确性 |
| `FlowScheduler` 1→0 单调、shift 生效、自定义步数 | `tests/test_flow.py` | 调度 |
| WanVAE encode→decode 时空形状往返(空间8×、时间4×) | `tests/test_vae.py` | VAE 下采样 |
| `WanDiT` 一次去噪:z_t→v_pred 形状 `(seq//P, out_dim=16)` | `tests/test_dit.py` | DiT 前向 |
| T2V/I2V 端到端 mock(无真实权重,小 dim) | `tests/test_pipeline.py` | pipeline 形状 |
| 数值对齐 diffusers `WanPipeline`(@gpu,小分辨率短步) | `tests/test_parity_wan.py` | 余弦/误差容差 |
| 单卡 vs 8 卡输出一致 `1e-4` | `tests/test_usp.py`(@gpu) | USP 数值一致 |
| 现有 LLM/embedding 测试全绿 | (不变) | 回归 |

## 7. 风险与开放问题

- **R-1 —— DiT block 精确调制与 RoPE 布局**。Wan2.1 官方 `wan/modules` 的 block 结构
  (adaLN vs 直接 inject、3D RoPE vs 纯 timestep freq、cross-attn 用 umt5 还是 CLIP 图像
  特征)未从源码核对。`freq_dim=256` 同时用于 timestep 与可能的位置,需读源码分清。实现前
  必须核对,否则数值对齐失败。
- **R-2 —— USP 权重按 head 分发**。加载器需把 q/k/v/o 的 `num_heads` 维按 `head_idx % P`
  切到对应 rank。若权重布局是 `[3, num_heads, head_dim, dim]`(QKV 合并)则按 head 切第 1 维;
  需核对 checkpoint 的实际权重名与布局。
- **R-3 —— VAE 的 USP 兼容**。3D 因果 VAE 的 encode/decode 在时间轴有因果依赖,不能简单
  按时间切片并行;v1 让 rank0 跑 VAE 再 broadcast latent(简单但非并行),v2 再做时空分片。
- **R-4 —— `num_heads % world_size` 约束**。14B=40 头,P∈{1,2,4,5,8,10,20,40};用户若要
  P=3 则需 1.3B(12 头)或 repad。文档需明示。
- **R-5 —— 显存峰值**。720P×81f 即便 USP P=8,每 rank flash 的 `seq_full` 仍 ~38k,O(N²/P)
  attention 中间量在 head_dim=128、heads=5 时约 `5×38k²×128×4B ≈ 4.6GB`,需 4090 24GB 或
  A100。用 ring 或降分辨率兜底。
- **R-6 —— T5/CLIP 依赖体积**。umt5-xxl ~10GB,需下载;v1 支持本地路径与镜像端点。

## 8. 启动命令与调用方式

### 8.1 启动

```bash
uv sync                       # 新增 diffusers/transformers/imageio/einops
# 模型获取(USP 下每 rank 都需权重,USP 分发在内存)
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run python -c "from huggingface_hub import snapshot_download; [snapshot_download(r, allow_patterns=['*.safetensors','*.json','*.txt']) for r in ['Wan-AI/Wan2.1-T2V-14B','Wan-AI/Wan2.1-I2V-14B-720P']]"
uv run python main_wan_t2v.py             # 8 卡示例(USP)
uv run python main_wan_i2v.py
uv run pytest tests/test_usp.py tests/test_flow.py -v
```

### 8.2 调用

```python
config = {
    "model_name_or_path": "Wan-AI/Wan2.1-T2V-14B",
    "world_size": 8, "runner_type": "diffusion",
    "diffusion": {"steps": 50, "shift": 5.0},
    "size": (1280, 720), "num_frames": 81,
}
engine = DiffusionEngine(config)
video = engine.t2v("a cat playing piano on a beach at sunset")   # (T,3,H,W) tensor
# I2V
engine = DiffusionEngine({**config, "model_name_or_path":"Wan-AI/Wan2.1-I2V-14B-720P"})
video = engine.i2v(image="./assets/cat.jpg", prompt="the cat starts playing piano")
```

## 9. 被否决的替代方案

- **直接用 diffusers `WanPipeline`**。否决:项目定位是自研 Triton 内核 + USP 栈;但可借
  diffusers 的 VAE/调度器作参考实现核对。
- **用 TP(列分片)而非 USP**。否决:TP 切权重列不解决 O(N²) 长序列;USP 才是序列维度
  并行,匹配 Wan 训练时的切分。
- **v1 即上 ring attention**。否决:实现复杂度大;USP 已覆盖官方默认分辨率,ring 留 v2。
- **训练支持**。超出范围;USP 路径训练/推理对称故"训练并行"可复用,但不实现反向。
