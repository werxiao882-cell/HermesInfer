# 实施任务:Wan2.1 T2V/I2V DiT 扩散推理(USP + 长序列)

按依赖顺序;每条标注设计 § 与风险 R。先调研核对,再 USP 基础设施,再 DiT/VAE/pipeline,
最后对齐与回归。

## 1. 调研与依赖

- [ ] 读 Wan2.1 官方 `wan/modules` 源码,核对 DiT block 精确结构:adaLN 调制方式、3D RoPE
      布局 vs timestep freq、cross-attn 用 umt5 还是 CLIP(R-1);回填 design §3.3、§3.4
- [ ] 核对 checkpoint 权重名与 QKV 合并布局,确定 USP 按 head 切的维度(R-2)
- [ ] `pyproject.toml` 新增 `diffusers`(参考)、`transformers`(T5/CLIP)、`imageio`、
      `torchvision`、`einops`;`uv sync`
- [ ] `AGENTS.md` 增"Diffusion / USP"小节占位

## 2. USP 基础设施(`usp/`)

- [ ] `usp/group.py`:`init_usp_group(world_size, rank)` 新建 NCCL 组(不与 TP 组共用)
- [ ] `usp/ulysses.py`:`all_to_all_seq2head` / `all_to_all_head2seq`,校验 `num_heads % P == 0`
- [ ] `usp/usp_attention.py`:`USPAttention` forward:seq2head → flash(非因果)→ head2seq
- [ ] `tests/test_usp.py`:往返保形保值;USP 与单卡等价(P=2,固定 seed)

## 3. 层扩展

- [ ] `layers/video_rope.py`:3D 视频 RoPE(freq_dim=256,T/H/W 坐标);R-1 核对布局
- [ ] 确认 `flash_attention_prefill(..., is_causal=False)` 在 DiT 全序列上正确(已由
      Qwen3-VL 验证非因果分支)
- [ ] `layers/__init__.py` 重导出 USP/3D RoPE

## 4. Wan DiT 模型(`models/wan.py` + `diffusion/`)

- [ ] `diffusion/patch_embed.py`:`PatchEmbed3D` patchify(1,2,2)/unpatchify
- [ ] `diffusion/block.py`:`WanDiTBlock`:adaLN 调制 + USP self-attn + cross-attn + MLP
      (按 R-1 结论实现调制)
- [ ] `diffusion/dit.py`:`WanDiT`:patch embed + timestep emb + N 层 block + unpatchify,
      按 config 构 1.3B/14B、t2v(in_dim=16)/i2v(in_dim=36)
- [ ] `tests/test_dit.py`:一次去噪 z_t→v_pred 形状 `(seq//P, out_dim=16)`

## 5. VAE 与调度器(`diffusion/`)

- [ ] `diffusion/vae.py`:`WanVAE` 3D 因果 encode/decode,空间 8×、时间 4×,16 通道
- [ ] `diffusion/scheduler.py`:`FlowScheduler` rectified flow 1→0、shifted、可配步数
- [ ] `tests/test_vae.py`:encode→decode 时空形状往返
- [ ] `tests/test_flow.py`:1→0 单调、shift 生效、自定义步数

## 6. 文本/图像编码器与 pipeline(`diffusion/pipeline.py`)

- [ ] 复用 `transformers` 加载 umt5-xxl(文本)、CLIP(I2V 图像 pooled)
- [ ] `T2VPipeline`:文本编码 → init z_1 → USP 切片 → 采样循环 → VAE decode → 视频
- [ ] `I2VPipeline`:图像 VAE 编码 + mask 拼通道(in_dim=36)+ CLIP pooled cross-attn
- [ ] `tests/test_pipeline.py`:T2V/I2V 端到端 mock(无真实权重,小 dim)

## 7. 引擎与权重加载

- [ ] `DiffusionEngine`:`t2v()` / `i2v()` 入口;`runner_type="diffusion"`
- [ ] `utils/loader.py` Wan 分支:DiT 权重按 head 切到对应 rank(USP)、VAE/T5/CLIP 复制
- [ ] VAE encode/decode 的多卡策略(v1:rank0 解码后 broadcast;R-3)

## 8. 长序列扩展(v2,可选)

- [ ] `usp/ring_attention.py`:zigzag 序列切分 + ring 通信 flash(数百帧)
- [ ] context chunking 滑窗注意力(超长视频兜底)

## 9. 启动脚本与文档

- [ ] `main_wan_t2v.py`、`main_wan_i2v.py`(8 卡 USP 示例)
- [ ] `README.md` 增 T2V/I2V quickstart;`AGENTS.md` 补全"Diffusion / USP"小节
      (USP 拓扑表 §4、num_heads%P 约束 R-4)

## 10. 数值对齐与回归

- [ ] `tests/test_parity_wan.py`(@gpu,小分辨率短步):与 diffusers `WanPipeline` 误差容差
- [ ] 单卡 vs 8 卡输出 `1e-4` 一致(@gpu)
- [ ] 显存峰值验证:720P×81f @ 8×4090 或 8×A100(R-5)
- [ ] 现有 LLM/embedding 测试全绿(回归)

## 11. 收尾

- [ ] `git add -A && git commit -m "Add Wan2.1 T2V/I2V DiT diffusion inference with USP"`
- [ ] 评审:对照本清单逐条 `[ ]` → `[x]`
