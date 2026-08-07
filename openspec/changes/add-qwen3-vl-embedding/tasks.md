# 实施任务:Qwen3-VL-Embedding-2B 推理支持

按依赖顺序排列的实现清单。每条任务末尾标注对应的设计章节(§)与风险项(R)。先做
调研与基础设施,再做视觉/多模态层,最后接引擎与数值对齐。

## 1. 调研与依赖奠基

- [ ] 读 HF 仓库 `1_Pooling/config.json` 与 `sentence_bert_config.json`,确认 embedding head
      是否带投影层(R-1);把结论回填到 design §3.2 `embedding_head.py` 一行
- [ ] 通读 `transformers>=4.57.0` 的 `Qwen3VLModel` 源码,记录 MRoPE 位置规则
      (T 是否每文本 token 前进、图像 token 的 T/H/W 派生方式)(R-4)
- [ ] 通读 `layers/attention.py:112` `flash_attention_varlen_kernel`,确认 prefill 是否带
      causal mask(R-3);把结论回填到 design §3.3
- [ ] 核查 TP 加载现状:确认 `loader.py` 绕过 per-param `weight_loader`(`loader.py:94,119,165`)
      导致生成路径 `world_size>1` 加载期就坏(§1.2、R-7);设计只对 VL 路径改走
      `weight_loader`(§3.8.2),记录生成路径 TP>1 为已知遗留
- [ ] 读 `linear.py:84-230` 的 `ColumnParallelLinear`/`QKVColumnParallelLinear`/
      `MergedColumnParallelLinear`/`RowParallelLinear` 的 `weight_loader` 切片与
      GQA 头边界,确认 VL 复用方式(§3.8.2)
- [ ] 在 `pyproject.toml`(`pyproject.toml:7-12`)新增 `Pillow`、`qwen-vl-utils>=0.0.14`,
      锁 `transformers>=4.57.0`;`uv sync` 重新生成 `uv.lock`;跑 `uv run python main.py`
      冒烟(R-6)
- [ ] 在 `AGENTS.md` Architecture 下新增"Pooling 模式"小节占位,列出待跳过的子系统

## 2. 旋转位置嵌入 MRoPE 扩展(§3.3)

- [ ] 给 `RotaryEmbedding`(`rotary_embedding.py:48`)加 `rope_type` 构造参数,
      支持 `"default" | "llama3" | "mrope"`
- [ ] 实现 `mrope` 分支:接收 3D 位置 `(T, H, W)`,按 `mrope_section=[24,20,20]` 切段,
      按 `mrope_interleaved=true` 做交错 cos/sin 布局
- [ ] 扩展 `apply_rotary_pos_emb`(`rotary_embedding.py:4`)按 shape rank 分发到 3D 位置形式,
      保留现有 3D varlen / 4D batched 行为
- [ ] 写 `tests/test_mrope.py`:cos/sin 分段与交错 vs 手算参考;
      `compute_mrope_positions` 对纯文本与含图 prompt 的位置正确性

## 3. 多模态输入管线(§3.2 multimodal/)

- [ ] 新建 `src/myvllm/multimodal/processor.py` 的 `build_image_inputs`:用 HF
      `AutoImageProcessor` 做 resize/normalize,patchify 出 `image_grid_thw` 与
      pinned `pixel_values`
- [ ] 新建 `src/myvllm/multimodal/positions.py` 的 `compute_mrope_positions`:产出
      `(T, H, W)` 三组 `(num_tokens,)` 位置,对标 §3.2 与 R-4 调研结论
- [ ] 新建 `src/myvllm/multimodal/registry.py` 的 `MultimodalRegistry`,挂在 `Sequence`
      上(新增可选字段)
- [ ] `Sequence.__getstate__`/`__setstate__`(`sequence.py:88-114`)确保 `mm_data`
      跨进程 pickle 正确(rank-0 → worker 的 H2D 前只传 `pixel_values`)

## 4. 视觉塔(§3.2 vision/)

- [ ] 新建 `vision/patch_embed.py` 的 `PatchEmbed3D`:卷积 + 3D 位置嵌入
      (`num_position_embeddings=2304`)
- [ ] 新建 `vision/vit.py` 的 `Qwen3VLVisionBlock` ×24:自注意力(无 causal mask)+
      MLP(`gelu_pytorch_tanh`)+ RMSNorm + QK-norm;复用 `layers/attention.py:Attention`
- [ ] 新建 `vision/merger.py` 的 `SpatialMerger`:`spatial_merge_size=2` → 线性
      `4*1024 → 2048`
- [ ] 新建 `vision/deepstack.py` 的 `DeepstackProj`:线性 `1024 → 2048`,作用于选定中间特征
- [ ] 新建 `vision/vision_tower.py` 的 `VisionTower`:编排 patch embed → 24 block →
      返回中间特征列表 + merged embedding
- [ ] 写 `tests/test_vision.py`:`SpatialMerger` shape 算术;deepstack 恰在 `{5,11,17}`
      注入;ViT 一次 varlen prefill 跑通

## 5. 新模型 `models/qwen3_vl.py`(§3.4)

- [ ] 实现 `Qwen3VLForEmbedding`:decoder 栈对标 `qwen3.py:285`,但用 MRoPE、在
      `{5,11,17}` 加 `DeepstackProj` 残差注入、无 `lm_head`、保留 Q/K-norm
- [ ] 扩展 `packed_module_mapping` 加入 `visual.*` / `merger.*` / `deepstack.*` 条目
- [ ] `models/__init__.py`(当前空)按需重导出

## 6. Pooling 输出头(§3.2 pooling/)

- [ ] 新建 `pooling/pooling.py`:`Pooling` 枚举(`LAST_TOKEN`/`MEAN`/`CLS`);
      `last_token` 用 `cu_seqlens_q[1:]-1` gather
- [ ] 新建 `pooling/embedding_head.py` 的 `EmbeddingHead`:按 R-1 结论决定是否带投影;
      池化 → L2 → 可选 MRL
- [ ] 新建 `pooling/mrl.py` 的 `mrl_truncate`:切前 `dim` 维并重新归一化

## 7. 引擎 pooling 路径(§3.5)

- [ ] 给 `LLMEngine` 配置加 `runner_type`/`pooling`/`multimodal` 字段(§3.5.1)
- [ ] `Scheduler`(`scheduler.py:6,35`)加纯 prefill 模式:始终走 prefill 分支、
      `running` 视空、`preempt()` 加 assert、绕过 no-progress guard 返回完成信号
- [ ] `BlockManager` 在 pooling 模式下的 `allocate`/`deallocate` 处理(R-2,任务 T-9 定夺):
      优先用 `runner_type` 守卫干净跳过;次选 dummy block
- [ ] `ModelRunner.__init__`(`model_runner.py:16,32-69`)新增 `case "Qwen3-VL-Embedding-2B"`,
      构造 `Qwen3VLForEmbedding`
- [ ] pooling 模式下跳过 `allocate_kv_cache()`(`model_runner.py:197`)与
      `capture_cudagraph()`(`model_runner.py:406`);`run_model` 改为直接
      `model(**inputs)`(`model_runner.py:354-378`)
- [ ] `prepare_prefill`(`model_runner.py:265-314`)加多模态分支:跑 vision tower、
      scatter visual_emb 到 `<|image_pad|>` span、算 3D 位置、`set_context`
- [ ] `ModelRunner.run()`(`model_runner.py:386`)pooling 分支:返回 pooled embedding
      (rank-0,对标 `model_runner.py:394`)
- [ ] 新增 `LLMEngine.encode(inputs)` 公开 API(§3.5.4):`AutoProcessor` chat template +
      `MultimodalData` + `step()` 循环到完成
- [ ] `Sequence.__getstate__`/`__setstate__`(`sequence.py:88-114`)在 prefill 分支
      附带 `mm_data`(`pixel_values`+`image_grid_thw`+`image_token_spans`),pin_memory
      零拷贝跨进程序列化(§3.8.3)
- [ ] 写 `tests/test_pooling_scheduler.py`:每序列一步完成;遵守 `max_image_patches`
      预算;`tests/test_encode_e2e.py`:CPU mock 前向,shape `(B, 2048)`

## 8. 权重加载器(§3.6,§3.8.2)

- [ ] 新增 `_load_param(model, hf_name, hf_weight, *, merged_id=None)` 分发函数:有
      `weight_loader` 属性则调 `param.weight_loader(param, hf_weight[, merged_id])`,
      否则 `default_weight_loader`
- [ ] VL 文本 decoder 的 QKV/ gate_up 改走 `_load_param` + per-param `weight_loader`
      (`linear.py:97-107,126,152`),**不再** `torch.cat` 后 `copy_(full)`——让
      `ColumnParallelLinear`/`QKVColumnParallelLinear` 按 `tp_rank` 正确切(GQA 感知)
- [ ] ViT(`visual.*`)、merger、deepstack、RMSNorm、EmbeddingHead 投影走 `default_weight_loader`
      复制全量(整塔复制,§3.8.1);ViT 的 q/k/v 不合并成 qkv_projection
- [ ] `VocabParallelEmbedding` 走其 `weight_loader` 做词表分片
- [ ] **不动生成路径**的 `torch.cat` merge + `copy_(full)` 分支(`loader.py:76-146`)(R-7)
- [ ] 复用 `loader.py:179-213` 摘要打印,新增"vision tower"小节,标注哪些 param 走了
      `weight_loader`(分片)、哪些复制

## 9. 入口脚本与文档(§3.7,§9,§10)

- [ ] 新增 `main_embedding.py`:配置 `runner_type="pooling"`,对 文本/图像/图像+文本 输入
      调 `encode()`,打印余弦相似度矩阵
- [ ] `main_embedding.py` 支持 `world_size` 从环境变量读取(§9.5 多卡启动)
- [ ] `README.md` 新增多模态 quickstart 块(含 §10.1–10.6 调用示例)
- [ ] `AGENTS.md` 把"Pooling 模式"小节从占位补全:列出被跳过的子系统、`encode()` API、
      TP 拓扑(§3.8.1 表)、VL 加载走 `weight_loader`(§3.8.2)、生成路径 TP>1 现状(R-7)

## 10. 数值对齐与回归(§5)

- [ ] 写 `tests/test_mrope.py`:A 组 5 个用例(分段/交错/纯文本/图像/混合位置)
- [ ] 写 `tests/test_vision.py`:B 组 4 个用例(patch_embed shape、merger shape、
      deepstack 注入索引、ViT 双向注意力 R-3)
- [ ] 写 `tests/test_pooling_scheduler.py`:C 组 4 个用例(一步完成、image 预算、
      preempt 不可达、终止信号)
- [ ] 写 `tests/test_encode_e2e.py`:D 组 6 个用例(文本/图像/混合返回 2048、L2、MRL、
      批保序)
- [ ] 写 `tests/test_parity_qwen.py`(`@pytest.mark.gpu` + 联网):E 组,复现 README 的
      4 查询/3 文档样例相似度矩阵,余弦容差 `1e-3`
- [ ] 写 `tests/test_tp.py`(`@pytest.mark.gpu and tp`,需 ≥2 卡):G 组 5 个用例
      —— TP 文本 decoder QKV 切片正确(守卫 §1.2 bug)、ViT 复制两 rank 一致、
      残差 full hidden、两 rank embedding 相等、单卡 vs 多卡 `1e-4` 一致
- [ ] 用 README 最大样例图做显存尖峰验证(R-5)
- [ ] 跑 `uv run pytest tests/ -v`,确认 `tests/test_scheduler.py` 仍全绿(生成路径回归)
- [ ] 跑 `uv run python main.py` 与 `uv run python main_llama32.py`,确认生成路径逐字节不变

## 11. 收尾

- [ ] `git add -A && git commit -m "Add Qwen3-VL-Embedding-2B pooling inference support"`
- [ ] 评审:对照本 tasks 清单逐条核对,把 `[ ]` 改 `[x]`
