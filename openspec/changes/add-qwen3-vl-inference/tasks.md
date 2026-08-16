# 实施任务:Qwen3-VL 生成式推理 + VL 专项优化

按依赖顺序;每条标注 design § 与风险 R。先模型/引擎接线,再优化,再对齐与回归。

## 1. 调研与接线

- [ ] 确认 `Qwen3-VL-2B-Instruct`/`-8B` 的 config.json(网络瞬断时未取到,需复核层数等;
      8B 可能 36 层)(§1.1)
- [ ] 核对生成路径加载器对 VL 名(visual.*/language_model.*)的 remap 是否够用(R-1)

## 2. 生成式 VL 模型(§3.3)

- [ ] `models/qwen3_vl.py` 增 `Qwen3VLForCausalLM`:复用 vision/decoder/deepstack,换
      `lm_head: ParallelLMHead`(tie 到 embed_tokens),`forward` 返回 hidden,
      `compute_logits(hidden)` 走 lm_head(对标 qwen3.py:Qwen3ForCausalLM)
- [ ] `tests/test_vl_model.py`:`compute_logits` 输出形状 `(total_tokens, vocab)`

## 3. 引擎:多模态 prefill 写 KV(§3.1,§3.4)

- [ ] `ModelRunner.__init__` 分发新增 `case "Qwen3-VL-2B-Instruct"`/
      `"Qwen3-VL-8B-Instruct"` → 构造 `Qwen3VLForCausalLM`(`runner_type='generation'`)
- [ ] 新增 `prepare_prefill_vl_gen(seqs)`:复用 Embedding 的多模态构造,但补
      `block_manager.allocate` → `slot_mapping` + `block_tables`(图像 token 占 KV slot)
- [ ] `run()` 判定:序列带 `mm_data` → prefill 走 `prepare_prefill_vl_gen`,否则原 `prepare_prefill`
- [ ] `prepare_decode` 不改(decode 无图像 token,与纯文本一致)
- [ ] `capture_cudagraph` 不改(VL decode batch 形状同文本;R-4 核对 deepstack 不在 decode 注入)
- [ ] `tests/test_vl_prefill.py`:slot_mapping 非 None,图像 token K/V 进 cache
- [ ] `tests/test_vl_decode.py`:paged attention 读回出 token

## 4. chat 入口(§3.4,§7)

- [ ] `LLMEngine.chat(inputs, sampling_params)`:每条 `build_multimodal_inputs` → Sequence
      (带 mm_data)→ `step()` 循环到完成 → tokenizer.decode → 文本(对标 `generate`)
- [ ] `main_vl_chat.py` 演示(文/图/图文 → 文本)
- [ ] `tests/test_vl_chat.py`:端到端 mock(小 dim)

## 5. 调度器:VL 连续批(§3.4,O3)

- [ ] 确认 `Scheduler.add_sequence` 对带 mm_data 序列走 KV 容量校验(图像 token 计入
      `num_tokens`/`num_blocks`);token 预算口径含图像 token
- [ ] `tests/test_vl_scheduler.py`:带图 + 纯文本混合调度不丢序列

## 6. 权重加载(§3.5,R-1)

- [ ] 把 `loader_vl.py` 的 `_candidate_custom_names`/`_load_param` 抽共用
- [ ] `model_runner.py` 生成路径对 VL 模型(`model_name` 含 `Qwen3-VL`)改调
      `load_weights_vl`(单卡可用;多卡 TP>1 生成路径加载修复留 R-7)
- [ ] 核对 `Qwen3VLForCausalLM` 的参数名与 HF checkpoint 名 remap 命中

## 7. 优化 O1:视觉 prefix 缓存(§4 O1,R-3)

- [ ] `engine/visual_cache.py`:按 `xxh64(pixel_values.tobytes()+grid_thw)` 缓存
      `(visual_emb, deepstack)`,LRU 按字节限
- [ ] `prepare_prefill_vl_gen`:命中跳过 `self.model.visual(...)`
- [ ] `tests/test_visual_cache.py`:同图二次命中跳过(计次)

## 8. 优化 O4:decode CUDA graph(§4 O4)

- [ ] 确认 VL decode 复用 `capture_cudagraph`(无改动);`block_tables` 含图像 block
      时图重放分支正确(R-4)

## 9. (v2)分离式编码 O2 + chunked prefill O6

- [ ] O2:`Scheduler` 加 encode 队列,带图序列先编码入视觉缓存再入 waiting
- [ ] O6:chunk 边界不切图像 span(R-2)

## 10. 数值对齐与回归

- [ ] `tests/test_parity_vlgen.py`(@gpu):与 transformers/vLLM 短输出 FP 容差
- [ ] 现有 Embedding / 纯文本生成测试全绿(回归)
- [ ] 显存验证(R-5):长图 + 长上下文,2B 单卡 / 8B TP

## 11. 文档与收尾

- [ ] `AGENTS.md` 增"VL 生成式"小节(§3 数据流、优化清单 §4)
- [ ] `README.md` 增 chat quickstart
- [ ] 评审:对照本清单逐条 `[ ]` → `[x]`
