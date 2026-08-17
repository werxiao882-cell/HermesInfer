import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams

# 生成式 Qwen3-VL-2B-Instruct 配置(world_size>1 启用 TP)。
# runner_type="generation":走 KV cache / 连续批 / decode / cuda graph / 采样;
# model_name 含 "Qwen3-VL" 触发 VL 生成式路径(MRoPE + 多模态 prefill 写 KV)。
config = {
    "model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
    "runner_type": "generation",
    "block_size": 256,
    "max_num_sequences": 16,
    "max_num_batched_tokens": 8192,
    "max_cached_blocks": 1024,
    "enforce_eager": False,                # 用 cuda graph 加速 decode
    "max_model_length": 4096,
    "max_num_seqs": 16,
    "multimodal": {"max_image_patches": 16384},
    # VL 模型默认值(对齐 config.json;8B 把 num_layers 改 36 等)
    "vocab_size": 151936, "hidden_size": 2048, "num_heads": 16, "head_dim": 128,
    "num_kv_heads": 8, "intermediate_size": 6144, "num_layers": 28,
    "rms_norm_epsilon": 1e-6, "base": 5_000_000, "mrope_section": [24, 20, 20],
    "tie_word_embeddings": True,
    "vision_depth": 24, "vision_hidden_size": 1024,
    "vision_intermediate_size": 4096, "vision_num_heads": 16,
    "patch_size": 16, "temporal_patch_size": 2, "in_channels": 3,
    "out_hidden_size": 2048, "spatial_merge_size": 2,
    "num_position_embeddings": 2304, "deepstack_visual_indexes": [5, 11, 17],
}


def main():
    engine = LLMEngine(config)
    sampling = SamplingParams(temperature=0.6, max_tokens=256)

    inputs = [
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
         "text": "描述这张图。"},
        {"text": "用一句话介绍杭州。"},
    ]
    out = engine.chat(inputs, sampling)
    for text in out["text"]:
        print("==== output ====")
        print(text)


if __name__ == "__main__":
    main()
