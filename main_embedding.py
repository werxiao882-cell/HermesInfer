import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from myvllm.engine.llm_engine import LLMEngine

# Qwen3-VL-Embedding-2B 配置(pooling / 纯 prefill)。world_size>1 启用 TP。
# runner_type="pooling" 切到 embedding 路径(无 KV cache/cuda graph/采样)。
config = {
    "model_name_or_path": "Qwen/Qwen3-VL-Embedding-2B",
    "world_size": int(os.environ.get("WORLD_SIZE", 1)),   # 多卡:设环境变量 WORLD_SIZE
    "runner_type": "pooling",                             # 关键:走 embedding 路径
    "block_size": 256,
    "max_num_sequences": 16,                               # 每批最大序列数
    "max_num_batched_tokens": 8192,                        # 每批最大打包 token 数
    "enforce_eager": True,                                 # pooling 无 cuda graph,这里无所谓
    "pooling": {"mode": "last_token", "normalize": True, "mrl_dim": None},  # last-token + L2
    "multimodal": {"max_image_patches": 16384},            # vision tower 的 OOM 守卫
    # ---- VL 模型默认值(对齐 config.json;可覆盖)----
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
    llm = LLMEngine(config)

    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        {"text": "Pet owner training dog outdoors near water."},
        {"text": "Woman surfing on waves during a sunny day."},
        {"text": "City skyline view from a high-rise building at night."},
    ]
    documents = [
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    ]

    q_emb = llm.encode(queries)
    d_emb = llm.encode(documents)
    sim = torch.stack(q_emb) @ torch.stack(d_emb).T
    print("\nSimilarity (query x doc):")
    print(sim)


if __name__ == "__main__":
    main()
