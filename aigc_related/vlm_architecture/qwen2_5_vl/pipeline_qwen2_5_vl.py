"""
Qwen2.5-VL 推理入口

用法:
    cd vlm_architecture
    python -m qwen2_5_vl.pipeline_qwen2_5_vl

该文件仿照 diffusion_architecture/flux/pipeline_flux.py 的风格，
可以直接在本地 debug，逐步跟踪 Qwen2.5-VL 的完整推理过程。

模型结构概览（相比 Qwen2-VL 的主要变化）:
    Qwen2_5_VLForConditionalGeneration
      ├── model: Qwen2_5_VLModel
      │     ├── visual: Qwen2_5_VisionTransformerPretrainedModel  (ViT 视觉编码器)
      │     │     ├── patch_embed: Qwen2_5_VisionPatchEmbed  (窗口注意力 + 3D RoPE)
      │     │     ├── rotary_pos_emb: Qwen2_5_VisionRotaryEmbedding
      │     │     ├── blocks: N × Qwen2_5_VLVisionBlock
      │     │     └── merger: Qwen2_5_VLPatchMerger
      │     └── language_model: Qwen2_5_VLTextModel
      │           ├── embed_tokens
      │           ├── layers: N × Qwen2_5_VLDecoderLayer
      │           │     ├── self_attn: Qwen2_5_VLAttention  (MRoPE)
      │           │     └── mlp: Qwen2MLP
      │           └── norm: Qwen2_5_VLRMSNorm
      └── lm_head: Linear(hidden_size → vocab_size)

    image_processor : Qwen2VLImageProcessorFast  (复用 qwen2_vl)
    video_processor : Qwen2VLVideoProcessor       (复用 qwen2_vl)

推理流程:
    1. Processor.apply_chat_template  →  格式化 messages 为带特殊 token 的文本串
    2. Processor.__call__             →  联合编码文本 + 图像，输出:
         - input_ids      : 含 <|image_pad|> 占位的 token ids
         - attention_mask : padding mask
         - pixel_values   : 图像 patch 像素值  (N_patches, C × patch_size²)
         - image_grid_thw : 每张图的 grid 尺寸 (T, H_grid, W_grid)
    3. model.generate                 →  自回归解码:
         a. ViT 编码 pixel_values → 视觉 token 特征序列
         b. PatchMerger 压缩相邻 patch
         c. 将视觉特征填入 <|image_pad|> 位置
         d. Qwen2.5 LLM causal attention + 逐 token 解码
    4. processor.batch_decode         →  token ids → 文本
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../qwen2_5_vl/
_VLM_DIR  = os.path.dirname(_THIS_DIR)                        # .../vlm_architecture/
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

import torch
from PIL import Image
from transformers import AutoTokenizer

from qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor


def build_processor(model_path: str) -> Qwen2_5_VLProcessor:
    """从本地路径加载 Qwen2_5_VLProcessor，使用本地类以便 debug 时可以 step into。

    Qwen2.5-VL 的 image/video processor 复用 qwen2_vl 的实现，
    故从本地 qwen2_vl 目录导入。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    video_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    processor = Qwen2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    return processor


def build_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Qwen2_5_VLForConditionalGeneration:
    """从本地路径或 HuggingFace Hub 加载 Qwen2_5_VLForConditionalGeneration。

    Args:
        model_path : 本地目录或 HuggingFace Hub 模型 ID
        device     : "cuda" / "cpu" / "auto"
        torch_dtype: 模型权重精度，推荐 torch.bfloat16
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    messages: list[dict],
    images: Optional[list[Image.Image]] = None,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """执行 Qwen2.5-VL 完整推理流程。"""
    # Step 1: apply_chat_template
    # 将 messages 格式化为模型期望的文本，其中 <image> 占位符会被
    # 替换为 <|vision_start|><|image_pad|>...<|vision_end|> 序列
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Step 2: Processor 联合编码文本 + 图像
    # - image_processor (Qwen2VLImageProcessorFast) 负责:
    #   smart_resize → normalize → patch 提取
    #   输出 pixel_values: (total_patches, C × patch_size²)
    #   image_grid_thw: (num_images, 3) 每图的 (T, H_grid, W_grid)
    # - tokenizer 负责: 文本 → input_ids，<|image_pad|> 数量由 grid_thw 决定
    inputs = processor(
        text=[text_input],
        images=images,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Step 3: 自回归生成
    # generate 内部调用链:
    #   model.prepare_inputs_for_generation
    #     → model.forward (首次完整前向)
    #         → visual encoder: pixel_values → patch features
    #         → patch merger 压缩
    #         → embed_tokens + 视觉特征拼接
    #         → N × decoder layers (MRoPE self-attention)
    #         → lm_head → logits
    #     → greedy/beam search 采样下一个 token
    #     → model.forward (后续步用 KV Cache 增量推理)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    # Step 4: 截取新生成的 token（去掉 prompt 部分）
    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


if __name__ == "__main__":
    MODEL_PATH = "/home/lixin/workspace/checkpoint_space/Qwen2.5-VL-7B-Instruct"
    DEMO_IMAGE = os.path.join(_THIS_DIR, "assets", "demo.png")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")
    print(f"Loading: {MODEL_PATH}")

    processor = build_processor(MODEL_PATH)
    model = build_model(MODEL_PATH, device=DEVICE)

    image = Image.open(DEMO_IMAGE).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": "请你描述一下这张图片。"},
            ],
        }
    ]

    result = run_inference(
        model=model,
        processor=processor,
        messages=messages,
        images=[image],
        max_new_tokens=256,
        device=DEVICE,
    )
    print(f"回复: {result}")
