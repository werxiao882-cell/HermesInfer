from __future__ import annotations
import os
import sys
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../qwen2_vl/
_VLM_DIR  = os.path.dirname(_THIS_DIR)                        # .../vlm_architecture/
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

import torch
from PIL import Image
from transformers import AutoTokenizer

from qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

def build_processor(model_path: str) -> Qwen2VLProcessor:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    video_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    processor = Qwen2VLProcessor(
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
) -> Qwen2VLForConditionalGeneration:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
    messages: list[dict],
    images: Optional[list[Image.Image]] = None,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """
    Step 1: 聊天模板格式化 (apply_chat_template)
    
    该步骤将结构化的多轮对话 (messages 列表) 转化为大模型底层能够直接理解的纯文本 Prompt 字符串。
    
    【处理过程】
    - 解析消息结构：遍历 messages 中的内容。假设用户输入的内容包含一张图像和文本问题 (例如 "Describe this image")。
    - 插入 ChatML 特殊标记：遵循 Qwen 的指令微调格式规范，为不同角色 (如 system/user/assistant) 的发言
      添加 `<|im_start|>` 和 `<|im_end|>` 对话边界控制符。
    - 替换多模态占位符：当解析到 `{"type": "image"}` 时，将其转换为模型专属的视觉特征输入标记序列
      `<|vision_start|><|image_pad|><|vision_end|>`。注意此时只插入了一个 `<|image_pad|>` 占位符，
      它会在后续的 Step 2 中被根据真实图像的 Patch 数量动态扩充。
    - 附加生成提示引导：参数 `add_generation_prompt=True` 会在最终拼接的文本末尾自动加上 
      `<|im_start|>assistant\n`，明确提示模型接下来该由 Assistant 开始输出文字。
      
    【最终输出 (text_input)】
    最终得到一个拼接好的字符串，形式类似于：
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image<|im_end|>\n<|im_start|>assistant\n"
    """
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    """
    Step 2: Processor 联合编码文本 + 图像
    
    针对输入的 RGB 图像，会经历以下详细过程：
    
    【1. 图像预处理 (Image Processor)】
    - 动态分辨率调整 (smart_resize): 计算最佳尺寸。在保持原始长宽比的前提下, 限制总像素数在min_pixels 和 max_pixels 之间，并严格保证高和宽都是 patch_size * merge_size (例如 14 * 2 = 28) 的整数倍。
    - 图像重采样与归一化：通过插值算法(如 Bicubic)将原图缩放到计算出的最佳尺寸；将像素值从[0, 255] 缩放至 [0, 1] 区间，接着使用指定的均值和方差 (如 CLIP 的 mean/std) 进行标准化。
    - Patch 切分与展平：将图像转换为张量并进行分块，按照 temporal_patch_size 和 patch_size 提取图像块 (patch)，并最终展平成形状为 (total_patches, patch_dim) 的 pixel_values 序列。
    - 记录网格拓扑 (image_grid_thw)：记录该图像切分后的三维网格尺寸 (grid_t, grid_h, grid_w)。
      这个网格信息不仅用于计算最终传递给大模型的 token 数量，还会被模型用于给图像特征计算 2D 旋转位置编码 (2D RoPE)。
    
    【2. 文本与多模态对齐 (Processor 整合)】
    - 动态扩展图像占位符：基于 image_grid_thw 计算出图像经过 Vision Encoder 和 Patch Merger 
      压缩后在 LLM 中对应的最终视觉 token 数量。程序会自动在 text_input 中，将原始文本里的 
      `<|image_pad|>` 标签替换为恰好等于该数量的连续 `<|image_pad|>` 序列，实现视觉与文本序列在长度上的精确对齐。
    - 文本序列化 (Tokenizer)：调用 Tokenizer 将扩展好占位符的文本转化为数字序列 input_ids。
    
    最终生成的 inputs 字典包含：文本相关的 input_ids、attention_mask，以及视觉特征的 pixel_values 和 image_grid_thw。
    """
    inputs = processor(text=[text_input], images=images, return_tensors="pt")
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
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

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
    MODEL_PATH = "/home/lixin/workspace/checkpoint_space/Qwen2-VL-7B-Instruct"
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
