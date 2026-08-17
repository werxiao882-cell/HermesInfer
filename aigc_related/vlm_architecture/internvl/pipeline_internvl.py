from __future__ import annotations
import os
import sys
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../internvl/
_VLM_DIR  = os.path.dirname(_THIS_DIR)                        # .../vlm_architecture/
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

import torch
from PIL import Image
from transformers import AutoTokenizer

from internvl.modeling_internvl import InternVLForConditionalGeneration
from internvl.processing_internvl import InternVLProcessor
from internvl.video_processing_internvl import InternVLVideoProcessor

def build_processor(model_path: str) -> InternVLProcessor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    video_processor = InternVLVideoProcessor()
    
    chat_template = getattr(tokenizer, "chat_template", None)
    
    # We construct the main processor
    # Note: Using AutoImageProcessor internally or we need to define InternVLImageProcessor
    # InternVL usually relies on standard image processors (e.g. CLIPImageProcessor)
    # The InternVLProcessor class definition uses `image_processor` 
    from transformers import AutoImageProcessor
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    except:
        image_processor = None
        
    processor = InternVLProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    return processor

def build_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> InternVLForConditionalGeneration:
    model = InternVLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: InternVLForConditionalGeneration,
    processor: InternVLProcessor,
    messages: list[dict],
    images: Optional[list[Image.Image]] = None,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """
    Step 1: 聊天模板格式化 (apply_chat_template)
    
    该步骤将结构化的多轮对话 (messages 列表) 转化为大模型底层能够直接理解的纯文本 Prompt 字符串。
    """
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    """
    Step 2: Processor 联合编码文本 + 图像
    """
    # InternVL processor might need some specific args depending on version
    inputs = processor(text=[text_input], images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Step 3: 自回归生成
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
    MODEL_PATH = "/home/lixin/workspace/checkpoint_space/InternVL2-8B" # Please update the actual model path
    
    # We will create an empty assets directory and a dummy image path
    assets_dir = os.path.join(_THIS_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    DEMO_IMAGE = os.path.join(assets_dir, "demo.png")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")
    print(f"Loading: {MODEL_PATH}")

    try:
        processor = build_processor(MODEL_PATH)
        model = build_model(MODEL_PATH, device=DEVICE)

        # Only run inference if demo.png exists
        if os.path.exists(DEMO_IMAGE):
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
        else:
            print(f"Please place a test image at {DEMO_IMAGE} to run inference.")
            
    except Exception as e:
        print(f"Could not run the demo: {e}")
        print("Make sure you have the correct model path and dependencies installed.")

