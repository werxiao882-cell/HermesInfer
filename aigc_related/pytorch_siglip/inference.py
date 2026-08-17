from pathlib import Path
from typing import List
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer


def load_models(
    checkpoint_path: str,
    vision_model_path: str,
    text_model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """加载预训练的模型和处理器"""
    print(f"Loading models on device: {device}")
    
    processor = AutoProcessor.from_pretrained(vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    model = AutoModel.from_pretrained(checkpoint_path)
    
    model = model.to(device)
    model.eval()
    
    print("Models loaded successfully!")
    return processor, tokenizer, model, device


def predict(image_path: str, texts: List[str], processor, tokenizer, model, device: str, max_length: int = 64):
    """对图像和文本进行推理预测"""
    # 加载并预处理图像
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
    
    # 预处理文本 (使用 max_length padding，与训练时保持一致)
    tokenized = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)
    return probs


def main():    
    CHECKPOINT_PATH = "./output/final_model"
    VISION_MODEL_PATH = "./pretrained_models/vit-base-patch16-224"
    TEXT_MODEL_PATH = "./pretrained_models/bert-base-chinese"
    IMAGE_PATH = "./assets/eval_data/4.png"
    
    # 验证路径
    if os.path.exists(IMAGE_PATH) is False:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if os.path.exists(CHECKPOINT_PATH) is False:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    # 加载模型
    processor, tokenizer, model, device = load_models(
        checkpoint_path=str(CHECKPOINT_PATH),
        vision_model_path=str(VISION_MODEL_PATH),
        text_model_path=str(TEXT_MODEL_PATH)
    )
    
    texts = ["一只狗",  "汽车在路上行驶"]
    probs = predict(
        image_path=str(IMAGE_PATH),
        texts=texts,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        device=device
    )
    
    print("=" * 60)
    print("Prediction Results:")
    print("=" * 60)
    print(f"Raw probabilities: {probs[0].cpu().numpy()}\n")
    
    for idx, (text, prob) in enumerate(zip(texts, probs[0])):
        print(f"[{idx+1}] {prob:.2%} - '{text}'")
    
    best_idx = probs[0].argmax().item()
    print("\n" + "=" * 60)
    print(f"Best match: '{texts[best_idx]}' ({probs[0][best_idx]:.2%})")
    print("=" * 60)


if __name__ == "__main__":
    main()