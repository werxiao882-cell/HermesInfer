import json
from PIL import Image
import os

from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset, DataLoader

class SiglipDataset(Dataset):
    def __init__(self, image_folder_path, json_file_path, tokenizer, max_length, processor) -> None:
        super().__init__()
        self.image_folder_path = image_folder_path
        self.json_file_path = json_file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = processor

        self.datasets = []
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    item = json.loads(line)
                    image_path = os.path.join(self.image_folder_path, item['image'])
                    text = item['content']  
                    self.datasets.append((image_path, text))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        image_path, text = self.datasets[idx]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt", use_fast=True)["pixel_values"]
        
        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Squeeze to remove batch dimension
        input_ids = text_inputs.input_ids.squeeze(0)
        attention_mask = text_inputs.attention_mask.squeeze(0)
        
        return {
            "pixel_values": image_inputs.squeeze(0),
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert-base-chinese")
    processor = AutoProcessor.from_pretrained("./pretrained_models/vit-base-patch16-224")
    dataset = SiglipDataset(
        image_folder_path="./dataset/pretrain_images",
        json_file_path="./dataset/processed_data.jsonl",
        tokenizer=tokenizer,
        max_length=64,
        processor=processor
    )
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for index, batch in enumerate(dataloader):
        print(f"Batch {index}:")
        print(f"  pixel_values shape: {batch['pixel_values'].shape}")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        break