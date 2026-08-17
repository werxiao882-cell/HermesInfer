from transformers import Trainer, TrainingArguments
from model import SiglipModel, SiglipConfig
from dataset import SiglipDataset
from transformers import AutoTokenizer, AutoProcessor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train():
    train_config = SiglipConfig(
        text_model_name="./pretrained_models/bert-base-chinese",
        image_model_name="./pretrained_models/vit-base-patch16-224",
    )
    model = SiglipModel(train_config)

    tokenizer = AutoTokenizer.from_pretrained(train_config.text_model_name)
    processor = AutoProcessor.from_pretrained(train_config.image_model_name)

    train_args = TrainingArguments(
        output_dir="./output",
        do_train=True,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=30,
        logging_steps=100,
        save_steps=10000,
        save_total_limit=10,
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        report_to='tensorboard',

        # Logging
        logging_strategy="steps",
        logging_first_step=True,
        label_names=["loss"],
        greater_is_better=False
    )

    dataset = SiglipDataset(
        image_folder_path="./dataset/pretrain_images",
        json_file_path="./dataset/processed_data.jsonl",
        tokenizer=tokenizer,
        max_length=64,
        processor=processor)
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("./output/final_model")
    trainer.save_state("./output/final_state")

if __name__ == "__main__":
    train()