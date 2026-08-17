# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-3B-Instruct
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions

# For HuggingFaceTB/SmolVLM2-2.2B-Instruct
pip install num2words==0.5.14

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir grpo-SmolVLM2-2.2B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 2

"""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/home/lixin/workspace/personal_learning/aigc_related/huggingface_hub"
os.environ["HF_HUB_CACHE"] = "/home/lixin/workspace/personal_learning/aigc_related/huggingface_hub"
import torch
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
trl_path = os.path.join(current_dir, "trl")
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import accuracy_reward, think_format_reward


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
    "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
    "reasoning.\n</think>\nThis is my answer."
)

def make_conversation(example):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]
    return {"prompt": prompt}

# Filter have big images
def filter_big_images(example):
    image = example["image"]
    return image.size[0] < 512 and image.size[1] < 512

def convert_to_rgb(example):
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["image"] = image
    return example

if __name__ == "__main__":
    ################################################################################################
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--model_name_or_path", "/home/lixin/workspace/checkpoint_space/Qwen2.5-VL-3B-Instruct/",
            "--output_dir", "grpo-Qwen2.5-VL-3B-Instruct-debug",
            "--learning_rate", "1e-5",
            "--dtype", "bfloat16",
            "--max_completion_length", "128",
            "--vllm_mode", "colocate",
            "--use_peft", "True",
            "--lora_target_modules", "q_proj", "v_proj",
            "--log_completions", "True",
            "--report_to", "none",
        ])
    ################################################################################################

    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################################################################################################
    # 1. Model
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config
    ################################################################################################

    ################################################################################################
    # 2. Dataset
    dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train")

    # Dataset Preprocessing: split train/test, make conversation, filter big images, convert to RGB
    dataset = dataset.train_test_split(test_size=100, seed=42)
    dataset = dataset.map(make_conversation)
    dataset = dataset.filter(filter_big_images)
    dataset = dataset.map(convert_to_rgb)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None
    ################################################################################################

    ################################################################################################
    # 3. Training
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    ################################################################################################

    ################################################################################################
    # 4. Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    ################################################################################################