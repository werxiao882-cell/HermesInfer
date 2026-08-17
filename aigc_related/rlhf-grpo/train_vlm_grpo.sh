accelerate launch \
    --config_file /home/lixin/workspace/personal_learning/aigc_related/rlhf-grpo/deepspeed_zero3.yaml \
    train_vlm_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions