import torch
from transformers import pipeline, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

# 1. 配置路径
base_model_name = "lvwerra/gpt2-imdb" # 原始模型
tuned_model_path = "./gpt2-imdb-pos-v2" # 你保存的新模型路径
device = 0 if torch.cuda.is_available() else -1

# 2. 加载奖励模型（用于打分对比）
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

# 3. 定义对比函数
def compare_models(prompts):
    # 加载两个模型
    model_base = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name).to("cuda")
    model_tuned = AutoModelForCausalLMWithValueHead.from_pretrained(tuned_model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    gen_kwargs = {"max_new_tokens": 32, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

    print(f"{'Prompt':<30} | {'Base Score':<10} | {'Tuned Score':<10}")
    print("-" * 60)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 生成文本
        out_base = model_base.generate(**inputs, **gen_kwargs)
        out_tuned = model_tuned.generate(**inputs, **gen_kwargs)

        text_base = tokenizer.decode(out_base[0])
        text_tuned = tokenizer.decode(out_tuned[0])

        # 打分 (POSITIVE 标签的分数)
        score_base = sentiment_pipe(text_base, top_k=None)
        score_tuned = sentiment_pipe(text_tuned, top_k=None)
        
        # 提取 POSITIVE 的分数
        s_b = next(x['score'] for x in score_base if x['label'] == 'POSITIVE')
        s_t = next(x['score'] for x in score_tuned if x['label'] == 'POSITIVE')

        print(f"{prompt:<30} | {s_b:.4f}     | {s_t:.4f}")
        print(f"Base output: {text_base}")
        print(f"Tuned output: {text_tuned}\n")

# 4. 测试一些测试集外的提示词（更能体现泛化能力）
test_prompts = [
    "The movie was",
    "I thought the acting was",
    "The director managed to",
    "Overall, I felt that"
]

if __name__ == "__main__":
    compare_models(test_prompts)