import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
trl_path = os.path.join(current_dir, "trl")
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    构建用于训练的数据集。
    
    该函数执行以下操作：
    1. 加载分词器并设置 padding token。
    2. 从 HuggingFace 加载指定的（默认 IMDB）数据集。
    3. 过滤掉长度不足（字符数小于200）的评论值。
    4. 随机截取评论的开头作为 Prompt（查询）。
    
    参数:
        config: PPO配置对象，包含模型名称。
        dataset_name: 数据集名称。
        input_min_text_length: 随机采样的 Prompt 最小长度（token数量）。
        input_max_text_length: 随机采样的 Prompt 最大长度（token数量）。
    """
    # 初始化分词器，#NOTE: GPT-2 等模型没有默认的 pad_token，通常设为 eos_token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load the IMDB dataset
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    # Only choose reviews with more than 200 tokens
    # #NOTE: 过滤掉太短的评论，确保我们有足够的上下文来截取 Prompt
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # 从每条评论中截取前 `input_size` 个 token 作为 Prompt
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    """
    数据整理函数，将 list of dicts 转换为 dict of lists。
    PPO 训练中需要这种格式来处理变长的 sequences。
    """
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':
    # 1. 配置PPO训练参数
    # NOTE: 使用lvwerra/gpt2-imdb作为pretrained_model, 它已经学会了如何写像IMDB电影评论。我们将通过PPO微调它，使其生成更积极的评论
    config = PPOConfig(model_name="lvwerra/gpt2-imdb", learning_rate=1.41e-5, log_with="wandb", steps=10000, ppo_epochs=4)

    import wandb
    wandb.init()

    # 2. 准备数据集
    dataset = build_dataset(config)

    # 3. 初始化模型
    # NOTE: AutoModelForCausalLMWithValueHead是TRL库提供的一个类，它在原始语言模型上添加了一个线性层[Value Head]，用于预测期望奖励值。
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    
    # NOTE: 参考模型[Reference Model]，它的作用是计算KL散度，防止更新后的策略模型偏离原始模型太远。
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 初始化PPOTrainer
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # 5. 设置奖励模型，在PPOTrainer中, 我们需要使用Reward Model来为生成的文本打分来引导模型在更新参数时向生成更高奖励的内容靠拢
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    # 6. 测试奖励模型是否正常工作
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    text = "this movie was really bad!!"
    print(sentiment_pipe(text, **sent_kwargs))

    text = "this movie was really good!!"
    print(sentiment_pipe(text, **sent_kwargs)) 

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # 7. 配置生成回复的策略
    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # 8. 开始PPO迭代训练
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Preprocess
        """
        PPO的重要性采样指的是, 在每次迭代中, 先使用当前策略与环境交互收集一批数据, 然后在这批数据上进行有限次数的策略更新[通常通过多轮mini-batch训练], 之后丢弃该批数据。
        然后必须使用更新后的策略重新收集新数据。形成：先收集→多次更新→丢弃→重新收集的循环。
        """
        ####################################################################################################################################################
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            response_generation_kwargs["max_new_tokens"] = gen_len 
            response = ppo_trainer.generate(query, **response_generation_kwargs) 
            response_tensors.append(response.squeeze()[-gen_len:]) 
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # 拼接query和response形成完整文本
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        # NOTE: 将文本传入奖励模型，获取每条文本的奖励分数, 得到Reward Score
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs) 
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        ####################################################################################################################################################

        ####################################################################################################################################################
        """
        使用PPOTrainer的step方法进行一次PPO优化步骤, 传入的参数包括：
        1. query_tensors: 模型输入的查询信息。
        2. response_tensors: 模型生成的回复信息。
        3. rewards: 奖励模型计算得到的奖励分数信息。
        """  
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ####################################################################################################################################################

        ppo_trainer.log_stats(stats, batch, rewards)

    # 9. 保存优化后的模型和分词器
    model.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    tokenizer.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)