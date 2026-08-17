from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Dict, Any
import json
import os
from torch.utils.tensorboard import SummaryWriter

class Config():
    def __init__(self, llm_model_path="", predict_token_num=5):
        super().__init__()
        self.llm_model_path = llm_model_path
        self.predict_token_num = predict_token_num

class MTPBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        if hasattr(nn, 'RMSNorm'):
            self.norm1 = nn.RMSNorm(hidden_size)
            self.norm2 = nn.RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

        self.transformer_block = nn.TransformerDecoderLayer(
            d_model=hidden_size, 
            nhead=32, 
            dim_feedforward=hidden_size * 4, 
            batch_first=True
        )

    def forward(self, prev_hidden_states: torch.Tensor, current_token_embeddings: torch.Tensor, **kwargs):
        normed_prev = self.norm1(prev_hidden_states)
        normed_curr = self.norm2(current_token_embeddings)
        x = torch.cat([normed_prev, normed_curr], dim=-1)
        x = self.proj(x)
        out = self.transformer_block(x, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out

class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path).base_model
        self.mtp_model = nn.ModuleList([MTPBlock(self.main_model.config.hidden_size) for _ in range(len(self.config.predict_token_num) - 1)])
        self.output_proj = nn.Linear(self.main_model.config.hidden_size, self.main_model.config.vocab_size)

    def forward_main(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        main_hidden_output = self.main_model(input_ids, attention_mask, output_hidden_states=True).last_hidden_state
        main_head_output = self.output_proj(main_hidden_output)
        return main_hidden_output, main_head_output

    def forward_mtp(self, input_ids: torch.Tensor, prev_hidden_states: torch.Tensor, head_index: int):
        input_embeddings = self.main_model.get_input_embeddings()(input_ids)
        mtp_hidden_output = self.mtp_model[head_index](prev_hidden_states, input_embeddings)
        mtp_head_output = self.output_proj(mtp_hidden_output)
        return mtp_hidden_output, mtp_head_output

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = {}
        main_hidden_output, main_head_output = self.main_model(input_ids, attention_mask, output_hidden_states=True)
        previous_hidden_states = main_hidden_output
        outputs["head_main"] = main_head_output
        for i, mtp_block in enumerate(self.mtp_model):
            previous_hidden_states, mtp_head_output = mtp_block(input_ids, previous_hidden_states, i)
            outputs[f"head_mtp_{i}"] = mtp_head_output
        return outputs

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_length: int = 100, threshold: float = 1e-6, **kwargs):
        self.eval()
        seq = input_ids.clone()
        
        with torch.no_grad():
            while seq.size(1) < max_length:
                # 1. 获得当前输入下所有头 (主模型 + MTP) 的输出
                # 注意：为了简化，当前注意力掩码(attention_mask)在生成过程中暂不动态更新，实际中需要
                outputs = self.forward(seq, attention_mask=None)
                
                speculative_tokens = []
                
                # 2. 收集各个头预测的 next_token
                # 主模型的预测 (第 1 个 token)
                logits_main = outputs['head_main'][:, -1, :]
                token_main = torch.argmax(logits_main, dim=-1)
                speculative_tokens.append(token_main)
                
                # MTP 模型的预测 (后续的 predict_token_num 个 token)
                for i in range(self.config.predict_token_num):
                    logits_mtp = outputs[f'head_mtp_{i}'][:, -1, :]
                    token_mtp = torch.argmax(logits_mtp, dim=-1)
                    speculative_tokens.append(token_mtp)
                
                # 拼接入选的投机 tokens: shape (batch_size, num_speculative_tokens)
                speculative_tokens = torch.stack(speculative_tokens, dim=1)
                
                # 3. 验证阶段：将生成的 tokens 与原序列拼接，用主模型进行验证
                all_tokens = torch.cat([seq, speculative_tokens], dim=-1)
                
                # 主模型跑一次完整前向，获取验证 logits
                # 只需拿到最后几个投机 tokens 对应位置的验证输出
                validation_outputs = self.main_model(all_tokens)
                # hasattr 检查以兼容不同版本的 transformers 返回格式，这里假设返回 logits
                all_logits = validation_outputs.logits if hasattr(validation_outputs, "logits") else validation_outputs[0]
                
                # 投机了 N 个 token，取出对应这些 token 输入时的输出 logits 用来验证
                validation_logits = all_logits[:, -speculative_tokens.size(1):, :]
                
                # 4. 计算各个投机 token 在主模型验证下的概率
                accept_probs = []
                for i in range(speculative_tokens.size(1)):
                    logits_i = validation_logits[:, i, :]
                    probs_i = torch.softmax(logits_i, dim=-1)
                    token_i = speculative_tokens[:, i].unsqueeze(-1)
                    
                    # 取出所选 token 的概率
                    token_prob = probs_i.gather(1, token_i).squeeze(-1)
                    accept_probs.append(token_prob)
                
                # shape: (batch_size, num_speculative_tokens)
                accept_probs = torch.stack(accept_probs, dim=1)
                
                # 5. 决定接受哪些 tokens (串行接受：遇到第一个不满足条件的就截断)
                # 接受条件：验证概率大于设定的 threshold
                accept_mask = accept_probs > threshold
                
                # 处理 batch 中的每条序列
                batch_size = seq.size(0)
                accepted_tokens_list = []
                
                for b in range(batch_size):
                    mask_b = accept_mask[b]
                    
                    # 找出第一个拒绝(False)的位置
                    reject_indices = (~mask_b).nonzero(as_tuple=True)[0]
                    
                    if len(reject_indices) > 0:
                        accept_num = reject_indices[0].item()
                    else:
                        accept_num = speculative_tokens.size(1)
                    
                    # 至少保证如果 accept_num == 0 的后备方案
                    # 如果第一个 token (主模型的预测) 都被拒绝了(罕见情况因为阈值通常较低)，回退到生成主模型预测即可
                    if accept_num > 0:
                        accepted_tokens_list.append(speculative_tokens[b, :accept_num])
                    else:
                        # Fallback：只接受主模型正常预测出来的那一个 token
                        accepted_tokens_list.append(speculative_tokens[b, :1])
                
                # 将本轮接受的 tokens 拼接到序列中
                # 如果是 batch_size > 1 且各个样本接受的数量不同，需要 pad 对齐。这里假设通常处理 batch_size=1
                # 为了简便，目前处理 batch_size=1 或所有样本接受相同数量
                if batch_size == 1:
                    accepted_tensor = accepted_tokens_list[0].unsqueeze(0)
                    seq = torch.cat([seq, accepted_tensor], dim=1)
                else:
                    # 如果批处理大小 > 1 且接受长度不一致，需要复杂的 padding/attention_mask 更新
                    # 这里暂不展开复杂的 batch 处理
                    raise NotImplementedError("Batched speculative decoding with varying accepted lengths is not fully supported in this snippet.")
                    
        return seq

class MyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = f.readlines()
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = json.loads(self.datas[index].strip())
        conversations = sample.get('conversations', [])
        
        if len(conversations) < 2:
            raise ValueError("Conversations must have at least user and assistant turns.")
            
        user = conversations[0]['content']
        assistant = conversations[1]['content']
        
        # 构造用户 Prompt
        q = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        # 构造助手回复，加上 EOS Token
        a = assistant + self.tokenizer.eos_token
        
        q_input_ids = self.tokenizer(q)['input_ids']
        a_input_ids = self.tokenizer(a)['input_ids']
        
        input_ids = q_input_ids + a_input_ids
        
        # 对于指令微调，只在 assistant 的回复部分计算 loss，user 输入部分 label 设为 -100
        labels = [-100] * len(q_input_ids) + a_input_ids
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
        
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        
        for feature in features:
            pad_len = max_len - len(feature['input_ids'])
            # 右填充 pad_token_id (input_ids) 和 -100 (labels)
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(feature['labels'] + [-100] * pad_len)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def train(config, model, dataloader, optimizer, writer, device, epochs, print_step, save_step, save_path):
    steps = 0
    model.train()
    
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 1. 计算主模型的输出
            main_hidden_output, main_head_output = model.main_model(input_ids, output_hidden_states=True)
            previous_hidden_output = main_hidden_output
            
            # 2. 依次计算各个 MTP Block 的 loss
            total_mtp_loss = 0
            for index in range(config.predict_token_num):
                # 调用 MTP Block 得到下一层的隐状态和预测 logits
                previous_hidden_output, mtp_head_output = model.mtp_model[index](input_ids, previous_hidden_output, index)
                
                # 对齐 labels 和 predictions
                # 第 index 个 MTP head 预测的是往后数第 index+1 个 token
                shift = index + 2  # index=0 预测下一个(偏移2)，index=1预测下下个(偏移3)
                
                # 截断以匹配对应的 labels 长度
                mtp_head_output = mtp_head_output[:, :-shift, :].contiguous()
                target = labels[:, shift:].contiguous()
                
                mtp_head_output = mtp_head_output.view(-1, model.main_model.config.vocab_size)
                target = target.view(-1)
                
                mtp_loss = F.cross_entropy(mtp_head_output, target, ignore_index=-100)
                total_mtp_loss += mtp_loss
                
                # 论文建议 retain_graph=True 或累加 loss 后最后一起 backward，这里采用累加方式
                # 如果显存不够，也可以按你的方式逐个 backward(retain_graph=True)
                
            # 3. 计算主模型的 loss (预测下一个 token)
            main_head_output = main_head_output[:, :-1, :].contiguous()
            main_target = labels[:, 1:].contiguous()
            
            main_loss = F.cross_entropy(
                main_head_output.view(-1, model.main_model.config.vocab_size), 
                main_target.view(-1), 
                ignore_index=-100
            )
            
            # 4. 汇总所有 Loss 并反向传播
            # loss 比例可以根据需要调整，这里赋予相同的权重 1.0
            total_loss = main_loss + total_mtp_loss
            total_loss.backward()
            
            optimizer.step()
            
            # 5. 日志打印与模型保存
            if (steps + 1) % print_step == 0:
                avg_mtp_loss = (total_mtp_loss / config.predict_token_num).item()
                writer.add_scalar('Loss/main_loss', main_loss.item(), steps)
                writer.add_scalar('Loss/mtp_loss_avg', avg_mtp_loss, steps)
                writer.add_scalar('Loss/total_loss', total_loss.item(), steps)
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(dataloader)}], "
                      f"Total Loss: {total_loss.item():.4f}, Main Loss: {main_loss.item():.4f}, "
                      f"Avg MTP Loss: {avg_mtp_loss:.4f}")
                
            if (steps + 1) % save_step == 0:
                save_file = os.path.join(save_path, f"model_step_{steps+1}.pth")
                torch.save(model.state_dict(), save_file)
                print(f"Model saved at step {steps+1} to {save_file}")
            
            steps += 1  

if __name__ == '__main__':
    # 1. 初始化配置与日志
    writer = SummaryWriter('./runs')
    config = Config(llm_model_path="your_llm_path", predict_token_num=1) # Replace with actual path
    
    # 2. 准备模型
    model = MTP(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params / 1e6:.2f} M')
    
    # 3. 准备数据
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    # 若模型没有pad_token，设置eos_token为pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = MyDataset('lora_medical.jsonl', tokenizer)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=MyDataCollator(tokenizer)
    )
    
    # 4. 准备优化器并开始训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_path = './mtp_checkpoints'
    os.makedirs(save_path, exist_ok=True)
    
    print("Starting training...")
    train(
        config=config, 
        model=model, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        writer=writer, 
        device=device, 
        epochs=10, 
        print_step=10, 
        save_step=500, 
        save_path=save_path
    )