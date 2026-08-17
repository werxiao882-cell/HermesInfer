import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekV3Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, bias_update_rate=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate # 偏置调整的步长

        # 1. 路由层 (只负责算原始分 s)
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 2. 动态偏置 (Bias) - 这是一个可学习或动态更新的参数
        # 这里的 bias 不参与主模型的梯度下降，所以我们要么设为 Buffer，要么在 step 时手动改
        self.register_buffer('expert_bias', torch.zeros(num_experts))

    def forward(self, x, training=True):
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        flat_x = x.view(-1, dim)

        # --- A. 计算原始分数 (logits) ---
        # logits: [Total_Tokens, Num_Experts]
        original_logits = self.gate(flat_x)

        # --- B. 加上动态偏置 (Bias) 进行路由选择 ---
        # 这里的 bias 广播加到每一个 token 的分数上
        # biased_logits 用于 Top-K 选择
        biased_logits = original_logits + self.expert_bias.unsqueeze(0)

        # --- C. Top-K 选择 ---
        # 选 Top-K 时用的是"修正后的分" (biased_logits)
        topk_probs, topk_indices = torch.topk(torch.sigmoid(biased_logits), k=self.top_k, dim=-1)
        # 注意：DeepSeek V3 使用 Sigmoid 而不是 Softmax，且通常只归一化选中的 K 个
        # 这里为了演示简单，保留标准 Softmax 后的归一化逻辑
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        aux_loss = 0.0
        if training:
            # --- D. 动态更新 Bias (核心策略) ---
            # 统计当前 Batch 每个专家的负载
            # mask: [Total_Tokens, Num_Experts]
            mask = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts).float()
            # current_load: 每个专家实际拿到的 Token 比例
            current_load = mask.sum(dim=0) # [Num_Experts]
            
            # 目标负载: 大家平分
            target_load = (batch_size * seq_len * self.top_k) / self.num_experts
            
            # 误差: 实际 - 目标 (正数说明过载，负数说明空闲)
            load_error = current_load - target_load
            
            # 调整 Bias: 
            # 负载太高 -> load_error > 0 -> 减 bias
            # 负载太低 -> load_error < 0 -> 加 bias
            # 注意：这通常是一个无梯度的原地操作 (In-place update)
            with torch.no_grad():
                self.expert_bias -= self.bias_update_rate * torch.sign(load_error)
                # 或者用 self.expert_bias -= self.bias_update_rate * load_error (比例更新)

            # --- E. 序列级互补损失 (Sequence-Wise Aux Loss) ---
            # 为了防止单条序列内偏科
            # 我们只在一个 Sequence 内部计算负载均衡 Loss
            
            # P: 原始分数的 Sigmoid/Softmax (代表 Router 真实意图)
            # 注意用 original_logits 计算，不要被 bias 影响梯度的方向
            P = torch.sigmoid(original_logits).view(batch_size, seq_len, -1)
            
            # 计算每个序列内部的平均概率
            # seq_P: [Batch, Num_Experts]
            seq_P = P.mean(dim=1) 
            
            # f: 每个序列内部的实际选择频率
            # seq_f: [Batch, Num_Experts]
            seq_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
            seq_f = seq_mask.mean(dim=1)
            
            # 计算 Loss: 对每个序列算点积，然后取 Batch 平均
            # 这里的 Loss 系数通常非常小 (比如 0.001)
            seq_aux_loss = torch.sum(seq_P * seq_f, dim=1).mean() * self.num_experts
            aux_loss = 0.1 * seq_aux_loss # 假设系数 0.1

        return topk_indices, topk_weights, aux_loss

# --- 模拟运行 ---
router = DeepSeekV3Router(input_dim=128, num_experts=8, top_k=2)
x = torch.randn(4, 10, 128) # 4条序列，每条10个词

# 打印初始 Bias
print(f"初始 Bias: {router.expert_bias}")

# 模拟一步训练
indices, weights, loss = router(x)

# 打印更新后的 Bias
print(f"更新后 Bias: {router.expert_bias}")
print(f"序列级 Loss: {loss:.6f}")