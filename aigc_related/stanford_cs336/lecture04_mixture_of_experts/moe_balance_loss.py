import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

        # 1. 路由层 (Router / Gate)
        # 输入: [..., input_dim] -> 输出: [..., num_experts]
        self.router = nn.Linear(input_dim, num_experts)

        # 2. 专家层 (Experts)
        # 这里的专家是个简单的 MLP: input -> 4*input -> input
        # 我们用 ModuleList 存一堆小网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.ReLU(),
                nn.Linear(input_dim * 4, input_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x 的形状: [Batch_Size, Seq_Len, Dim] -> [B, L, D]
        batch_size, seq_len, dim = x.shape
        
        # 为了方便计算，先把 B 和 L 展平: [B*L, D]
        # flat_x 代表所有的 Token
        flat_x = x.view(-1, dim) 
        total_tokens = flat_x.size(0)

        # ==================================================
        # 第一步: 路由打分 (Router)
        # ==================================================
        # router_logits: [Total_Tokens, Num_Experts]
        router_logits = self.router(flat_x)
        
        # 计算概率 (Softmax)
        router_probs = F.softmax(router_logits, dim=-1)

        # ==================================================
        # 第二步: 选 Top-K
        # ==================================================
        # weights: 选中的 K 个专家的权重 (概率值)
        # indices: 选中的 K 个专家的 ID
        # 形状都是: [Total_Tokens, Top_K]
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # ==================================================
        # 第三步: 计算辅助损失 (Auxiliary Loss)
        # 公式: alpha * N * sum(f_i * P_i)
        # ==================================================
        
        # 1. P_i (Intention): Router 对每个专家的平均概率预期
        # 形状: [Num_Experts]
        P = router_probs.mean(dim=0)
        
        # 2. f_i (Reality): 实际选中每个专家的频率
        # 我们需要把 topk_indices 变成 one-hot 编码来统计次数
        # topk_indices 形状 [Total_Tokens, Top_K] -> 展平看所有选票
        # mask 形状: [Total_Tokens * Top_K, Num_Experts]
        mask = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts).float()
        
        # 统计每个专家得到了多少张票，除以总票数 (或总Token数)
        # 注意: 这里通常分母用 total_tokens
        f = mask.sum(dim=0) / total_tokens
        
        # 3. 计算点积并放大
        # 这里 alpha 设为 0.01 (常见的超参数)
        aux_loss = 0.01 * self.num_experts * torch.sum(P * f)

        # ==================================================
        # 第四步: 专家干活 (Expert Computation)
        # ==================================================
        
        # 这是一个用来存结果的容器，形状和输入一样
        final_output = torch.zeros_like(flat_x)
        
        # 我们重新归一化一下选中的权重，让它们加起来等于1 (这是一个常见Trick，但不是必须)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 循环遍历每一个专家 (这是 Python 循环实现的简单版，实际 CUDA 会并行)
        for i, expert in enumerate(self.experts):
            # 找出哪些 Token 选中了当前专家 i
            # topk_indices: [Total_Tokens, K]
            # expert_mask: [Total_Tokens, K] (布尔值)
            expert_mask = (topk_indices == i)
            
            # 只要这一行里有 True，就说明这个 Token 的 K 个选择里包含专家 i
            # token_indices_for_expert: [N_selected] (选了专家i的 token 的行号)
            batch_idx, kth_choice = torch.where(expert_mask)
            
            if len(batch_idx) == 0:
                continue # 如果没人选这个专家，就跳过
            
            # 1. 拿出这些 Token 的输入数据
            # inputs: [N_selected, Dim]
            inputs = flat_x[batch_idx]
            
            # 2. 让专家 i 计算
            # output: [N_selected, Dim]
            expert_out = expert(inputs)
            
            # 3. 乘以路由权重
            # 我们需要找到对应的权重。
            # 比如 Token 5 在 Top-K 的第 0 个位置选了专家 i，那就要取 weights[5, 0]
            current_weights = topk_weights[batch_idx, kth_choice].unsqueeze(1) # [N_selected, 1]
            
            weighted_output = expert_out * current_weights
            
            # 4. 把结果加回到总结果里 (Scatter / Add)
            # 可能会有多个专家处理同一个 Token，所以用 index_add (累加)
            final_output.index_add_(0, batch_idx, weighted_output)

        # 变回 [B, L, D]
        final_output = final_output.view(batch_size, seq_len, dim)
        
        return final_output, aux_loss, router_logits

# ==========================================
# 模拟运行与讲解
# ==========================================

# 1. 设置参数
B, L, dim = 2, 16, 128   # Batch=2, Length=3, Dim=4 (总共 6 个 Token)
num_experts = 4          # 4 个专家
top_k = 2                # 每个 Token 选 2 个专家

# 2. 初始化模型
moe = SimpleMoE(input_dim=dim, num_experts=num_experts, top_k=top_k)

# 3. 随机生成输入数据 [B, L, dim]
x = torch.randn(B, L, dim)
print(f"输入 x 形状: {x.shape} \n{x}\n")

# 4. 前向传播
output, loss, logits = moe(x)

print("-" * 50)
print("1. Router输出 (Logits) [Total_Tokens, Num_Experts]:")
# 展平以便观察
print(logits.detach()) 
# 这里你可以看到每个 Token 对 4 个专家的打分

print("\n2. 辅助损失 (Aux Loss):")
print(f"Loss = {loss.item():.6f}")

print("\n3. 最终输出 [B, L, dim]:")
print(output.shape)
print(output.detach())