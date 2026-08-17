import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedAttention(nn.Module):
    """
    统一的注意力机制实现，通过控制 num_kv_heads 来支持 MHA, MQA, GQA。
    
    MHA (Multi-Head Attention): num_kv_heads = num_heads
    MQA (Multi-Query Attention): num_kv_heads = 1
    GQA (Grouped-Query Attention): 1 < num_kv_heads < num_heads 且能被 num_heads 整除
    """
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 如果未指定，则默认为标准多头注意力 MHA
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        assert self.num_heads % self.num_kv_heads == 0, "num_heads 必须被 num_kv_heads 整除"
        
        self.head_dim = embed_dim // num_heads
        self.num_groups = self.num_heads // self.num_kv_heads  # 每个 KV head 对应的 Q head 数量
        
        # 定义线性映射层
        # 注意: Q 需要映射到 num_heads，而 K 和 V 只映射到 num_kv_heads
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        用于将 KV 的 head 复制 n_rep 次，以匹配 Query 的 head 数量。
        hidden_states shape: (batch_size, num_kv_heads, seq_len, head_dim)
        return shape: (batch_size, num_heads, seq_len, head_dim)
        """
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        # (batch_size, num_kv_heads, 1, seq_len, head_dim)
        hidden_states = hidden_states[:, :, None, :, :]
        # 扩展并展平以匹配 num_heads
        hidden_states = hidden_states.expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影并 reshape
        # 结果 shape: (batch_size, num_heads(或 num_kv_heads), seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. 如果是 GQA/MQA，需要重复 KV 的 heads 以匹配 Q 的 heads
        k = self.repeat_kv(k, self.num_groups)  # (batch_size, num_heads, seq_len, head_dim)
        v = self.repeat_kv(v, self.num_groups)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 3. 计算注意力分数: Q * K^T / sqrt(d_k)
        # q: (batch_size, num_heads, seq_len, head_dim)
        # k.transpose(-2, -1): (batch_size, num_heads, head_dim, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # 假设传入的 attention_mask shape 是 (batch_size, 1, 1, seq_len) 或者相容的 shape
            scores = scores + attention_mask
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. 乘以 V 得到上下文向量
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim) 
        # -> (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)
        
        # 5. 拼接多头并进行输出投影
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        
        return output

# ==========================================
# 面试时可以提供的三个包装类，更清晰地展示区别
# ==========================================

class MultiHeadAttention(UnifiedAttention):
    """标准的 MHA，num_kv_heads 等于 num_heads"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__(embed_dim, num_heads, num_kv_heads=num_heads, dropout=dropout)

class MultiQueryAttention(UnifiedAttention):
    """MQA，所有 Query head 共享 1 个 KV head"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__(embed_dim, num_heads, num_kv_heads=1, dropout=dropout)

class GroupedQueryAttention(UnifiedAttention):
    """GQA，多组 Query 共享一个 KV head"""
    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__(embed_dim, num_heads, num_kv_heads=num_kv_heads, dropout=dropout)

# ==========================================
# 测试代码（方便面试官运行验证）
# ==========================================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # MHA: 8个Q头，8个KV头
    mha = MultiHeadAttention(embed_dim, num_heads)
    out_mha = mha(x)
    print("MHA output shape:", out_mha.shape)  # (2, 10, 64)
    
    # MQA: 8个Q头，1个KV头
    mqa = MultiQueryAttention(embed_dim, num_heads)
    out_mqa = mqa(x)
    print("MQA output shape:", out_mqa.shape)  # (2, 10, 64)
    
    # GQA: 8个Q头，2个KV头 (每4个Q头共享1个KV头)
    gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=2)
    out_gqa = gqa(x)
    print("GQA output shape:", out_gqa.shape)  # (2, 10, 64)
