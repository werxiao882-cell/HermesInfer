"""
KV Cache —— 自回归推理加速的核心思想。

  问题：生成第 t 个 token 时，前面 t-1 个 token 的 K/V 每次都重算，浪费。
  解法：缓存历史 K/V，每步只算新 token 的 Q/K/V，把新 K/V append 到 cache。

  复杂度：无 cache O(T^2)，有 cache 每步 O(T)，总 O(T^2) 但常数小很多。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    def __init__(self):
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """k/v: (B, num_heads, seq_len, head_dim)"""
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache = self.v_cache = None


class CachedAttention(nn.Module):
    """带 KV Cache 的单头注意力（面试简化版，逻辑与多头一致）。"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ):
        B, seq_len, _ = x.shape
        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache and cache is not None:
            k, v = cache.update(k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(out), cache


def autoregressive_generate(model: CachedAttention, prompt: torch.Tensor, max_new: int = 5):
    """逐 token 生成，演示 KV Cache 用法。"""
    cache = KVCache()
    # Prefill：一次性处理 prompt
    out, cache = model(prompt, cache=cache, use_cache=True)

    generated = [prompt]
    next_token = out[:, -1:, :]  # 取最后一个 hidden state 作为下一个 token 的输入（简化）

    for _ in range(max_new):
        out, cache = model(next_token, cache=cache, use_cache=True)
        generated.append(next_token)
        next_token = out  # 简化：直接用 output 作为下一步输入

    return torch.cat(generated, dim=1)


if __name__ == "__main__":
    B, dim, heads = 1, 64, 4
    attn = CachedAttention(dim, heads)

    prompt = torch.randn(B, 3, dim)
    cache = KVCache()

    # Prefill
    out1, cache = attn(prompt, cache=cache, use_cache=True)
    cached_len = cache.k_cache.size(2)

    # Decode one step
    new_token = torch.randn(B, 1, dim)
    out2, cache = attn(new_token, cache=cache, use_cache=True)

    print(f"After prefill,  cache seq_len: {cached_len}")
    print(f"After 1 decode, cache seq_len: {cache.k_cache.size(2)}")
    assert cache.k_cache.size(2) == cached_len + 1
    print("[验证成功] KV Cache append 正常")
