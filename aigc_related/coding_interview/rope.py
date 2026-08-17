"""
RoPE (Rotary Position Embedding) 面试核心：

  对 Q/K 的每对维度 (2i, 2i+1) 做 2D 旋转，编码相对位置信息。
  绝对位置 p 的旋转角 = p * theta_i，其中 theta_i = 10000^(-2i/d)

  关键性质：RoPE(q_m, m) · RoPE(k_n, n) 只依赖 (m - n)，即相对位置。
"""
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 cos/sin 表，dim 必须是偶数。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, num_heads, seq_len, head_dim)
    cos/sin: (seq_len, head_dim/2) — 广播到 batch 和 head
    """
    # 拆成偶/奇维度对
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # 2D 旋转: [x1', x2'] = [cos*x1 - sin*x2, sin*x1 + cos*x2]
    rotated = torch.stack(
        [cos * x1 - sin * x2, sin * x1 + cos * x2],
        dim=-1,
    )
    return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        cos, sin = precompute_freqs_cis(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(2)
        return (
            apply_rotary_emb(q, self.cos[:seq_len], self.sin[:seq_len]),
            apply_rotary_emb(k, self.cos[:seq_len], self.sin[:seq_len]),
        )


if __name__ == "__main__":
    batch, heads, seq, head_dim = 2, 4, 16, 32
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)

    rope = RotaryEmbedding(head_dim)
    q_rot, k_rot = rope(q, k)

    print("RoPE output shape:", q_rot.shape)
    assert q_rot.shape == q.shape

    # 相对位置性质：RoPE(q, pos_m) · RoPE(k, pos_n) 只依赖 m-n
    # 简单 sanity check：不同位置的 q/k 点积应随距离变化
    dot_same = (q_rot[0, 0, 5] * k_rot[0, 0, 5]).sum()
    dot_diff = (q_rot[0, 0, 5] * k_rot[0, 0, 8]).sum()
    print(f"Dot product (same pos): {dot_same.item():.4f}")
    print(f"Dot product (diff pos): {dot_diff.item():.4f}")
    print("[验证成功] RoPE 前向正常")
