"""
Cross-Attention (Encoder-Decoder Attention) 面试核心：

  Q 来自 Decoder，K/V 来自 Encoder。
  Decoder 的每个位置通过注意力「查询」源序列中最相关的部分。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        decoder_hidden: (B, tgt_len, D)  — Q 的来源
        encoder_output: (B, src_len, D)  — K/V 的来源
        encoder_mask:   (B, 1, 1, src_len)  padding 位置填 -inf
        """
        B, tgt_len, _ = decoder_hidden.shape
        src_len = encoder_output.size(1)

        q = self.q_proj(decoder_hidden).view(B, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_output).view(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_output).view(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if encoder_mask is not None:
            scores = scores + encoder_mask

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(B, tgt_len, -1)
        return self.o_proj(context)


if __name__ == "__main__":
    B, src_len, tgt_len, dim, heads = 2, 12, 8, 64, 8
    enc = torch.randn(B, src_len, dim)
    dec = torch.randn(B, tgt_len, dim)

    # 模拟 padding mask：最后 3 个 src token 是 padding
    mask = torch.zeros(B, 1, 1, src_len)
    mask[:, :, :, -3:] = float("-inf")

    cross_attn = CrossAttention(dim, heads)
    out = cross_attn(dec, enc, encoder_mask=mask)
    print("Cross-Attention output shape:", out.shape)
    assert out.shape == (B, tgt_len, dim)
    print("[验证成功] Cross-Attention 前向正常")
