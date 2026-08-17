import torch
import torch.nn as nn
import torch.nn.functional as F
from myvllm.layers import LayerNorm, SiluAndMul
from myvllm.usp import USPAttention


class WanDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, ffn_dim, freq_dim=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm1 = LayerNorm(torch.ones(dim))
        self.attn = USPAttention(dim, num_heads, head_dim)

        self.norm2 = LayerNorm(torch.ones(dim))
        self.cross_attn_q = nn.Linear(dim, dim, bias=False)
        self.cross_attn_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_attn_o = nn.Linear(dim, dim, bias=False)
        self.cross_attn_norm = LayerNorm(torch.ones(dim))

        self.ffn_norm = LayerNorm(torch.ones(dim))
        self.ffn_gate_up = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_act = SiluAndMul()
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(freq_dim, 6 * dim, bias=True),
        )

    def forward(self, x, t_emb, text_emb, cu_seqlens, positions_3d=None, rope_fn=None):
        """
        x: (seq//P, dim)
        t_emb: (B, freq_dim) timestep embedding
        text_emb: (B, text_len, dim) text embeddings for cross-attention
        cu_seqlens: cumulative sequence lengths
        positions_3d: (3, seq//P) for 3D RoPE
        rope_fn: callable for applying RoPE
        """
        modulation = self.adaLN_modulation(t_emb)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)

        shift1 = shift1.squeeze(0).unsqueeze(0)
        scale1 = scale1.squeeze(0).unsqueeze(0)
        gate1 = gate1.squeeze(0).unsqueeze(0)
        shift2 = shift2.squeeze(0).unsqueeze(0)
        scale2 = scale2.squeeze(0).unsqueeze(0)
        gate2 = gate2.squeeze(0).unsqueeze(0)

        residual = x
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale1) + shift1
        attn_out = self.attn(x_mod, cu_seqlens, positions_3d, rope_fn)
        x = residual + gate1 * attn_out

        residual = x
        x_norm = self.norm2(x)
        cross_out = self._cross_attention(x_norm, text_emb)
        x = residual + cross_out

        residual = x
        x_norm = self.ffn_norm(x)
        x_mod = x_norm * (1 + scale2) + shift2
        gate_up = self.ffn_gate_up(x_mod)
        act = self.ffn_act(gate_up)
        ffn_out = self.ffn_down(act)
        x = residual + gate2 * ffn_out

        return x

    def _cross_attention(self, x, text_emb):
        """
        x: (seq//P, dim)
        text_emb: (B, text_len, dim)
        """
        seq_per_rank, dim = x.shape
        B, text_len, _ = text_emb.shape

        q = self.cross_attn_q(x).view(seq_per_rank, self.num_heads, self.head_dim)
        kv = self.cross_attn_kv(text_emb.squeeze(0))
        k, v = kv.chunk(2, dim=-1)
        k = k.view(text_len, self.num_heads, self.head_dim)
        v = v.view(text_len, self.head_dim * self.num_heads)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.view(text_len, self.num_heads, self.head_dim).transpose(0, 1)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(0, 1).reshape(seq_per_rank, dim)
        return self.cross_attn_o(out)
