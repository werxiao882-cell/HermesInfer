import torch
import torch.nn as nn
from .group import usp_world_size
from .ulysses import all_to_all_seq2head, all_to_all_head2seq
from myvllm.layers.attention import flash_attention_prefill


class USPAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        P = usp_world_size()
        assert num_heads % P == 0, f"num_heads ({num_heads}) must be divisible by world_size ({P})"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_heads = num_heads // P
        self.scale = 1.0 / (head_dim ** 0.5)

        self.qkv = nn.Linear(dim, 3 * self.local_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.local_heads * head_dim, dim, bias=False)

    def forward(self, x, cu_seqlens, positions_3d=None, rope_fn=None):
        """
        x: (seq//P, dim) — sequence-sharded input
        cu_seqlens: cumulative sequence lengths for varlen flash attention
        positions_3d: (3, seq//P) T/H/W positions for 3D RoPE (optional)
        rope_fn: callable(positions, q, k) -> (q, k) for applying RoPE (optional)

        Returns: (seq//P, dim) — sequence-sharded output
        """
        seq_per_rank, dim = x.shape
        P = usp_world_size()
        seq_full = seq_per_rank * P

        qkv = self.qkv(x)
        q, k, v = qkv.split([self.local_heads * self.head_dim] * 3, dim=-1)

        q = q.view(seq_per_rank, self.local_heads, self.head_dim)
        k = k.view(seq_per_rank, self.local_heads, self.head_dim)
        v = v.view(seq_per_rank, self.local_heads, self.head_dim)

        if rope_fn is not None and positions_3d is not None:
            q, k = rope_fn(positions_3d, q, k)

        q = all_to_all_seq2head(q, self.num_heads)
        k = all_to_all_seq2head(k, self.num_heads)
        v = all_to_all_seq2head(v, self.num_heads)

        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        cu_seqlens_full = cu_seqlens
        if P > 1:
            cu_seqlens_full = cu_seqlens * P

        o = flash_attention_prefill(
            q, k, v, cu_seqlens_full, self.scale,
            num_heads=self.local_heads,
            num_kv_heads=self.local_heads,
            head_dim=self.head_dim,
            is_causal=False,
        )

        o = o.transpose(0, 1).contiguous()
        o = all_to_all_head2seq(o, self.num_heads)
        o = o.reshape(seq_per_rank, self.local_heads * self.head_dim)

        return self.o_proj(o)
