import torch
import torch.nn as nn
import math


class VideoRotaryEmbedding(nn.Module):
    def __init__(self, freq_dim: int = 256, head_dim: int = 128, base: float = 10000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, freq_dim, 2, dtype=torch.float32) / freq_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions_3d, query, key):
        """
        positions_3d: (3, seq) = (T, H, W) 三条位置轴
        query: (seq, num_heads, head_dim)
        key: (seq, num_heads, head_dim)

        Returns: (rotated_query, rotated_key)
        """
        t_pos, h_pos, w_pos = positions_3d[0], positions_3d[1], positions_3d[2]

        freq_len = self.freq_dim // 2
        t_freqs = torch.einsum("i,j->ij", t_pos.float(), self.inv_freq)
        h_freqs = torch.einsum("i,j->ij", h_pos.float(), self.inv_freq)
        w_freqs = torch.einsum("i,j->ij", w_pos.float(), self.inv_freq)

        t_cos, t_sin = t_freqs.cos(), t_freqs.sin()
        h_cos, h_sin = h_freqs.cos(), h_freqs.sin()
        w_cos, w_sin = w_freqs.cos(), w_freqs.sin()

        cos = torch.cat([t_cos, h_cos, w_cos], dim=-1)
        sin = torch.cat([t_sin, h_sin, w_sin], dim=-1)

        pad_len = self.head_dim // 2 - cos.shape[-1]
        if pad_len > 0:
            cos = torch.nn.functional.pad(cos, (0, pad_len), value=1.0)
            sin = torch.nn.functional.pad(sin, (0, pad_len), value=0.0)

        cos = cos[:self.head_dim // 2]
        sin = sin[:self.head_dim // 2]

        query = _apply_rotary(query, cos, sin)
        key = _apply_rotary(key, cos, sin)

        return query, key


def _apply_rotary(x, cos, sin):
    """
    x: (seq, num_heads, head_dim)
    cos: (seq, head_dim//2)
    sin: (seq, head_dim//2)
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    x1, x2 = x.chunk(2, dim=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)
