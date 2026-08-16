import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Handle both 3D varlen (total_tokens, num_heads, head_dim) and 4D batched (B, seq_len, num_heads, head_dim)
    if x.dim() == 3:
        # Varlen mode: (total_tokens, num_heads, head_dim)
        total_tokens, num_heads, head_dim = x.shape
        # cos, sin shape: (total_tokens, head_dim/2)
        # Expand to (total_tokens, 1, head_dim/2) for broadcasting
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embedding
        # x1, x2 shape: (total_tokens, num_heads, head_dim/2)
        # cos, sin shape: (total_tokens, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)
    else:
        # Batched mode: (B, seq_len, num_heads, head_dim)
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # Expand cos and sin to match the batch and head dimensions
        # cos, sin shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embedding with proper broadcasting
        # x1, x2 shape: (B, seq_len, num_heads, head_dim/2)
        # cos, sin shape: (1, seq_len, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        base:int,
        rotary_embedding: int,
        max_position: int = 2048,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.base = base
        # how many dimensions to apply rotary embedding
        self.rotary_embedding = rotary_embedding
        # max position that the long context can reach
        self.max_position = max_position
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # specifically for llama3.2
            import math
            inv_freq = self.inv_freq
            # no smooth if low_freq_factor == high_freq_factor
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        positions = torch.arange(self.max_position).float()
        # (max_position, rotary_embedding/2)
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # (max_position, rotary_embedding)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    # tell the position index of the token
    # apply rotary embedding to query and key
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # (seq_len, rotary_embedding)
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


# Multimodal Rotary Embedding (M-RoPE) for Qwen3-VL.
# Splits head_dim into 3 sections (T, H, W) per mrope_section and interleaves the
# frequencies: [T0,H0,W0,T1,H1,W1,...,T19,H19,W19,T20,T21,T22,T23] for section
# [24,20,20]. Positions are 3D (3, seq) for T/H/W. The resulting cos/sin are
# (seq, head_dim//2) and reuse apply_rotary_pos_emb (GPT-J rotation), which is
# mathematically equivalent to transformers' cat(freqs,freqs) + rotate_half.
class MRotaryEmbedding(nn.Module):
    def __init__(self, base: int, head_dim: int, mrope_section: list[int]):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        self.mrope_section = mrope_section  # e.g. [24, 20, 20]
        # inv_freq for the full head_dim; sections index into it
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_perm()

    def _build_perm(self):
        # 构造两个查找表(都是 head_dim//2 维):
        #   mrope_perm[out_i]      = 该输出槽取哪个 inv_freq 源索引
        #   mrope_section_id[out_i]= 该输出槽属于哪段(0=T,1=H,2=W),用于选位置
        #
        # 输出布局(对齐 transformers apply_interleaved_mrope):
        #   [T0,H0,W0, T1,H1,W1, ..., T(min-1),H(min-1),W(min-1),  尾部 T...]
        # section=[24,20,20] 时:min=20,前 20 组三轴交错占 60 槽,尾部 T 占 4 槽(T20..T23),共 64=head_dim//2。
        # 即:输出槽 0..59 是 [T0,H0,W0,...,T19,H19,W19],槽 60..63 是 [T20,T21,T22,T23]。
        s = self.mrope_section
        t_idx = list(range(0, s[0]))                       # inv_freq[0..23]  -> T 的 24 个频率
        h_idx = list(range(s[0], s[0] + s[1]))            # inv_freq[24..43] -> H 的 20 个
        w_idx = list(range(s[0] + s[1], sum(s)))          # inv_freq[44..63] -> W 的 20 个
        perm = []
        for i in range(min(s)):                            # 前 20 组:三轴交错
            perm += [t_idx[i], h_idx[i], w_idx[i]]
        perm += t_idx[min(s):]                             # 尾部 T(24-20=4 个)
        # 由源索引反推段号(用于在 forward 里选 T/H/W 哪条位置轴)
        section = torch.zeros(len(perm), dtype=torch.long)
        for out_i, src in enumerate(perm):
            if src < s[0]:
                section[out_i] = 0      # T
            elif src < s[0] + s[1]:
                section[out_i] = 1      # H
            else:
                section[out_i] = 2      # W
        self.register_buffer("mrope_perm", torch.tensor(perm, dtype=torch.long), persistent=False)
        self.register_buffer("mrope_section_id", section, persistent=False)

    @torch.compile
    def forward(self, positions_3d, query, key):
        # positions_3d: (3, seq) = (T, H, W) 三条位置轴(由 compute_mrope_positions 算出)
        # 目标:产出 (seq, head_dim//2) 的 cos/sin,布局为交错 T/H/W,复用
        # apply_rotary_pos_emb(GPT-J 旋转),与 transformers 的 cat(freqs,freqs)+rotate_half 等价。
        #   证明:transformers cos[i]=cos[i+d/2]=cos(freq_i),project cos[i]=cos(freq_i),同;
        #   旋转都是 pair(x[i],x[i+d/2]) 用 θ_i,故两者等价。
        inv_freq_perm = self.inv_freq[self.mrope_perm]            # (head_dim//2,) 每槽的频率
        # 按段号选位置:positions_3d[section] -> (head_dim//2, seq),每槽用对应 T/H/W 的位置
        pos = positions_3d[self.mrope_section_id]                 # (head_dim//2, seq)
        freqs = inv_freq_perm[:, None] * pos                      # (head_dim//2, seq) 角度
        freqs = freqs.t()                                         # (seq, head_dim//2)
        cos = freqs.cos()
        sin = freqs.sin()
        # 复用既有 GPT-J 旋转(x1,x2=chunk(2);out=cat([x1*cos-x2*sin, x1*sin+x2*cos]))
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin),
        )


# Rotary embedding for the ViT tower (Qwen3VLVisionRotaryEmbedding).
# dim = head_dim (full), theta=10000. Produces cos/sin (seq, head_dim) via
# cat(freqs, freqs); rotation is rotate-half (GPT-NeoX). Provided separately
# because ViT positions come from the patch grid, not text positions.
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids:(seq, 2) = [hpos, wpos],来自 patch 网格的 2D 坐标(行主序,
        # 对齐 Conv3d 的 row-major patch 顺序)。dim=head_dim//2,inv_freq 有 dim//2 个。
        # (seq,2,1) * inv_freq(dim//2,) -> (seq,2,dim//2) -> flatten(1) -> (seq, dim):
        #   前 dim//2 维用 h 位置,后 dim//2 维用 w 位置;再 cat 自身成 (seq, 2*dim=head_dim),
        #   供 rotate-half 旋转。故 H/W 各占 head_dim/4 频率(对齐 transformers
        #   Qwen3VLVisionRotaryEmbedding(head_dim//2) + get_vision_position_ids 的 2D 布局)。
        # 注:transformers 用 block-major 顺序(为 merger 的 view(-1,4*hidden));本实现
        # 用 row-major(对齐本项目的 _merge2x2 按 (t,h,w) reshape)。二者自洽,注意力全双向,
        # 数值与 transformers 严格对齐需 GPU 核对(test_parity_qwen)。
        freqs = (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)  # (seq, dim)
        return torch.cat([freqs, freqs], dim=-1)  # (seq, 2*dim) = (seq, head_dim)


if __name__ == "__main__":
    base = 5
    # how many dimensions to apply rotary embedding
    rotary_dim = 16
    # maximum position that the long context can reach
    max_position = 100
    print(torch.arange(0, rotary_dim, 2))
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    t = torch.arange(max_position).float()

    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    print(freqs[2])

