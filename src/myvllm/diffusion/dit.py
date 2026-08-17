import torch
import torch.nn as nn
import math
from .patch_embed import PatchEmbed3D
from .block import WanDiTBlock
from myvllm.layers import LayerNorm
from myvllm.layers.video_rope import VideoRotaryEmbedding


class TimestepEmbedding(nn.Module):
    def __init__(self, freq_dim=256, dim=1536):
        super().__init__()
        self.freq_dim = freq_dim
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        """
        t: (B,) scalar timesteps
        Returns: (B, dim) timestep embedding
        """
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class WanDiT(nn.Module):
    def __init__(
        self,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        in_dim=16,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        text_len=512,
        patch_size=(1, 2, 2),
        head_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.freq_dim = freq_dim

        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim

        self.patch_embed = PatchEmbed3D(in_dim=in_dim, dim=dim, patch_size=patch_size)
        self.timestep_embed = TimestepEmbedding(freq_dim=freq_dim, dim=dim)

        self.text_proj = nn.Linear(dim, dim, bias=False)

        self.rope = VideoRotaryEmbedding(freq_dim=freq_dim, head_dim=head_dim)

        self.blocks = nn.ModuleList([
            WanDiTBlock(dim, num_heads, head_dim, ffn_dim, freq_dim)
            for _ in range(num_layers)
        ])

        self.norm_out = LayerNorm(torch.ones(dim))
        self.final_proj = nn.Linear(dim, out_dim * patch_size[0] * patch_size[1] * patch_size[2])

        self.patch_size = patch_size

    def forward(self, z_t, t, text_emb, positions_3d, cu_seqlens):
        """
        z_t: (B, in_dim, T', H', W') noisy latent
        t: (B,) timesteps
        text_emb: (B, text_len, dim) text embeddings
        positions_3d: (3, seq//P) T/H/W positions
        cu_seqlens: cumulative sequence lengths

        Returns: (B, out_dim, T', H', W') velocity prediction
        """
        B, C, T, H, W = z_t.shape

        x = self.patch_embed(z_t)
        x = x.squeeze(0)

        t_emb = self.timestep_embed(t)

        text_emb = self.text_proj(text_emb)

        for block in self.blocks:
            x = block(x, t_emb, text_emb, cu_seqlens, positions_3d, self.rope)

        x = self.norm_out(x)
        x = self.final_proj(x)

        x = x.unsqueeze(0)
        pt, ph, pw = self.patch_size
        out_dim = self.out_dim
        x = x.view(B, T // pt, H // ph, W // pw, pt, ph, pw, out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, out_dim, T, H, W)

        return x
