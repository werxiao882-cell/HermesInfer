import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, in_dim=16, dim=1536, patch_size=(1, 2, 2)):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.proj = nn.Linear(in_dim * pt * ph * pw, dim)

    def forward(self, x):
        """
        x: (B, C, T, H, W) latent tensor
        Returns: (B, num_tokens, dim) patch tokens
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size
        assert T % pt == 0 and H % ph == 0 and W % pw == 0

        x = x.view(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, (T // pt) * (H // ph) * (W // pw), C * pt * ph * pw)

        return self.proj(x)

    def unpatchify(self, x, T, H, W):
        """
        x: (B, num_tokens, out_dim) patch tokens
        Returns: (B, out_dim, T, H, W) unpatchified tensor
        """
        B, N, out_dim = x.shape
        pt, ph, pw = self.patch_size
        Tt, Ht, Wt = T // pt, H // ph, W // pw

        x = x.view(B, Tt, Ht, Wt, pt, ph, pw, out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, out_dim, T, H, W)

        return x
