import torch
import torch.nn as nn


class WanVAE(nn.Module):
    def __init__(self, model_name_or_path="Wan-AI/Wan2.1-T2V-14B"):
        super().__init__()
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained(model_name_or_path, subfolder="vae")
        self.vae.eval()

    @torch.no_grad()
    def encode(self, video, device="cuda"):
        """
        video: (B, T, H, W, 3) uint8 or (B, 3, T, H, W) float [-1, 1]
        Returns: (B, 16, T', H', W') latent
        """
        if video.dim() == 5 and video.shape[-1] == 3:
            video = video.permute(0, 4, 1, 2, 3).float() / 127.5 - 1.0

        video = video.to(device)
        latents = self.vae.encode(video).latent_dist.sample()
        return latents

    @torch.no_grad()
    def decode(self, latents, device="cuda"):
        """
        latents: (B, 16, T', H', W')
        Returns: (B, 3, T, H, W) float [-1, 1]
        """
        latents = latents.to(device)
        video = self.vae.decode(latents).sample
        return video

    def to(self, device):
        self.vae = self.vae.to(device)
        return self
