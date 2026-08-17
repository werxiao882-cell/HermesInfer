import torch
import torch.distributed as dist
from .dit import WanDiT
from .vae import WanVAE
from .scheduler import FlowScheduler
from myvllm.usp.group import usp_rank, usp_world_size


class T2VPipeline:
    def __init__(self, dit, vae, text_encoder, tokenizer, scheduler, device="cuda"):
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def encode_text(self, prompt, max_len=512):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state

    @torch.no_grad()
    def __call__(self, prompt, num_frames=81, height=480, width=832, seed=42):
        rank = usp_rank()
        P = usp_world_size()

        text_emb = self.encode_text(prompt)

        T_latent = (num_frames - 1) // 4 + 1
        H_latent = height // 8
        W_latent = width // 8

        generator = torch.Generator(device=self.device).manual_seed(seed)
        z_t = torch.randn(1, 16, T_latent, H_latent, W_latent, generator=generator, device=self.device)

        if P > 1:
            z_t = z_t.chunk(P, dim=2)[rank].contiguous()

        pt, ph, pw = self.dit.patch_size
        T_tokens = T_latent // pt
        H_tokens = H_latent // ph
        W_tokens = W_latent // pw
        seq_full = T_tokens * H_tokens * W_tokens
        seq_per_rank = seq_full // P

        positions_3d = self._build_positions(T_tokens, H_tokens, W_tokens, rank, P)
        positions_3d = positions_3d.to(self.device)

        cu_seqlens = torch.tensor([0, seq_full], dtype=torch.int32, device=self.device)

        timesteps = self.scheduler.get_timesteps(device=self.device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i].unsqueeze(0)
            v_pred = self.dit(z_t, t, text_emb, positions_3d, cu_seqlens)
            z_t = self.scheduler.step(z_t, v_pred, i, timesteps)

        if P > 1:
            z_all = [torch.empty_like(z_t) for _ in range(P)]
            dist.all_gather(z_all, z_t)
            z_t = torch.cat(z_all, dim=2)

        if rank == 0:
            video = self.vae.decode(z_t, device=self.device)
            return video
        return None

    def _build_positions(self, T, H, W, rank, P):
        t_pos = torch.arange(T).unsqueeze(-1).unsqueeze(-1).expand(T, H, W).reshape(-1)
        h_pos = torch.arange(H).unsqueeze(0).unsqueeze(-1).expand(T, H, W).reshape(-1)
        w_pos = torch.arange(W).unsqueeze(0).unsqueeze(0).expand(T, H, W).reshape(-1)

        seq_per_rank = T * H * W // P
        start = rank * seq_per_rank
        end = start + seq_per_rank

        return torch.stack([t_pos[start:end], h_pos[start:end], w_pos[start:end]], dim=0)


class I2VPipeline(T2VPipeline):
    def __init__(self, dit, vae, text_encoder, tokenizer, image_encoder, scheduler, device="cuda"):
        super().__init__(dit, vae, text_encoder, tokenizer, scheduler, device)
        self.image_encoder = image_encoder

    @torch.no_grad()
    def __call__(self, image, prompt, num_frames=81, height=480, width=832, seed=42):
        rank = usp_rank()
        P = usp_world_size()

        text_emb = self.encode_text(prompt)

        T_latent = (num_frames - 1) // 4 + 1
        H_latent = height // 8
        W_latent = width // 8

        if rank == 0:
            image_latent = self.vae.encode(image, device=self.device)
        else:
            image_latent = torch.zeros(1, 16, 1, H_latent, W_latent, device=self.device)

        if P > 1:
            dist.broadcast(image_latent, src=0)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(1, 16, T_latent, H_latent, W_latent, generator=generator, device=self.device)

        mask = torch.zeros(1, 4, T_latent, H_latent, W_latent, device=self.device)
        mask[:, :, 1:, :, :] = 1.0

        image_latent_expanded = image_latent.expand(1, 16, T_latent, H_latent, W_latent).clone()
        image_latent_expanded[:, :, 1:, :, :] = 0.0

        z_t = torch.cat([noise, image_latent_expanded, mask], dim=1)

        if P > 1:
            z_t = z_t.chunk(P, dim=2)[rank].contiguous()

        pt, ph, pw = self.dit.patch_size
        T_tokens = T_latent // pt
        H_tokens = H_latent // ph
        W_tokens = W_latent // pw
        seq_full = T_tokens * H_tokens * W_tokens

        positions_3d = self._build_positions(T_tokens, H_tokens, W_tokens, rank, P)
        positions_3d = positions_3d.to(self.device)

        cu_seqlens = torch.tensor([0, seq_full], dtype=torch.int32, device=self.device)

        timesteps = self.scheduler.get_timesteps(device=self.device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i].unsqueeze(0)
            v_pred = self.dit(z_t, t, text_emb, positions_3d, cu_seqlens)
            z_t_noise = z_t[:, :16]
            z_t_noise = self.scheduler.step(z_t_noise, v_pred, i, timesteps)
            z_t = torch.cat([z_t_noise, z_t[:, 16:]], dim=1)

        z_final = z_t[:, :16]

        if P > 1:
            z_all = [torch.empty_like(z_final) for _ in range(P)]
            dist.all_gather(z_all, z_final)
            z_final = torch.cat(z_all, dim=2)

        if rank == 0:
            video = self.vae.decode(z_final, device=self.device)
            return video
        return None
