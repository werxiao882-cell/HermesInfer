import torch


class FlowScheduler:
    def __init__(self, steps=50, shift=5.0, eps=1e-6):
        self.steps = steps
        self.shift = shift
        self.eps = eps

    def get_timesteps(self, device="cpu"):
        t = torch.linspace(1.0, 0.0, self.steps + 1, device=device)
        if self.shift != 1.0:
            t = self.shift * t / (1 + (self.shift - 1) * t)
        return t

    def step(self, z_t, v_pred, t_idx, timesteps):
        """
        z_t: current latent (B, C, T, H, W)
        v_pred: velocity prediction from DiT (B, C, T, H, W)
        t_idx: current step index
        timesteps: full timestep schedule from get_timesteps()

        Returns: z_{t-dt}
        """
        t_curr = timesteps[t_idx]
        t_next = timesteps[t_idx + 1]
        dt = t_curr - t_next

        z_next = z_t - dt * v_pred
        return z_next
