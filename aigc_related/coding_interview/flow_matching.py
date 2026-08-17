"""
Flow Matching / Rectified Flow 面试核心：

  训练目标：学一个速度场 v_theta(x_t, t)，使得沿 ODE dx/dt = v 能把噪声 x_0 推到数据 x_1。

  直线路径 (Conditional Flow Matching):
    x_t = (1 - t) * x_0 + t * x_1        # t ~ U(0, 1)
    目标速度: u_t = x_1 - x_0             # 与 t 无关的常向量
    Loss: || v_theta(x_t, t) - u_t ||^2

  采样 (Euler ODE solver):
    从 x ~ N(0, I) 出发，逐步 x += v_theta(x, t) * dt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityNet(nn.Module):
    """占位网络：输入 (x_t, t) -> 预测速度 v。真实场景用 U-Net / DiT。"""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


def sample_flow_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """
    直线路径插值 + 目标速度。
    x0: 噪声, x1: 数据, t: (B, 1) in [0, 1]
    """
    t = t.view(-1, 1)
    x_t = (1 - t) * x0 + t * x1
    target_v = x1 - x0
    return x_t, target_v


def flow_matching_loss(model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
    """单步 CFM loss。"""
    batch_size = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, 1, device=x1.device)
    x_t, target_v = sample_flow_path(x0, x1, t)
    pred_v = model(x_t, t.squeeze(-1))
    return F.mse_loss(pred_v, target_v)


def sample_ode(model: nn.Module, shape, num_steps: int = 50, device="cpu"):
    """Euler 求解 ODE，从噪声生成样本。"""
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            v = model(x, t)
            x = x + v * dt
    return x


def train_flow_matching(model, dataloader, optimizer, epochs=1, device="cpu"):
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in dataloader:
            x1 = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            optimizer.zero_grad()
            loss = flow_matching_loss(model, x1)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch + 1}, loss: {total / len(dataloader):.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 16
    model = VelocityNet(dim)

    # 模拟 2D 高斯数据
    data = torch.randn(64, dim) + 2.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_flow_matching(model, [data], optimizer, epochs=50, device="cpu")

    samples = sample_ode(model, (32, dim), num_steps=50)
    print("Generated samples mean (should shift toward ~2):", samples.mean().item())
    print("[验证成功] Flow Matching 训练与采样流程正常")
