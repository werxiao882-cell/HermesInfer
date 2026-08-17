import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================
# 1. 极简噪声预测网络 (MLP)
# ======================
class Network(nn.Module):
    """输入: (展平图像 [B,784], 时间 t [B,1]) → 输出: 预测噪声 ε [B,784]"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(785, 512),  # 784 (图像) + 1 (时间)
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 784)
        )

    def forward(self, x_t, t):
        t = t.view(-1, 1)  # [B, 1]
        inp = torch.cat([x_t, t], dim=1)
        return self.net(inp)


# ======================
# 2. DDPM 噪声调度器
# ======================
class DDPMScheduler:
    """
    对应 demo.py 中 DDPMScheduler 的核心逻辑:
    - 前向扩散: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    - 训练目标: 预测 ε (epsilon prediction)
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        for name in ("betas", "alphas", "alphas_cumprod",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def add_noise(self, x0, noise, timesteps):
        """对应 demo.py: noise_scheduler.add_noise(latents, noise, timesteps)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def step(self, x_t, pred_noise, t):
        """单步反向去噪: x_t → x_{t-1}"""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        # μ_θ = 1/sqrt(α_t) * (x_t - β_t / sqrt(1-ᾱ_t) * ε_θ)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(beta_t) * noise
        else:
            x_prev = mean
        return x_prev


# ======================
# 3. MNIST 数据加载 (归一化到 [-1, 1])
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)

# ======================
# 4. DDPM 训练
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().to(device)
scheduler = DDPMScheduler(num_timesteps=1000).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_steps = 2000  # 演示用，实际建议 10k+

print("🚀 开始训练 DDPM (MNIST)...")
for step in range(num_steps):
    # 采样真实数据 x0 ~ p_data
    x0, _ = next(iter(train_loader))
    x0 = x0.view(x0.size(0), -1).to(device)

    # 采样噪声 ε ~ N(0, I)
    noise = torch.randn_like(x0)

    # 随机采样时间步 t ~ U{0, ..., T-1}
    timesteps = torch.randint(0, scheduler.num_timesteps, (x0.size(0),), device=device)

    # 前向扩散: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
    x_t = scheduler.add_noise(x0, noise, timesteps)

    # 时间归一化到 [0, 1] 供网络使用
    t_norm = timesteps.float() / scheduler.num_timesteps

    # 模型预测噪声 ε_θ(x_t, t)
    pred_noise = model(x_t, t_norm)

    # epsilon prediction 损失 (对应 demo.py prediction_type="epsilon")
    loss = nn.functional.mse_loss(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}/{num_steps} | Loss: {loss.item():.6f}")

print("✅ 训练完成！")

# ======================
# 5. 推理生成 (逐步反向去噪)
# ======================
def generate_samples(model, scheduler, n_samples=16):
    """从纯噪声 x_T ~ N(0,I) 逐步去噪生成 MNIST 样本"""
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, 784, device=device)

        for t in reversed(range(scheduler.num_timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_norm = t_batch.float() / scheduler.num_timesteps
            pred_noise = model(x, t_norm)
            x = scheduler.step(x, pred_noise, t)

        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        return x.view(n_samples, 1, 28, 28).cpu()


generated = generate_samples(model, scheduler, n_samples=16)

# ======================
# 6. 可视化
# ======================
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, img in zip(axes.flat, generated):
    ax.imshow(img.squeeze(), cmap='gray')
    ax.axis('off')
plt.suptitle('DDPM Generated MNIST Samples')
plt.tight_layout()
plt.savefig('ddpm_samples.png', dpi=150)
print("📷 生成结果已保存至 ddpm_samples.png")
