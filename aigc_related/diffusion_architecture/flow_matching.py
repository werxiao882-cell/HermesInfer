import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================
# 1. 极简向量场网络 (MLP)
# ======================
class Network(nn.Module):
    """输入: (展平图像 [B,784], 时间 t [B,1]) → 输出: 速度 [B,784]"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(785, 512),  # 784 (图像) + 1 (时间)
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 784)   # 输出与输入同维度的速度
        )
    
    def forward(self, z, t):
        # z: [B, 784], t: [B] → 拼接为 [B, 785]
        t = t.view(-1, 1)  # [B,1]
        inp = torch.cat([z, t], dim=1)  # [B, 785]
        return self.net(inp)

# ======================
# 2. MNIST 数据加载 (归一化到 [-1, 1])
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0,1] → [-1,1]
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)

# ======================
# 3. Flow Matching 训练
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_steps = 2000  # 演示用，实际建议 10k+

print("🚀 开始训练 Flow Matching (MNIST)...")
for step in range(num_steps):
    # 采样真实数据 x1 ~ p_data
    x1, _ = next(iter(train_loader))
    x1 = x1.view(x1.size(0), -1).to(device)  # [B, 784]
    
    # 采样噪声 z0 ~ N(0, I) normal: 均值为0.，
    z0 = torch.randn_like(x1).to(device)  # [B, 784]
    
    # 采样时间 t ~ U[0,1]
    t = torch.rand(x1.size(0), device=device)  # [B]
    
    # 构造直线路径上的点: φ_t = (1-t)z0 + t x1
    z_t = (1 - t.unsqueeze(1)) * z0 + t.unsqueeze(1) * x1  # [B, 784]
    
    # 目标速度 = 路径导数 = x1 - z0 (直线路径)
    target_vel = x1 - z0  # [B, 784]
    
    # 模型预测速度
    pred_vel = model(z_t, t)  # [B, 784]
    
    # L2 损失
    loss = nn.functional.mse_loss(pred_vel, target_vel)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print(f"Step {step}/{num_steps} | Loss: {loss.item():.6f}")

print("✅ 训练完成！")

# ======================
# 4. 推理生成 (欧拉法解 ODE)
# ======================
def generate_samples(model, n_samples=16, n_steps=100):
    """从噪声生成 MNIST 样本"""
    model.eval()
    with torch.no_grad():
        # 初始化: z0 ~ N(0, I)
        z = torch.randn(n_samples, 784).to(device)
        dt = 1.0 / n_steps
        
        # 欧拉积分: z_{t+dt} = z_t + v_θ(z_t, t) * dt
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=device)
            vel = model(z, t, condition)  # [B, 784]
            z = z + vel * dt   # 欧拉更新
        
        # 反归一化: [-1,1] → [0,1] 用于显示
        z = (z + 1) / 2
        z = torch.clamp(z, 0, 1)
        return z.view(n_samples, 1, 28, 28).cpu()

generated = generate_samples(model, n_samples=16)