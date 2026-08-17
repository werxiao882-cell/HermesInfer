import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    LayerNorm: 减均值 + 除标准差 + 仿射变换 (gamma, beta)
    对最后一维 (hidden_dim) 做归一化，独立于 batch / seq 其他样本。
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """
    RMSNorm: 只除均方根，不减均值；通常只有 weight，没有 bias。
    Llama / Qwen 等现代 LLM 的标准选择。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x^2)); 用 rsqrt 避免除法
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 计算再转回，与 Llama 源码一致，提升数值稳定性
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


if __name__ == "__main__":
    batch, seq, dim = 2, 8, 64
    x = torch.randn(batch, seq, dim)

    ln = LayerNorm(dim)
    rms = RMSNorm(dim)

    out_ln = ln(x)
    out_rms = rms(x)

    print("LayerNorm output shape:", out_ln.shape)
    print("RMSNorm  output shape:", out_rms.shape)

    # 与 PyTorch 官方实现对比
    ref_ln = nn.LayerNorm(dim)
    with torch.no_grad():
        ref_ln.weight.copy_(ln.gamma)
        ref_ln.bias.copy_(ln.beta)
    assert torch.allclose(ln(x), ref_ln(x), atol=1e-5)
    print("[验证成功] LayerNorm 与 nn.LayerNorm 一致")
