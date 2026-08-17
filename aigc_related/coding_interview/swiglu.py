"""
SwiGLU FFN —— Llama / Qwen 等现代 LLM 的标准 FFN。

  标准 FFN:  FFN(x) = W2 * ReLU(W1 * x)
  SwiGLU:    FFN(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)

  门控机制 (GLU) 让模型选择性通过信息；SiLU 比 ReLU 更平滑。
  通常 hidden_dim = 4 * d_model 再 * 2/3 以控制参数量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            # Llama 惯例：4*d 再乘 2/3，并对 256 对齐
            hidden_dim = int(d_model * 8 / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)    # down
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)    # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FeedForward(nn.Module):
    """Pre-Norm Transformer Block 中的 FFN 子层示例。"""

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


if __name__ == "__main__":
    batch, seq, dim = 2, 16, 512
    x = torch.randn(batch, seq, dim)
    ffn = SwiGLU(dim)
    out = ffn(x)
    print("SwiGLU output shape:", out.shape)
    assert out.shape == x.shape
    print(f"Hidden dim: {ffn.w1.out_features}")
    print("[验证成功] SwiGLU 前向正常")
