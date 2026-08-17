"""
LoRA (Low-Rank Adaptation) 面试核心：

  冻结预训练权重 W，只训练低秩增量 ΔW = B @ A
  其中 A: (r, d_in), B: (d_out, r), r << min(d_in, d_out)

  前向: y = W @ x + (alpha/r) * B @ A @ x
  参数量从 d_in*d_out 降到 r*(d_in + d_out)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # 预训练权重（冻结）
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad = False

        # LoRA 低秩矩阵（可训练）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性变换 + LoRA 增量
        base = F.linear(x, self.weight)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def merge_weights(self):
        """推理时可合并权重：W' = W + scaling * B @ A"""
        with torch.no_grad():
            self.weight += self.scaling * (self.lora_B @ self.lora_A)
            self.lora_A.zero_()
            self.lora_B.zero_()


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    in_f, out_f, rank = 512, 512, 8
    layer = LoRALinear(in_f, out_f, rank=rank)

    full_params = in_f * out_f
    lora_params = rank * (in_f + out_f)
    trainable = count_trainable_params(layer)

    print(f"Full weight params:  {full_params:,}")
    print(f"LoRA trainable:      {lora_params:,}  (ratio: {lora_params/full_params:.2%})")
    assert trainable == lora_params

    x = torch.randn(2, 16, in_f)
    out = layer(x)
    print("LoRA output shape:", out.shape)

    # 验证 merge 后输出一致
    out_before = layer(x).detach()
    layer.merge_weights()
    out_after = F.linear(x, layer.weight).detach()
    assert torch.allclose(out_before, out_after, atol=1e-5)
    print("[验证成功] LoRA 前向与 merge 一致")
