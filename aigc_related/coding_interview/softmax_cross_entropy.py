"""
Softmax 与 Cross-Entropy 手搓实现 —— 面试高频，重点在数值稳定性。

  Softmax:  减 max 防溢出 -> exp -> 归一化
  CE Loss:  -log(softmax(logits)[target])
           = log(sum(exp(logits))) - logits[target]   (配合 log-sum-exp trick)
"""
import torch
import torch.nn.functional as F


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_logits = logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True)


def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (B, C)
    targets: (B,)  类别索引
    """
    max_logits = logits.max(dim=1, keepdim=True).values
    stable = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(stable), dim=1))
    target_logits = stable[torch.arange(logits.size(0)), targets]
    return (log_sum_exp - target_logits).mean()


def log_softmax_stable(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log_softmax(x) = x - log(sum(exp(x)))，同样减 max 保稳定。"""
    max_logits = logits.max(dim=dim, keepdim=True).values
    stable = logits - max_logits
    return stable - torch.log(torch.sum(torch.exp(stable), dim=dim, keepdim=True))


if __name__ == "__main__":
    torch.manual_seed(42)
    logits = torch.randn(4, 10) * 100  # 故意放大，测试数值稳定性
    targets = torch.tensor([0, 3, 7, 1])

    # Softmax
    manual_sm = stable_softmax(logits)
    official_sm = F.softmax(logits, dim=-1)
    assert torch.allclose(manual_sm, official_sm, atol=1e-5)
    print("[验证成功] Softmax 与 F.softmax 一致")

    # Cross-Entropy
    manual_ce = stable_cross_entropy(logits, targets)
    official_ce = F.cross_entropy(logits, targets)
    assert torch.allclose(manual_ce, official_ce, atol=1e-5)
    print(f"Manual CE:  {manual_ce.item():.6f}")
    print(f"Official CE: {official_ce.item():.6f}")
    print("[验证成功] Cross-Entropy 与 F.cross_entropy 一致")

    # log_softmax
    manual_lsm = log_softmax_stable(logits)
    official_lsm = F.log_softmax(logits, dim=-1)
    assert torch.allclose(manual_lsm, official_lsm, atol=1e-5)
    print("[验证成功] log_softmax 与 F.log_softmax 一致")
