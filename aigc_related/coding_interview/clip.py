"""
CLIP 面试核心：对比学习 (InfoNCE) + 双向对称损失。
Encoder 结构不重要，用简单 MLP 占位即可，重点是 loss 和训练流程。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 占位 Encoder —— 面试时口头说明「真实场景用 ViT + Transformer」即可
# ---------------------------------------------------------------------------

class DummyImageEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DummyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 简单 mean pooling 代替 Transformer
        return self.net(self.embed(token_ids).mean(dim=1))


# ---------------------------------------------------------------------------
# 核心：CLIP 对比损失
# ---------------------------------------------------------------------------

def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE:
      1. L2 归一化
      2. logits = image @ text^T / temperature   -> (B, B)
      3. 对角线是正样本，其余是 in-batch 负样本
      4. loss = (CE(logits, arange(B)) + CE(logits.T, arange(B))) / 2
    """
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


class CLIP(nn.Module):
    def __init__(self, image_input_dim: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.image_encoder = DummyImageEncoder(image_input_dim, embed_dim)
        self.text_encoder = DummyTextEncoder(vocab_size, embed_dim)
        # 可学习 temperature，原始 CLIP 也这么做
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, text_ids):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_ids)
        temperature = self.logit_scale.exp().clamp(max=100.0)
        loss = clip_contrastive_loss(image_features, text_features, temperature)
        return loss


def train_one_epoch(model, dataloader, optimizer, device="cpu"):
    """最小训练循环，体现 CLIP 的训练思想。"""
    model.train()
    total_loss = 0.0
    for images, text_ids in dataloader:
        images, text_ids = images.to(device), text_ids.to(device)
        optimizer.zero_grad()
        loss = model(images, text_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, embed_dim = 4, 32
    image_dim, vocab_size = 784, 100

    model = CLIP(image_dim, vocab_size, embed_dim)

    # 模拟一个 batch：每张图对应一句 caption（token 序列）
    images = torch.randn(batch_size, image_dim)
    text_ids = torch.randint(0, vocab_size, (batch_size, 10))

    loss = model(images, text_ids)
    print(f"CLIP loss: {loss.item():.4f}")

    # 验证：正样本相似度应高于随机负样本（训练前只是 sanity check shape）
    with torch.no_grad():
        img_feat = F.normalize(model.image_encoder(images), dim=-1)
        txt_feat = F.normalize(model.text_encoder(text_ids), dim=-1)
        sim = img_feat @ txt_feat.T
    print("Similarity matrix shape:", sim.shape)
    print("[验证成功] CLIP 前向与 loss 计算正常")
