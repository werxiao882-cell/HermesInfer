import torch
import torch.nn.functional as F

def manual_cross_entropy(logits, targets):
    """
    手动实现交叉熵损失函数。
    CrossEntropyLoss = Softmax + Log + NLLLoss
    或者直接使用公式: loss = -log(exp(x_i) / sum(exp(x_j)))
    为了数值稳定性，通常使用: loss = -x_i + log(sum(exp(x_j)))
    """
    # 1. 为了数值稳定性，减去每行的最大值 (防止 exp 指数爆炸)
    # logits shape: [batch_size, num_classes]
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    stable_logits = logits - max_logits
    
    # 2. 计算 Softmax 的分母: sum(exp(logits))
    exp_sum = torch.sum(torch.exp(stable_logits), dim=1)
    
    # 3. 获取正样本对应的 logit 值 (x_i)
    # targets 是 [0, 1, 2] 这种索引
    batch_size = logits.shape[0]
    # 使用 range 配合 targets 索引出对角线上的值（即正样本的 logits）
    target_logits = stable_logits[torch.arange(batch_size), targets]
    
    # 4. 计算每个样本的 loss: -log(exp(x_i) / sum(exp(x_j)))
    # 等价于: -(target_logits - log(exp_sum))
    # 即: log(exp_sum) - target_logits
    sample_losses = torch.log(exp_sum) - target_logits
    
    # 5. 返回平均损失
    return sample_losses.mean()

def clip_loss_demo_v2():
    # 1. 模拟初始化数据 (Batch Size = 3)
    batch_size = 3
    feature_dim = 8
    
    # 固定随机种子以便对比结果
    torch.manual_seed(42)
    
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    
    print("1. 原始特征维度:")
    print(f"Image Features Shape: {image_features.shape}")
    print(f"Text Features Shape: {text_features.shape}\n")

    # 2. L2 归一化
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    # 3. 计算相似度矩阵 (Logits)
    temperature = 0.07
    logits = (image_features @ text_features.T) / temperature
    
    print("2. 相似度矩阵 (Logits):")
    print(logits)
    print(f"Logits Shape: {logits.shape}\n")

    # 4. 构建 Ground Truth (标签)
    targets = torch.arange(batch_size)
    print(f"3. 目标标签 (Targets): {targets}\n")

    # 5. 计算双向交叉熵损失 (Symmetric Loss)
    
    # 使用官方函数计算 (用于对比)
    loss_i_official = F.cross_entropy(logits, targets)
    loss_t_official = F.cross_entropy(logits.T, targets)
    
    # 使用手动实现的函数计算
    loss_i_manual = manual_cross_entropy(logits, targets)
    loss_t_manual = manual_cross_entropy(logits.T, targets)
    
    # 总损失
    loss_official = (loss_i_official + loss_t_official) / 2
    loss_manual = (loss_i_manual + loss_t_manual) / 2
    
    print("4. Loss 计算结果对比:")
    print(f"Official Loss (I->T): {loss_i_official.item():.6f}")
    print(f"Manual   Loss (I->T): {loss_i_manual.item():.6f}")
    print("-" * 30)
    print(f"Official Loss (T->I): {loss_t_official.item():.6f}")
    print(f"Manual   Loss (T->I): {loss_t_manual.item():.6f}")
    print("-" * 30)
    print(f"Total Official Loss: {loss_official.item():.6f}")
    print(f"Total Manual   Loss: {loss_manual.item():.6f}")

    # 验证是否一致
    assert torch.allclose(loss_official, loss_manual), "手动实现的 Loss 与官方不一致！"
    print("\n[验证成功] 手动实现的交叉熵函数与 PyTorch 官方函数结果完全一致。")

if __name__ == "__main__":
    clip_loss_demo_v2()
