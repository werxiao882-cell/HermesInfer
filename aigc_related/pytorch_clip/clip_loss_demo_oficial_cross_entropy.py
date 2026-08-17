import torch
import torch.nn.functional as F

def clip_loss_demo():
    # 1. 模拟初始化数据 (Batch Size = 3)
    # 假设我们有 3 张图片和 3 段对应的文本
    batch_size = 3
    feature_dim = 8  # 投影后的特征维度
    
    # 随机生成一些特征向量 (模拟 Encoder + Projection Head 的输出)
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    
    print("1. 原始特征维度:")
    print(f"Image Features Shape: {image_features.shape}")
    print(f"Text Features Shape: {text_features.shape}\n")

    # 2. L2 归一化 (Normalization)
    # CLIP 的核心是计算余弦相似度，归一化后，点积就等于余弦相似度
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    # 3. 计算相似度矩阵 (Logits)
    # 矩阵乘法: [batch_size, dim] @ [dim, batch_size] -> [batch_size, batch_size]
    # logits[i, j] 表示第 i 张图片和第 j 段文本的相似度
    temperature = 0.07
    logits = (image_features @ text_features.T) / temperature
    
    print("2. 相似度矩阵 (Logits):")
    print(logits)
    print(f"Logits Shape: {logits.shape}\n")

    # 4. 构建 Ground Truth (标签)
    # 在 CLIP 训练中，对角线上的 (i, i) 是正样本对，其余都是负样本
    # 所以标签就是 [0, 1, 2, ..., batch_size-1]
    targets = torch.arange(batch_size)
    print(f"3. 目标标签 (Targets): {targets}\n")

    # 5. 计算双向交叉熵损失 (Symmetric Loss)
    
    # (A) Image-to-Text Loss: 
    # 对于每一张图片，看它在所有文本中是否准确找到了匹配的那一个（按行计算）
    loss_i = F.cross_entropy(logits, targets)
    
    # (B) Text-to-Image Loss: 
    # 对于每一段文本，看它在所有图片中是否准确找到了匹配的那一个（按列计算，所以转置）
    loss_t = F.cross_entropy(logits.T, targets)
    
    # (C) 总损失：取平均
    loss = (loss_i + loss_t) / 2
    
    print("4. Loss 计算结果:")
    print(f"Loss (Image -> Text): {loss_i.item():.4f}")
    print(f"Loss (Text -> Image): {loss_t.item():.4f}")
    print(f"Total CLIP Loss: {loss.item():.4f}")

    # --- Debug 提示 ---
    print("\n[Debug 提示]")
    print("- 对角线元素 (logits.diag()) 越大，Loss 越小。")
    print("- 非对角线元素越大，Loss 越大。")
    print(f"- 当前对角线相似度值: {logits.diag().detach().numpy()}")

if __name__ == "__main__":
    clip_loss_demo()