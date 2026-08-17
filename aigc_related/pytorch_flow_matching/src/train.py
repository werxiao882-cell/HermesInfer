import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from model import ConditionalFlowMatchingUNet


class ConditionalFlowMatching:
    """条件流匹配训练器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def sample_conditional_flow(self, x0, x1, t):
        """
        构造条件流：从噪声 x0 到数据 x1 的插值路径
        x0: 噪声 (batch_size, 1, 28, 28)
        x1: 数据 (batch_size, 1, 28, 28)  
        t: 时间 (batch_size, 1, 1, 1)
        """
        # 线性插值：x_t = (1-t) * x0 + t * x1
        x_t = (1 - t) * x0 + t * x1
        
        # 条件速度场：dx/dt = x1 - x0
        u_t = x1 - x0
        
        return x_t, u_t
    
    def compute_loss(self, x_batch, class_labels):
        """
        计算条件流匹配损失
        x_batch: 真实数据 (batch_size, 1, 28, 28)
        class_labels: 类别标签 (batch_size,)
        """
        batch_size = x_batch.shape[0]
        
        # 采样时间 t ~ U[0,1]
        t = torch.rand(batch_size, device=self.device)
        t_expanded = t.view(batch_size, 1, 1, 1)
        
        # 采样噪声 x0 ~ N(0, I)
        x0 = torch.randn_like(x_batch)
        
        # 构造条件流
        x_t, u_t = self.sample_conditional_flow(x0, x_batch, t_expanded)
        
        # 模型预测速度场
        v_pred = self.model(x_t, t, class_labels)
        
        # 计算损失：||v_pred - u_t||^2
        loss = F.mse_loss(v_pred, u_t)
        
        return loss


def get_data_loaders(batch_size=128, num_workers=4):
    """获取 MNIST 数据加载器"""
    # 数据预处理：归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 将 [0,1] 映射到 [-1,1]
    ])
    
    # 训练集
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 测试集
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 也保存最新的检查点
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def train_epoch(model, flow_matcher, train_loader, optimizer, scheduler, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        loss = flow_matcher.compute_loss(data, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # 记录到TensorBoard（每100个batch记录一次）
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / num_batches


def validate_epoch(model, flow_matcher, test_loader, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Validation"):
            data = data.to(device)
            labels = labels.to(device)
            
            loss = flow_matcher.compute_loss(data, labels)
            total_loss += loss.item()
    
    return total_loss / num_batches




def main():
    parser = argparse.ArgumentParser(description='Train Conditional Flow Matching on MNIST')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # TensorBoard writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logs will be saved to: {args.log_dir}")
    
    # 数据加载器
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(args.batch_size, args.num_workers)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 模型
    print("Initializing model...")
    model = ConditionalFlowMatchingUNet(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[16],
        channel_mult=[1, 2, 4],
        num_classes=10,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 流匹配器
    flow_matcher = ConditionalFlowMatching(model, device)
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 加载检查点
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss = train_epoch(model, flow_matcher, train_loader, optimizer, scheduler, device, epoch, writer)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate_epoch(model, flow_matcher, test_loader, device)
        val_losses.append(val_loss)
        
        # 记录epoch级别的指标到TensorBoard
        writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
        writer.add_scalar('Validation/Loss_Epoch', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or val_loss < best_val_loss:
            save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存最佳模型
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                writer.add_scalar('Validation/Best_Loss', best_val_loss, epoch)

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # 关闭TensorBoard writer
    writer.close()


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from torch.nn import functional as F
    
    main()