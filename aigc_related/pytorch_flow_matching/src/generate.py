import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import argparse
from tqdm import tqdm
from model import ConditionalFlowMatchingUNet


class FlowMatchingSampler:
    """Flow Matching采样器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def euler_sampler(self, shape, class_labels, num_steps=50, save_intermediates=False):
        """
        使用Euler方法求解ODE进行采样
        
        Args:
            shape: 生成图像的形状 (batch_size, channels, height, width)
            class_labels: 类别标签 (batch_size,)
            num_steps: 采样步数
            save_intermediates: 是否保存中间步骤
            
        Returns:
            samples: 最终生成的样本
            intermediates: 中间步骤的样本（如果save_intermediates=True）
        """
        batch_size = shape[0]
        
        # 从标准高斯噪声开始
        x = torch.randn(shape, device=self.device)
        
        # 存储中间结果
        intermediates = []
        if save_intermediates:
            intermediates.append(x.detach().cpu().clone())
        
        # Euler方法求解ODE
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for step in tqdm(range(num_steps), desc="Generating"):
                t = torch.full((batch_size,), step * dt, device=self.device)
                
                # 模型预测速度场
                v = self.model(x, t, class_labels)
                
                # Euler步骤：x_{t+dt} = x_t + v * dt
                x = x + v * dt
                
                # 保存中间结果
                if save_intermediates:
                    intermediates.append(x.detach().cpu().clone())
        
        if save_intermediates:
            return x.detach().cpu(), intermediates
        else:
            return x.detach().cpu()
    
    def rk4_sampler(self, shape, class_labels, num_steps=50, save_intermediates=False):
        """
        使用4阶Runge-Kutta方法求解ODE进行采样（更精确但更慢）
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        
        intermediates = []
        if save_intermediates:
            intermediates.append(x.detach().cpu().clone())
        
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for step in tqdm(range(num_steps), desc="Generating (RK4)"):
                t = torch.full((batch_size,), step * dt, device=self.device)
                
                # RK4方法
                k1 = self.model(x, t, class_labels)
                k2 = self.model(x + 0.5 * dt * k1, t + 0.5 * dt, class_labels)
                k3 = self.model(x + 0.5 * dt * k2, t + 0.5 * dt, class_labels)
                k4 = self.model(x + dt * k3, t + dt, class_labels)
                
                x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                
                if save_intermediates:
                    intermediates.append(x.detach().cpu().clone())
        
        if save_intermediates:
            return x.detach().cpu(), intermediates
        else:
            return x.detach().cpu()


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"Loading model from {checkpoint_path}")
    
    # 创建模型
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
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        # 检查是否是完整的检查点格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                print("Loaded model from checkpoint")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model from checkpoint (state_dict format)")
        else:
            # 假设整个字典就是state_dict
            model.load_state_dict(checkpoint)
            print("Loaded model weights (direct state_dict)")
    else:
        # 如果不是字典，可能是直接的模型权重
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    return model


def generate_progression_grid(sampler, num_samples_per_class=1, num_steps=50, save_path=None):
    """
    生成0-9数字的生成过程网格图
    
    Args:
        sampler: FlowMatchingSampler实例
        num_samples_per_class: 每类生成多少个样本
        num_steps: 生成步数
        save_path: 保存路径
    """
    device = sampler.device
    
    # 准备类别标签：0-9每类生成num_samples_per_class个
    all_class_labels = []
    for class_id in range(10):
        all_class_labels.extend([class_id] * num_samples_per_class)
    
    class_labels = torch.tensor(all_class_labels, device=device)
    batch_size = len(class_labels)
    
    print(f"Generating {batch_size} samples ({num_samples_per_class} per digit)...")
    
    # 生成带中间步骤的样本
    shape = (batch_size, 1, 28, 28)
    final_samples, intermediates = sampler.euler_sampler(
        shape, class_labels, num_steps=num_steps, save_intermediates=True
    )
    
    # 选择要显示的10个步骤（从开始到结束）
    num_display_steps = 10
    step_indices = [int(i * (len(intermediates) - 1) / (num_display_steps - 1)) for i in range(num_display_steps)]
    
    # 创建无间隙网格图
    fig, axes = plt.subplots(10, num_display_steps, figsize=(num_display_steps, 10))
    plt.subplots_adjust(hspace=0, wspace=0)  # 移除间隙
    
    for row in range(10):  # 10个数字类别
        for col, step_idx in enumerate(step_indices):
            ax = axes[row, col]
            
            # 获取当前数字类别的第一个样本
            sample_idx = row * num_samples_per_class
            img = intermediates[step_idx][sample_idx, 0].numpy()
            
            # 显示图像，不添加任何标注
            ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            # 移除所有边框和标注
            ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved progression grid to {save_path}")
    
    plt.close(fig)  # 释放内存
    
    return final_samples, intermediates


def generate_samples(sampler, class_labels, num_steps=50, save_path=None):
    """
    生成指定类别的样本
    
    Args:
        sampler: FlowMatchingSampler实例
        class_labels: 要生成的类别标签列表
        num_steps: 生成步数
        save_path: 保存路径
    """
    device = sampler.device
    class_labels = torch.tensor(class_labels, device=device)
    batch_size = len(class_labels)
    
    print(f"Generating {batch_size} samples...")
    
    # 生成样本
    shape = (batch_size, 1, 28, 28)
    samples = sampler.euler_sampler(shape, class_labels, num_steps=num_steps)
    
    # 显示结果
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))
    if batch_size == 1:
        axes = [axes]
    
    for i, (sample, label) in enumerate(zip(samples, class_labels.cpu())):
        img = sample[0].numpy()
        axes[i].imshow(img, cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(f'{label.item()}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved samples to {save_path}")
    
    plt.close(fig)  # 释放内存
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate MNIST digits using Conditional Flow Matching')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--sampler', type=str, default='euler', choices=['euler', 'rk4'],
                        help='ODE solver method')
    parser.add_argument('--output_dir', type=str, default='./samples',
                        help='Output directory for generated images')
    parser.add_argument('--generate_progression', action='store_true',
                        help='Generate progression grid for digits 0-9')
    parser.add_argument('--generate_samples', type=str, default=None,
                        help='Generate samples for specific digits (e.g., "0,1,2,3,4")')

    parser.add_argument('--samples_per_class', type=int, default=1,
                        help='Number of samples per class for progression grid')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 创建采样器
    sampler = FlowMatchingSampler(model, device)
    
    # 生成进度网格
    if args.generate_progression:
        print("Generating progression grid...")
        save_path = os.path.join(args.output_dir, 'progression_grid.png')
        final_samples, intermediates = generate_progression_grid(
            sampler, 
            num_samples_per_class=args.samples_per_class,
            num_steps=args.num_steps,
            save_path=save_path
        )
        
    # 生成指定数字的样本
    if args.generate_samples:
        digits = [int(d.strip()) for d in args.generate_samples.split(',')]
        print(f"Generating samples for digits: {digits}")
        save_path = os.path.join(args.output_dir, f'samples_{"_".join(map(str, digits))}.png')
        samples = generate_samples(sampler, digits, args.num_steps, save_path)
    
    print("Generation completed!")

if __name__ == "__main__":
    main()