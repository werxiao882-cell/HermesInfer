import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalGroupNorm(nn.Module):
    """条件组归一化"""
    def __init__(self, num_groups, num_channels, num_classes):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.class_emb = nn.Embedding(num_classes, num_channels * 2)
        
    def forward(self, x, class_labels):
        # x: (batch_size, channels, height, width)
        # class_labels: (batch_size,)
        
        x = self.norm(x)  # 标准化
        
        # 获取类别嵌入
        emb = self.class_emb(class_labels)  # (batch_size, channels * 2)
        emb = emb.view(-1, self.num_channels * 2, 1, 1)  # 重塑为 (batch_size, channels * 2, 1, 1)
        
        # 分离缩放和偏移参数
        scale, shift = emb.chunk(2, dim=1)  # 每个都是 (batch_size, channels, 1, 1)
        
        return x * (1 + scale) + shift


class ResNetBlock(nn.Module):
    """条件 ResNet 块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 时间嵌入投影
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 第一个卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = ConditionalGroupNorm(8, out_channels, num_classes)
        
        # 第二个卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = ConditionalGroupNorm(8, out_channels, num_classes)
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, time_emb, class_labels):
        residual = self.residual_conv(x)
        
        # 第一个卷积块
        h = self.conv1(x)
        h = self.norm1(h, class_labels)
        h = F.silu(h)
        
        # 添加时间嵌入
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # 第二个卷积块
        h = self.conv2(h)
        h = self.norm2(h, class_labels)
        h = F.silu(h)
        h = self.dropout(h)
        
        return h + residual


class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # 重塑为多头注意力格式
        qkv = qkv.view(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch_size, num_heads, height*width, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, channels, height, width)
        
        out = self.proj_out(out)
        return out + residual


class ConditionalFlowMatchingUNet(nn.Module):
    """条件流匹配 U-Net 模型"""
    def __init__(self, 
                 in_channels=1, 
                 model_channels=64,
                 out_channels=1,
                 num_res_blocks=2,
                 attention_resolutions=[16],
                 channel_mult=[1, 2, 4],
                 num_classes=10,
                 dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 类别嵌入
        self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # 输入投影
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # 下采样块
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResNetBlock(ch, mult * model_channels, time_embed_dim, num_classes, dropout)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                    
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
                ds *= 2
                
        # 中间块
        self.middle_block = nn.Sequential(
            ResNetBlock(ch, ch, time_embed_dim, num_classes, dropout),
            AttentionBlock(ch),
            ResNetBlock(ch, ch, time_embed_dim, num_classes, dropout)
        )
        
        # 上采样块
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResNetBlock(ch + ich, mult * model_channels, time_embed_dim, num_classes, dropout)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                    
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                    
                self.output_blocks.append(nn.Sequential(*layers))
                
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
        
    def forward(self, x, timesteps, class_labels):
        """
        x: (batch_size, channels, height, width) - 输入图像
        timesteps: (batch_size,) - 时间步
        class_labels: (batch_size,) - 类别标签
        """
        # 嵌入时间和类别
        t_emb = self.time_embed(timesteps)
        c_emb = self.class_embed(class_labels)
        emb = t_emb + c_emb
        
        # 下采样
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], ResNetBlock):
                h = module[0](h, emb, class_labels)
                if len(module) > 1:  # 有注意力层
                    h = module[1](h)
            else:
                h = module(h)
            hs.append(h)
            
        # 中间块
        h = self.middle_block[0](h, emb, class_labels)  # ResNetBlock
        h = self.middle_block[1](h)  # AttentionBlock
        h = self.middle_block[2](h, emb, class_labels)  # ResNetBlock
        
        # 上采样
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], ResNetBlock):
                h = module[0](h, emb, class_labels)
                for layer in module[1:]:
                    h = layer(h)
            else:
                h = module(h)
                
        return self.out(h)


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalFlowMatchingUNet().to(device)
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    t = torch.rand(batch_size).to(device)
    classes = torch.randint(0, 10, (batch_size,)).to(device)
    
    with torch.no_grad():
        output = model(x, t, classes)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")