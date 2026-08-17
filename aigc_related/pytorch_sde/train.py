import torch
import torch.nn as nn
import numpy as np
import functools
import argparse
from copy import deepcopy
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import sys
import os

# Add current directory to path to allow importing model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ScoreNet

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def marginal_prob_std(t, sigma, device='cpu'):
    """ 计算任意t时刻的扰动后条件高斯分布的标准差 """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device='cpu'):
    """ 计算任意t时刻的扩散系数，本例定义的SDE没有漂移系数 """
    return torch.tensor(sigma**t, device=device)

def loss_fn(score_model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        score_model: A PyTorch model instance that represents a
            time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    # Step1 从[0.00001, 0.9999]中随机生成batchsize个浮点型t
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # Step2 基于重参数技巧采样出分布p_t(x)的一个随机样本perturbed_x
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]

    # Step3 将当前的加噪样本和时间输入到Score Network中预测出分数score
    score = score_model(perturbed_x, random_t)

    # Step4 计算score matching loss
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss

def main():
    parser = argparse.ArgumentParser(description='SDE Training Script')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cuda or cpu')
    parser.add_argument('--sigma', type=float, default=25.0, help='Sigma value for SDE')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    device = args.device
    sigma = args.sigma
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr

    print(f"Running on device: {device}")
    print(f"Sigma: {sigma}")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}, LR: {lr}")

    # 构建无参函数 (partial functions with fixed sigma and device)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)

    # Initialize model
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    # Dataset and DataLoader
    dataset = MNIST('./pytorch_sde/', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = Adam(score_model.parameters(), lr=lr)
    
    # EMA
    ema = EMA(score_model)

    # Training loop
    tqdm_epoch = tqdm.tqdm(range(n_epochs))
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(score_model)
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        
        # Print average loss for the epoch
        tqdm_epoch.set_description('Average ScoreMatching Loss: {:5f}'.format(avg_loss / num_items))
        
        # Save checkpoint
        torch.save(score_model.state_dict(), f'./pytorch_sde/ckpt_{epoch}.pth')

if __name__ == "__main__":
    main()
