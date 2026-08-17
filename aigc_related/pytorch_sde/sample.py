import torch
import functools
import argparse
import tqdm
import numpy as np
from torchvision.utils import make_grid, save_image
import sys
import os

# Add current directory to path to allow importing model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ScoreNet
from train import marginal_prob_std, diffusion_coeff

def euler_sampler(score_model,
                  marginal_prob_std,
                  diffusion_coeff,
                  batch_size=64,
                  num_steps=500,
                  device='cuda',
                  eps=1e-3):
    """
    Generate samples from score-based models with the Euler-Maruyama solver.
    
    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by parallel processing.
        num_steps: The number of sampling steps.
        device: The device to run the sampling on.
        eps: The smallest time step for numerical stability.
    
    Returns:
        Samples.
    """
    # Step1 定义初始时间1和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
        * marginal_prob_std(t)[:, None, None, None]

    # Step2 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # Step3 根据欧拉算法来求解逆时间SDE
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    # Step4 取最后一步的期望值作为生成的样本
    return mean_x

def main():
    parser = argparse.ArgumentParser(description='SDE Sampling Script')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cuda or cpu')
    parser.add_argument('--sigma', type=float, default=25.0, help='Sigma value for SDE')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for sampling')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of sampling steps')
    parser.add_argument('--ckpt', type=str, default='ckpt.pth', help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='samples.png', help='Output image file')
    args = parser.parse_args()

    device = args.device
    sigma = args.sigma
    batch_size = args.batch_size
    num_steps = args.num_steps
    ckpt_path = args.ckpt
    output_path = args.output

    print(f"Running on device: {device}")
    print(f"Sigma: {sigma}")
    print(f"Loading checkpoint from: {ckpt_path}")

    # Partial functions
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)

    # Initialize model
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    # Load checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file {ckpt_path} not found!")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device)
    score_model.load_state_dict(ckpt)
    score_model.eval()

    # Sample
    print("Sampling...")
    samples = euler_sampler(score_model, 
                           marginal_prob_std_fn, 
                           diffusion_coeff_fn, 
                           batch_size=batch_size, 
                           num_steps=num_steps, 
                           device=device)

    # Save samples
    samples = samples.clamp(0.0, 1.0)
    save_image(samples, output_path, nrow=int(np.sqrt(batch_size)))
    print(f"Samples saved to {output_path}")

if __name__ == "__main__":
    main()
