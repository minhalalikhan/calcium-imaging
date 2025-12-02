import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=1):
    # Assumes img1 and img2 are 4D tensors: (N, C, H, W) with pixel values in [0, val_range]
    channel = img1.size(1)
    if window is None:
        real_size = min(window_size, img1.size(2), img1.size(3))
        window = create_window(real_size, channel).to(img1.device)
        
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.5, window_size=11, val_range=1.0):
        super(SSIM_MSE_Loss, self).__init__()
        self.alpha = alpha  # weight for MSE loss
        self.window_size = window_size
        self.val_range = val_range
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        # Normalize images to [0,1] if val_range > 1
        if self.val_range != 1:
            img1 = img1 / self.val_range
            img2 = img2 / self.val_range
            
        ssim_loss = 1 - ssim(img1, img2, window_size=self.window_size, val_range=1.0)
        mse_loss = self.mse(img1, img2)
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
        return total_loss
