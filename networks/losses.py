import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(ColorLoss, self).__init__()

        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, x, y):
        loss = self.loss(x, y)
        return loss


class LaplacianLoss(nn.Module):
    def __init__(self, reduction='sum', max_level=5):
        super(LaplacianLoss, self).__init__()

        self.criterion = nn.L1Loss(reduction=reduction)
        self.lap = LaplacianPyramid(max_level=max_level)

    def forward(self, x, y):
        x_lap, y_lap = self.lap(x), self.lap(y)
        diff_levels = [self.criterion(a, b) for a, b in zip(x_lap, y_lap)]
        return sum(2**(j - 1) * diff_levels[j] for j in range(len(diff_levels))) * 1e-2


class GaussianConv(nn.Module):
    def __init__(self, kernel_size=5, channels=3, sigma=2.0):
        super(GaussianConv, self).__init__()
        kernel = self.gauss_kernel(kernel_size, sigma)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(kernel, requires_grad=False).cuda()
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2, groups=self.channels)

    def gauss_kernel(self, size=5, sigma=1.0):
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        kernel = np.sum(np.exp((grid - size // 2)**2 / (-2 * sigma**2))**2, axis=2)
        kernel /= np.sum(kernel)
        return kernel


class LaplacianPyramid(nn.Module):
    def __init__(self, max_level=5):
        super(LaplacianPyramid, self).__init__()
        self.gaussian_conv = GaussianConv()
        self.max_level = max_level

    def forward(self, x):
        t_pyr = []
        current = x
        for level in range(self.max_level - 1):
            t_guass = self.gaussian_conv(current)
            t_diff = current - t_guass
            t_pyr.append(t_diff)
            current = F.avg_pool2d(t_guass, 2)
        t_pyr.append(current)

        return t_pyr
