import torch
from torch import nn
from torch.nn import functional as F


class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x = x.clamp(min=self.eps).pow(exponent=self.p)
        kernel_size = x.size()[2:]
        y = F.avg_pool2d(input=x, kernel_size=kernel_size).pow(exponent=1.0 / self.p)
        return y


class MaxAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)

        return torch.cat((max_f, avg_f), 1)
