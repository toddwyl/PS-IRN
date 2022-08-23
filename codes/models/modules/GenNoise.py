import torch
import torch.nn as nn
from models.modules.Subnet_constructor import DenseBlock


class GenNoise(nn.Module):
    def __init__(self, in_ch=3, out_ch=9, gc=64, clamp=0.7):
        super(GenNoise, self).__init__()
        self.R = nn.Sequential(
            DenseBlock(in_ch, gc),
            DenseBlock(gc, 2*gc),
            DenseBlock(2*gc, 4*gc),
            DenseBlock(4*gc, out_ch)
        )
        self.short = DenseBlock(in_ch, out_ch)
        self.clamp = clamp

    def forward(self, x):
        noise = self.R(x)
        output = self.clamp * torch.tanh(self.short(x) + noise)
        return output
