import torch
import torch.nn as nn
import models.modules.module_util as mutil


class ConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        mutil.initialize_weights_xavier(self.blocks, 0.1)

    def forward(self, x):
        output = self.blocks(x)
        return output

# Nested Unet


class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1, n1=32, clamp=0.003):
        super(NestedUNet, self).__init__()
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_ch, filters[0], filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3], filters[3])
        # self.conv4_0 = ConvBlock(filters[3], filters[4], filters[4])

        self.conv0_1 = ConvBlock(
            filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = ConvBlock(
            filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = ConvBlock(
            filters[2] + filters[3], filters[2], filters[2])
        # self.conv3_1 = ConvBlock(
        #     filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = ConvBlock(
            filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = ConvBlock(
            filters[1]*2 + filters[2], filters[1], filters[1])
        # self.conv2_2 = ConvBlock(
        #     filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = ConvBlock(
            filters[0]*3 + filters[1], filters[0], filters[0])
        # self.conv1_3 = ConvBlock(
        #     filters[1]*3 + filters[2], filters[1], filters[1])
        # self.conv0_4 = ConvBlock(
        #     filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.clamp = clamp

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        # x0_4 = self.conv0_4(
        #     torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.clamp * self.final(x0_3)
        return output
