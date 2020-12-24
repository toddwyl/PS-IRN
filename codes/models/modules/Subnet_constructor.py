import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.RDB import RDNBlock


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, instance_norm=False):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc,
                               channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm2d(channel_out)
        if init == 'xavier':
            mutil.initialize_weights_xavier(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.instance_norm:
            x5_norm = self.norm(x5)
            return x5_norm
        else:
            return x5


class PANBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, instance_norm=False):
        super(PANBlock, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, 3, 1, 1, bias=bias)
        self.gate_conv = nn.Conv2d(
            channel_in, channel_out, 1, 1, 0, bias=bias)
        self.out_conv = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=bias)

        if init == 'xavier':
            mutil.initialize_weights_xavier(
                [self.conv, self.gate_conv, self.out_conv], 0.1)
        else:
            mutil.initialize_weights(
                [self.conv, self.gate_conv, self.out_conv], 0.1)

    def forward(self, x):
        # pdb.set_trace()
        h = self.conv(x)
        w = self.gate_conv(x)
        pa = h * torch.sigmoid(w)
        return self.out_conv(pa)


class PhyBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(PhyBlock, self).__init__()
        self.channel_in = channel_in
        self.F_hidden_dim = 49
        self.filter_size = 3
        self.padding = 3//2
        self.bias = bias
        self.F = nn.Sequential()
        self.F.add_module('bn1', nn.GroupNorm(4, channel_in))
        self.F.add_module('conv1', nn.Conv2d(in_channels=channel_in, out_channels=self.F_hidden_dim,
                                             kernel_size=self.filter_size, stride=(1, 1), padding=self.padding))
        self.F.add_module('conv2', nn.Conv2d(in_channels=self.F_hidden_dim,
                                             out_channels=channel_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.conv = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.convgate = nn.Conv2d(in_channels=channel_out + channel_out,
                                  out_channels=channel_out,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

    def forward(self, x):
        # x [batch_size, hidden_dim, height, width]
        gain = self.F(x)
        h = self.conv(x)
        x_add = h + gain        # prediction
        # concatenate along channel axis
        combined = torch.cat([h, x_add], dim=1)
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        # correction , Haddamard product
        o = x_add + K * gain
        return o


class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, instance_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, 3, 1, 1, bias=bias)
        self.out_conv = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=bias)

        if init == 'xavier':
            mutil.initialize_weights_xavier(
                [self.conv, self.out_conv], 0.1)
        else:
            mutil.initialize_weights(
                [self.conv, self.out_conv], 0.1)

    def forward(self, x):
        h = self.conv(x)
        return self.out_conv(h)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, instance_norm=False):
        super(ResidualDenseBlock, self).__init__()
        self.conv_1x1 = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5_1x1 = nn.Conv2d(channel_in + 4 * gc,
                                   channel_out, 1, 1, 0, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm2d(channel_out)
        if init == 'xavier':
            mutil.initialize_weights_xavier(
                [self.conv1, self.conv2, self.conv3, self.conv4, self.conv_1x1, self.conv5_1x1], 0.1)
        else:
            mutil.initialize_weights(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):
        xr = self.conv_1x1(x)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5_1x1(torch.cat((x, x1, x2, x3, x4), 1))
        if self.instance_norm:
            return self.norm(x5+xr)
        else:
            return x5+xr


# def subnet(net_structure, init='xavier', instance_norm=False):
#     def constructor(channel_in, channel_out):
#         if net_structure == 'DBNet':
#             if init == 'xavier':
#                 return DenseBlock(channel_in, channel_out, init)
#             else:
#                 return DenseBlock(channel_in, channel_out)
#         elif net_structure == 'RDBNet':
#             if init == 'xavier':
#                 return ResidualDenseBlock(channel_in, channel_out, init)
#             else:
#                 return ResidualDenseBlock(channel_in, channel_out)
#         else:
#             return None

#     return constructor

def subnet(net_structure_dict, init='xavier', instance_norm=False):
    def constructor_wraper(net_structure):
        def constructor(channel_in, channel_out):
            if net_structure == 'DBNet':
                if init == 'xavier':
                    return DenseBlock(channel_in, channel_out, init)
                else:
                    return DenseBlock(channel_in, channel_out)
            elif net_structure == 'RDBNet':
                if init == 'xavier':
                    return ResidualDenseBlock(channel_in, channel_out, init)
                else:
                    return ResidualDenseBlock(channel_in, channel_out)
            elif net_structure == 'PAN':
                if init == 'xavier':
                    return PANBlock(channel_in, channel_out, init)
                else:
                    return PANBlock(channel_in, channel_out)
            elif net_structure == 'Conv':
                if init == 'xavier':
                    return ConvBlock(channel_in, channel_out, init)
                else:
                    return ConvBlock(channel_in, channel_out)
            elif net_structure == 'RDN':
                if init == 'xavier':
                    return RDNBlock(channel_in, channel_out, init)
                else:
                    return RDNBlock(channel_in, channel_out)
            elif net_structure == 'Phy':
                if init == 'xavier':
                    return PhyBlock(channel_in, channel_out, init)
                else:
                    return PhyBlock(channel_in, channel_out)
            else:
                return None
        return constructor

    if isinstance(net_structure_dict, str):
        # single model or all the blocks are the same
        return constructor_wraper(net_structure_dict)
    else:
        return {k: constructor_wraper(v) for k, v in net_structure_dict.items()}
