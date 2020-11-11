import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.jacobian_calc import JacobianReg


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.name = 'psi'
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, inpt):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, h, w = inpt.shape
        return inpt.view(bs, bl, bl, int(d // bl_sq), h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, -1, h * bl, w * bl)

    def forward(self, inpt, rev=False):
        if not rev:
            bl, bl_sq = self.block_size, self.block_size_sq
            bs, d, new_h, new_w = inpt.shape[0], inpt.shape[1], int(
                inpt.shape[2] // bl), int(inpt.shape[3] // bl)
            out = inpt.view(bs, d, new_h, bl, new_w, bl).permute(
                0, 3, 5, 1, 2, 4).reshape(bs, d*bl_sq, new_h, new_w)
        else:
            out = self.inverse(inpt)
        return out


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()
        self.name = 'InvBlockExp'
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        self.K = subnet_constructor(self.split_len2, self.split_len1)
        self.Jacobian = JacobianReg()

    def forward(self, x1, x2=None, rev=False, cat=False):
        if x2 is None:
            x1, x2 = (x1.narrow(1, 0, self.split_len1), x1.narrow(
                1, self.split_len1, self.split_len2))

        if not rev:
            self.s2 = self.clamp * (torch.sigmoid(self.K(x2)) * 2 - 1)
            y1 = x1.mul(torch.exp(self.s2)) + self.F(x2)
            # y1 = x1.mul(torch.exp(self.beta)) + self.F(x2)
            self.s1 = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s1)) + self.G(y1)
        else:
            self.s1 = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s1))
            self.s2 = self.clamp * (torch.sigmoid(self.K(y2)) * 2 - 1)
            y1 = (x1 - self.F(y2)).div(torch.exp(self.s2))
            # y1 = (x1 - self.F(y2)).div(torch.exp(self.beta))
        if cat:
            return torch.cat((y1, y2), 1)
        else:
            return y1, y2


class InvZipBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvZipBlock, self).__init__()
        self.name = 'InvZipBlock'
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.beta1 = nn.Parameter(torch.FloatTensor([1.]), requires_grad=True)
        self.beta2 = nn.Parameter(torch.FloatTensor([1.]), requires_grad=True)
        self.alpha1 = nn.Parameter(torch.FloatTensor([1.]), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.FloatTensor([1.]), requires_grad=True)
        self.Jacobian = JacobianReg()

    def forward(self, x1, x2=None, rev=False, cat=False):
        if x2 is None:
            x1, x2 = (x1.narrow(1, 0, self.split_len1), x1.narrow(
                1, self.split_len1, self.split_len2))

        if not rev:
            y1 = self.alpha1 * self.beta2 * (x1 + self.F(x2))
            y2 = self.beta1 * x2 + self.alpha2 * self.G(y1.div(self.beta2))
        else:
            y2 = (x2 - self.alpha2 * self.G(x1.div(self.beta2))).div(self.beta1)
            y1 = x1.div(self.alpha1 * self.beta2) - self.F(y2)
        if cat:
            return torch.cat((y1, y2), 1)
        else:
            return y1, y2


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.name = 'HaarDownsampling'
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None,
                           stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in,
                               4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4,
                               x.shape[2] // 2, x.shape[3] // 2])
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape(
                [x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape(
                [x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2,
                                     groups=self.channel_in)
            return out

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[],
                 down_num=1, scale=2, preprocess_op='Haar'):
        super(InvRescaleNet, self).__init__()
        self.name = 'InvRescaleNet'
        operations = []
        current_channel = channel_in
        self.split_len1 = []
        self.split_len2 = []
        if preprocess_op == 'Haar':
            op = HaarDownsampling(current_channel)
        elif preprocess_op == 'PixelShuffle':
            op = psi(scale)
        else:
            assert False, 'Unknown preprocess_op'
        for i in range(down_num):
            # b = HaarDownsampling(current_channel)
            b = op
            operations.append(b)
            self.split_len1.append(0)
            self.split_len2.append(0)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor,
                                current_channel, channel_out)
                operations.append(b)
                self.split_len1.append(channel_out)
                self.split_len2.append(current_channel - channel_out)
        self.Jacobian = JacobianReg()
        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, calc_jac=False):
        if not rev:
            x_ps = self.operations[0].forward(x, rev)
            x1, x2 = (x_ps.narrow(1, 0, self.split_len1[1]), x_ps.narrow(
                1, self.split_len1[1], self.split_len2[1]))
            y1, y2 = x1, x2
            for idx, op in enumerate(self.operations[1:]):
                i = idx+1
                if op.name == 'HaarDownsampling' or op.name == 'psi':
                    y_ps = op.forward(torch.cat((y1, y2), dim=1), rev)
                    y1, y2 = (y_ps.narrow(1, 0, self.split_len1[i + 1]), y_ps.narrow(
                        1, self.split_len1[i + 1], self.split_len2[i + 1]))
                else:
                    y1, y2 = op.forward(y1, y2, rev)
            out = torch.cat((y1, y2), dim=1)
        else:
            len_op = len(self.operations)
            x1, x2 = (x.narrow(1, 0, self.split_len1[len_op-1]), x.narrow(
                1, self.split_len1[len_op-1], self.split_len2[len_op-1]))
            y1, y2 = x1, x2
            for idx, op in enumerate(reversed(self.operations[1:])):
                i = len_op - idx - 1
                if op.name == 'HaarDownsampling' or op.name == 'psi':
                    y_ps = op.forward(torch.cat((y1, y2), dim=1), rev)
                    y1, y2 = (y_ps.narrow(1, 0, self.split_len1[i - 1]), y_ps.narrow(
                        1, self.split_len1[i - 1], self.split_len2[i - 1]))
                else:
                    y1, y2 = op.forward(y1, y2, rev)
            out = self.operations[0].forward(torch.cat((y1, y2), dim=1), rev)
        if calc_jac:
            jac = self.Jacobian(x1, y1) + self.Jacobian(x1, y2) + \
                self.Jacobian(x2, y1) + self.Jacobian(x2, y2)
            return out, jac
        else:
            return out


class InvZipNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[],
                 down_num=1, scale=2, preprocess_op='Haar'):
        super(InvZipNet, self).__init__()

        operations = []
        self.name = 'InvZipNet'
        current_channel = channel_in
        if preprocess_op == 'Haar':
            op = HaarDownsampling(current_channel)
        elif preprocess_op == 'PixelShuffle':
            op = psi(scale)
        else:
            assert False, 'Unknown preprocess_op'
        for i in range(down_num):
            b = op
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvZipBlock(subnet_constructor,
                                current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, calc_jac=False):
        out = x
        if not rev:
            for idx, op in enumerate(self.operations):
                out = op.forward(out, rev)
        else:
            if calc_jac:
                Jac_G_inv = 1.
                for idx, op in enumerate(reversed(self.operations)):
                    out, Jac_B = op.forward(out, rev, calc_jac=calc_jac)
                    Jac_G_inv *= Jac_B
                return out, Jac_G_inv
            else:
                for idx, op in enumerate(reversed(self.operations)):
                    out = op.forward(out, rev)
        return out
