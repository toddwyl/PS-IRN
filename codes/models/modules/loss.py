import torch
import torch.nn as nn
import numpy as np


class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                'GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


def charb(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.mean(torch.sum(torch.pow(x.pow(2) + eps, 1. / alpha), (1, 2, 3)))
    # return torch.mean(torch.pow(x.pow(2) + eps, 1. / alpha))


class CharbonnierLoss(torch.nn.Module):
    '''From Back to Basics: 
    Unsupervised Learning of Optical Flow
    via Brightness Constancy and Motion Smoothness'''

    def __init__(self, alpha=2, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        return charb(input - target, self.alpha, self.eps)


class MagnitudeLoss(torch.nn.Module):

    def __init__(self, loss):
        super(MagnitudeLoss, self).__init__()
        self.loss = loss

    def forward(self, w):
        return self.loss(w, w.detach() * 0)


class SmoothnessLoss(torch.nn.Module):
    '''From Back to Basics: 
    Unsupervised Learning of Optical Flow
    via Brightness Constancy and Motion Smoothness'''

    def __init__(self, loss, delta=1):
        super(SmoothnessLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def forward(self, w):
        ldudx = self.loss((w[:, 0, 1:, :] - w[:, 0, :-1, :]) /
                          self.delta, w[:, 0, 1:, :].detach() * 0)
        ldudy = self.loss((w[:, 0, :, 1:] - w[:, 0, :, :-1]) /
                          self.delta, w[:, 0, :, 1:].detach() * 0)
        ldvdx = self.loss((w[:, 1, 1:, :] - w[:, 1, :-1, :]) /
                          self.delta, w[:, 1, 1:, :].detach() * 0)
        ldvdy = self.loss((w[:, 1, :, 1:] - w[:, 1, :, :-1]) /
                          self.delta, w[:, 1, :, 1:].detach() * 0)
        return ldudx + ldudy + ldvdx + ldvdy


class DivergenceLoss(torch.nn.Module):

    def __init__(self, loss, delta=1):
        super(DivergenceLoss, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, w):
        dudx = (w[:, 0, 1:] - w[:, 0, :-1]) / self.delta
        dvdy = (w[:, 1, 1:] - w[:, 1, :-1]) / self.delta
        return self.loss(dudx + dvdy, dudx.detach() * 0)


class WeightedSpatialMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedSpatialMSELoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduce=False, size_average=False)

    def forward(self, input, target, weights=1):
        return self.loss(input, target).mean(3).mean(2).mean(1) * weights
