'''
    Copyright (c) Facebook, Inc. and its affiliates.
    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    PyTorch implementation of Jacobian regularization described in [1].
    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
'''
from __future__ import division
import torch
import torch.nn as nn
import torch.autograd


class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self):
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        B, C, H, W = y.shape
        # random properly-normalized vector for each sample
        v = torch.ones((B, C, H, W)).to(x.device)
        Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
        # print("Jv:", Jv)
        # J2 = torch.norm(Jv)**2 / (B * C * H * W)
        # R = torch.log1p((1 / 2) * J2)
        # R = J2
        return torch.mean(Jv**2)

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        '''
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''
        # flat_y = y.reshape(-1)
        # flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=create_graph)
        # if grad_x is None:
        #     grad_x = 0.
        return grad_x
