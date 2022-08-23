import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, CharbonnierLoss
from models.modules.Quantization import Quantization
from models.modules.jacobian_calc import JacobianReg
logger = logging.getLogger('base')


class IRNModel(BaseModel):
    def __init__(self, opt):
        super(IRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.var = opt['var']
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        get_batch_method_dict = {
            'zeros': self.zeros_batch, 'repeat': self.z_repeat_batch}
        self.get_batch_method = get_batch_method_dict[self.train_opt['get_batch_method']]
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.CharbonnierLoss = CharbonnierLoss()
            self.Reconstruction_forw = ReconstructionLoss(
                losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(
                losstype=self.train_opt['pixel_criterion_back'])
            # self.Jacobian = JacobianReg()
            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            'Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']), amsgrad=True)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == 'LinearLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.LinearLR_Restart(
                            optimizer, train_opt['lr_steps']))
            else:
                raise NotImplementedError(
                    'MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)
        # return torch.zeros(tuple(dims)).to(self.device)

    def z_repeat_batch(self, dims, LR=None):
        repeat_num = int(dims[1] / LR.shape[1])
        return LR.repeat(1, repeat_num, 1, 1)

    def zeros_batch(self, dims, LR=None):
        return torch.zeros(tuple(dims)).to(self.device)

    def ChannelAdd(self, y, channel=3):
        # y:(16, 12, 72, 72)
        B, C, H, W = y.shape
        y_channel = torch.zeros((B, channel, H, W)).to(self.device)
        repeat_num = float(C / channel)
        for i in range(C):
            y_channel[:, i % channel] += y[:, i]
        return y_channel/repeat_num

    def AddNoise(self, y, zdim):
        B, channel, H, W = y.shape
        C = int(zdim)
        repeat_num = int(C/channel)
        lrs = y.repeat(1, repeat_num, 1, 1)
        # return lrs + torch.randn(lrs.shape).to(self.device)
        # add RGB channel by randn

        if self.var is None:
            self.var = 1/255.
        Randns = [self.var * torch.randn((B, repeat_num, H, W)).to(
            self.device) for _ in range(channel)]
        for i in range(repeat_num):
            for j in range(channel):
                lrs[:, i + j] = lrs[:, i + j] + Randns[j][:, i]

        # TODO: clamp or not clamp ?
        # lrs = torch.clamp(lrs, min=0, max=1)
        return lrs

    def loss_forward(self, y, y_sum, LR):
        # l_forw_fit = self.train_opt['lambda_fit_forw'] * \
        #     self.Reconstruction_forw(y, LR)
        # if y.shape != LR.shape:
        repeat_num = int(y.shape[1]/LR.shape[1])
        l_forw_fit = self.train_opt['lambda_fit_forw'] * \
            self.Reconstruction_forw(y, LR.repeat(1, repeat_num, 1, 1))
        l_forw_sum = self.train_opt['lambda_fit_forw'] * \
            self.Reconstruction_forw(y_sum, LR)
        # else:
        #     l_forw_fit = self.train_opt['lambda_fit_forw'] * \
        #         self.CharbonnierLoss(y, LR)
        return l_forw_fit, l_forw_sum

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.CharbonnierLoss(
            x, x_samples_image)
        l2_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_forw(
            x, x_samples_image)
        return l_back_rec, l2_back_rec

    def loss_Jac_backward(self, output):
        input_inv, Jac_G_inv = self.netG(x=output, rev=True, calc_jac=True)
        return self.train_opt['lambda_Jac_G'] * Jac_G_inv

    def warmup(self, step):
        quant = False
        self.train_opt['train_LR_ref_start'] = 0.
        train_LR_ref = False
        calc_jac = False
        if step > self.train_opt['quant_start']:
            quant = True
        if step > self.train_opt['train_LR_ref_start']:
            train_LR_ref = True
        if step > self.train_opt['calc_jac_start']:
            calc_jac = True
        return quant, train_LR_ref, calc_jac

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        quant, train_LR_ref, calc_jac = self.warmup(step)
        # forward downscaling
        self.input = self.real_H
        self.input.requires_grad = True
        self.output = self.netG(x=self.input)
        y1, y2 = self.output[:, :3, :, :], self.output[:, 3:, :, :]

        noise = self.netR(y1)
        y = y2 + noise

        # TODO: split y and calc loss in repeat mode
        if calc_jac:
            Jac_G_inv = self.loss_Jac_backward(self.output)

        # zshape = y2.shape
        zdim = self.output.shape[1]
        LR_ref = self.ref_L.detach()
        y = self.ChannelAdd(self.output)
        # l_forw_fit = self.loss_forward(y, LR_ref)
        l_forw_fit, l_forw_sum = self.loss_forward(self.output, y, LR_ref)

        # backward upscaling
        if not train_LR_ref:
            if quant:
                LR = self.Quantization(y)
            else:
                LR = y
            y_ = self.AddNoise(LR, zdim)
        else:
            y_ = self.AddNoise(LR_ref, zdim)

        l_back_rec, l2_back_rec = self.loss_backward(self.real_H, y_)

        # total loss
        # loss = l_forw_fit + l_back_rec
        loss = l_forw_fit + l_forw_sum + l_back_rec
        if calc_jac:
            loss += Jac_G_inv
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(
                self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()
        # set log
        # self.log_dict['l2_forw'] = l_forw_fit.item() + l_forw_ce.item()
        # self.log_dict['l2_forw'] = l_forw_fit.item()
        self.log_dict['l2_back_rec'] = l2_back_rec.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_sum'] = l_forw_sum.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()
        if calc_jac:
            self.log_dict['Jac_G_inv'] = Jac_G_inv.item()

    def test(self, test_LR_ref=False):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zdim = input_dim * (self.opt['scale']**2)

        self.netG.eval()
        with torch.no_grad():
            if test_LR_ref:
                LR_ref = self.ref_L.detach()
                self.forw_L = LR_ref
            else:
                out = self.netG(x=self.input)
                self.forw_L = self.ChannelAdd(out)
                self.forw_L = self.Quantization(self.forw_L)
            y_forw = self.AddNoise(self.forw_L, zdim)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            out = self.netG(x=HR_img)
            LR_img = self.ChannelAdd(out)
            LR_img = self.Quantization(LR_img)
        self.netG.train()

        return LR_img

    # def upscale(self, LR_img, scale, gaussian_scale=1):
    #     Lshape = LR_img.shape
    #     zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
    #     y_ = torch.cat((LR_img, gaussian_scale *
    #                     self.gaussian_batch(zshape)), dim=1)

    #     self.netG.eval()
    #     with torch.no_grad():
    #         HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
    #     self.netG.train()

    #     return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG,
                              self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
