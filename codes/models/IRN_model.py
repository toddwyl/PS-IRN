import logging
from collections import OrderedDict
import itertools
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
logger = logging.getLogger('base')


class IRNModel(BaseModel):
    def __init__(self, opt):
        super(IRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        get_batch_method_dict = {
            'zeros': self.zeros_batch, 'repeat': self.z_repeat_batch}
        self.get_batch_method = get_batch_method_dict[self.train_opt['get_batch_method']]
        self.netG = networks.define_G(opt).to(self.device)
        self.netR = networks.define_R(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()])
            self.netR = DistributedDataParallel(
                self.netR, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.netR = DataParallel(self.netR)
        # print network
        self.models = [self.netG, self.netR]
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()
            self.netR.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(
                losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(
                losstype=self.train_opt['pixel_criterion_back'])
            # self.Jacobian = JacobianReg()
            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in itertools.chain(self.netG.named_parameters(), self.netR.named_parameters()):
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            'Params [{:s}] will not optimize.'.format(k))
            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                              weight_decay=wd_G,
                                              betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)

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
        ratio = int(dims[1] / LR.shape[1])
        return LR.repeat(1, ratio, 1, 1)

    def zeros_batch(self, dims, LR=None):
        return torch.zeros(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z, z_gt=None):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * \
            self.Reconstruction_forw(out, y)
        if z_gt is None:
            z_gt = torch.zeros_like(z)
        l_forw_ce = self.train_opt['lambda_ce_forw'] * \
            self.Reconstruction_forw(z, z_gt)

        return l_forw_fit, l_forw_ce

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(
            x, x_samples_image)
        l2_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_forw(
            x, x_samples_image)
        return l_back_rec, l2_back_rec

    def loss_Jac_backward(self, output):
        input_inv, Jac_G_inv = self.netG(x=output, rev=True, calc_jac=True)
        return self.train_opt['lambda_Jac_G'] * Jac_G_inv

    def warmup(self, step):
        quant = False
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
        self.optimizer.zero_grad()
        quant, train_LR_ref, calc_jac = self.warmup(step)
        # forward downscaling
        self.input = self.real_H
        self.input.requires_grad = True
        self.output = self.netG(x=self.input)
        y1, y2 = self.output[:, :3, :, :], self.output[:, 3:, :, :]
        if quant:
            y1 = self.Quantization(y1)
        noise = self.netR(y1)
        y2 = y2 - noise
        if calc_jac:
            Jac_G_inv = self.loss_Jac_backward(self.output)
        zshape = y2.shape
        LR_ref = self.ref_L.detach()
        z_gt = self.get_batch_method(zshape, LR_ref)
        l_forw_fit, l_forw_ce = self.loss_forward(y1, LR_ref, y2, z_gt)

        if not train_LR_ref:
            # backward upscaling
            LR = y1
            noise_hat = self.netR(LR).detach()
            y_ = torch.cat(
                (LR, self.get_batch_method(zshape, LR)+noise_hat), dim=1)
            l_back_rec, l2_back_rec = self.loss_backward(self.real_H, y_)
        else:
            noise_hat = self.netR(LR_ref).detach()
            y_gt = torch.cat((LR_ref, z_gt+noise_hat), dim=1)
            l_back_rec, l2_back_rec = self.loss_backward(self.real_H, y_gt)

        # total loss
        loss = l_forw_fit + l_back_rec + l_forw_ce
        if calc_jac:
            loss += Jac_G_inv
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(
                self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer.step()
        # set log
        self.log_dict['l2_forw'] = l_forw_fit.item() + l_forw_ce.item()
        self.log_dict['l2_back_rec'] = l2_back_rec.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_ce'] = l_forw_ce.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()
        if calc_jac:
            self.log_dict['Jac_G_inv'] = Jac_G_inv.item()

    def test(self, test_LR_ref=False):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim *
                  (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        self.netG.eval()
        with torch.no_grad():
            if test_LR_ref:
                LR_ref = self.ref_L.detach()
                self.forw_L = LR_ref
            else:
                self.forw_L = self.netG(x=self.input)[:, :3, :, :]
                self.forw_L = self.Quantization(self.forw_L)
            noise_L = self.netR(self.forw_L)
            y_forw = torch.cat(
                (self.forw_L, self.get_batch_method(zshape, self.forw_L) + noise_L), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

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
        models = self.models
        for model in models:
            s, n = self.get_network_description(model)
            if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(model.__class__.__name__,
                                                 model.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(model.__class__.__name__)
            if self.rank <= 0:
                logger.info(
                    'Network G or R structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG,
                              self.opt['path']['strict_load'])

        load_path_R = self.opt['path']['pretrain_model_R']
        if load_path_R is not None:
            logger.info(
                'Loading model for R [{:s}] ...'.format(load_path_R))
            self.load_network(load_path_R, self.netR,
                              self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netR, 'R', iter_label)
