import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.GenNoise import GenNoise
from models.modules.Inv_arch import InvRescaleNet, InvZipNet
from models.modules.Subnet_constructor import subnet
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    model_name = which_model['model_name']
    if model_name is None:
        model_name = 'InvRescaleNet'
    subnet_type = which_model['subnet_type']
    print(subnet_type)
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    if opt_net['clamp']:
        clamp = opt_net['clamp']
    else:
        clamp = 1.
    model_dict = {'InvRescaleNet': InvRescaleNet, 'InvZipNet': InvZipNet}
    down_num = int(math.log(opt_net['scale'], 2))
    netG = model_dict[model_name](opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init, instance_norm=opt_net['instance_norm']),
                                  opt_net['block_num'], down_num=down_num, scale=opt_net['scale'], preprocess_op=opt_net['preprocess_op'], clamp=clamp)

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(
            in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError(
            'Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


def define_R(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_R']

    if which_model == 'GenNoise':
        in_ch = opt_net['in_nc']
        out_ch = in_ch * (opt_net['scale'] ** 2) - opt_net['out_nc']
        clamp = 1 if opt_net['clamp_R'] is None else opt_net['clamp_R']
        netR = GenNoise(in_ch, out_ch, clamp=clamp)
    else:
        raise NotImplementedError(
            'Noise Add Rmodel [{:s}] not recognized'.format(which_model))
    return netR


# Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
