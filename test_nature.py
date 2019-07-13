from __future__ import print_function
import argparse
import time
import datetime
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import math
import cv2
from scipy.signal import convolve2d

import models.DPGANs as net
from misc import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--testDataroot', required=False,
  default=' ./facades/dataset/rain/test_nature', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--testBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=286, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
parser.add_argument('--netG_BEGAN', default='', help="path to netG (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='check', help='folder to output images and model checkpoints')
parser.add_argument('--pretrained_model', type=int, default=90)
opt = parser.parse_args()
print(opt)



def load_pretrained_model(opt):
    netG.load_state_dict(torch.load(os.path.join(
        '%s/netG_epoch_%d.pth' % (opt.exp, opt.pretrained_model))))
    print('loaded trained models (epoch: {})..!'.format(opt.pretrained_model))



if __name__ == '__main__':
    create_exp_dir(opt.exp)
    opt.manualSeed = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    # get dataloader
    dataloader = getLoader(opt.dataset,
                           opt.testDataroot,
                           opt.originalSize,
                           opt.imageSize,
                           opt.testBatchSize,
                           opt.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='test',
                           shuffle=False,
                           seed=opt.manualSeed
                           )

    # get logger
    trainLogger = open('%s/test.log' % opt.exp, 'w')

    ngf = opt.ngf
    ndf = opt.ndf
    inputChannelSize = opt.inputChannelSize
    outputChannelSize = opt.outputChannelSize

    # get models
    netG = net.G(inputChannelSize, outputChannelSize, ngf)
    netG.apply(weights_init)

    netG.train()
    val_target = torch.FloatTensor(opt.testBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.testBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    netG.cuda()
    val_target, val_input = val_target.cuda(), val_input.cuda()


    # Start with trained model从训练模型开始
    if opt.pretrained_model:
        start = opt.pretrained_model + 1
        load_pretrained_model(opt)
    else:
        start = 0
    print(netG)

    # get randomly sampled validation images and save it

    for i in range(0,10):
        val_iter = iter(dataloader)
        data_val = val_iter.next()
        if opt.mode == 'B2A':
            val_target_cpu, val_input_cpu = data_val  # 此处暂时用不到雨量标签。
        elif opt.mode == 'A2B':
            val_input_cpu, val_target_cpu, _ = data_val
        val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
        val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
        val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

        output = torch.FloatTensor(val_input.size(0) * 2, 3, val_input.size(2), val_input.size(3)).fill_(0)


        for idx in range(val_input.size(0)):
            input_img = val_input[idx, :, :, :].unsqueeze(0)
            target_img = val_target[idx, :, :, :].unsqueeze(0)
            with torch.no_grad():
                input = Variable(input_img)
                real = Variable(target_img)
            # fake_BEGAN = netG_BEGAN(input)
            fake_GAN, _, _ = netG(input)

            output[idx * 2 + 0, :, :, :].copy_(input.data.squeeze(0))
            # output[idx * 3 + 1, :, :, :].copy_(real.data.squeeze(0))
            output[idx * 2 + 1, :, :, :].copy_(fake_GAN.data.squeeze(0))
            # output[idx * 4 + 3, :, :, :].copy_(fake_GAN.data.squeeze(0))
            vutils.save_image(fake_GAN.data.squeeze(0), '%s/%4dfake.png' % (opt.exp, i), normalize=True)
            vutils.save_image(input.data.squeeze(0), '%s/%4dinput.png' % (opt.exp, i), normalize=True)
        vutils.save_image(output, '%s/%dgenerated_nature.png' % (opt.exp, i), padding=10, nrow=2, normalize=True, pad_value=1.0)
