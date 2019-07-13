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
  default='/facades/dataset/rain/test_syn', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--testBatchSize', type=int, default=14, help='input batch size')
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


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def psnr1(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))





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
                              opt.imageSize,  # opt.originalSize,
                              opt.imageSize,
                              opt.testBatchSize,
                              opt.workers,
                              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                              split='val',
                              shuffle=False,
                              seed=opt.manualSeed)

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
    val_iter = iter(dataloader)
    data_val = val_iter.next()
    for i, data in enumerate(dataloader, 0):
        if opt.mode == 'B2A':
            val_target_cpu, val_input_cpu = data_val  # 此处暂时用不到雨量标签。
        elif opt.mode == 'A2B':
            val_input_cpu, val_target_cpu, _ = data_val
        val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
        val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
        val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

        output = torch.FloatTensor(val_input.size(0) * 3, 3, val_input.size(2), val_input.size(3)).fill_(0)

        total_input_psnr = 0
        total_generate_psnr = 0
        total_input_ssim = 0
        total_generate_ssim = 0


        for idx in range(val_input.size(0)):
            input_img = val_input[idx, :, :, :].unsqueeze(0)
            target_img = val_target[idx, :, :, :].unsqueeze(0)
            with torch.no_grad():
                input = Variable(input_img)
                real = Variable(target_img)
            # fake_BEGAN = netG_BEGAN(input)
            fake_GAN, _, _ = netG(input)

            output[idx * 3 + 0, :, :, :].copy_(input.data.squeeze(0))
            output[idx * 3 + 1, :, :, :].copy_(real.data.squeeze(0))
            output[idx * 3 + 2, :, :, :].copy_(fake_GAN.data.squeeze(0))
            # output[idx * 4 + 3, :, :, :].copy_(fake_GAN.data.squeeze(0))

            vutils.save_image(output[idx * 3 + 0, :, :, :], 'quality_img/input_%04d.png' % idx, normalize=True)
            vutils.save_image(output[idx * 3 + 1, :, :, :], 'quality_img/real_%04d.png' % idx, normalize=True)
            vutils.save_image(output[idx * 3 + 2, :, :, :], 'quality_img/generate_%04d.png' % idx, normalize=True)
            input_t = cv2.imread('quality_img/input_%04d.png' % idx)
            real_t = cv2.imread('quality_img/real_%04d.png' % idx)
            generate_t = cv2.imread('quality_img/generate_%04d.png' % idx)

            input_psnr = psnr1(input_t, real_t)
            generate_psnr = psnr1(generate_t, real_t)
            print("input_t psnr %d:" % idx, input_psnr)
            print("generate_t psnr %d:" % idx, generate_psnr)

            im1 = cv2.cvtColor(input_t, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(real_t, cv2.COLOR_BGR2GRAY)
            im3 = cv2.cvtColor(generate_t, cv2.COLOR_BGR2GRAY)
            input_ssim = compute_ssim(im1, im2)
            generate_ssim = compute_ssim(im3, im2)
            print("input_t ssim %d:" % idx, input_ssim)
            print("generate_t ssim %d:" % idx, generate_ssim)
            total_input_psnr += input_psnr
            total_generate_psnr += generate_psnr
            total_input_ssim += input_ssim
            total_generate_ssim += generate_ssim


        vutils.save_image(output, '%s/1generated.png' % opt.exp, nrow=3, normalize=True)
        print("total_input_psnr :", total_input_psnr/opt.testBatchSize)
        print("total_generate_psnr :", total_generate_psnr/opt.testBatchSize)
        print("total_input_ssim :", total_input_ssim/opt.testBatchSize)
        print("total_generate_ssim :", total_generate_ssim/opt.testBatchSize)
