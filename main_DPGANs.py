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
parser.add_argument('--dataroot', required=False,
  default='./facades/DID-MDN-training/Rain_Medium/train2018new', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='/facades/github', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=7, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=14, help='input batch size')
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
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=200, help='epoch to reaching at learning rate of 0')

parser.add_argument('--lambdaL1', type=float, default=25., help='lambdaL1')
parser.add_argument('--lambdaGAN', type=float, default=1., help='lambdaGAN')
parser.add_argument('--lambdaP1', type=float, default=5., help='lambdaP1')
parser.add_argument('--lambdaP2', type=float, default=1.5, help='lambdaP2')
parser.add_argument('--lambdaP3', type=float, default=1.5, help='lambdaP3')
parser.add_argument('--lambdaP4', type=float, default=1, help='lambdaP4')

parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--exp', default='check', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=200, help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--saveIter', type=int, default=5, help='interval for save network')
# using pretrained
parser.add_argument('--pretrained_model', type=int, default=None)
parser.add_argument('--Perceptual', type=int, default=1, help="0or1")
parser.add_argument('--margin', type=float, default=0.5, help='margin for D perceprtual loss')

opt = parser.parse_args()
print(opt)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
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
  C1 = (k1 * L) ** 2
  C2 = (k2 * L) ** 2
  window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
  window = window / np.sum(np.sum(window))

  if im1.dtype == np.uint8:
    im1 = np.double(im1)
  if im2.dtype == np.uint8:
    im2 = np.double(im2)

  mu1 = filter2(im1, window, 'valid')
  mu2 = filter2(im2, window, 'valid')
  mu1_sq = mu1 * mu1
  mu2_sq = mu2 * mu2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
  sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
  sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

  return np.mean(np.mean(ssim_map))


def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def load_pretrained_model(opt):
    netG.load_state_dict(torch.load(os.path.join(
        '%s/netG_epoch_%d.pth' % (opt.exp, opt.pretrained_model))))
    netD.load_state_dict(torch.load(os.path.join(
        '%s/netD_epoch_%d.pth' % (opt.exp, opt.pretrained_model))))
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
                           opt.dataroot,
                           opt.originalSize,
                           opt.imageSize,
                           opt.batchSize,
                           opt.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='train',
                           shuffle=True,
                           seed=opt.manualSeed
                           )

    opt.dataset = 'pix2pix'
    valDataloader = getLoader(opt.dataset,
                              opt.valDataroot,
                              opt.imageSize,    # opt.originalSize,
                              opt.imageSize,
                              opt.valBatchSize,
                              opt.workers,
                              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                              split='val',
                              shuffle=False,
                              seed=opt.manualSeed)

    # get logger
    trainLogger = open('%s/train.log' % opt.exp, 'w')

    ngf = opt.ngf
    ndf = opt.ndf
    inputChannelSize = opt.inputChannelSize
    outputChannelSize= opt.outputChannelSize

    # get models
    netG = net.G(inputChannelSize, outputChannelSize, ngf)
    netG.apply(weights_init)
    netD = net.D(inputChannelSize + outputChannelSize, ndf)
    netD.apply(weights_init)

    # NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper) 注L_cGAN和L_L1的权重(即本文式(4))
    lambdaGAN = opt.lambdaGAN
    lambdaL1 = opt.lambdaL1
    lambdaP1 = opt.lambdaP1
    lambdaP2 = opt.lambdaP2
    lambdaP3 = opt.lambdaP3
    lambdaP4 = opt.lambdaP4

    netG.train()
    netD.train()
    criterionBCE = nn.BCELoss()
    criterionL1 = nn.L1Loss()  # L1损失函数
    criterionMSE = nn.MSELoss()  # l2损失函数

    target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
    val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
    label_d = torch.FloatTensor(opt.batchSize)
    # NOTE: size of 2D output maps in the discriminator
    sizePatchGAN = 1
    real_label = 1
    fake_label = 0

    # image pool storing previously generated samples from G 图像池存储以前从G生成的样本
    imagePool = ImagePool(opt.poolSize)

    #  cuda
    netD.cuda()
    netG.cuda()
    criterionBCE.cuda()
    criterionL1.cuda()
    criterionMSE.cuda()
    target, input, label_d = target.cuda(), input.cuda(), label_d.cuda()
    val_target, val_input = val_target.cuda(), val_input.cuda()

    target = Variable(target)
    input = Variable(input)
    label_d = Variable(label_d)

    # get randomly sampled validation images and save it 获取随机采样的验证图像并保存
    val_iter = iter(valDataloader)
    data_val = val_iter.next()
    if opt.mode == 'B2A':
        val_target_cpu, val_input_cpu = data_val  # 此处暂时用不到雨量标签。
    elif opt.mode == 'A2B':
        val_input_cpu, val_target_cpu = data_val
    val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
    val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
    val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
    vutils.save_image(val_target, '%s/real_target.png' % opt.exp, nrow=7, normalize=True)
    vutils.save_image(val_input, '%s/real_input.png' % opt.exp, nrow=7, normalize=True)
    print(val_input.size())
    print(val_target.size())
    for idz in range(val_target.size(0)):
        val_target_img = val_target[idz, :, :, :]
        val_input_img = val_input[idz, :, :, :]
        vutils.save_image(val_target_img, 'quality_img/val_target_idz_%04d.png' % idz, normalize=True)
        vutils.save_image(val_input_img, 'quality_img/val_input_idz_%04d.png' % idz, normalize=True)

    # get optimizer
    optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
    optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.0)

    # Start with trained model从训练模型开始
    if opt.pretrained_model:
      start = opt.pretrained_model + 1
      load_pretrained_model(opt)
    else:
      start = 0
    print(netG)
    print(netD)
    # Start time
    start_time = time.time()
    # NOTE training loop训练循环
    ganIterations = 0
    for epoch in range(start, opt.niter + 1):
        if epoch > opt.annealStart:
            adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
            adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
        for i, data in enumerate(dataloader, 0):
            if opt.mode == 'B2A':
                target_cpu, input_cpu = data
            elif opt.mode == 'A2B':
                input_cpu, target_cpu = data
            batch_size = target_cpu.size(0)
            if epoch == 0 and i % 14 == 0:
                vutils.save_image(target_cpu, 'quality_img/target_cpu%04d.png' % (i/14), normalize=True)
                vutils.save_image(input_cpu, 'quality_img/input_cpu%04d.png' % (i/14), normalize=True)
            target_cpu, input_cpu = target_cpu.cuda(), input_cpu.cuda()
            # NOTE paired samples 成双样本
            target.data.resize_as_(target_cpu).copy_(target_cpu)
            input.data.resize_as_(input_cpu).copy_(input_cpu)

            # max_D first 首先最大化分类器
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            # NOTE: compute L_cGAN in eq.(2) 在式(2)中计算L_cGAN     式（1）？
            label_d.data.resize_((batch_size, 1, sizePatchGAN, sizePatchGAN)).fill_(real_label)  # 填1
            output, per1, per2, per3, per4 = netD(torch.cat([target, input], 1))  # conditional

            errD_real = criterionBCE(output, label_d)
            errD_real.backward()
            D_x = output.data.mean()

            x_hat, _, _ = netG(input)
            fake = x_hat.detach()  # 返回一个新变量，与当前图形分离。  返回的 Variable 永远不会需要梯度
            fake = Variable(imagePool.query(fake.data))
            label_d.data.fill_(fake_label)

            output, _, _, _, _ = netD(torch.cat([fake, input], 1))  # conditional
            _, per1_hat, per2_hat, per3_hat, per4_hat = netD(torch.cat([x_hat, input], 1))
            errD_fake = criterionBCE(output, label_d)
            errD_fake.backward()
            D_G_z1 = output.data.mean()

            # perceptual loss

            if opt.Perceptual == 1:
                errD_Per_1 = criterionL1(per1_hat, per1)  # L1
                errD_Per_2 = criterionL1(per2_hat, per2)  # L1
                errD_Per_3 = criterionL1(per3_hat, per3)  # L1
                errD_Per_4 = criterionL1(per4_hat, per4)  # L1
                errD_Per = opt.margin - (lambdaP1 * errD_Per_1 + lambdaP2 * errD_Per_2 +
                                         lambdaP3 * errD_Per_3 + lambdaP4 * errD_Per_4)
                if errD_Per < 0:
                    errD_Per = 0
                elif lambdaP1 != 0 or lambdaP2 != 0 or lambdaP3 != 0 or lambdaP4 != 0:
                    errD_Per.backward()
            else:
                errD_Per = 0

            errD = errD_real + errD_fake + errD_Per
            optimizerD.step()  # update parameters

            # prevent computing gradients of weights in Discriminator
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad() # start to update G

            # compute L_L1 (eq.(4) in the paper
            _, per1, per2, per3, per4 = netD(torch.cat([target, input], 1))  # conditional
            output, per1_hat, per2_hat, per3_hat, per4_hat = netD(torch.cat([x_hat, input], 1))

            L_img_ = criterionL1(x_hat, target)  #L1
            L_img = lambdaL1 * L_img_     # 乘以L1的权重
            if lambdaL1 != 0:
                L_img.backward(retain_graph=True) # in case of current version of pytorch

            if opt.Perceptual == 1:
                errPer_1 = criterionL1(per1_hat, per1)  # L1
                errPer_2 = criterionL1(per2_hat, per2)  # L1
                errPer_3 = criterionL1(per3_hat, per3)  # L1
                errPer_4 = criterionL1(per4_hat, per4)  # L1
                errPer = lambdaP1 * errPer_1 + lambdaP2 * errPer_2 + lambdaP3 * errPer_3 + lambdaP4 * errPer_4
                if lambdaP1 != 0 or lambdaP2 != 0 or lambdaP3 != 0 or lambdaP4 != 0:
                    errPer.backward(retain_graph=True)
            else:
                errPer = 0

            # compute L_cGAN (eq.(2) in the paper
            label_d.data.fill_(real_label)
            errG_ = criterionBCE(output, label_d)
            errG = lambdaGAN * errG_
            if lambdaGAN != 0:
                # errG.backward(retain_graph=True)  # in case of current version of pytorch
                errG.backward()
            D_G_z2 = output.data.mean()

            optimizerG.step()
            ganIterations += 1

            if ganIterations % opt.display == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], epoch [{}/{}], step [{}/{}], "
                      "L_D:{:.6f} L_DP:{:.6f} L_P:{:.6f} L_img:{:.6f} L_G:{:.6f}  D(x):{:.6f} D(G(z)):[{:.6f}/{:.6f}]".
                      format(elapsed, epoch, opt.niter, i, len(dataloader),
                             errD.data, errD_Per, errPer, L_img.data, errG.data, D_x, D_G_z1, D_G_z2))

                sys.stdout.flush()
                trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                                  (i, errD.data, errG.data, L_img.data, D_x, D_G_z1, D_G_z2))
                trainLogger.flush()

            if i % opt.evalIter == 0:
                val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
                val_batch_mask = torch.FloatTensor(val_input.size()).fill_(0)
                val_batch_detail = torch.FloatTensor(val_input.size()).fill_(0)
                for idx in range(val_input.size(0)):
                    single_img = val_input[idx, :, :, :].unsqueeze(0)
                    with torch.no_grad():
                        val_inputv = Variable(single_img)
                    x_hat_val, mask, detail = netG(val_inputv)
                    x_hat_val = x_hat_val.view(3, opt.imageSize, opt.imageSize)
                    mask = mask.view(3, opt.imageSize, opt.imageSize)
                    detail = detail.view(3, opt.imageSize, opt.imageSize)
                    val_batch_output[idx, :, :, :].copy_(x_hat_val.data)
                    val_batch_mask[idx, :, :, :].copy_(mask.data)
                    val_batch_detail[idx, :, :, :].copy_(detail.data)
                vutils.save_image(val_batch_output, '%s/generated_epoch_%04d_iter%04d.png' % \
                                  (opt.exp, epoch, i), nrow=7, normalize=True)
                vutils.save_image(val_batch_mask, '%s/mask_epoch_%04d_iter%04d.png' % \
                                  (opt.exp, epoch, i), nrow=7, normalize=True)
                if epoch == 3:
                    vutils.save_image(val_batch_detail, '%s/detail_epoch_%04d_iter%04d.png' % \
                                      (opt.exp, epoch, i), nrow=7, normalize=True)

            # do checkpointing
        if epoch % opt.saveIter == 0:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.exp, epoch))
            for idy in range(val_batch_output.size(0)):
                quality_img = val_batch_output[idy, :, :, :]
                vutils.save_image(quality_img, 'quality_img/generated_epoch_%04d_idy_%04d.png' % \
                                  (epoch, idy), normalize=True)
                gt = cv2.imread('quality_img/generated_epoch_%04d_idy_%04d.png' % \
                                (epoch, idy))
                img = cv2.imread('quality_img/val_target_idz_%04d.png' % idy)
                print("psnr%d:" % idy, psnr1(gt, img))

                im1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print("ssim%d:" % idy, compute_ssim(im1, im2))
    trainLogger.close()


