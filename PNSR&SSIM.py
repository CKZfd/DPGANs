import numpy as np
import math
import cv2
from scipy.signal import convolve2d

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



if __name__ == "__main__":
    psnr_total = 0
    ssim_total = 0
    for idy in range(10):
        gt = cv2.imread('quality_img/val_input_idz_%04d.png' % idy)
        # gt = cv2.imread('quality_img512/val_input_idz_%04d.png' % idy)
        img = cv2.imread('quality_img/val_target_idz_%04d.png' % idy)
        pnsr_now = psnr1(gt, img)
        # print("psnr%d:" % idy, pnsr_now)
        psnr_total += pnsr_now

        im1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ssim_now = compute_ssim(im1, im2)
        # print("ssim%d:" % idy, ssim_now)
        ssim_total += ssim_now
    psnr_total /= 10
    ssim_total /= 10
    print(psnr_total)
    print(ssim_total)

    epoch = 0
    for epoch in range(0,102,5):
        psnr_total =0
        ssim_total = 0
        for idy in range(14):
            gt = cv2.imread('quality_img/generated_epoch_%04d_idy_%04d.png' % (epoch, idy))
            # gt = cv2.imread('quality_img512/val_input_idz_%04d.png' % idy)
            img = cv2.imread('quality_img/val_target_idz_%04d.png' % idy)
            pnsr_now = psnr1(gt, img)
            # print("psnr%d:" % idy, pnsr_now)
            psnr_total += pnsr_now

            im1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ssim_now = compute_ssim(im1, im2)
            # print("ssim%d:" % idy, ssim_now)
            ssim_total += ssim_now
        psnr_total /= 14
        ssim_total /= 14
        print(epoch,psnr_total)
        # print(ssim_total)
        # print("epoch %d Aver psnr: "% epoch, psnr_total)
        # print("epoch %d Aver ssim: "% epoch, ssim_total)
    # gt = cv2.imread('quality_img512/generated_epoch_0000_idy_0000.png')
    # img = cv2.imread('quality_img512/val_target_idz_0000.png')
    # print(gt.shape)
    # print(psnr1(gt, img))
    #
    # im1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    # im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(im1.shape)
    # print(compute_ssim(im1, im2))









