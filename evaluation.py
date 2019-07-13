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
    psnr_cnn_total = 0
    psnr_dpgan_total = 0
    psnr_gmm_total = 0
    psnr_input_total = 0
    ssim_cnn_total = 0
    ssim_dpgan_total = 0
    ssim_gmm_total = 0
    ssim_input_total = 0
    max_key = []
    max_value1 = []
    max_value2 = []

    for i in range(0, 100):
        if i < 10:
            file_cnn = 'new/000' + str(i) + 'cnn.png'
            file_gmm = 'new/000' + str(i) + 'gmm.png'
            file_dpgan = 'new/000' + str(i) + 'generate.png'
            file_input = 'new/000' + str(i) + 'input.png'
            file_real = 'new/000' + str(i) + 'real.png'
        else:
            file_cnn = 'new/00' + str(i) + 'cnn.png'
            file_gmm = 'new/00' + str(i) + 'gmm.png'
            file_dpgan = 'new/00' + str(i) + 'generate.png'
            file_input = 'new/00' + str(i) + 'input.png'
            file_real = 'new/00' + str(i) + 'real.png'
        cnn = cv2.imread(file_cnn)
        gmm = cv2.imread(file_gmm)
        dpgan = cv2.imread(file_dpgan)
        input = cv2.imread(file_input)
        real = cv2.imread(file_real)

        psnr_cnn = psnr1(cnn, real)
        psnr_gmm = psnr1(gmm, real)
        psnr_dpgan = psnr1(dpgan, real)
        psnr_input = psnr1(input, real)
        psnr_real = psnr1(real, real)
        print(i,": ", psnr_input, psnr_gmm, psnr_cnn, psnr_dpgan,psnr_real)
        psnr_cnn_total += psnr_cnn
        psnr_gmm_total += psnr_gmm
        psnr_dpgan_total += psnr_dpgan
        psnr_input_total += psnr_input

        cnn1 = cv2.cvtColor(cnn, cv2.COLOR_BGR2GRAY)
        gmm1 = cv2.cvtColor(gmm, cv2.COLOR_BGR2GRAY)
        dpgan1 = cv2.cvtColor(dpgan, cv2.COLOR_BGR2GRAY)
        input1 = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        real1 = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
        ssim_cnn = compute_ssim(cnn1, real1)
        ssim_gmm = compute_ssim(gmm1, real1)
        ssim_dpgan = compute_ssim(dpgan1, real1)
        ssim_input = compute_ssim(input1, real1)
        ssim_real = compute_ssim(real1, real1)

        print(i, ": ", ssim_input, ssim_gmm, ssim_cnn, ssim_dpgan, ssim_real)
        ssim_cnn_total += ssim_cnn
        ssim_gmm_total += ssim_gmm
        ssim_dpgan_total += ssim_dpgan
        ssim_input_total += ssim_input
        if psnr_dpgan > psnr_cnn and psnr_dpgan > psnr_gmm and ssim_dpgan >ssim_cnn and ssim_dpgan > ssim_gmm:
            max_key.append(i)
            max_value1.append((psnr_input, psnr_gmm, psnr_cnn, psnr_dpgan))
            max_value2.append((ssim_input, ssim_gmm, ssim_cnn, ssim_dpgan))

    print("total: ", psnr_input_total/100, psnr_gmm_total/100, psnr_cnn_total/100, psnr_dpgan_total/100)
    print("total: ", ssim_input_total/100, ssim_gmm_total/100, ssim_cnn_total/100, ssim_dpgan_total/100)
    print(max_key)
    for i in range(0, len(max_value1)):
        print(max_key[i], max_value1[i])
        print(max_key[i], max_value2[i])