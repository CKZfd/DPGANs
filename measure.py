import numpy
from numpy import sqrt, pi
import scipy.ndimage.filters
import math
from scipy.ndimage import gaussian_filter
import scipy.signal
import scipy.ndimage
from numpy.lib.stride_tricks import as_strided as ast


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map = (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

# FIXME there seems to be a problem with this code
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return numpy.mean(ssim_map)


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den
    return vifp


def Laguerre_Gauss_Circular_Harmonic_3_0(size, sigma):
    x = numpy.linspace(-size / 2.0, size / 2.0, size)
    y = numpy.linspace(-size / 2.0, size / 2.0, size)
    xx, yy = numpy.meshgrid(x, y)

    r = numpy.sqrt(xx * xx + yy * yy)
    gamma = numpy.arctan2(yy, xx)
    l30 = - (1 / 6.0) * (1 / (sigma * sqrt(pi))) * numpy.exp(-r * r / (2 * sigma * sigma)) * (
                sqrt(r * r / (sigma * sigma)) ** 3) * numpy.exp(-1j * 3 * gamma)
    return l30


def Laguerre_Gauss_Circular_Harmonic_1_0(size, sigma):
    x = numpy.linspace(-size / 2.0, size / 2.0, size)
    y = numpy.linspace(-size / 2.0, size / 2.0, size)
    xx, yy = numpy.meshgrid(x, y)

    r = numpy.sqrt(xx * xx + yy * yy)
    gamma = numpy.arctan2(yy, xx)
    l10 = - (1 / (sigma * sqrt(pi))) * numpy.exp(-r * r / (2 * sigma * sigma)) * sqrt(
        r * r / (sigma * sigma)) * numpy.exp(-1j * gamma)
    return l10


"""
Polar edge coherence map
Same size as source image
"""


def pec(img):
    # TODO scale parameter should depend on resolution
    l10 = Laguerre_Gauss_Circular_Harmonic_1_0(17, 2)
    l30 = Laguerre_Gauss_Circular_Harmonic_3_0(17, 2)
    y10 = scipy.ndimage.filters.convolve(img, numpy.real(l10)) + 1j * scipy.ndimage.filters.convolve(img,
                                                                                                     numpy.imag(l10))
    y30 = scipy.ndimage.filters.convolve(img, numpy.real(l30)) + 1j * scipy.ndimage.filters.convolve(img,
                                                                                                     numpy.imag(l30))
    pec_map = - (numpy.absolute(y30) / numpy.absolute(y10)) * numpy.cos(numpy.angle(y30) - 3 * numpy.angle(y10))
    return pec_map


"""
Edge coherence metric
Just one number summarizing typical edge coherence in this image.
"""


def eco(img):
    l10 = Laguerre_Gauss_Circular_Harmonic_1_0(17, 2)
    l30 = Laguerre_Gauss_Circular_Harmonic_3_0(17, 2)
    y10 = scipy.ndimage.filters.convolve(img, numpy.real(l10)) + 1j * scipy.ndimage.filters.convolve(img,
                                                                                                     numpy.imag(l10))
    y30 = scipy.ndimage.filters.convolve(img, numpy.real(l30)) + 1j * scipy.ndimage.filters.convolve(img,
                                                                                                     numpy.imag(l30))
    eco = numpy.sum(- (numpy.absolute(y30) * numpy.absolute(y10)) * numpy.cos(numpy.angle(y30) - 3 * numpy.angle(y10)))
    return eco


"""
Relative edge coherence
Ratio of ECO
"""
def reco(img1, img2):
    C = 1  # TODO what is a good value?
    return (eco(img2) + C) / (eco(img1) + C)


if __name__ == "__main__":
    # Inputs are image files
    epoch = 500
    for idy in range(10):
        print("idy: ", idy)
        ref_file = 'quality_img/val_target_idz_%04d.png' % idy
        dist_file = 'quality_img/generated_epoch_%04d_idy_%04d.png' % (epoch, idy)
        ref = scipy.misc.imread(ref_file, flatten=True).astype(numpy.float32)
        dist = scipy.misc.imread(dist_file, flatten=True).astype(numpy.float32)

        width, height = ref.shape[1], ref.shape[0]
        print("Comparing %s to %s, resolution %d x %d" % (ref_file, dist_file, width, height))

        vifp_value = vifp_mscale(ref, dist)
        print("VIFP=%f" % (vifp_value))

        ssim_value = ssim_exact(ref / 255, dist / 255)
        print("SSIM=%f" % (ssim_value))

        # FIXME this is buggy, disable for now
        # ssim_value2 = ssim.ssim(ref/255, dist/255)
        # print "SSIM approx=%f" % (ssim_value2)

        psnr_value = psnr(ref, dist)
        print("PSNR=%f" % (psnr_value))

        # niqe_value = niqe.niqe(dist/255)
        # print "NIQE=%f" % (niqe_value)

        reco_value = reco(ref / 255, dist / 255)
        print("RECO=%f" % (reco_value))