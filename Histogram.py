import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def image_hist(image): #画三通道图像的直方图
   color = ("blue", "green", "red")#画笔颜色的值可以为大写或小写或只写首字母或大小写混合
   for i, color in enumerate(color):
       hist = cv.calcHist([image], [i], None, [256], [0, 256])
       plt.plot(hist, color=color)
       plt.xlim([0, 256])
   plt.show()


def image_hist_norm(image): #画三通道图像的直方图
   color = ("blue", "green", "red")#画笔颜色的值可以为大写或小写或只写首字母或大小写混合
   for i, color in enumerate(color):
       hist = cv.calcHist([image], [i], None, [256], [0.0, 1.0])
       plt.plot(hist, color=color)
       plt.xlim([0.0, 1.0])
   plt.show()


for j in range(0, 12):
    image = cv.imread('D:/PYTHON/GANs\pix2pixBEGAN.pytorch-master/sample/'+str(j)+'input.jpg', 1)
    color = ("blue", "green", "red")  # 画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(221)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    image = cv.imread('D:/PYTHON/GANs\pix2pixBEGAN.pytorch-master/sample/'+str(j)+'target.jpg', 1)
    color = ("blue", "green", "red")  # 画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(222)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    image = cv.imread('D:/PYTHON/GANs\pix2pixBEGAN.pytorch-master/sample/'+str(j)+'mask.jpg', 1)
    color = ("blue", "green", "red")  # 画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(223)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    image = cv.imread('D:/PYTHON/GANs\pix2pixBEGAN.pytorch-master/sample/'+str(j)+'detail2.jpg', 1)
    color = ("blue", "green", "red")  # 画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(224)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


# image = cv.imread('D:/PYTHON/GANs\pix2pixBEGAN.pytorch-master/sample/4mask.jpg', 1)
# cv.imshow('souce image', image)
# # result = np.zeros(image.shape, dtype=np.float32)
# # cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# # cv.imshow('image', result)
# image_hist(image)
# for i in range(0, 5):
#     rain = cv.imread('D:/PYTHON/GANs\pix2pixBEGAN.pytorch-master//sample/4detail'+str(i)+'.jpg', 1)
#     # result = np.zeros(image.shape, dtype=np.float32)
#     # cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
#     # cv.imshow('image', result)
#     image_hist(rain)
# # image_hist_norm(result)
# cv.waitKey(0)
# cv.destroyAllWindows()
