import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from models.guide_filter import GuidedFilter


def conv_Block(in_c, out_c, stride=1, bn=True):
    block = []
    if stride==1:
        block.append(nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
    else:
        block.append(nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False))
    if bn:
        block.append(nn.BatchNorm2d(out_c))
    block.append(nn.LeakyReLU(0.2, inplace=True))
    conv_Block = nn.Sequential(*block)
    return conv_Block


def unet_Block(in_c, out_c, transposed=False, bn=True, relu=True):
    block = []
    if transposed:
        if relu:
            block.append(nn.ReLU(inplace=True))
        block.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
        if bn:
            block.append(nn.BatchNorm2d(out_c))
    else:
        if relu:
            block.append(nn.LeakyReLU(0.2, inplace=True))
        block.append(nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False))
        if bn:
            block.append(nn.BatchNorm2d(out_c))
    unet_Block = nn.Sequential(*block)
    return unet_Block


def blockUNet(in_c, out_c, transposed=False, bn=True, relu=True, dropout=False):
    block = []
    if relu:
        block.append(nn.ReLU(inplace=True))
    else:
        block.append(nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.append(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.append(nn.BatchNorm2d(out_c))
    if dropout:
        block.append(nn.Dropout2d(0.5, inplace=True))
    blockUNet = nn.Sequential(*block)
    return blockUNet


class D(nn.Module):
    def __init__(self, nc, nf):
        super(D, self).__init__()

        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer1.append(conv_Block(nc, nf, stride=1, bn=False))  # (1, 64, 256, 256)

        layer2.append(conv_Block(nf, nf*2, stride=2, bn=True)) # (1, 128, 128, 128)
        nf = nf * 2  # 128
        layer2.append(conv_Block(nf, nf, stride=1, bn=True))  # (1, 128, 128, 128)
        layer2.append(conv_Block(nf, nf * 2, stride=2, bn=True))   # (1, 256, 64, 64)

        nf = nf * 2  # 256
        layer3.append(conv_Block(nf, nf, stride=1, bn=True))  # (1, 256, 64, 64)
        layer3.append(conv_Block(nf, nf * 2, stride=2, bn=True))  # (1, 512, 32, 32)

        nf = nf * 2  # 512
        layer4.append(conv_Block(nf, nf, stride=2, bn=True))  # (1, 512, 16, 16)
        layer4.append(conv_Block(nf, nf, stride=2, bn=True))  # (1, 512, 8, 8)

        layer5.append(conv_Block(nf, 8, stride=2, bn=True))  # (1, 8, 4, 4)
        layer5.append(nn.Conv2d(8, 1, 4, 1, 0, bias=False))
        layer5.append(nn.Sigmoid())

        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)

    def forward(self, x):
        per1 = self.layer1(x)
        per2 = self.layer2(per1)
        per3 = self.layer3(per2)
        per4 = self.layer4(per3)
        output = self.layer5(per4)
        return output, per1, per2, per3, per4


class G(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G, self).__init__()

        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []
        dlayer6 = []
        dlayer5 = []
        dlayer4 = []
        dlayer3 = []
        dlayer2 = []
        dlayer1 = []

        layer1.append(unet_Block(input_nc, nf, transposed=False, bn=False, relu=False))  # (1, 64, 128, 128)
        layer2.append(unet_Block(nf, nf*2, transposed=False, bn=True, relu=True))  # (1, 128, 64, 64)
        layer3.append(unet_Block(nf * 2, nf * 4, transposed=False, bn=True, relu=True))  # (1, 256, 32, 32)
        layer4.append(unet_Block(nf * 4, nf * 8, transposed=False, bn=True, relu=True))  # (1, 512, 16, 16)
        layer5.append(unet_Block(nf * 8, nf * 8, transposed=False, bn=True, relu=True))  # (1, 512, 8, 8)
        layer6.append(unet_Block(nf * 8, nf * 8, transposed=False, bn=True, relu=True))  # (1, 512, 4, 4)
        layer6.append(nn.LeakyReLU(0.2, inplace=True))

        d_inc = nf * 8
        dlayer6.append(unet_Block(d_inc, nf * 8, transposed=True, bn=True, relu=False))  # (1, 512, 8, 8)
        d_inc = nf * 8 * 2
        dlayer5.append(unet_Block(d_inc, nf * 8, transposed=True, bn=True, relu=True))  # (1, 512, 16, 16)
        d_inc = nf * 8 * 2
        dlayer4.append(unet_Block(d_inc, nf * 4, transposed=True, bn=True, relu=True))  # (1, 256, 32, 32)
        d_inc = nf * 4 * 2
        dlayer3.append(unet_Block(d_inc, nf * 2, transposed=True, bn=True, relu=True))   # (1, 128, 64, 64)
        d_inc = nf * 2 * 2
        dlayer2.append(unet_Block(d_inc, nf, transposed=True, bn=True, relu=True))   # (1, 64, 128, 128)
        d_inc = nf * 2
        dlayer1.append(unet_Block(d_inc, output_nc, transposed=True, bn=False, relu=True))  # (1, 3, 256, 256)
        dlayer1.append(nn.Tanh())

        self.guide_filter = GuidedFilter(15, 1e-2)
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)
        self.layer6 = nn.Sequential(*layer6)
        self.dlayer6 = nn.Sequential(*dlayer6)
        self.dlayer5 = nn.Sequential(*dlayer5)
        self.dlayer4 = nn.Sequential(*dlayer4)
        self.dlayer3 = nn.Sequential(*dlayer3)
        self.dlayer2 = nn.Sequential(*dlayer2)
        self.dlayer1 = nn.Sequential(*dlayer1)

    def forward(self, x):
        base = self.guide_filter(x, x)  # using guided filter for obtaining base layer
        detail = x - base  # detail layer
        out1 = self.layer1(detail)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)

        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        out = x - dout1

        return out, dout1, detail

