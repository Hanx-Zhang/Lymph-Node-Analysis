import numpy as np
import torch.nn as nn
import torch

class double_conv(nn.Module):
    ''' Applies (conv => BN => ReLU) two times. '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(int(out_ch / 2), out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(int(out_ch / 2), out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    ''' First Section of U-Net. '''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    ''' Applies a MaxPool with a Kernel of 2x2,
        then applies a double convolution pack. '''

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class dense_down(nn.Module):
    ''' Applies a MaxPool with a Kernel of 2x2,
        then applies a double convolution pack. '''

    def __init__(self, in_ch, out_ch):
        super(dense_down, self).__init__()

        self.dense1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2)
        )

        self.dense2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2)
        )

        self.mpconv = nn.Sequential(
            double_conv(in_ch, out_ch)
        )

    def forward(self, x1, x2):
        x1 = self.dense1(x1)
        x2 = self.dense2(x2)
        x_dense = torch.cat([x1, x2], dim=1)

        x = self.mpconv(x_dense)
        return x_dense, x

class dense_down_bottom(nn.Module):
    ''' Applies a MaxPool with a Kernel of 2x2,
        then applies a double convolution pack. '''

    def __init__(self, in_ch, out_ch, out_ch_2):
        super(dense_down_bottom, self).__init__()

        self.dense1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2)
        )

        self.dense2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2)
        )

        self.mpconv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(int(out_ch / 2), out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch_2, kernel_size=3, padding=1),
            nn.GroupNorm(int(out_ch_2 / 2), out_ch_2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.dense1(x1)
        x2 = self.dense2(x2)
        x_dense = torch.cat([x1, x2], dim=1)

        x = self.mpconv(x_dense)
        return x_dense, x

class up(nn.Module):
    ''' Applies a Deconvolution and then applies applies a double convolution pack. '''

    def __init__(self, in_up, in_ch, out_ch, trilinear=True):
        super(up, self).__init__()

        if trilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_up, in_up, kernel_size=2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x

class outconv(nn.Module):
    ''' Applies the last Convolution to give an answer. '''

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Dense_UNet(nn.Module):
    ''' This Object defines the architecture of U-Net. '''

    def __init__(self, n_channels, n_classes):
        super(Dense_UNet, self).__init__()

        self.inc = inconv(n_channels, 16)
        self.down1 = dense_down(16+n_channels, 32)
        self.down2 = dense_down(48+n_channels, 64)
        self.down3 = dense_down_bottom(112+n_channels, 128, 64)

        self.up1 = up(64, 128, 32)
        self.up2 = up(32, 64, 16)
        self.up3 = up(16, 32, 16)
        self.outc = outconv(16, n_classes)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2_dense, x2 = self.down1(x,x1)
        x3_dense, x3 = self.down2(x2_dense, x2)
        x4_dense, x4 = self.down3(x3_dense, x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outc(x)
        x = self.out_act(x)
        return x

def preprocess_vessel_normalization(img):
    img = (img - (-1000)) / (600 - (-1000))
    img = np.clip(img, 0, 1)
    return img

