import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding='same', dilation=dilation)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding='same', dilation=dilation)
        self.bnorm1 = nn.BatchNorm2d(channel_out)
        self.bnorm2 = nn.BatchNorm2d(channel_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.activation(self.bnorm1(conv1))
        conv2 = self.conv2(conv1)
        conv2 = self.activation(self.bnorm2(conv2))
        return conv2

class Downsample(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.downsample = nn.Conv2d(channel_in, channel_in * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv_transpose(x)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.activation == 'relu':
            return self.relu(self.layer(x))
        return self.layer(x)  

class UNet(nn.Module):
    def __init__(self, channels, classes=1, subpixels=9):
        super(UNet, self).__init__()
        self.CHANNELS = channels
        self.CLASSES = classes
        self.SUBPIXELS = subpixels

        self.inp = ConvBlock(self.CHANNELS, 64)

        self.stage1 = ConvBlock(128, 128, dilation=1)
        self.stage2 = ConvBlock(256, 256, dilation=1)
        self.stage3 = ConvBlock(512, 512, dilation=2)

        self.down1 = Downsample(64)
        self.down2 = Downsample(128)
        self.down3 = Downsample(256)

        self.up1 = Upsample(512, 256)
        self.up2 = Upsample(256, 128)
        self.up3 = Upsample(128, 64)

        self.stage3i = ConvBlock(512, 256, dilation=2)
        self.stage2i = ConvBlock(256, 128, dilation=1)
        self.stage1i = ConvBlock(128, 64, dilation=1)

        self.out = nn.Conv2d(64, self.CLASSES, kernel_size=1)

        # subpixel classifier
        self.linear1 = Linear(512 * 4 * 4, 512, activation='relu')
        self.linear2 = Linear(512, 256, activation='relu')
        self.linear3 = Linear(256, self.SUBPIXELS + 1, activation=None)

    def forward(self, x):
        size = x.shape[-1]
        
        a1 = self.inp(x)
        d1 = self.down1(a1)

        a2 = self.stage1(d1)
        d2 = self.down2(a2)

        a3 = self.stage2(d2)
        d3 = self.down3(a3)

        a4 = self.stage3(d3)
        u1 = self.up1(a4)

        c1 = self.stage3i(torch.cat([a3, u1], dim=1))
        u2 = self.up2(c1)

        c2 = self.stage2i(torch.cat([a2, u2], dim=1))
        u3 = self.up3(c2)
        u3 = u3[:, :, :size, :size]                                             # resize for odd-sized images

        c3 = self.stage1i(torch.cat([a1, u3], dim=1))
        mask = self.out(c3)

        # subpixel classifier
        flat = torch.flatten(a4, start_dim=1)
        
        l1 = self.linear1(flat)
        l2 = self.linear2(l1)
        subp = self.linear3(l2)

        return mask, subp
