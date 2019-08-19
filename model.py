import torch
from torch import nn

class convx2(nn.Module):
    def __init__(self, *ch):
        super(convx2, self).__init__()
        self.conv_number = len(ch)-1
        self.model = nn.Sequential()
        for i in range(self.conv_number):
            self.model.add_module('conv{0}'.format(i),nn.Conv2d(ch[i], ch[i+1], 3, 1, 1))

    def forward(self, x):
        y = self.model(x)
        return y

class FC_EF(nn.Module):
    def __init__(self, in_ch = 3):
        super(FC_EF, self).__init__()
        self.conv1 = convx2(*[in_ch*2, 16, 16])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = convx2(*[16, 32, 32])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = convx2(*[32, 64, 64, 64])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = convx2(*[64, 128, 128, 128])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv5 = convx2(*[256, 128, 128, 64])
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = convx2(*[128, 64, 64, 32])
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = convx2(*[64, 32, 16])
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8 = convx2(*[32, 16, 2])

    def forward(self, x1, x2):
        h1 = self.conv1(torch.cat((x1,x2), 1))
        h = self.pool1(h1)
        h2 = self.conv2(h)
        h = self.pool2(h2)
        h3 = self.conv3(h)
        h = self.pool3(h3)
        h4 = self.conv4(h)
        h = self.pool4(h4)
        h = self.deconv1(h)
        h = self.conv5(torch.cat((h, h4), 1))
        h = self.deconv2(h)
        h = self.conv6(torch.cat((h, h3), 1))
        h = self.deconv3(h)
        h = self.conv7(torch.cat((h, h2), 1))
        h = self.deconv4(h)
        h = self.conv8(torch.cat((h, h1), 1))
        y = h
        return y