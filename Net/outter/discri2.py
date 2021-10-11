
import torch.nn as nn
import torch

class D2(nn.Module):

    def __init__(self):
        super(D2,self).__init__()
        self.d1=nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
        )
    def forward(self,x):
        x1=self.d1(x)
        return x1