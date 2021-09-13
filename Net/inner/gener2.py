#输入的图片需要是偶数
#奇数不行
import torch.nn as nn
import torch.nn.functional as F
from cincgan.Net.inner.Block import Block
class G2(nn.Module):
    def __init__(self):
        super(G2,self).__init__()
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.block=self.make_layer(Block,6)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv5=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv6=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=7,stride=1,padding=3)
    def forward(self,x):
        x=self.leakyrelu(self.conv1(x))
        x=self.leakyrelu(self.conv2(x))
        x=self.leakyrelu(self.conv3(x))
        x=self.block(x)
        x=self.leakyrelu(self.conv4(x))
        x=self.leakyrelu(self.conv5(x))
        x=self.conv6(x)
        return x
    def make_layer(self,block,num):
        y=[]
        for _ in range(num):
            y.append(block())
        return nn.Sequential(*y)