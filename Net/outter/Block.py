import torch.nn as nn
import torch.nn.functional as F
class Block(nn.Module):
    def __init__(self):
        #利用super进行初始化
        super(Block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        y=x
        x1=self.leakyrelu(self.conv1(x))
        x2=self.leakyrelu(self.conv2(x1))
        y1=y+x2
        return y1