import torch
import torch.nn as nn
class tvloss(nn.Module):
    def __init__(self,tvloss_weight=0.5,batch_size=16):
        super(tvloss,self).__init__()
        self.tvloss_weight=tvloss_weight
        self.batch_size=batch_size

    def forward(self,x):
        batch_size=x.size()[0]
        x_height=x.size()[2]
        x_width=x.size()[3]
        g_h_size=self.total_size(x[:,:,1:,:])
        g_w_size=self.total_size(x[:, :, :, 1:])
        g_h = torch.pow(x[:,:,1:,:]-x[:,:,:x_height-1,:],2).sum()
        g_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :x_width - 1], 2).sum()
        loss=self.tvloss_weight*(g_h/g_h_size+g_w/g_w_size)/float(self.batch_size)
        return loss

    def total_size(self,x):
        return x.size()[1]*x.size()[2]*x.size()[3]



