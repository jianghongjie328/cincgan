import torch
import torch.nn as nn

if __name__=="__main__":
    print(torch.__version__)
    # test=torch.rand(1,3,5,5)
    # m=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=4,stride=2,padding=1)
    # out=m(test)
    # print(out.size())

    #padding: size+2
    #kernal_size=k stride=s [size+2*padding-(kernal_size-stride)]/stride  n+1为最终大小

    #5+2-(4-2)  9/2=4