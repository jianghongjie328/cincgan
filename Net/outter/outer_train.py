import torch
from torch.utils.data import Dataset,DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gener3 import G3
from discri2 import D2
from gener2 import G2
from gener1 import G1
from discri1 import D1
from EDSR import Edsr
import torch.optim as optim
from weight_init import weight_init
from Tvloss import tvloss

torch.autograd.set_detect_anomaly(True)
criteration_GAN_LR=nn.MSELoss()
criteration_cyc_LR=nn.MSELoss()
criteration_idt_LR=nn.L1Loss()



class Load_LR_difficult(Dataset):  # 继承Dataset类
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Load_LR_difficult, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            img_folder = root + '/DIV2K_train_LR_difficult_crop/'
        else:
            img_folder = root
        num_data = 800
        self.filenames = []
        self.img_folder = img_folder
        self.filenames = os.listdir(self.img_folder)

    def __getitem__(self, index):  # 返回目标图
        img_name = self.img_folder + self.filenames[index]
        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):  # 总的数据长度
        return (len(self.filenames))


class Load_LR(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Load_LR, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            img_folder = root + '/DIV2K_train_LR_crop/'
        else:
            img_folder = root
        num_data = 800
        self.filenames = []
        self.img_folder = img_folder
        self.filenames = os.listdir(self.img_folder)

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return (len(self.filenames))

class Load_HR(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Load_LR, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            img_folder = root + '/DIV2K_train_HR_crop/'
        else:
            img_folder = root
        num_data = 800
        self.filenames = []
        self.img_folder = img_folder
        self.filenames = os.listdir(self.img_folder)

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return (len(self.filenames))
if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sets_x = Load_LR_difficult('C:/Users/maker/Desktop/cincgan/DIV2K_train', transform=transforms.ToTensor())
    sets_x1 = Load_LR('C:/Users/maker/Desktop/cincgan/DIV2K_train', transform=transforms.ToTensor())  # 预先载入：从哪来，进行的转换
    sets_z = Load_HR('C:/Users/maker/Desktop/cincgan/DIV2K_train', transform=transforms.ToTensor())
    x = DataLoader(dataset=sets_x, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)
    x1 = DataLoader(dataset=sets_x1, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)  # batch，是否打乱
    z=DataLoader(dataset=sets_z, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)
    d1=D1()
    g1=G1()
    g2=G2()
    g3=G3()
    d2=D2()
    edsr=Edsr()
    d1.apply(weight_init)
    g1.apply(weight_init)
    g2.apply(weight_init)
    d2.apply(weight_init)
    g3.apply(weight_init)
    edsr.apply(weight_init)
    if torch.cuda.is_available():
        g1.cuda()
        g2.cuda()
        d1.cuda()
        d2.cuda()
        g3.cuda()
        edsr.cuda()
    tv=tvloss(batch_size=16)
    for i in range(40):#400000个迭代
        for step,x2 in enumerate(zip(x,x1,z)):  #这里可以通过x[0],x[1]赖访问两部分的图像
            x_train=x2[0].to(device)
            y_train=x2[1].to(device)
            z_train=x2[2].to(device)
            rate=(10**(-4))/(2**(int(i/40000)))
            #是否需要训练G1和G2
            #是否需要reso过程和clean过程是否分别需要两个optim
            optimg=optim.Adam(list(g1.parameters())+list(g2.parameters()),lr=rate,betas=(0.5,0.999))
            optimd = optim.Adam(list(d1.parameters()) , lr=rate, betas=(0.5, 0.999))
            optimd.zero_grad()
            optimg.zero_grad()

            gan_lr=torch.Tensor([0.0]).float().cuda()
            gan_lr1 = torch.Tensor([0.0]).float().cuda()
            cyc_lr=torch.Tensor([0.0]).float().cuda()
            idt_lr=torch.Tensor([0.0]).float().cuda()
            tv_lr=torch.Tensor([0.0]).float().cuda()
            y_fake=torch.Tensor([0.0]).float().cuda()
            y_fake1=torch.Tensor([0.0]).float().cuda()
            y_return = torch.Tensor([0.0]).float().cuda()
            y_dit = torch.Tensor([0.0]).float().cuda()
            y_tv = torch.Tensor([0.0]).float().cuda()

            x_train = torch.where(torch.isnan(x_train), torch.full_like(x_train, 0), x_train)
            x_train = torch.where(torch.isinf(x_train), torch.full_like(x_train, 1), x_train)

            y_train = torch.where(torch.isnan(y_train), torch.full_like(y_train, 0), y_train)
            y_train = torch.where(torch.isinf(y_train), torch.full_like(y_train, 1), y_train)

            y_fake = d1(g1(x_train))
            y_ones=torch.ones_like(y_fake,dtype=torch.float32)
            y_ones.to(device)
            gan_lr=criteration_GAN_LR(y_fake,y_ones)
            gan_lr.backward()
            optimd.step()
            optimd.zero_grad()
            optimg.zero_grad()

            y_fake1=d1(g1(x_train))
            y_ones1 = torch.ones_like(y_fake1, dtype=torch.float32)
            gan_lr1 = criteration_GAN_LR(y_fake1, y_ones1)
            gan_lr1.backward()


            y_return=g2(g1(x_train))
            cyc_lr=10*criteration_cyc_LR(y_return,x_train)
            cyc_lr.backward()

            y_idt=g2(g1(y_train))
            idt_lr=5*criteration_idt_LR(y_idt,y_train)
            idt_lr.backward()

            y_tv=g1(x_train)
            tv_lr=tv(y_tv)
            tv_lr.backward()

            total_loss=gan_lr+10*cyc_lr+5*idt_lr+tv_lr
            if step%5==0:
                print("we are in iter: ",i," we are in batch: ",step," Now,our loss is: ",total_loss)
            optimg.step()
            if i%2==0:
                torch.save(g1.state_dict(),"g1.pth")
                torch.save(g2.state_dict(),"g2.pth")
                torch.save(d1.state_dict(),"d1.pth")
            #初始化模型参数
            #定义优化器
            #计算loss
            #每100个batch，打印loss
            #优化





# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))

#
# for step, b_x in enumerate(x1):
#     if step < 5:
#         imgs = torchvision.utils.make_grid(b_x)
#         print(imgs.shape)
#         imgs = np.transpose(imgs,(1,2,0))
#         print(imgs.shape)
#         plt.imshow(imgs)
#         plt.show()