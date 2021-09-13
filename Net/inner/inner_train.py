import torch
from torch.utils.data import Dataset,DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gener1 import G1
from gener2 import G2
from discri1 import D1
import torch.optim as optim
from weight_init import weight_init
from Tvloss import tvloss

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

if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sets_x = Load_LR_difficult('D:/super-reso/DIV2K_train', transform=transforms.ToTensor())
    sets_x1 = Load_LR('D:/super-reso/DIV2K_train', transform=transforms.ToTensor())  # 预先载入：从哪来，进行的转换
    x = DataLoader(dataset=sets_x, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)
    x1 = DataLoader(dataset=sets_x1, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)  # batch，是否打乱
    d1=D1()
    g1=G1()
    g2=G2()
    tv=tvloss(batch_size=16)
    for i in range(400000):#60000个迭代
        for step,x2 in enumerate(zip(x,x1)):  #这里可以通过x[0],x[1]赖访问两部分的图像
            x_train=x2[0].to(device)
            y_train=x2[1].to(device)
            rate=(2*10**(-4))/(2**(int(i/40000)))
            optim=optim.Adam(lr=rate,betas=(0.5,0.999))
            optim.zero_grad()
            d1.apply(weight_init)
            g1.apply(weight_init)
            g2.apply(weight_init)
            gan_lr=criteration_GAN_LR(d1(g1(x_train)),torch.ones_like(x_train))
            cyc_lr=criteration_cyc_LR(g2(g1(x_train))-x_train)
            idt_lr=criteration_idt_LR(g2(g1(y_train))-y_train)
            tv_lr=tv(g1(x_train))
            total_loss=gan_lr+10*cyc_lr+5*idt_lr+tv_lr
            if step%100==0:
                print("we are in iter: ",i," we are in batch: ",step," Now,our loss is: ",total_loss)
            total_loss.backward()
            optim.step()
            if i%4==0:
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