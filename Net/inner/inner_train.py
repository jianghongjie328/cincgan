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
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sets_x = Load_LR_difficult('D:/project/super-reso/DIV2K_train', transform=transforms.ToTensor())
    sets_x1 = Load_LR('D:/project/super-reso/DIV2K_train', transform=transforms.ToTensor())  # 预先载入：从哪来，进行的转换
    x = DataLoader(dataset=sets_x, batch_size=64, shuffle=True,num_workers=0,pin_memory=True)
    x1 = DataLoader(dataset=sets_x1, batch_size=64, shuffle=True,num_workers=0,pin_memory=True)  # 这里的num_worker2需要调整，找出最合适的
    d1=D1()
    g1=G1()
    g2=G2()
    d1.apply(weight_init)
    g1.apply(weight_init)
    g2.apply(weight_init)
    if torch.cuda.is_available():
        g1.cuda()
        g2.cuda()
        d1.cuda()
    tv=tvloss(batch_size=64)
    rate=2*10**(-4)
    plt.figure("temp_result")
    for i in range(20000):#目前的问题是，迭代次数偏大，内存没有被充分应用，目前把数据量扩大300倍，使用1.37g，360000条数据，
        # 原先是一天800*2400。考虑到如果收敛的话，我们的将每20次迭代缩小lr
        #1.比较一天左右跑的数据量。目前参数为360000对数据1.37g*2，大概能跑2534400，效率上升大概百分之30左右
        #2.gpu-copy 20%，其他都是0？
        #3.变量申明部分的代码，参考一下源代码。
        rate = rate / (2 ** (int(i))) #根据明早11点跑的结果调整loss多久收敛一次
        for step,x2 in enumerate(zip(x,x1)):  #这里可以通过x[0],x[1]赖访问两部分的图像
            x_train=x2[0].to(device)
            y_train=x2[1].to(device)
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

            if step % 2 == 0:   #将generator的训练降低到判断器的1/2
                y_fake1=d1(g1(x_train))
                y_ones1 = torch.ones_like(y_fake1, dtype=torch.float32)
                gan_lr1 = criteration_GAN_LR(y_fake1, y_ones1)
                gan_lr1.backward()


                y_return=g2(g1(x_train))
                cyc_lr=10*criteration_cyc_LR(y_return,x_train)
                cyc_lr.backward()

                y_idt=g1(y_train)
                idt_lr=5*criteration_idt_LR(y_idt,y_train)
                idt_lr.backward()

                y_tv=g1(x_train)
                tv_lr=0.5*tv(y_tv)
                tv_lr.backward()

                total_loss=gan_lr+10*cyc_lr+5*idt_lr+0.5*tv_lr #将进行颜色协调的部分从5调到了10
                if step%10==0:
                    print("we are in iter: ",i," we are in batch: ",step," Now,our loss is: ",total_loss)
                optimg.step()
                optimd.zero_grad()
                optimg.zero_grad()

            if step%10==0:
                torch.save(g1.state_dict(),"g1.pth")
                torch.save(g2.state_dict(),"g2.pth")
                torch.save(d1.state_dict(),"d1.pth")
                #进行效果测试
                perform_in=cv2.imread("D:\\project\\super-reso\\DIV2K_train\\DIV2K_train_LR_difficult\\0001x4d.png")
                perform_in_copy=perform_in
                standard=cv2.imread("D:\\project\\super-reso\\DIV2K_train\\DIV2K_train_LR\\1.png")
                #维度交换，维度扩张,归一化
                perform_in=perform_in/255.0
                perform_in=np.transpose(perform_in,(2,0,1))
                perform_in=np.expand_dims(perform_in,axis=0)
                #转化未tensor
                perform_in=torch.Tensor(perform_in)
                #加入cuda
                perform_in=perform_in.cuda()
                #放入网络进行处理
                perform_out=g1(perform_in)
                perform_return = g2(perform_out)
                #从cuda中取出
                perform_out=perform_out.cpu().detach().numpy()
                perform_return=perform_return.cpu().detach().numpy()
                #减少维度，维度交换，反归一化
                perform_out=perform_out[0]
                perform_return=perform_return[0]
                perform_out=np.transpose(perform_out,(1,2,0))
                perform_return=np.transpose(perform_return,(1,2,0))
                # perform_out=perform_out*255
                # perform_return=perform_return*255.0  #RGB图像在进行显示时需要归一化图像进行显示显示结果才正常
                #显示，原图，降噪图片，降噪后还原图，参考标准图
                plt.subplot(2,2,1)
                plt.title("raw noise")
                plt.imshow(perform_in_copy)
                plt.subplot(2,2,2)
                plt.title("denoise")
                plt.imshow(perform_out)
                plt.subplot(2,2,3)
                plt.title("return noise")
                plt.imshow(perform_return)
                plt.subplot(2,2,4)
                plt.title("standard")
                plt.imshow(standard)          #这里显示为蓝色很正常，因为默认为bgr图像
                plt.savefig("C:\\Users\\MI\\Desktop\\temp_result.png")
                cv2.imwrite("C:\\Users\\MI\\Desktop\\denoise.png",perform_out*255)
                # plt.clf()#cv2的读入是BGR和存储是RGB。而plt的存储不变该是BGR就是BGR。



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