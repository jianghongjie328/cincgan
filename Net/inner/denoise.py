import numpy as np

from gener1 import G1
from gener2 import G2
from discri1 import D1
import torch
import cv2
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import math
import os

class Load_LR_test(Dataset):  # 继承Dataset类
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Load_LR_test, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            img_folder = root + '/DIV2K_train_LR_difficult/'
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

if __name__=="__main__":
    #载入数据
    sets_x = Load_LR_test('D:\\project\\super-reso\\DIV2K_train', transform=transforms.ToTensor())
    x = DataLoader(dataset=sets_x, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g1=G1()
    g2=G2()
    g1.load_state_dict(torch.load(r"D:\project\super-reso\cincgan\self-write\cincgan\Net\inner\g1.pth"))
    g2.load_state_dict(torch.load(r"D:\project\super-reso\cincgan\self-write\cincgan\Net\inner\g2.pth"))
    # if torch.cuda.is_available():
    #     g1.cuda()
    #     g2.cuda()
    name=0
    for step,x_test in enumerate(x):
        # x_test=x_test.to(device)
        picy=g1(x_test)
        picy_raw=g2(picy)
        for i in range(1):
            y_sort = picy
            y_sort1 = picy_raw
            y_sort = y_sort.detach().numpy()
            y_sort1 = y_sort1.detach().numpy()
            y_sort=y_sort[0]
            y_sort1=y_sort1[0]
            y_sort = np.transpose(y_sort, (1, 2, 0))
            y_sort1 = np.transpose(y_sort1, (1, 2, 0))
            y_sort = y_sort * 255
            y_sort1 = y_sort1 * 255

            cv2.imwrite("D:\\project\\super-reso\\DIV2K_train\\\DIV2K_train_LR_difficult_denoise\\" + str(name) + ".png",
                        y_sort)
            cv2.imwrite("D:\\project\\super-reso\\DIV2K_train\\\DIV2K_train_LR_difficult_noise\\" + str(name) + ".png",
                        y_sort1)
            name = name + 1
    #送进网络
    # name=1
    # for step,x_test in enumerate(x):
    #     #获取单张图片，转为numpy
    #     x_test=x_test.numpy()
    #     x_test=x_test[0]
    #     height=x_test.shape[1]
    #     width=x_test.shape[2]
    #     #获取x_test的高和宽，然后获取，并且除以32向上取整
    #     height_num=math.floor(height/32)
    #     width_num=math.floor(width/32)
    #     #根据这些数据将图片分割为32*32，最后剩下的就让它剩下,放进一个列表中。
    #     x_in=[]
    #     for i in range(height_num):
    #         for j in range(width_num):
    #             # if(i*32+31<=height-1):
    #             #     endx=i*32+32
    #             # else:
    #             #     endx=height
    #             # if(j*32+31<=width-1):
    #             #     endy=j*32+32
    #             # else:
    #             #     endy=width
    #             endx = i * 32 + 32
    #             endy = j * 32 + 32
    #             x_test1=x_test[:,i*32:endx,j*32:endy]
    #             x_in.append(x_test1)
    #     #图片升高一维，并且转为tensor
    #     # print()
    #     x_in=np.array(x_in)
    #     x_test=torch.from_numpy(x_in)
    #     x_test = x_test.to(device)
    #     y=g1(x_test)
    #     y_raw=g2(y)
    #     #将结果放进一张列表中，并且进行拼接
    # #存储数据
    #     y=y.cpu().detach().numpy()
    #     y_raw=y_raw.cpu().detach().numpy()
    #     picy=np.zeros((3,height,width))
    #     picy_raw=np.zeros((3,height,width))
    #     for i in range(height_num):
    #         for j in range(width_num):
    #             picy[:,i*32:i*32+32,j*32:j*32+32]=y[i*width_num+j]
    #             picy_raw[:, i * 32:i * 32 + 32, j * 32:j * 32 + 32] = y_raw[i * width_num + j]

