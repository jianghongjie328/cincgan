import cv2
import os
import random


class Data_deal:
    # 对LR,LR_difficult，HR的数据分到新的文件夹
    # 对LR,LR_difficult,HR进行90°的旋转和翻转
    # 进行随机裁剪并且到新的文件夹
    def __init__(self,rawfolder,newfolder,newfolder_enforce,newfolder_crop,scale):
        self.rawfolder=rawfolder
        self.newfolder=newfolder
        self.newfolder_crop=newfolder_crop
        self.newfolder_enforce = newfolder_enforce
        self.scale=scale
    def return_file(self,path):
        for root,folder,files in os.walk(path):
            return files
    def to_new_folder(self,ini_range,end_range):
        files=self.return_file(self.rawfolder)
        files=files[ini_range-1:end_range]
        name=1
        for file in files:
            a=cv2.imread(self.rawfolder+'\\'+file)
            cv2.imwrite(self.newfolder+"\\"+str(name)+".png",a)
            name=name+1
    def enforcment(self):
        files=self.return_file(self.newfolder)
        name=1
        for file in files:
            a=cv2.imread(self.newfolder+'\\'+file)
            cv2.imwrite(self.newfolder_enforce+'\\'+str(name)+'.png',a)
            name=name+1
            a1=cv2.rotate(a, rotateCode =cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(self.newfolder_enforce + '\\' + str(name) + '.png', a1)
            name = name + 1
            a2=cv2.flip(a,flipCode=1)  #水平翻转
            cv2.imwrite(self.newfolder_enforce + '\\' + str(name) + '.png', a2)
            name = name + 1
        #对newfolder做处理
    def randomly_crop(self):
        #对原本的newfoler_enforce做处理，并且存到newfoldercrop中
        #os.walk，对每一张图获取shape，随机裁剪shape[0]/32*shape[1]/32张,生成的随机裁剪点要在0---shape[0]-32,0---shape[0]-32
        files=self.return_file(self.newfolder_enforce)
        name=1
        for file in files:
            a=cv2.imread(self.newfolder_enforce+'\\'+file)
            x=a.shape[0]
            y=a.shape[1]
            num=int(x/self.scale)*int(y/self.scale)
            random_end_x=x-self.scale
            random_end_y=y-self.scale
            for i in range(num):
                pointx = random.randint(0, random_end_x)
                pointy = random.randint(0, random_end_y)
                a1=a[pointx:pointx+self.scale,pointy:pointy+self.scale,:]
                cv2.imwrite(self.newfolder_crop+"\\"+str(name)+".png",a1)
                name=name+1

if __name__=="__main__":
    rawpath="D:\\super-reso\\DIV2K_train\\DIV2K_train_HR"
    newpath="D:\\super-reso\\DIV2K_train\\DIV2K_train_HR_new"
    newpath_enforce="D:\\super-reso\\DIV2K_train\\DIV2K_train_HR_enforce"
    newpath_crop = "D:\\super-reso\\DIV2K_train\\DIV2K_train_HR_crop"
    deal=Data_deal(rawpath,newpath,newpath_enforce,newpath_crop,128)
    deal.to_new_folder(1,400)
    deal.enforcment()
    deal.randomly_crop()
    # for i in range(100):
    #     x=random.randint(0,100)
    #     print(x)







