import os

import cv2
import os
import  tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Data:
    def __init__(self,srcpath,dstpath,scale):
        self.srcpath=srcpath
        self.dstpath=dstpath
        self.scale=scale
        self.imgname=[]
    def load(self):
        for root,dirs,files in os.walk(self.srcpath):
            #print(files)
            for  file in files:
                self.imgname.append(self.srcpath+'\\'+file)
        #存进self.imgname
        #负责获取该文件夹下的文件名
    def save(self,img,name):
        cv2.imwrite(self.dstpath+'\\'+name+'.png',img)
        #放到目标文件夹下，单张图片
    def reshape(self):
        name=1
        for path in self.imgname:
            a=cv2.imread(path)
            x=int(a.shape[0]/float(self.scale))*self.scale
            y=int(a.shape[1]/float(self.scale))*self.scale
            a=a[:x:][:y:]
            a=tf.image.resize(a,[int(x/self.scale),int(y/self.scale)],method='bicubic')
            a=a.numpy()
            cv2.imshow("1",a)
            cv2.waitKey(0)
            self.save(a,str(name))
            name=name+1
        #从文件名中读取每一张图片并且reshape，调用save放进文件夹中
if __name__=="__main__":
    data=Data(r'D:\\super-reso\\DIV2K_train\\DIV2K_train_HR',r'D:\\super-reso\\DIV2K_train\\DIV2K_train_LR',4)
    data.load()
    data.reshape()
    #print(str(1))
    # a = cv2.imread(r'D:\\super-reso\\DIV2K_train\\DIV2K_train_HR\\0001.png')
    # scale=7
    # print(a.shape)
    # x = int(a.shape[0] / float(scale)) * scale
    # y = int(a.shape[1] / float(scale)) * scale
    # print(x," ",y)
    # a = a[:x:][:y:]
    # print(a.shape)
    # cv2.imshow("1",a)
    # cv2.waitKey(0)

