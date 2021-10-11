# cincgan
##本人刚刚开始学习pytorch  
因为我发现github上的cincgan代码我基本上看不懂，所以只能对应着Junshk前辈的cincgan代码的一部分进行复现，  
接下来我会慢慢的对应着Junshk前辈的代码对我自己的代码进行优化。  
目前优化效果不是很好，降噪声的部分loss处在0.5左右，但视觉效果不好  
这个是降噪声的图  
![denoise](https://user-images.githubusercontent.com/55622672/136787276-a2882e77-f21f-44e1-ab44-3d34de06f9af.png)  
这个是标准图  
![1](https://user-images.githubusercontent.com/55622672/136790273-dea0192f-b910-40d4-b7c8-59f19933d059.png)  
##文件构如下：
cincgan  
&nbsp;Net  
&ensp;inner 去除噪声部分的代码  
&emsp;Block.py     
&emsp;gener1.py  
&emsp;gener2.py  
&emsp;denoise.py 使用网络降噪的代码
&emsp;discri1.py 以上都是网络结构  
&emsp;Tvloss.py 定义的Tvloss损失函数  
&emsp;test.py   
&emsp;weight_init.py 用于初始化权值  
&emsp;inner_train.py 包含了载入数据，进行内层循环的代码  
&ensp;outer  
&nbsp;tools  
&emsp;bicubic.py 用于对HR图像进行bicubic降采样。  
&emsp;data_enforcement.py 用于进行裁剪，翻转，对称等操作
    
  
