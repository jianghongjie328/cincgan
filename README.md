# cincgan
本人刚刚开始学习pytorch
因为我发现github上的cincgan代码我基本上看不懂，所以只能对应着Junshk前辈的cincgan代码的一部分进行复现，
接下来我会慢慢的对应着Junshk前辈的代码对我自己的代码进行优化。
文件结构如下：
cincgan
  Net
    inner 去除噪声部分的代码
      Block.py 
      gener1.py
      gener2.py
      discri1.py 以上都是网络结构
      Tvloss.py 定义的Tvloss损失函数
      test.py 
      weight_init.py 用于初始化权值
      inner_train.py 包含了载入数据，进行内层循环的代码
    outer
  tools
    bicubic.py 用于对HR图像进行bicubic降采样。
    data_enforcement.py 用于进行裁剪，翻转，对称等操作
    
  
