import torch.nn as nn
import torch
from Tvloss import tvloss
from gener1 import G1
from weight_init import weight_init

if __name__=="__main__":
    g1=G1()
    g1.apply(weight_init)


