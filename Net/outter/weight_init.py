import torch.nn as nn
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.dirac_(m.weight)