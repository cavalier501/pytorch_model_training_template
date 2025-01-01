import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.0001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.normal(m.weight, 1.0, 0.02)
            nn.init.constant(m.bias, 0)


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        pass
        
    def forward(self,x):
        pass
        return x