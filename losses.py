import torch.nn.functional as F
import torch.nn as nn

class my_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse=nn.MSELoss().requires_grad_(True);
        # TODO 一些需要的变量，如landmark索引等
        # 保存在self属性中

    
    def other_function(self,input):
        # 一些可能需要实现的函数
        # 如取出landmark、取出人脸front部分等
        # 注意：
        # 定义处有self
        # 使用处无self
        return 

    def forward(self,input,gt):

        
        self.other_function(input);
        result=self.mse(input,gt)
        return result;