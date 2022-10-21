import torch
import numpy as np

class Reshape(torch.nn.Module):

    def __init__(self,size):
        super(Reshape,self).__init__()
        size.reverse()
        self.size = size
        print(self.size)

    def forward(self,x):
        b,_ = x.shape
        self.size.insert(0,b)
        x = torch.reshape(x,shape = self.size)
        return x