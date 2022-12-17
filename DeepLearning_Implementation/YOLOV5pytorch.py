from turtle import forward
import torch
from torch import nn 
import torch.functional as TF

class CSPDarknet53(nn.Module):
    def __init__(self,input,output):
        super(CSPDarknet53,self).__init__()
        #TODO implement the modulelist
    def forward(self,x):
        #TODO: implement the forward pass
        return x