from re import I
from pytorch_implementation import VGG
import torch
from torchinfo import summary

modello = VGG(input=3,output=2)

test = torch.rand((1,224,224,3))

x = modello(test)
#summary(modello,input_size=[1,224,224,3])
