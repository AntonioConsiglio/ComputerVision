from re import I
from VGG16pytorch import VGG
import torch
from torchinfo import summary

modello = VGG(input=3,output=2)
test = torch.rand((1,3,224,224))
x = modello(test)
summary(modello,input_size=[1,3,224,224])
