import torch
from torch import nn 
import torch.functional as TF


class VGG(nn.Module):
    
    def __init__(self,input,output,inputsize = None):
        super(VGG,self).__init__()
        if inputsize is None:
            inputsize = 224
        filtersize = 64
        self.convblock1 = MultipleConvBlock(input,filtersize,3,conv_number=2)
        self.max1 = nn.MaxPool2d(2,2)
        self.convblock2 = MultipleConvBlock(filtersize,filtersize*2,3,conv_number=2)
        self.max2 = nn.MaxPool2d(2,2)
        self.convblock3 = MultipleConvBlock(filtersize*2,filtersize*4,3,conv_number=3)
        self.max3 = nn.MaxPool2d(2,2)
        self.convblock4 = MultipleConvBlock(filtersize*4,filtersize*8,3,conv_number=3)
        self.max4 = nn.MaxPool2d(2,2)
        self.convblock5 = MultipleConvBlock(filtersize*8,filtersize*8,3,conv_number=3)
        self.endmax5 = nn.MaxPool2d(2,2)
        inputsizes = (inputsize//32)**2*filtersize*8
        self.linear1 = nn.Linear(inputsizes,4096)
        self.drop1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(4096,output)

    def forward(self,x):
        
        x = self.convblock1(x)
        x = self.max1(x)
        x = self.convblock2(x)
        x = self.max2(x)
        x = self.convblock3(x)
        x = self.max3(x)
        x = self.convblock4(x)
        x = self.max4(x)
        x = self.convblock5(x)
        x = self.endmax5(x)
        x = torch.flatten(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.output_layer(x)

        return nn.functional.softmax(x,dim=0)

class MultipleConvBlock(nn.Module):

    def __init__(self,input_size,output_size,kernel_size,conv_number):
        super(MultipleConvBlock,self).__init__()
        self.modulelist = nn.ModuleList()
        for _ in range(conv_number):
            self.modulelist.append(ConvBlock(input_size,output_size,kernel_size,padd=1))
            input_size = output_size

    def forward(self,x):
        for layer in self.modulelist:
            x = layer(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self,input_size,output_size,kernel_size,padd = 0,strd = 1):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(input_size,output_size,kernel_size,strd,padd)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

