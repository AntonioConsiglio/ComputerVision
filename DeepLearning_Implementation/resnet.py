import torch
from torch import nn
from dataclasses import dataclass

import torch.nn.functional as F

@dataclass
class ConvRBConfig:
    in_channel:int
    out_channel:int
    kernel_size:int
    stride:int
    padding:int = 1
    bias:bool = False
    activation:str ="ReLU"
    n_conv:int=2

@dataclass
class ResNetConfig:
    in_channel:int = 3
    n_class:int = 1000
    architecture:int = 50
    activation:str = "ReLU"
    first_conv_out_channel:int = 64


class ResidualBlock(nn.Module):
    def __init__(self,config:ConvRBConfig,is_first=False):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()
        self.activation = getattr(nn,self.config.activation)(inplace=True)
       # First conv2d block
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.config.in_channel,
                          out_channels=self.config.out_channel,
                          kernel_size=self.config.kernel_size if self.config.n_conv == 2 else 1,
                          stride=self.config.stride,
                          padding=self.config.padding if self.config.n_conv == 2 else 0,
                          bias=self.config.bias),
                nn.BatchNorm2d(self.config.out_channel)
            )
        )

        # Second conv2d block

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.config.out_channel,
                          out_channels=self.config.out_channel,
                          kernel_size=self.config.kernel_size,
                          stride=1,
                          padding=self.config.padding,
                          bias=self.config.bias),
                nn.BatchNorm2d(self.config.out_channel)
            )
        )
        
         # Third conv2d block
        if self.config.n_conv == 3:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.config.out_channel,
                                out_channels=self.config.out_channel*4,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=self.config.bias),
                    nn.BatchNorm2d(self.config.out_channel*4)
                )
            )
         
            self.config.out_channel = self.config.out_channel*4
        
        self.projection_layer = nn.Sequential(
            nn.Conv2d(self.config.in_channel,self.config.out_channel,
                      kernel_size=1,stride=self.config.stride,bias=False),
            nn.BatchNorm2d(self.config.out_channel)) \
            if (self.config.stride == 2 or self.config.n_conv == 3 and is_first) else nn.Identity()
    
    def forward(self,x):
        
        residual = x
        for n, layer in enumerate(self.layers,start=1):
<<<<<<< HEAD
            x = layer(x)
=======
            print(f"Input_shape: {x.size()}")
            x = layer(x)
            print(f"Output_shape: {x.size()}")
>>>>>>> 3310702 (resnet implementation)
            if self.config.n_conv == n:
                residual = self.projection_layer(residual)
                x = x + residual
            x =  self.activation(x)  
        return x
    

class ResNet(nn.Module):
    def __init__(self,config:ResNetConfig):
        super().__init__()
        self.config = config

        self.conv1_block = nn.Sequential(
                            nn.Conv2d(self.config.in_channel,
                            self.config.first_conv_out_channel,
                            kernel_size=7,
                            padding=3,
                            stride=2,
                            bias=False),
                            nn.BatchNorm2d(self.config.first_conv_out_channel),
                            getattr(nn,self.config.activation)()
        )
       
        self.max_pool = nn.MaxPool2d(3,2,padding=1)
        self.small_net = True
        self.residual_block = []
        if self.config.architecture == 18:
            self.residual_block = [2,2,2,2]
        if self.config.architecture == 34:
            self.residual_block = [2,4,6,3]
        if self.config.architecture == 50:
            self.residual_block = [3,4,6,3]
            self.small_net = False
        if self.config.architecture == 101:
            self.residual_block = [3,4,23,3]
            self.small_net = False

        self.residual_layers = nn.ModuleList()

        in_channel = self.config.first_conv_out_channel
        out_channel = self.config.first_conv_out_channel
        for i,r_n in enumerate(self.residual_block):
            block = nn.ModuleList()

            for j in range(r_n):
                config = ConvRBConfig(in_channel,
                                      out_channel,
                                      kernel_size=3,
                                      stride = 1 if i == 0 or j > 0 else 2,
                                      n_conv=2 if self.small_net else 3)
                block.append(
                    ResidualBlock(config,j==0)
                )
                if not self.small_net:
                    in_channel = out_channel*4
                else:
                    in_channel = out_channel
            out_channel = out_channel*2

            self.residual_layers.append(nn.Sequential(*block))
        
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channel,self.config.n_class)
        
    def forward(self,x):
        x = self.conv1_block(x)
        x = self.max_pool(x)

        for i, res_layer in enumerate(self.residual_layers):
            x = res_layer(x)

        x = self.avg_pooling(x).flatten(1)
        out = self.classifier(x)

        return out


if __name__ == "__main__":
<<<<<<< HEAD
    config = ResNetConfig(architecture=18)
=======
    config = ResNetConfig()
>>>>>>> 3310702 (resnet implementation)
    model = ResNet(config)

    print(f"{model=}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n Total number of parameters for model: {total_params}')
    x = torch.randn((1,3,224,224))

    result = model(x)
    print(result.shape)