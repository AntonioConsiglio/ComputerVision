import torch
from torch import nn
from typing import Optional

class SiglipVisionConfig:
    def __init__(self,
                 hidden_size = 768,
                 intermediate_size = 3072,
                 num_hidden_layers = 12,
                 num_attention_heads=12,
                 num_channels=3,
                 image_size = 224,
                 patch_size = 16,
                 layer_norm_eps=1e-6,
                 attention_droput = 0.0,
                 num_image_tokens:Optional[int] = None,
                 **kwargs,
                 ):
        
        self.hidden_size = hidden_size # output dimension of embeddings
        self.intermediate_size = intermediate_size # output dimension of first MLP linear layer 
        self.num_hidden_layers = num_hidden_layers # Number of Transforerm blocks
        self.num_attention_heads = num_attention_heads # Number of attention heads of Transformer block
        self.num_channels = num_channels # input channels
        self.image_size = image_size # input image size
        self.patch_size = patch_size # Number of paches, this number set the sequence len or number of tokes x image ( img_size / numb_paches) = dimension of patch => dp*dp = n_of_tokens x image
        self.layer_norm_eps = layer_norm_eps # eps for normalizaion to avoid infinite
        self.attention_droput = attention_droput # dropout for attention mask, only during training
        self.num_image_tokens = num_image_tokens


class SigLipVisionEmbedding(nn.Module):
    
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.num_channels
        self.out_channels = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # positional embeddings
        self.num_patches = (config.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_patches,config.hidden_size)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1,-1)),
                             persistent=False)


    def forward(self,input_imgs:torch.Tensor) -> torch.Tensor:
        _, _, h, w = input_imgs.shape
        # [batch, channel, height, width] -> [batch, hidden_size, n_tokens( (img_size / pathc_size)^2 )]
        patch_embeddings = self.patch_embedding(input_imgs).flatten(2)
        # [batch, hidden_size,  n_tokens] -> [batch, n_tokens, hidden_size] 
        embeddings = patch_embeddings.transpose(1,2)
        # add position embeddings learned during training
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        return embeddings


class SigLipMLP(nn.Module):

    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size,True)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size,True)

    def forward(self,x):

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x


class SigLipAttention(nn.Module):

    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config  = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**0.5
        self.dropout = config.attention_droput

        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)

        self.out_prok = nn.Linear(self.embed_dim,self.embed_dim)

    def forward(self,x):
        B, C, _ = x.size()
        q = self.q_proj(x) 
        k = self.k_proj(x) 
        v = self.v_proj(x)

        q = q.view(B,C,self.num_heads,self.head_dim).transpose(1,2) 
        k = k.view(B,C,self.num_heads,self.head_dim).transpose(1,2) 
        v = v.view(B,C,self.num_heads,self.head_dim).transpose(1,2)

        attn_weights = torch.nn.functional.softmax((q @ k.transpose(2,3)) / self.scale, dim=-1)

        attn_weights = torch.nn.functional.dropout(attn_weights,self.config.attention_droput,training=self.training)

        attn = attn_weights @ v

        attn = attn.transpose(1,2).contiguous().reshape(B,C,self.embed_dim)

        return attn, attn_weights


class SigLipVisionTransformer(nn.Module):

    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # input layer norm
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        # Self attention
        self.self_attn = SigLipAttention(config)
        # Second layer norm
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        # Multi layer perceptron
        self.mlp = SigLipMLP(config=config)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.layer_norm1(x)
        x, _ = self.self_attn(x)
        x = x + residual

        residual = x

        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class SigLipEncoder(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        # Transformer blocks
        self.layers = nn.Sequential(*[SigLipVisionTransformer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for layer in self.layers:
        #     x = layer(x)
        return self.layers(x)


class SigLipVModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # transform image -> batch -> embeddings vectors
        self.embeddings = SigLipVisionEmbedding(config) 
        # encode embeddings vector -> transformer encoder
        self.encoder = SigLipEncoder(config) 
        # layer norm after encoding -> usually after this layer we have a classifier to classify the image for example
        self.post_layers_norm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        hidden_state = self.embeddings(x)
        out_hidden_state = self.encoder(hidden_state)
        out_hidden_state = self.post_layers_norm(out_hidden_state)

        return out_hidden_state
    


if __name__ == "__main__":

    config = SiglipVisionConfig()

    model = SigLipVModel(config)

    tot_parameters = 0
    # for k,v in model.state_dict().items():
    #     print(f"{k} : {v.size()}")
    #     tot_parameters += (v.flatten()).size()[0]

    # print(f"Total parameters: {tot_parameters/1e6:.3f} M")

    x = torch.randn((4,3,224,224))

    result = model(x)

    print(result.size())