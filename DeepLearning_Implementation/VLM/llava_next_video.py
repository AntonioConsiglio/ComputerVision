from typing import List,Union
import torch

class LLaVaNeXTProcessor:

    IMAGE_TOKEN = '<image>'

    def __init__(self,tokenizer,num_image_tokes:int, image_size:int, video_):
        self.image_net_mean = [] 
        self.image_net_std = []

    def __call__(self,
                 text: List[str],
                 images:Union[str,torch.Tensor]):
        pass

    def process_image(self,images,size,resample):
        if isinstance(images,list):
