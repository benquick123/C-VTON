import torch
from torch import nn
from torch.nn import functional as F

    
class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
    
class Down(nn.Module):
    
    def __init__(self, *args, pooling=nn.MaxPool2d, **kwargs):
        super(Down, self).__init__()
        
        self.blocks = nn.Sequential(
            pooling(2),
            ConvBlock(*args, **kwargs)   
        )
        
    def forward(self, x):
        return self.blocks(x)
    
    
class ConvBlock(nn.Module):
    """
    https://github.com/rosinality/progressive-gan-pytorch/blob/master/model.py#L137
    """
    
    def __init__(self, in_channel, out_channel, kernel_size, padding, mid_channel=None, conv=nn.Conv2d, normalization=nn.BatchNorm2d, spectral_norm=False, relu_slope=0.01):
        super(ConvBlock, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        if mid_channel is None:
            mid_channel = out_channel
            
        if spectral_norm:
            spectral_norm = nn.utils.spectral_norm
        else:
            spectral_norm = lambda x, *args, **kwargs : x
        
        self.blocks = nn.Sequential(
            spectral_norm(conv(in_channel, mid_channel, kernel_size=kernel_size[0], padding=padding[0])),
            normalization(out_channel),
            nn.LeakyReLU(relu_slope),
            spectral_norm(conv(mid_channel, out_channel, kernel_size=kernel_size[1], padding=padding[1])),
            normalization(out_channel),
            nn.LeakyReLU(relu_slope)
        )
        
    def forward(self, x):
        return self.blocks(x)