from torch import nn
from torch.nn import functional as F
import torch


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


class Down(nn.Module):
    
    def __init__(self, *args, pooling=nn.MaxPool2d, block=ConvBlock, **kwargs):
        super(Down, self).__init__()
        
        self.pool = pooling(2)
        self.block = block(*args, **kwargs)
        
    def forward(self, *args):
        xs = [self.pool(x) for x in args]
        return self.block(*xs)


class Encoder(nn.Module):
    
    def __init__(self, in_channels, return_activations, **kwargs):
        super(Encoder, self).__init__()
        self.return_activations = return_activations
        
        from_rgb = []
        from_rgb.append(nn.Conv2d(in_channels, 64, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 128, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        self.from_rgb = nn.ModuleList(from_rgb)
        
        modules_down = []
        modules_down.append(Down(64, 128, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(128, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        self.modules_down = nn.ModuleList(modules_down)
        
    def forward(self, input_batch, step=0, alpha=1.0):
        if self.return_activations:
            activations = []
        else:
            activations = None
        
        x = self.from_rgb[-(step+1)](input_batch)
        if self.return_activations:
            activations.append(x)
        
        if step > 0 and alpha < 1.0:
            residual_x = F.interpolate(input_batch, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            residual_x = self.from_rgb[-step](residual_x)
        else:
            residual_x = None
        
        for module_index in range(-(step+1), 0, 1):
            x = self.modules_down[module_index](x)
            
            if module_index == -(step+1) and residual_x is not None:
                x = (1 - alpha) * residual_x + alpha * x
                
            if self.return_activations:
                activations.append(x)
                
        if self.return_activations:
            return x, activations
        else:
            return x
        
