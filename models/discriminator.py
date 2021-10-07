import numpy as np

import torch
import torch.nn as nn
import models.norms as norms

from torch.nn import functional as F


class OASIS_Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        semantic_nc = np.sum([nc for mode, nc in zip(["body", "cloth", "densepose"], opt.semantic_nc) if mode in opt.segmentation])
        output_channel = semantic_nc
        if opt.img_size[0] == 64:
            self.channels = [3, 128, 128, 256, 512]
        elif opt.img_size[0] == 256:     
            self.channels = [3, 128, 128, 256, 256, 512, 512]
        elif self.opt.img_size[0] == 512:
            self.channels = [3, 128, 128, 256, 256, 512, 512, 512]
        elif opt.img_size[0] == 1024:
            self.channels = [3, 128, 128, 256, 256, 256, 512, 512, 512]
        else:
            raise NotImplementedError
        
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
            
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        for i in range(1, opt.num_res_blocks-1):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
            
        self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, x):
        # encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
            
        # decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            if x.shape[-2:] != encoder_res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=encoder_res[-i-1].shape[-2:], mode="bilinear", align_corners=False)
            
            # print(x.shape, encoder_res[-i-1].shape)
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))
        ans = self.layer_up_last(x)
        
        return ans


class CDiscriminator(nn.Module):
    
    def __init__(self, opt):
        super(CDiscriminator, self).__init__()
        self.opt = opt
        
        if opt.img_size[0] == 64:
            self.channels = [3, 128, 128, 256, 512]
        elif opt.img_size[0] == 256:     
            self.channels = [3, 128, 128, 256, 256, 512, 512]
        elif self.opt.img_size[0] == 512:
            self.channels = [3, 128, 128, 256, 256, 512, 512, 512]
        elif self.opt.img_size[0] == 1024:
            self.channels = [3, 128, 128, 256, 256, 256, 512, 512, 512]
        else:
            raise NotImplementedError
        
        I_body_down = []
        C_t_body_down = []
        
        # encoder part
        for i in range(opt.num_res_blocks):
            I_body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
            C_t_body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
            
        self.I_body_down = nn.Sequential(*I_body_down)
        self.C_t_body_down = nn.Sequential(*C_t_body_down)
        
        norm_layer = norms.get_spectral_norm(opt)
        self.I_end = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            norm_layer(nn.Conv2d(512, 64, kernel_size=1))
        )
        
        self.C_t_end = nn.Sequential(
            nn.LeakyReLU(0.2, False), 
            norm_layer(nn.Conv2d(512, 64, kernel_size=1))
        )
        
        self.linear = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Flatten(),
            # norm_layer(nn.Linear(2 * 64 * 4 * (2 + (opt.dataset == "viton")), out_features=1))
            norm_layer(nn.Linear(2 * 64 * 4 * 3, out_features=1))
        )
        
        
    def forward(self, I, C_t):
        I_enc = self.I_end(self.I_body_down(I))
        C_t_enc = self.C_t_end(self.C_t_body_down(C_t))

        x = self.linear(torch.cat((I_enc, C_t_enc), dim=1))
        return x
    

class PDiscriminator(nn.Module):
    
    def __init__(self, opt):
        super(PDiscriminator, self).__init__()
        
        self.opt = opt
        if opt.patch_size == 16:
            self.channels = [3, 256, 512]
        elif opt.patch_size == 64:
            self.channels = [3, 128, 128, 256, 512]
        elif opt.patch_size == 128:
            self.channels = [3, 129, 129, 256, 256, 512]
        elif opt.patch_size == 256:     
            self.channels = [3, 128, 128, 256, 256, 512, 512]
        else:
            raise NotImplementedError
        
        body_down = []
        for i in range(len(self.channels) - 1):
            body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
            
        self.body_down = nn.Sequential(*body_down)
        
        norm_layer = norms.get_spectral_norm(opt)
        self.end = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            norm_layer(nn.Conv2d(512, 64, kernel_size=1)),
            nn.LeakyReLU(0.2, False),
            nn.Flatten(),
            norm_layer(nn.Linear(64 * 4 * 4, 128)),
            nn.LeakyReLU(0.2, False),
            norm_layer(nn.Linear(128, 1))
        )
        
    def forward(self, x):
        x = self.body_down(x)
        x = self.end(x)
        return x


class residual_block_D(nn.Module):
    def __init__(self, fin, fout, opt, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = norms.get_spectral_norm(opt)
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), 
                                           nn.Upsample(scale_factor=2), 
                                           norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), 
                                           norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
                
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), 
                                   norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
