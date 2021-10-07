import math
from models.sync_batchnorm.replicate import DataParallelWithCallback

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.norms as norms
from models.blocks import Encoder
from bpgm.model.models import BPGM


class Simple_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        ch = opt.channels_G
        if opt.img_size[0] == 64:
            self.channels = [16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif opt.img_size[0] == 256:
            self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif opt.img_size[0] == 512:
            self.channels = [16*ch, 16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif opt.img_size[0] == 1024:
            self.channels = [16*ch, 16*ch, 16*ch, 16*ch, 8*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        else:
            raise NotImplementedError
        
        semantic_nc = np.sum([nc for mode, nc in zip(["body", "cloth", "densepose"], opt.semantic_nc) if mode in opt.segmentation])
        
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock(self.channels[i] + opt.z_dim + semantic_nc, self.channels[i+1], opt))
            
        self.fc = nn.Conv2d(semantic_nc + self.opt.z_dim, self.channels[0], 3, padding=1)
        self.conv_img = nn.Conv2d(self.channels[-1] + opt.z_dim + semantic_nc, 3, 3, padding=1)
        
    def forward(self, seg, z=None):
        scale = 1 / math.pow(2, self.opt.num_res_blocks-1)
        _z = F.interpolate(z, scale_factor=scale, recompute_scale_factor=False)
        _seg = F.interpolate(seg, scale_factor=scale, recompute_scale_factor=False, mode="nearest")
        
        x = torch.cat((_z, _seg), dim=1)
        x = self.fc(x)
        
        for i in range(self.opt.num_res_blocks-1, -1, -1):
            # remember, we go i = n -> 0
            scale = 1 / math.pow(2, i)
            _seg = F.interpolate(seg, scale_factor=scale, mode="nearest", recompute_scale_factor=False)
            _z = F.interpolate(z, scale_factor=scale, recompute_scale_factor=False)

            x = torch.cat((x, _z, _seg), dim=1)
            
            x = self.body[self.opt.num_res_blocks - 1 - i](x)
            if i > 0:
                x = self.up(x)
        
        x = torch.cat((x, z, seg), dim=1)        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
            
        return x


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        ch = opt.channels_G
        if opt.img_size[0] == 64:
            self.channels = [16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif opt.img_size[0] == 256:
            self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif opt.img_size[0] == 512:
            self.channels = [16*ch, 16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif opt.img_size[0] == 1024:
            self.channels = [16*ch, 16*ch, 16*ch, 16*ch, 8*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        else:
            raise NotImplementedError
        
        semantic_nc = np.sum([nc for mode, nc in zip(["body", "cloth", "densepose"], opt.semantic_nc) if mode in opt.segmentation])
        
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            # self.body.append(ResnetBlock_with_SPADE(self.channels[i] + opt.z_dim, self.channels[i+1], opt))
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            
        self.fc = nn.Conv2d(semantic_nc + self.opt.z_dim, self.channels[0], 3, padding=1)
        self.conv_img = nn.Conv2d(self.channels[-1] + opt.z_dim, 3, 3, padding=1)

    def forward(self, seg, z=None):
        scale = 1 / math.pow(2, self.opt.num_res_blocks-1)
        _z = F.interpolate(z, scale_factor=scale, recompute_scale_factor=False)
        _seg = F.interpolate(seg, scale_factor=scale, recompute_scale_factor=False, mode="nearest")
        
        x = torch.cat((_z, _seg), dim=1)
        x = self.fc(x)
        
        for i in range(self.opt.num_res_blocks-1, -1, -1):
            # remember, we go i = n -> 0
            scale = 1 / math.pow(2, i)
            _seg = F.interpolate(seg, scale_factor=scale, mode="nearest", recompute_scale_factor=False)
            _z = F.interpolate(z, scale_factor=scale, recompute_scale_factor=False)
            
            _cat = torch.cat((_seg, _z), dim=1)
            # x = torch.cat((x, _z), dim=1)
            
            x = self.body[self.opt.num_res_blocks - 1 - i](x, _cat)
            if i > 0:
                x = self.up(x)
        
        x = torch.cat((x, z), dim=1)        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
            
        return x


class OASIS_AE(nn.Module):
    
    def __init__(self, opt):    
        super(OASIS_AE, self).__init__()

        self.step = int(math.log(opt.img_size, 2)) - 4
        
        self.I_encoder = Encoder(3, return_activations=True)
        self.C_t_encoder = Encoder(3, return_activations=True)
        
        self.oasis = OASIS_Generator(opt)
    
    def forward(self, I_m, C_t, seg):
        I_feat, I_act = self.I_encoder(I_m, step=self.step)
        C_t_feat, C_t_act = self.C_t_encoder(C_t, step=self.step)
        
        act = []
        for _I_act, _C_t_act in zip(I_act, C_t_act):
            act.append(torch.cat((_I_act, _C_t_act), dim=1))
        
        x = torch.cat((I_feat, C_t_feat), dim=1)
        x = self.oasis(seg, x, act, mode="other")
        
        return x
    

class OASIS_Simple(nn.Module):
    
    def __init__(self, opt):
        super(OASIS_Simple, self).__init__()
        
        self.opt = opt
        self.oasis = OASIS_Generator(opt)
        # self.oasis = Simple_Generator(opt)
        
        if self.opt.transform_cloth:
            self.bpgm = BPGM(opt)
            self.bpgm.eval()
        else:
            self.bpgm = None
        
    def forward(self, I_m, C_t, body_seg, cloth_seg, densepose_seg, agnostic=None):
        if agnostic is not None:
            C_transformed = self.transform_cloth_old(agnostic, C_t)
        else:
            C_transformed = self.transform_cloth(densepose_seg, C_t)
        
        z = torch.cat((I_m, C_t, C_transformed), dim=1)
        
        seg_dict = {
            "body": body_seg,
            "cloth": cloth_seg,
            "densepose": densepose_seg
        }
        
        if len(self.opt.segmentation) == 1:
            seg = seg_dict[self.opt.segmentation[0]]
        else:
            seg = torch.cat([seg_dict[mode] for mode in sorted(seg_dict.keys()) if mode in self.opt.segmentation], axis=1)
            
        x = self.oasis(seg, z)
        return x
    
    def transform_cloth(self, seg, C_t):
        if self.bpgm is not None:
            with torch.no_grad():
                # grid, _ = self.bpgm(torch.cat((I_m, seg), dim=1), C_t)
                if self.bpgm.resolution != self.opt.img_size:
                    _seg = F.interpolate(seg, size=self.bpgm.resolution, mode="nearest")
                    _C_t = F.interpolate(C_t, size=self.bpgm.resolution, mode="bilinear", align_corners=False)
                    grid = self.bpgm(_seg, _C_t).permute(0, 3, 1, 2)

                    grid = F.interpolate(grid, size=self.opt.img_size, mode="bilinear", align_corners=False)
                    grid = grid.permute(0, 2, 3, 1)
                else:
                    grid = self.bpgm(seg, C_t)
                                    
                C_t = F.grid_sample(C_t, grid, padding_mode='border', align_corners=True)

            return C_t                
        else:
            return C_t

    def transform_cloth_old(self, agnostic, C_t):
        if self.bpgm is not None:
            with torch.no_grad():
                # grid, _ = self.bpgm(torch.cat((I_m, seg), dim=1), C_t)
                if self.bpgm.resolution != self.opt.img_size:
                    agnostic = F.interpolate(agnostic, size=self.bpgm.resolution, mode="nearest")
                    _C_t = F.interpolate(C_t, size=self.bpgm.resolution, mode="bilinear", align_corners=False)
                    grid = self.bpgm(agnostic, _C_t).permute(0, 3, 1, 2)

                    grid = F.interpolate(grid, size=self.opt.img_size, mode="bilinear", align_corners=False)
                    grid = grid.permute(0, 2, 3, 1)
                else:
                    grid = self.bpgm(agnostic, C_t)
                                    
                C_t = F.grid_sample(C_t, grid, padding_mode='border', align_corners=True)

            return C_t                
        else:
            return C_t


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        semantic_nc = np.sum([nc for mode, nc in zip(["body", "cloth", "densepose"], opt.semantic_nc) if mode in opt.segmentation])
        spade_conditional_input_dims = semantic_nc + opt.z_dim
        
        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x

        dx = self.conv_0(self.activ(x))
        dx = self.conv_1(self.activ(dx))
        out = x_s + dx
        return out
