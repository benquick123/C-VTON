import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

sys.path.append('../')

import os

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from bpgm.model.models import BPGM
from bpgm.model.utils import load_checkpoint, save_checkpoint
from bpgm.utils.args import get_opt
from bpgm.utils.dataset import DataLoader, MPVDataset, VitonDataset

if __name__ == "__main__":
    
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    if opt.dataset == "mpv":
        dataset = MPVDataset(opt)
    elif opt.dataset == "viton":
        dataset = VitonDataset(opt)
    else:
        raise NotImplementedError
    
    model = BPGM(opt)
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    else:
        raise NotImplementedError
    
    model.cuda()
    model.eval()

    for i in range(len(dataset.filepath_df)):
        images = dataset[i]
        images_swap = dataset[i]
        
        if images['im_name'] != "013418_0.jpg":
            continue
        
        # images = dataset[309]
        # images_swap = dataset[134]
        
        for key, im in images.items():
            if isinstance(im, torch.Tensor) and im.shape[0] in {1, 3}:
                im = im / 2 + 0.5
                im = im.permute(1, 2, 0).numpy()
                im = (im * 255).astype(np.uint8)
                
                if im.shape[-1] == 1:
                    im = np.repeat(im, 3, axis=-1)
                
                im = Image.fromarray(im)
                # im.save(os.path.join("sample", "bpgm_warp", key + ".png"))

        # DEAL WITH ORIGINAL
        tc = images['target_cloth'].unsqueeze(0).cuda()
        tcm = images['target_cloth_mask'].unsqueeze(0).cuda()
        im_bm = images['body_mask'].unsqueeze(0).cuda()
        im_label = images['body_label'].unsqueeze(0).cuda()
        # agnostic = images['agnostic'].unsqueeze(0).cuda()
            
        grid = model(im_label, tc)
        # grid = model(agnostic, tc)
        
        warped_cloth = F.grid_sample(tc, grid, padding_mode='border', align_corners=True)
        
        warped_cloth_masked = warped_cloth * im_bm
        warped_mask = F.grid_sample(tcm, grid, padding_mode='border', align_corners=True)
        
        warped_cloth = warped_cloth.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth = (warped_cloth * 255).astype(np.uint8)
        
        warped_cloth_masked = warped_cloth_masked.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth_masked = (warped_cloth_masked * 255).astype(np.uint8)
        
        warped_mask = warped_mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_mask = np.repeat(warped_mask, 3, axis=-1)
        warped_mask = (warped_mask * 255).astype(np.uint8)
        
        # im = Image.fromarray(warped_cloth).save(os.path.join("sample", "bpgm_warp", "warped_cloth.png"))
        # im = Image.fromarray(warped_cloth_masked).save(os.path.join("sample", "bpgm_warp", "warped_cloth_masked.png"))
        # im = Image.fromarray(warped_mask).save(os.path.join("sample", "bpgm_warp", "warped_mask.png"))
        
        # DEAL WITH SWAP
        tc = images_swap['target_cloth'].unsqueeze(0).cuda()
        tcm = images_swap['target_cloth_mask'].unsqueeze(0).cuda()
        im_bm = images['body_mask'].unsqueeze(0).cuda()
        im_label = images['body_label'].unsqueeze(0).cuda()
        # agnostic = images['agnostic'].unsqueeze(0).cuda()
        
        grid = model(im_label, tc)
        # grid = model(agnostic, tc)
        
        warped_cloth_swap = F.grid_sample(tc, grid, padding_mode='border', align_corners=True)
        
        warped_cloth_masked_swap = warped_cloth_swap * im_bm
        warped_mask_swap = F.grid_sample(tcm, grid, padding_mode='border', align_corners=True)
        
        warped_cloth_swap = warped_cloth_swap.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth_swap = (warped_cloth_swap * 255).astype(np.uint8)
        
        warped_cloth_masked_swap = warped_cloth_masked_swap.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth_masked_swap = (warped_cloth_masked_swap * 255).astype(np.uint8)
        
        warped_mask_swap = warped_mask_swap.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_mask_swap = np.repeat(warped_mask_swap, 3, axis=-1)
        warped_mask_swap = (warped_mask_swap * 255).astype(np.uint8)
        
        # im = Image.fromarray(warped_cloth_swap).save(os.path.join("sample", "bpgm_warp", "warped_cloth_swap.png"))
        # im = Image.fromarray(warped_cloth_swap).save(os.path.join("sample", "viton_bpgm_warp", images["im_name"]))
        im = Image.fromarray(warped_cloth_swap).save(os.path.join("tmp.jpg"))
        break
        
        # im = Image.fromarray(warped_cloth_masked_swap).save(os.path.join("sample", "bpgm_warp", "warped_cloth_masked_swap.png"))
        # im = Image.fromarray(warped_mask_swap).save(os.path.join("sample", "bpgm_warp", "warped_mask_swap.png"))
