#coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.append('../')

import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from bpgm.model.models import BPGM
from bpgm.model.utils import load_checkpoint, save_checkpoint
from bpgm.utils.args import get_opt
from bpgm.utils.dataset import DataLoader, MPVDataset, VitonDataset
from bpgm.utils.losses import VGGLoss
from bpgm.utils.visualization import board_add_images


def train_bpgm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    for step in range(opt.keep_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        tc = inputs['target_cloth'].cuda()
        tcm = inputs['target_cloth_mask'].cuda()
        
        im_c =  inputs['cloth'].cuda()
        im_bm = inputs['body_mask'].cuda()
        im_cm = inputs['cloth_mask'].cuda()
        
        im_label = inputs['body_label'].cuda()
        
        grid = model(im_label, tc)

        warped_cloth = F.grid_sample(tc, grid, padding_mode='border', align_corners=True)
        warped_cloth = warped_cloth * im_bm
        warped_mask = F.grid_sample(tcm, grid, padding_mode='border', align_corners=True)
        
        loss_cloth = criterionL1(warped_cloth, im_c) + 0.1 * criterionVGG(warped_cloth, im_c)
        loss_mask = criterionL1(warped_mask, im_cm) * 0.1
        loss = loss_cloth + loss_mask
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            label = inputs['label'].cuda()
            im_g = inputs['grid_image'].cuda()
            with torch.no_grad():
                warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)
            
            visuals = [[label, warped_grid, -torch.ones_like(label)], 
                    [tc, warped_cloth, im_c],
                    [tcm, warped_mask, im_cm]]
            
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_old_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['target_cloth'].cuda()
        im_c =  inputs['cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        im_pose = inputs['im_pose']
            
        grid = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)

        visuals = [[torch.zeros_like(im), torch.zeros_like(im), im_pose],
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        loss = criterionL1(warped_cloth, im_c)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    if opt.dataset == "mpv":
        train_dataset = MPVDataset(opt)
    elif opt.dataset == "viton":
        train_dataset = VitonDataset(opt)
    else:
        raise NotImplementedError
    
    # create dataloader
    train_loader = DataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    model = BPGM(opt)
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
        
    if opt.old:
        train_old_gmm(opt, train_loader, model, board)
    else:
        train_bpgm(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'bpgm_final.pth'))
  
    print('Finished training %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
