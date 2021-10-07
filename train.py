import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
import config

from torch.cuda.amp import GradScaler

#--- read options ---#
opt = config.read_arguments(train=True)

#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)
fid_computer = fid_pytorch(opt, dataloader_val)

#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(opt, model)

if opt.no_seg:
    # modify arguments for loading
    opt.phase, opt._phase = "test", opt.phase
    opt.name, opt._name = "DP-VTON-VITON_v2", opt.name
    
    model_aux = models.OASIS_model(opt)
    model_aux = models.put_on_multi_gpus(opt, model_aux)
    model_aux.eval()
    
    # return back to specified arguments
    opt.phase = opt._phase
    opt.name = opt._name
else:
    model_aux = None

#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
if opt.add_d_loss:
    optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
if opt.add_cd_loss:
    optimizerCD = torch.optim.Adam(model.module.netCD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
if opt.add_pd_loss:
    optimizerPD = torch.optim.Adam(model.module.netPD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

scaler = GradScaler()

#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        if not already_started and i < start_iter:
            continue
        
        already_started = True
        cur_iter = epoch*len(dataloader) + i
        image, label = models.preprocess_input(opt, data_i)

        label_centroid = data_i["label_centroid"] if opt.add_pd_loss else None
        agnostic = data_i["agnostic"].cuda() if opt.bpgm_id.find("old") >= 0 else None
        
        if model_aux is not None:
            image = models.generate_swapped_batch(image)
            # from PIL import Image
            # import numpy as np
            
            # im = ((image["I"][0].permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
            # im = Image.fromarray(im)
            # im.save("orig.png")
            
            with torch.no_grad():
                image["I_m"] = model_aux(image, label, "generate", None, agnostic=agnostic)
                
                
                # im = ((I_m[0].permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
                # im = Image.fromarray(im)
                # im.save("swap.png")
                
            image = models.generate_swapped_batch(image)

        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer, label_centroids=label_centroid, agnostic=agnostic)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
            
        scaler.scale(loss_G).backward()
        scaler.step(optimizerG)

        if opt.add_d_loss:
            #--- discriminator update ---#
            model.module.netD.zero_grad()
            loss_D, losses_D_list = model(image, label, "losses_D", losses_computer, agnostic=agnostic)
            loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
            
            scaler.scale(loss_D).backward()
            scaler.step(optimizerD)
        else:
            losses_D_list = [None] * 7
        
        if opt.add_cd_loss:
            #--- conditional discriminator update ---#
            model.module.netCD.zero_grad()
            loss_CD, losses_CD_list = model(image, label, "losses_CD", losses_computer, agnostic=agnostic)
            loss_CD, losses_CD_list = loss_CD.mean(), [loss.mean() if loss is not None else None for loss in losses_CD_list]
            
            scaler.scale(loss_CD).backward()
            scaler.step(optimizerCD)
        else:
            losses_CD_list = [None, None]
        
        if opt.add_pd_loss:
            #--- patch discriminator update ---#
            model.module.netPD.zero_grad()
            loss_PD, losses_PD_list = model(image, label, "losses_PD", losses_computer, label_centroids=label_centroid, agnostic=agnostic)
            loss_PD, losses_PD_list = loss_PD.mean(), [loss.mean() if loss is not None else None for loss in losses_PD_list]
            
            scaler.scale(loss_PD).backward()
            scaler.step(optimizerPD)
        else:
            losses_PD_list = [None, None]
        
        scaler.update()
        
        #--- stats update ---#
        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt)
            
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter, agnostic=agnostic)
            timer(epoch, cur_iter)
            visualizer_losses(cur_iter, losses_G_list + losses_D_list + losses_CD_list + losses_PD_list)
            
        if cur_iter % opt.freq_save_ckpt == 0:
            utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)

#--- after training ---#
if not opt.no_EMA:
    utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
    
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")

