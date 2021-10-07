from dataloaders.MPVDataset import MPVDataset
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

from datetime import datetime

import dataloaders.dataloaders as dataloaders
from utils.fid_scores import fid_pytorch
from utils.metrics import inception_score as is_fn
from utils.metrics import ssim as ssim_fn
from lpips import LPIPS

from models import models

fig, axes = None, None


def plot_simple_reconstructions(model, val_dataset, filename, opt, n_imgs=20, save=True, **kwargs):
    global fig, axes
    
    os.makedirs("/".join(filename.split("/")[:-1]), exist_ok=True)
    
    training = model.training
    if training:
        model.eval()
        
    with torch.no_grad():
        samples = None
        
        if isinstance(n_imgs, int):
            samples = list(range(n_imgs))
        elif isinstance(n_imgs, list):
            samples = list(n_imgs)
            n_imgs = len(samples)
        assert samples is not None, "Samples has to be int or list."
        
        input_im = []
        x0 = []
        x1 = []
        cloth_labels = []
        body_labels = []
        densepose_labels = []
        agnostic_im = []
        for i, s in enumerate(iter(val_dataset)):
            if i in samples:
                input_im.append(s["image"]["I"].unsqueeze(0))
                x0.append(s["image"]["I_m"].unsqueeze(0))
                x1.append(s["image"]["C_t"].unsqueeze(0))
                body_labels.append(s["body_label"].unsqueeze(0))
                cloth_labels.append(s["cloth_label"].unsqueeze(0))
                densepose_labels.append(s["densepose_label"].unsqueeze(0))
                if isinstance(s["agnostic"], str):
                    agnostic_im.append(torch.zeros((1, 1)))
                else:
                    agnostic_im.append(s["agnostic"].unsqueeze(0))
                
            if len(input_im) == n_imgs:
                break
                
        # dataset = TensorDataset(x0.cuda(), x1.cuda())
        input_im = torch.cat(input_im, dim=0)
        x0 = torch.cat(x0, dim=0)
        x1 = torch.cat(x1, dim=0)
        x1_flipped = x1.flip(0)
        body_labels = torch.cat(body_labels, dim=0)
        cloth_labels = torch.cat(cloth_labels, dim=0)
        densepose_labels = torch.cat(densepose_labels, dim=0)
        agnostic_im = torch.cat(agnostic_im, dim=0)
        
        # plt.imshow(torch.cat((x0[0], x1[0]), dim=1).permute(1, 2, 0).detach().cpu().numpy())
        # plt.savefig("tmp.png")
        # exit()
        
        dataset = TensorDataset(input_im, x0, x1, x1_flipped, body_labels, cloth_labels, densepose_labels, agnostic_im)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
        out = []
        swapped = []
        cloth_seg_swapped = []
        
        for I, I_m, C_t, C_t_flip, body_label, cloth_label, densepose_label, agnostic in dataloader:
            image, label = models.preprocess_input(opt, {"image": {"I_m": I if opt.no_seg else I_m, "C_t": C_t, "C_t_flip": C_t_flip}, 
                                                         "body_label": body_label, "cloth_label": cloth_label, "densepose_label": densepose_label})
            agnostic = agnostic if opt.bpgm_id.find("old") >= 0 else None
            
            out.append(model(image, label, "generate", None, agnostic=agnostic).detach().cpu())
            
            image["C_t"] = image["C_t_flip"]
            label["cloth_seg"] = model.module.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])
            cloth_seg_swapped.append(label["cloth_seg"])
            
            swapped.append(model(image, label, "generate", None, agnostic=agnostic).detach().cpu())
        
        input_im = (input_im + 1) / 2
        x0 = (x0 + 1) / 2
        x1 = (x1 + 1) / 2
        
        body_labels = body_labels * (255 // body_labels.shape[1])
        cloth_labels = cloth_labels * (255 // cloth_labels.shape[1])
        
        cloth_seg_swapped = torch.cat(cloth_seg_swapped, axis=0)
        cloth_label_swapped = torch.argmax(cloth_seg_swapped, axis=1, keepdim=True)
        cloth_label_swapped = cloth_label_swapped * (255 // cloth_label_swapped.shape[1])
        
        out = torch.cat(out, dim=0)
        out = (out + 1) / 2
        
        swapped = torch.cat(swapped, dim=0)
        swapped = (swapped + 1) / 2
        
        if fig is None or axes is None:
            fig, axes = plt.subplots(n_imgs, 8, figsize=(11, int(1.5 * n_imgs)), sharex=True, sharey=True)
            
        for i, _axes in enumerate(axes):
            if i == 0:
                _axes[0].set_title("Original")
                _axes[1].set_title("Original masked")
                _axes[2].set_title("Target (" + str(samples[i]) + ")")
                _axes[3].set_title("Body seg")
                _axes[4].set_title("Cloth seg")
                _axes[5].set_title("Output")
                _axes[6].set_title("Swapped cloth seg")
                _axes[7].set_title(str(samples[-1-i]))
            if i == 1:
                _axes[2].set_title(str(samples[i]))
                _axes[7].set_title("Swapped output (" + str(samples[-1-i]) + ")")
            else:
                _axes[2].set_title(str(samples[i]))
                _axes[7].set_title(str(samples[-1-i]))
            
            _axes[0].imshow(input_im[i].permute(1, 2, 0).numpy())
            _axes[1].imshow(x0[i].permute(1, 2, 0).numpy())
            _axes[2].imshow(x1[i].permute(1, 2, 0).numpy())
            _axes[3].imshow(body_labels[i].permute(1, 2, 0).numpy())
            _axes[4].imshow(cloth_labels[i].permute(1, 2, 0).numpy())
            _axes[5].imshow(out[i].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
            _axes[6].imshow(cloth_label_swapped[i].permute(1, 2, 0).cpu().numpy())
            _axes[7].imshow(swapped[i].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
            
    fig.tight_layout()
    if save:
        plt.savefig(filename)
    else:
        plt.show()

    if training:
        model.train()


print_fn = print
def evaluate(model, val_dataset, opt):
    
    os.makedirs(os.path.join("results", opt.name), exist_ok=True)
    metric_file = open(os.path.join("results", opt.name, "metrics.log"), "w")
    def print(*args, **kwargs):
        print_fn(*args, **kwargs)
        print_fn(*args, **kwargs, file=metric_file)
        
    dataloader_val, dataloader_test = dataloaders.get_dataloaders(opt, same=True)
    if opt.phase == "val":
        dataloader = dataloader_val
    elif opt.phase == "test":
        dataloader = dataloader_test
    else:
        assert False
    
    with torch.no_grad():
        lpips_fn = LPIPS(net='vgg', verbose=False).cuda()
        fid_computer = fid_pytorch(opt, dataloader)
        
        val_pred_y = []
        lpips = []
        for d in dataloader:
                image, label = models.preprocess_input(opt, d)
                agnostic = d["agnostic"] if opt.bpgm_id.find("old") >= 0 else None
                
                if opt.no_seg:
                    image["I_m"] = image["I"]
                
                pred_y = model(image, label, "generate", None, agnostic=agnostic).detach()
                
                val_pred_y.append(pred_y.cpu())
                lpips.append(lpips_fn(pred_y, image["I"]))
        
        val_pred_y = torch.cat(val_pred_y, dim=0)
        
        lpips = torch.cat(lpips, dim=0).cpu()
        print(datetime.now(), "- LPIPS: %.3f (+/- %.3f)" % (lpips.mean(), lpips.std()))
        
        # ssim = ssim_fn(val_pred_y, val_dataset)
        # print(datetime.now(), "- SSIM: %.3f" % ssim)
        
        fid = fid_computer.compute_fid_with_valid_path(model.module.netG, model.module.netEMA)
        print(datetime.now(), "- FID: %.3f" % fid)
    
    metric_file.close()
    
    # is_score = is_fn(TensorDataset(val_pred_y), batch_size=opt.batch_size, resize=True)
    # print(datetime.now(), "- IS: %.3f" % (is_score[0]))