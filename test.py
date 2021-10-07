import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2
import numpy as np
from torch.utils.data import DataLoader, dataloader

import config
import dataloaders.dataloaders as dataloaders
import models.models as models
import utils.utils as utils
from dataloaders.MPVDataset import MPVDataset
from dataloaders.VitonDataset import VitonDataset
from utils.plotter import evaluate, plot_simple_reconstructions

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader to populate opt ---#
opt.phase = "test"
dataloaders.get_dataloaders(opt)

assert opt.phase in {"val", "test"}

if opt.dataset == "mpv":
    dataset_cl = MPVDataset
elif opt.dataset == "viton":
    dataset_cl = VitonDataset
else:
    raise NotImplementedError

if (opt.phase == "val" or opt.phase == "test"):
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(opt, model)
    model.eval()
    
    image_indices = [2, 7, 8, 18, 35, 36, 38, 45, 47, 52, 56, 57, 58, 60, 63, 64, 66, 72, 74, 80]

    dataset = dataset_cl(opt, phase=opt.phase)
    evaluate(model, dataset, opt)

if opt.phase == "test":
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(opt, model)
    model.eval()

    dataset = dataset_cl(opt, phase=opt.phase)
    
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    
    os.makedirs(os.path.join("results", opt.name, opt.phase + "_images"), exist_ok=True)
    
    for i, data_i in enumerate(test_dataloader):
        print(i, "/", len(test_dataloader), end="\r")
        image, label = models.preprocess_input(opt, data_i)
        # label["cloth_seg"] = model.module.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])
        agnostic = data_i["agnostic"] if opt.bpgm_id.find("old") >= 0 else None
        
        if opt.no_seg:
            image["I_m"] = image["I"]
        
        pred = model(image, label, "generate", None, agnostic=agnostic).detach().cpu().squeeze().permute(1, 2, 0).numpy()
        pred = (pred + 1) / 2
        
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        # pred = cv2.resize(pred, (data_i['original_size'][1], data_i['original_size'][0]), interpolation=cv2.INTER_LINEAR)
        pred = cv2.resize(pred, opt.img_size[::-1], interpolation=cv2.INTER_AREA)
        
        if opt.dataset == "mpv":
            filename = data_i['name'][0].split("/")[-1].replace(".jpg", ".png")
        elif opt.dataset == "viton":
            filename = data_i['name'][0].split("/")[-1]
        cv2.imwrite(os.path.join("results", opt.name, opt.phase + "_images", filename), pred)
    
    print()


