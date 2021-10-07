import multiprocessing as mp

import torch

from dataloaders.MPVDataset import MPVDataset
from dataloaders.VitonDataset import VitonDataset


def get_dataloaders(opt, same=False):
    if opt.dataset == "mpv":
        dataset_cl = MPVDataset
    elif opt.dataset == "viton":
        dataset_cl = VitonDataset
    else:
        raise NotImplementedError
    
    if opt.phase == "train":
        dataset_0 = dataset_cl(opt, phase="train")
        dataset_1 = dataset_cl(opt, phase="val")
        print("Created %s, size train: %d, size val: %d" % (dataset_0.name(), len(dataset_0), len(dataset_1)))
        dataloader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=mp.cpu_count() // 3)
        dataloader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=mp.cpu_count() // 3)
    elif opt.phase == "train_whole":
        dataset_0 = dataset_cl(opt, phase="train_whole")
        dataset_1 = dataset_cl(opt, phase="test" + ("_same" if same else ""))
        print("Created %s, size train: %d, size test: %d" % (dataset_0.name(), len(dataset_0), len(dataset_1)))
        dataloader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=mp.cpu_count() // 3)
        dataloader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=mp.cpu_count() // 3)
    elif opt.phase in {"test", "val"}:
        dataset_0 = dataset_cl(opt, phase="val")
        dataset_1 = dataset_cl(opt, phase="test" + ("_same" if same else ""))
        print("Created %s, size val: %d, size test: %d" % (dataset_0.name(), len(dataset_0), len(dataset_1)))
        dataloader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=opt.batch_size, shuffle=True, drop_last=False, num_workers=mp.cpu_count() // 3)
        dataloader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=mp.cpu_count() // 3)


    return dataloader_0, dataloader_1