import argparse
from datetime import datetime
import multiprocessing as mp


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_"))
    parser.add_argument('--workers', type=int, default=mp.cpu_count() // 2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--old', action="store_true", help="whether to use the legacy gmm training routine")
    
    parser.add_argument('--dataset', default="viton")
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=1000)
    parser.add_argument("--keep_step", type=int, default=30000)
    parser.add_argument("--alpha_rampup", type=int, default=30000)
    parser.add_argument("--decay_step", type=int, default=0)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--result_dir", type=str, default="result", help="folder for saving images during testing.")
    
    opt = parser.parse_args()
    opt.stage = "BPGM"
    
    if opt.dataroot == None:
        opt.dataroot = "../data/%s" % opt.dataset
    
    opt.name = opt.name + "_" + opt.stage
    return opt
