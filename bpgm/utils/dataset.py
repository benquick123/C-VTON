#coding=utf-8
import json
import os.path as osp

import cv2
import numpy as np
from numpy.lib.polynomial import polyfit
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

"""
semantic_labels = [
    [127, 127, 127],
    [0, 255, 255],
    [255, 255, 0],
    [127, 127, 0],
    [255, 127, 127],
    [0, 255, 0],
    
    [0, 0, 0],
    
    [255, 127, 0],
    [0, 0, 255],
    [127, 255, 127],
    [0, 127, 255],
    [127, 0, 255],
    [255, 255, 127],
    [255, 0, 0],
    [255, 0, 255],
    [-1, -1, -1]
]
"""

semantic_labels = [
    [0, 0, 0],
	[105, 105, 105],
	[85, 107, 47],
	[139, 69, 19],
	[72, 61, 139],
	[0, 128, 0],
	[154, 205, 50],
	[0, 0, 139],
	[255, 69, 0],
	[255, 165, 0],
	[255, 255, 0],
	[0, 255, 0],
	[186, 85, 211],
	[0, 255, 127],
	[220, 20, 60],
	[0, 191, 255],
	[0, 0, 255],
	[216, 191, 216],
	[255, 0, 255],
	[30, 144, 255],
	[219, 112, 147],
	[240, 230, 140],
	[255, 20, 147],
	[255, 160, 122],
	[127, 255, 212],
    [-1, -1, -1]
]


class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


class MPVDataset(data.Dataset):
    
    def __init__(self, opt):
        super(MPVDataset, self).__init__()
        
        self.opt = opt
        self.db_path = opt.dataroot
        self.split = opt.datamode
        
        # opt.img_size = (opt.img_size, int(opt.img_size * (160 / 256)))
        opt.img_size = (opt.img_size, int(opt.img_size * (192 / 256)))
        
        self.filepath_df = pd.read_csv(osp.join(self.db_path, "all_poseA_poseB_clothes.txt"), sep="\t", names=["poseA", "poseB", "target", "split"])
        self.filepath_df = self.filepath_df.drop_duplicates("poseA")
        self.filepath_df = self.filepath_df[self.filepath_df["poseA"].str.contains("front")]
        
        self.filepath_df = self.filepath_df.drop(["poseB"], axis=1)
        self.filepath_df = self.filepath_df.sort_values("poseA")
        
        if self.split in {"train", "val"}:
            self.filepath_df = self.filepath_df[self.filepath_df.split == "train"]
            
            if self.split == "train":
                self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * opt.train_size)]
            elif self.split == "val":
                self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * opt.val_size):]
        else:
            self.filepath_df = self.filepath_df[self.filepath_df.split == "test"]
            del self.filepath_df["target"]
            
            filepath_df_new = pd.read_csv(osp.join(self.db_path, "test_unpaired_images.txt"), sep=" ", names=["poseA", "target"])
            self.filepath_df = pd.merge(self.filepath_df, filepath_df_new, how="left")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(opt.img_size),
            transforms.ToTensor()
        ])
        
    def name(self):
        return "MPVDataset"
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]
        
        c_name = df_row["target"].split("/")[-1]
        im_name = df_row["poseA"].split("/")[-1]
        
        # get original image of person
        image = cv2.imread(osp.join(self.db_path, df_row["poseA"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract non-warped cloth
        target_cloth_image = cv2.imread(osp.join(self.db_path, df_row["target"]))
        target_cloth_image = cv2.cvtColor(target_cloth_image, cv2.COLOR_BGR2RGB)
        
        # extract non-warped cloth mask
        target_cloth_mask = cv2.inRange(target_cloth_image, np.array([0, 0, 0]), np.array([253, 253, 253]))
        target_cloth_mask = cv2.morphologyEx(target_cloth_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        _cm = np.zeros((target_cloth_mask.shape[0]+2, target_cloth_mask.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(target_cloth_mask, _cm, (0, 0), 0)
        
        _cm *= 255
        target_cloth_mask = cv2.bitwise_not(_cm[1:-1, 1:-1])
        
        # load and process the body labels
        label = cv2.imread(osp.join(self.db_path, df_row["poseA"][:-4] + "_densepose.png"))
        try:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        except cv2.error:
            print(osp.join(self.db_path, df_row["poseA"][:-4] + "_densepose.png"))
            exit()
        
        label = cv2.resize(label, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        label_transf = np.zeros((*self.opt.img_size, len(semantic_labels)))
        for i, color in enumerate(semantic_labels):
            label_transf[np.all(label == color, axis=-1), i] = 1.0
        
        # convert the labels to torch.tensor
        label_transf = torch.tensor(label_transf, dtype=torch.float32).permute(2, 0, 1).contiguous()
        
        parse_body = label_transf[2, :, :].unsqueeze(0)
        
        # or (comment this in case segmentations should be cloth-based)
        _label = cv2.imread(osp.join(self.db_path, df_row["poseA"][:-4] + "_parsed.png"))
        _label = cv2.cvtColor(_label, cv2.COLOR_BGR2RGB)
        _label = cv2.resize(_label, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        cloth_mask = (torch.tensor(np.all(_label == [128, 0, 128], axis=2).astype(np.float32)).unsqueeze(0) - 0.5) / 0.5
        
        # convert image to tensor before extracting body-path of the image
        image = self.transform(image)
        image = (image - 0.5) / 0.5
        
        # mask the image to get desired inputs
        cloth_image = image * parse_body
        # cloth_image[parse_body == 0] = 1.0
        
        body_image = image * (1 - parse_body)
        
        # scale the inputs to range [-1, 1]
        label = self.transform(label)
        label = (label - 0.5) / 0.5
        # body_image = self.transform(body_image)
        # body_image = (body_image - 0.5) / 0.5
        # cloth_image = self.transform(cloth_image)
        # cloth_image = (cloth_image - 0.5) / 0.5
        target_cloth_image = self.transform(target_cloth_image)
        target_cloth_image = (target_cloth_image - 0.5) / 0.5
        target_cloth_mask = self.transform(target_cloth_mask)
        target_cloth_mask = (target_cloth_mask - 0.5) / 0.5
        
        # load grid image
        im_g = cv2.imread("./data/grid.png")
        im_g = cv2.cvtColor(im_g, cv2.COLOR_BGR2RGB)
        im_g = self.transform(im_g)
        im_g = (im_g - 0.5) / 0.5
        
        if self.opt.old:
            # load pose points
            pose_name = df_row["poseA"].replace('.jpg', '_keypoints.json')
            with open(osp.join(self.db_path, pose_name), 'r') as f:
                try:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1,3))
                
                except IndexError:
                    pose_data = np.zeros((25, 3))

            pose_data[:, 0] = pose_data[:, 0] * (self.opt.img_size[0] / 256)
            pose_data[:, 1] = pose_data[:, 1] * (self.opt.img_size[1] / 160)
            
            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, *self.opt.img_size)
            r = 5
            im_pose = Image.new('L', self.opt.img_size)
            pose_draw = ImageDraw.Draw(im_pose)
            for i in range(point_num):
                one_map = Image.new('L', self.opt.img_size)
                draw = ImageDraw.Draw(one_map)
                pointx = pose_data[i,0]
                pointy = pose_data[i,1]
                if pointx > 1 and pointy > 1:
                    draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                    pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                
                one_map = self.transform(np.array(one_map))
                pose_map[i] = one_map[0]
                
            # im_pose for visualization
            im_pose = self.transform(np.array(im_pose))
                            
            # save background-person mask
            shape = torch.tensor(1 - np.all(_label == [0, 0, 0], axis=2).astype(np.float32)) * 2 - 1
            shape = shape.unsqueeze(0)
            
            # extract just the head image
            head_label_colors = [0, 128, 0], [128, 0, 192]
            
            head_mask = torch.zeros(self.opt.img_size)
            for color in head_label_colors:
                head_mask += np.all(_label == color, axis=2)
            
            im_h = image * head_mask
                
            # cloth-agnostic representation
            agnostic = torch.cat([shape, im_h, pose_map], 0).float()

            cloth_image = image * (cloth_mask * 0.5 + 0.5) + torch.ones_like(image) * (1 - (cloth_mask * 0.5 + 0.5))
            
        else:
            agnostic = ""
            im_pose = ""
        
        result = {
            'c_name':               c_name,                     # for visualization
            'im_name':              im_name,                    # for visualization or ground truth
            
            'target_cloth':         target_cloth_image,         # for input
            'target_cloth_mask':    target_cloth_mask,          # for input
            
            'cloth':                cloth_image,                # for ground truth
            'cloth_mask':           cloth_mask,
            'body_mask':            parse_body,
            
            'body_label':           label_transf,
            'label':                label,
            
            'image':                image,                      # for visualization
            'body_image':           body_image,                 # for visualization
            
            'grid_image':           im_g,                       # for visualization
            
            'agnostic':             agnostic,
            'im_pose':              im_pose
        }
        
        return result
    
    def __len__(self):
        return len(self.filepath_df)
    
    
class VitonDataset(data.Dataset):
    
    def __init__(self, opt):
        super(VitonDataset, self).__init__()
        
        self.opt = opt
        self.db_path = opt.dataroot
        self.split = opt.datamode
        opt.img_size = (opt.img_size, int(opt.img_size * 0.75))
        
        self.filepath_df = pd.read_csv(osp.join(self.db_path, "viton_%s_pairs.txt" % ("test" if self.split == "test" else "train")), sep=" ", names=["poseA", "target"])
            
        if self.split == "train":
            self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * opt.train_size)]
        elif self.split == "val":
            self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * opt.val_size):]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(opt.img_size),
            transforms.ToTensor()
        ])
        
    def name(self):
        return "VitonDataset"
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]
        
        c_name = df_row["target"].split("/")[-1]
        im_name = df_row["poseA"].split("/")[-1]
        
        # get original image of person
        image = cv2.imread(osp.join(self.db_path, "data", "image", df_row["poseA"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract non-warped cloth
        target_cloth_image = cv2.imread(osp.join(self.db_path, "data", "cloth", df_row["target"]))
        target_cloth_image = cv2.cvtColor(target_cloth_image, cv2.COLOR_BGR2RGB)
        
        # extract non-warped cloth mask
        target_cloth_mask = cv2.inRange(target_cloth_image, np.array([0, 0, 0]), np.array([253, 253, 253]))
        target_cloth_mask = cv2.morphologyEx(target_cloth_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        _cm = np.zeros((target_cloth_mask.shape[0]+2, target_cloth_mask.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(target_cloth_mask, _cm, (0, 0), 0)
        
        _cm *= 255
        target_cloth_mask = cv2.bitwise_not(_cm[1:-1, 1:-1])
        
        # load and process the body labels
        label = cv2.imread(osp.join(self.db_path, "data", "image_densepose_parse", df_row["poseA"].replace(".jpg", ".png")))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        label = cv2.resize(label, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        label_transf = np.zeros((*self.opt.img_size, len(semantic_labels)))
        for i, color in enumerate(semantic_labels):
            label_transf[np.all(label == color, axis=-1), i] = 1.0
        
        # convert the labels to torch.tensor
        label_transf = torch.tensor(label_transf, dtype=torch.float32).permute(2, 0, 1).contiguous()
        
        parse_body = label_transf[2, :, :].unsqueeze(0)
        # or (comment this in case segmentations should be cloth-based)
        _label = cv2.imread(osp.join(self.db_path, "data", "image_parse_with_neck", df_row["poseA"].replace(".jpg", ".png")))
        _label = cv2.cvtColor(_label, cv2.COLOR_BGR2RGB)
        _label = cv2.resize(_label, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        cloth_mask = torch.tensor(np.all(_label == [128, 0, 128], axis=2).astype(np.float32)).unsqueeze(0) * 2 - 1
        
        # convert image to tensor before extracting body-path of the image
        image = self.transform(image)
        image = (image - 0.5) / 0.5
        
        # mask the image to get desired inputs
        cloth_image = image * parse_body
        
        body_image = image * (1 - parse_body)
        
        # scale the inputs to range [-1, 1]
        label = self.transform(label)
        label = (label - 0.5) / 0.5
        target_cloth_image = self.transform(target_cloth_image)
        target_cloth_image = (target_cloth_image - 0.5) / 0.5
        target_cloth_mask = self.transform(target_cloth_mask)
        target_cloth_mask = (target_cloth_mask - 0.5) / 0.5
        
        # load grid image
        im_g = cv2.imread("./data/grid.png")
        im_g = cv2.cvtColor(im_g, cv2.COLOR_BGR2RGB)
        im_g = self.transform(im_g)
        im_g = (im_g - 0.5) / 0.5
        
        if self.opt.old:
            # load pose points
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            with open(osp.join(self.db_path, "data", 'pose', pose_name), 'r') as f:
                try:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1,3))
                
                except IndexError:
                    pose_data = np.zeros((25, 3))

            pose_data[:, 0] = pose_data[:, 0] * (self.opt.img_size[0] / 1024)
            pose_data[:, 1] = pose_data[:, 1] * (self.opt.img_size[1] / 768)
            
            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, *self.opt.img_size)
            r = 5
            im_pose = Image.new('L', self.opt.img_size)
            pose_draw = ImageDraw.Draw(im_pose)
            for i in range(point_num):
                one_map = Image.new('L', self.opt.img_size)
                draw = ImageDraw.Draw(one_map)
                pointx = pose_data[i,0]
                pointy = pose_data[i,1]
                if pointx > 1 and pointy > 1:
                    draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                    pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                
                one_map = self.transform(np.array(one_map))
                pose_map[i] = one_map[0]
                
            # im_pose for visualization
            im_pose = self.transform(np.array(im_pose))
                            
            # save background-person mask
            shape = torch.tensor(1 - np.all(_label == [0, 0, 0], axis=2).astype(np.float32)) * 2 - 1
            shape = shape.unsqueeze(0)
            
            # extract just the head image
            head_label_colors = [0, 128, 0], [128, 0, 192]
            
            head_mask = torch.zeros(self.opt.img_size)
            for color in head_label_colors:
                head_mask += np.all(_label == color, axis=2)
            
            im_h = image * head_mask
                
            # cloth-agnostic representation
            agnostic = torch.cat([shape, im_h, pose_map], 0).float()

            cloth_image = image * (cloth_mask * 0.5 + 0.5) + torch.ones_like(image) * (1 - (cloth_mask * 0.5 + 0.5))
            
        else:
            agnostic = ""
            im_pose = ""
        
        result = {
            'c_name':               c_name,                     # for visualization
            'im_name':              im_name,                    # for visualization or ground truth
            
            'target_cloth':         target_cloth_image,         # for input
            'target_cloth_mask':    target_cloth_mask,          # for input
            
            'cloth':                cloth_image,                # for ground truth
            'cloth_mask':           cloth_mask,
            'body_mask':            parse_body,
            
            'body_label':           label_transf,
            'label':                label,
            
            'image':                image,                      # for visualization
            'body_image':           body_image,                 # for visualization
            
            'grid_image':           im_g,                       # for visualization
            
            'agnostic':             agnostic,
            'im_pose':              im_pose
        }
        
        return result
    
    def __len__(self):
        return len(self.filepath_df)
