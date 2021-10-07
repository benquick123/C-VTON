import os
import json

import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

semantic_cloth_labels = [
    [128, 0, 128],
    [128, 128, 64],
    [128, 128, 192],
    [0, 255, 0],
    [0, 128, 128], # dress
    [128, 128, 128], # something upper?
    
    [0, 0, 0], # bg
    
    [0, 128, 0], # hair
    [0, 64, 0], # left leg?
    [128, 128, 0], # right hand
    [0, 192, 0], # left foot
    [128, 0, 192], # head
    [0, 0, 192], # legs / skirt?
    [0, 64, 128], # skirt?
    [128, 0, 64], # left hand
    [0, 192, 128], # right foot
    [0, 0, 128],
    [0, 128, 64],
    [0, 0, 64],
    [0, 128, 192]
]

semantic_densepose_labels = [
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
	[127, 255, 212]
]

semantic_body_labels = [
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
    [255, 0, 255]
]


class VitonDataset(Dataset):
    
    def __init__(self, opt, phase, test_pairs=None):

        opt.label_nc = [len(semantic_body_labels), len(semantic_cloth_labels), len(semantic_densepose_labels)]
        opt.semantic_nc = [label_nc + 1 for label_nc in opt.label_nc]
        
        opt.offsets = [0]
        segmentation_modes = ["body", "cloth", "densepose"]
        for i, mode in enumerate(segmentation_modes):
            if mode in opt.segmentation:
                opt.offsets.append(opt.offsets[-1] + opt.semantic_nc[i] + 1) # to account for extra real/fake class in discriminator output
            else:
                opt.offsets.append(0)
        
        if isinstance(opt.img_size, int):
            opt.img_size = (opt.img_size, int(opt.img_size * 0.75))
        
        self.opt = opt
        self.phase = phase
        self.db_path = opt.dataroot
        
        test_pairs = "viton_%s_pairs.txt" % ("test" if phase in {"test", "test_same"} else "train") if test_pairs is None else test_pairs
        # test_pairs = "/home/benjamin/StyleTON/data/viton/viton_train_swap.txt"
        self.filepath_df = pd.read_csv(os.path.join(self.db_path, test_pairs), sep=" ", names=["poseA", "target"])
        # self.filepath_df.target = self.filepath_df.target.str.replace("_0", "_1")
        if phase == "test_same":
            self.filepath_df.target = self.filepath_df.poseA.str.replace("_0", "_1")
        
        if phase == "train":
            self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * opt.train_size)]
        elif phase == "val":
            self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * opt.val_size):]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
        ])
        
        if phase in {"train", "train_whole"} and self.opt.add_pd_loss:
            self.hand_indices = [2, 7, 11, 14]
            self.body_label_centroids = [None] * len(self.filepath_df)
        else:
            self.body_label_centroids = None
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]

        # get original image of person
        image = cv2.imread(os.path.join(self.db_path, "data", "image", df_row["poseA"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.opt.img_size[::-1], interpolation=cv2.INTER_AREA)
        
        original_size = image.shape[:2]
        
        # extract non-warped cloth
        cloth_image = cv2.imread(os.path.join(self.db_path, "data", "cloth", df_row["target"]))
        cloth_image = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
        
        # load cloth labels
        cloth_seg = cv2.imread(os.path.join(self.db_path, "data", "image_parse_with_hands", df_row["poseA"].replace(".jpg", ".png")))
        cloth_seg = cv2.cvtColor(cloth_seg, cv2.COLOR_BGR2RGB)
        cloth_seg = cv2.resize(cloth_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)

        # mask the image to get desired inputs
        
        # get the mask without upper clothes / dress, hands, neck
        # additionally, get cloth segmentations by cloth part
        cloth_seg_transf = np.zeros(self.opt.img_size)
        mask = np.zeros(self.opt.img_size)
        for i, color in enumerate(semantic_cloth_labels):
            cloth_seg_transf[np.all(cloth_seg == color, axis=-1)] = i
            if i < (6 + self.opt.no_bg):    # this works, because colors are sorted in a specific way with background being the 8th.
                mask[np.all(cloth_seg == color, axis=-1)] = 1.0
                
        cloth_seg_transf = np.expand_dims(cloth_seg_transf, 0)
        cloth_seg_transf = torch.tensor(cloth_seg_transf)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1).astype(np.uint8)
        masked_image = image * (1 - mask)
        
        # load and process the body labels
        body_seg = cv2.imread(os.path.join(self.db_path, "data", "image_body_parse", df_row["poseA"].replace(".jpg", ".png")))
        body_seg = cv2.cvtColor(body_seg, cv2.COLOR_BGR2RGB)
        body_seg = cv2.resize(body_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # get one-hot encoded body segmentations 
        body_seg_transf = np.zeros(self.opt.img_size)
        for i, color in enumerate(semantic_body_labels):
            body_seg_transf[np.all(body_seg == color, axis=-1)] = i
            
            # additionally, get body segmentation centroids.
            if self.phase == "train" and self.opt.add_pd_loss and (self.body_label_centroids[index] is None or len(self.body_label_centroids[index]) != len(self.hand_indices)) and i in self.hand_indices:
                if self.body_label_centroids[index] is None:
                    self.body_label_centroids[index] = []
                    
                non_zero = np.nonzero(np.all(body_seg == color, axis=-1))
                if len(non_zero[0]):
                    x = int(non_zero[0].mean())
                else:
                    x = -1
                    
                if len(non_zero[1]):
                    y = int(non_zero[1].mean())
                else:
                    y = -1
                    
                self.body_label_centroids[index].append([x, y])
                
        body_label_centroid = self.body_label_centroids[index] if self.body_label_centroids is not None else ""
        
        body_seg_transf = np.expand_dims(body_seg_transf, 0)
        body_seg_transf = torch.tensor(body_seg_transf)
        
        densepose_seg = cv2.imread(os.path.join(self.db_path, "data", "image_densepose_parse", df_row["poseA"].replace(".jpg", ".png")))
        densepose_seg = cv2.cvtColor(densepose_seg, cv2.COLOR_BGR2RGB)
        densepose_seg = cv2.resize(densepose_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        densepose_seg_transf = np.zeros(self.opt.img_size)
        for i, color in enumerate(semantic_densepose_labels):
            densepose_seg_transf[np.all(densepose_seg == color, axis=-1)] = i
            
        densepose_seg_transf = np.expand_dims(densepose_seg_transf, 0)
        densepose_seg_transf = torch.tensor(densepose_seg_transf)
        
        # scale the inputs to range [-1, 1]
        image = self.transform(image)
        image = (image - 0.5) / 0.5
        masked_image = self.transform(masked_image)
        masked_image = (masked_image - 0.5) / 0.5
        cloth_image = self.transform(cloth_image)
        cloth_image = (cloth_image - 0.5) / 0.5

        if self.opt.bpgm_id.find("old") >= 0:
            # load pose points
            pose_name = df_row["poseA"].replace('.jpg', '_keypoints.json')
            with open(os.path.join(self.db_path, "data", 'pose', pose_name), 'r') as f:
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
                   
            # save background-person mask
            shape = torch.tensor(1 - np.all(cloth_seg == [0, 0, 0], axis=2).astype(np.float32)) * 2 - 1
            shape = shape.unsqueeze(0)
            
            # extract just the head image
            head_label_colors = [0, 128, 0], [128, 0, 192]
            
            head_mask = torch.zeros(self.opt.img_size)
            for color in head_label_colors:
                head_mask += np.all(cloth_seg == color, axis=2)
            
            im_h = image * head_mask
                
            # cloth-agnostic representation
            agnostic = torch.cat([shape, im_h, pose_map], 0).float()
        else:
            agnostic = ""

        return {"image": {"I": image,
                        "C_t": cloth_image,
                        "I_m": masked_image },
                "cloth_label": cloth_seg_transf,
                "body_label": body_seg_transf,
                "densepose_label": densepose_seg_transf,
                "name": df_row["poseA"],
                "agnostic": agnostic,
                "original_size": original_size,
                "label_centroid": body_label_centroid}
            
    def __len__(self):
        return len(self.filepath_df)
    
    def name(self):
        return "VitonDataset"
