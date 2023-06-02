import os
import random
from math import ceil
from glob import glob
import torch

import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import json
import cv2

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    hi = random.randint(0, res_h)
    wj = random.randint(0, res_w)
    return hi, wj, crop_h, crop_w


class FSCData(data.Dataset):
    def __init__(self, data_path, crop_size=384, downsample_ratio=8, method='train'):

        anno_file = os.path.join(data_path, 'annotation_FSC147_384.json')
        data_split_file = os.path.join(data_path, 'Train_Test_Val_FSC_147.json')
        self.im_dir = os.path.join(data_path, 'images_384_VarV2')
        self.gt_dir = os.path.join(data_path, 'gt_density_map_adaptive_384_VarV2')

        with open(anno_file) as f:
            self.annotations = json.load(f)
        with open(data_split_file) as f:
            self.data_split = json.load(f)

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method
        self.im_ids = self.data_split[method]

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0

        self.trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.trans_dmap = transforms.ToTensor()

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, item):
        im_id = self.im_ids[item]
        img_path = os.path.join(self.im_dir, im_id)
        gd_path = os.path.join(self.gt_dir, im_id).replace('.jpg', '.npy')

        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
        rects = np.array(rects)
        points = np.array(anno['points'])

        try:
            img = Image.open(img_path).convert('RGB')
            dmap = np.load(gd_path)
            dmap = dmap.astype(np.float32, copy=False)  # np.float64 -> np.float32 to save memory
        except:
            raise Exception('Image open error {}'.format(im_id))
        
        sample = [img,rects,dmap,points,im_id]
        
        if self.method == 'train':
            return self.train_transform(sample)
        else:
            return self.val_transform(sample)

    def train_transform(self, sample):
        img, rects, dmap, points, name = sample
        wd, ht = img.size

        # crop examplar
        examplars = []
        rects = rects.astype(np.int)
        for y1, x1, y2, x2 in rects:
            tmp_ex = img.crop((x1, y1, x2, y2))
            examplars.append(tmp_ex)

        # rescale augmentation
        re_size = random.random() + 0.5   # 0.5 ~ 1.5
        wdd = int(wd*re_size)
        htt = int(ht*re_size)
        if min(wdd, htt) >= self.c_size:
            raw_size = (wd, ht)
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            dmap = cv2.resize(dmap, (wd, ht))
            ratio = (raw_size[0]*raw_size[1])/(wd*ht)
            dmap = dmap * ratio

        # random crop augmentation
        hi, wi, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = img.crop((wi, hi, wi+w, hi+h))
        dmap = dmap[hi:hi+h, wi:wi+w]

        '''
        # random horizontal flip
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = np.fliplr(dmap)
        '''

        dmap = Image.fromarray(dmap)

        # resize random 1-shot
        i = random.choice((0, 1, 2))
        rshot = examplars[i]
        shot_resizer = transforms.Resize((64, 64))
        rshot = shot_resizer(rshot)
        rshot = self.trans_img(rshot)

        # scale information
        rect = rects[i]
        dy = float(rect[2] - rect[0])
        dx = float(rect[3] - rect[1])
        if re_size > 1:
            dy = dy * re_size
            dx = dx * re_size
        scale_info = (dy / self.c_size, dx / self.c_size)
        scale_info = torch.tensor(scale_info)
        y = torch.linspace(0, scale_info[0], 64).view(64, 1)
        x = torch.linspace(0, scale_info[1], 64).view(1, 64)
        y = y.repeat(1, 64)
        x = x.repeat(64, 1)
        scale_embed = y + x

        return self.trans_img(img), self.trans_dmap(dmap), rshot, scale_embed
        # return img, dmap, 1-shot, scale_embed
    
    def val_transform(self, sample):
        img, rects, dmap, points, name = sample

        # crop examplar
        examplars = []
        rects = rects.astype(np.int)
        for y1, x1, y2, x2 in rects:
            tmp_ex = img.crop((x1, y1, x2, y2))
            examplars.append(tmp_ex)

        # img
        img = self.trans_img(img)

        # dmap
        dmap = self.trans_dmap(dmap)

        # count
        count = torch.sum(dmap)

        # resized random 1-shot
        i = random.choice((0, 1, 2))
        rshot = examplars[i]
        shot_resizer = transforms.Resize((64, 64))
        rshot = shot_resizer(rshot)
        rshot = self.trans_img(rshot)

        # scale information
        rect = rects[i]
        dy = float(rect[2] - rect[0])
        dx = float(rect[3] - rect[1])
        scale_info = (dy / self.c_size, dx / self.c_size)
        scale_info = torch.tensor(scale_info)
        y = torch.linspace(0, scale_info[0], 64).view(64, 1)
        x = torch.linspace(0, scale_info[1], 64).view(1, 64)
        y = y.repeat(1, 64)
        x = x.repeat(64, 1)
        scale_embed = y + x

        return img, count, rshot, scale_embed, name
