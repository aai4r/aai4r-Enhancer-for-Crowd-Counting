from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import h5py
import cv2

def random_crop(im_w, im_h, crop_w, crop_h):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    c_up = random.randint(0, res_h)
    c_left = random.randint(0, res_w)
    c_right = c_left + crop_w
    c_down = c_up + crop_h
    return c_up, c_down, c_left, c_right

# cal overlap area of each head
def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size=0, enhancer='no', repeat=5,
                method='train',
                 log_para = 1000):

        self.root_path = root_path
        self.log_para = log_para
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method    

        if self.method =='train':
            self.im_list = sorted(glob(os.path.join(self.root_path, 'train_data/images/*.jpg')))
        elif self.method =='val':
            self.im_list = sorted(glob(os.path.join(self.root_path, 'val_data/images/*.jpg')))
        else:
            self.im_list = sorted(glob(os.path.join(self.root_path, 'test_data/images/*.jpg')))

        self.c_size = crop_size
        
        # for gray image dataset
        # self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
        # for color image dataset
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        aug_range = (0.5, 1.5)
        self.colorjitter = transforms.ColorJitter(brightness=aug_range, contrast=aug_range)
        self.repeat = repeat
        self.enhancer = enhancer

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.im_list[index]
        gt_path = img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        img = self.trans(img)
        target = torch.Tensor(target)
        
        if self.method =='val' or self.method =='test':
            img = self.normalize(img)
            return img, target
        elif self.method == 'train':
            img_h = img.shape[1]
            img_w = img.shape[2]
            imgs = []
            targets = []
            for _ in range(4):
                c_up, c_down, c_left, c_right = random_crop(img_w, img_h, self.c_size, self.c_size)   
                img_ = img[:,c_up:c_down, c_left:c_right]
                target_ = target[c_up:c_down, c_left:c_right]
                if np.random.random() > 0.5:
                    img_ = img_.flip(dims=[2])
                    target_ = target_.flip(dims=[1])
        
                if self.enhancer =='yes':
                    for _ in range(self.repeat):
                        imgs.append(self.normalize(self.colorjitter(img_)))
                else:
                    imgs.append(self.normalize(img_))
                targets.append(target_)

            return torch.stack(imgs), torch.stack(targets)*self.log_para
   
