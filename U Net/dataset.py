import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2 
from torchvision import tv_tensors

class Data(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.imread(img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path,1)
        mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        mask[mask == 255.0] = 1.0
        return image, mask

class transformData(Dataset):
    def __init__(self, base_dataset, transform):
        super(transformData, self).__init__()
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = tv_tensors.Mask(y)
        x, y = self.transform(x, y)
        return x, y
