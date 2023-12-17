import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transforms_=None, img_size=128, mask_size=64):  # __init__是初始化该类的一些基础参数
        self.transform = transforms.Compose(transforms_)
        self.root_dir = root_dir  # 文件目录
        self.img_size = img_size
        self.mask_size = mask_size
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

    def apply_right_mask(self, img):
        """Mask right part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()  # 或许名字要变一下？
        masked_img[:,  : , i:] = 1

    def apply_bot_mask(self, img):
        """Mask bottom part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : , :] = 1
        
    def apply_right_bot_mask(self, img):
        """Mask right and bottom part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : , i:] = 1

        return masked_img, (i, self.mask_size)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path)  # 读取该图片
        img = img.convert('L')
        if self.transform:
            img = self.transform(img)
        masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    

