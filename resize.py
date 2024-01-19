import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets_resize import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

#  python resize.py D:/礁灰岩/crop256/ D:/礁灰岩/resize256/ False
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str, default='../../input1', help="root dir of img dataset")
parser.add_argument("out_dir", type=str, default='out', help="out dir to save the generated image")
parser.add_argument("rotate", type=bool, default=False, help="rotate the croped images")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
rotate=opt.rotate

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
]
dataloader = DataLoader(
    ImageDataset(opt.root_dir, transforms_=transforms_),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print("===Resize starts===")
for ind, (images_array) in enumerate(dataloader):
    resized= Variable(images_array[0].type(Tensor))
    save_image(resized, "%s/%s_0.png" % (opt.out_dir, ind), normalize=True)

    if False:
        rotated_90= Variable(images_array[1].type(Tensor))
        rotated_180= Variable(images_array[2].type(Tensor))
        rotated_270=Variable(images_array[3].type(Tensor))
        save_image(rotated_90, "%s/%s_1.png" % (opt.out_dir, ind), normalize=True)
        save_image(rotated_180, "%s/%s_2.png" % (opt.out_dir, ind), normalize=True)
        save_image(rotated_270, "%s/%s_3.png" % (opt.out_dir, ind), normalize=True)
    print(f"Saved all the resized images {ind}th") 