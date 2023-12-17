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

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
# python implementations/context_encoder/generate.py <input_dir> <out_dir> <pth_path>
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str, default='../../input1', help="root dir of img dataset")
parser.add_argument("out_dir", type=str, default='out', help="out dir to save the generated image")
parser.add_argument("ckpt", type=str, default='../../ckpt.pth', help="checkpoint of generator")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

generator = Generator(channels=opt.channels)
weights_dict = torch.load(opt.ckpt, map_location='cpu')
generator.load_state_dict(weights_dict, strict=False)

if cuda:
    generator.cuda()

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

for ind, (samples, masked_samples, _) in enumerate(dataloader):
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))

    z0 = Variable(Tensor(np.random.normal(0, 1, (samples.shape[0], opt.latent_dim))))
    gen_img = generator(masked_samples, z0)

    # Save sample
    save_image(samples[0], "%s/%s_ori.png" % (opt.out_dir, ind), normalize=True)
    save_image(gen_img[0], "%s/%s_gen.png" % (opt.out_dir, ind), normalize=True)