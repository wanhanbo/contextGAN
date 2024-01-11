"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""

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

from datasets2 import *
from models_new import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str, default='../../input2', help="root dir of img dataset")
parser.add_argument("out_dir", type=str, default='out', help="out dir to save the generated image")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=112, help="size of random mask")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
# patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch_h, patch_w = int(opt.img_size / 2 ** 4), int(opt.img_size / 2 ** 4)
patch = (1, patch_h, patch_w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels,batchsize=opt.batch_size,noise_dim=opt.latent_dim)
discriminator = Discriminator(channels=opt.channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
     transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
]
dataloader = DataLoader(
    ImageDataset(opt.root_dir, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# test_dataloader = DataLoader(
#     ImageDataset(opt.root_dir, transforms_=transforms_, mode="val"),
#     batch_size=12,
#     shuffle=True,
#     num_workers=1,
# )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def save_sample(batches_done):
    samples, masked_samples, i = next(iter(dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    
    # Generate inpainted image
    gen_img = generator(masked_samples)

    # Save sample
    save_image(samples[:25], "%s/%d_ori.png" % (opt.out_dir, batches_done), nrow=5, normalize=True)
    save_image(gen_img[:25], "%s/%d_gen.png" % (opt.out_dir, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, masked_size) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor)) # 用于输入encoder
        unmasked_parts = Variable(masked_imgs.type(Tensor)) # 用于计算g_pixel

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # gen_imgs = generator(masked_imgs, z)
        gen_imgs = generator(masked_imgs) # 新模型在内部处理噪音

    

        # Adversarial loss
        g_adv = adversarial_loss(discriminator(gen_imgs), valid)
        # Pixelwise loss
        start_pixel, ms = masked_size
        start_pixel = start_pixel[0] # eg:128-112
        ms = ms[0]
        gen_imgs_unmasked_parts = gen_imgs.clone()
        # g_pixel = pixelwise_loss(gen_imgs_unmasked_parts[:, :, :start_pixel, :], unmasked_parts[:, :, :start_pixel, :])
        # g_pixel += pixelwise_loss(gen_imgs_unmasked_parts[:, :, start_pixel:(start_pixel+ms), :start_pixel], unmasked_parts[:, :, start_pixel:(start_pixel+ms), :start_pixel])
        # g_pixel += pixelwise_loss(gen_imgs_unmasked_parts[:, :, start_pixel:(start_pixel+ms), (start_pixel+ms):], unmasked_parts[:, :, start_pixel:(start_pixel+ms),  (start_pixel+ms):])
        # g_pixel += pixelwise_loss(gen_imgs_unmasked_parts[:, :, start_pixel+ms:, :], unmasked_parts[:, :, start_pixel+ms:, :])
        # when right and bottom are masked, top and left are conditions
        # g_pixel = pixelwise_loss(gen_imgs_unmasked_parts[:, :, :start_pixel, :], unmasked_parts[:, :, :start_pixel, :]) #top_condition
        # g_pixel = pixelwise_loss(gen_imgs_unmasked_parts[:, :, :, :start_pixel], unmasked_parts[:, :, :, :start_pixel]) #left_condition
        # g_pixel = pixelwise_loss(gen_imgs_unmasked_parts[:, :, :, start_pixel:], unmasked_parts[:, :, :, start_pixel:]) #right_condition
        # g_pixel = pixelwise_loss(gen_imgs_unmasked_parts[:, :, :start_pixel, :], unmasked_parts[:, :, :start_pixel, :])
        # g_pixel += pixelwise_loss(gen_imgs_unmasked_parts[:, :, start_pixel:, :start_pixel], unmasked_parts[:, :, start_pixel:, :start_pixel]) 
        g_pixel = pixelwise_loss(gen_imgs_unmasked_parts[:, :, :start_pixel, :], unmasked_parts[:, :,:start_pixel, :]) # right_top_condition
        g_pixel += pixelwise_loss(gen_imgs_unmasked_parts[:, :, start_pixel:, ms:], unmasked_parts[:, :,start_pixel:, ms:]) 

        # Total loss
        g_loss = 0.1 * g_adv + 0.9 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)

torch.save(generator.state_dict(), "%s/%d.pth" % (opt.out_dir, batches_done))
