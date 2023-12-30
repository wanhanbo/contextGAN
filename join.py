
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
from models import * #这里引入不同的模型

import torch.nn as nn
import torch.nn.functional as F
import torch
# python implementations/context_encoder/generate.py <input_dir> <out_dir> <pth_path>
# python join.py 1_0.bmp ..\contextGan_res
parser = argparse.ArgumentParser()
parser.add_argument("--init_dir", type=str, default='d:\礁灰岩\input1\*.bmp', help="initial image")
parser.add_argument("out_dir", type=str, default='out', help="out dir to save the generated image")
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

path_left="D:/codes/gan/contextGan_data/contextGAN_bs128_all_82_left/93799.pth"
path_top="D:/codes/gan/contextGan_data/contextGAN_bs128_all_82_top/93799.pth"
path_left_top="D:/codes/gan/contextGan_data/contextGAN_bs128_all_82_top&left/93799.pth"
model1 = Generator(channels=opt.channels)
weights_dict1 = torch.load(path_left, map_location='cpu')
model1.load_state_dict(weights_dict1, strict=False)

model2 = Generator(channels=opt.channels)
weights_dict2 = torch.load(path_top, map_location='cpu')
model2.load_state_dict(weights_dict2, strict=False)


model3 = Generator(channels=opt.channels)
weights_dict3 = torch.load(path_left_top, map_location='cpu')
model3.load_state_dict(weights_dict3, strict=False)


if cuda:
   model1.cuda()
   model2.cuda()
   model3.cuda()
   

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
]

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# generate
# from left-top to right-bottom
# model1,2,3 for left,top,left-top condition
transform = transforms.Compose(transforms_)
def initImg():
    imgs = glob.glob(opt.init_dir)
    ind = random.randint(1,len(imgs))
    img = Image.open(imgs[ind])  # 读取初始图片
    img = img.convert('L')
    img = transform(img)
    return img.unsqueeze(0).cuda()


w=10 # 宽度方向的拼接图像数目
h=10
w_size=128
h_size=128 # 图像宽、高
left_size=32
top_size=32
channels = 1  # 图像的通道数
num_images = h * w  # 图像的数量
mem=torch.zeros((h,w,1,channels,w_size,h_size)).cuda()
mem[0][0]=initImg() #随机初始化一张图像，或者随便拿来一张
# image_tensor[:,:,0] = torch.from_numpy(image)

def generete_left_top_condition(left_img,top_img,lt_img):
    start_i=top_size
    start_j=left_size
    masked_img = torch.ones_like(left_img)
    masked_img[:,:,:start_i,:start_j]=lt_img[:,:,h_size-top_size:, w_size-left_size:]
    masked_img[:,:, start_j : , :left_size] = left_img[:,:, start_j : , w_size-left_size:]
    masked_img[:,:,  :top_size , start_i:] = top_img[:,:,  h_size-top_size: , start_i:]
    return masked_img

def generete_left_condition(left_img):
    masked_img = torch.ones_like(left_img)
    masked_img[:,:, : , :left_size] = left_img[:,:, : , w_size-left_size:]
    return masked_img

def generete_top_condition(top_img):
    masked_img = torch.ones_like(top_img)
    masked_img[:,:,  :top_size , :] = top_img[:,:, h_size-top_size: , :]
    return masked_img


print("===开始生成图像===")
for i in range(h):
    for j in range(w):
        noise= Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        if (i > 0 and j > 0):
            masked_img=generete_left_top_condition(mem[i][j-1],mem[i-1][j],mem[i-1][j-1])
            mem[i][j] = model3(masked_img,noise)
        elif j > 0:
            masked_img=generete_left_condition(mem[i][j-1])
            mem[i][j] = model1(masked_img,noise)
        elif(i > 0):
            masked_img=generete_top_condition(mem[i-1][j])
            mem[i][j] = model2(masked_img,noise)
        if(i>0 or j>0):
            save_image(masked_img[0], "%s/%s_masked.png" % (opt.out_dir, i*w+j), normalize=True)
            save_image(mem[i][j][0], "%s/%s_gen.png" % (opt.out_dir, i*w+j), normalize=True)



# join
print("===拼接图像===")
last_w_size=w*w_size-(w-1)*left_size
last_h_size=h*h_size-(h-1)*top_size
print(f"last_w_size={last_w_size}")
print(f"last_h_size={last_h_size}")
res=torch.zeros((1,channels,last_h_size,last_w_size)).cuda()
print(res.shape)
for i in range(h):
    for j in range(w):
        start_i=h_size+(i-1)*h_size-i*top_size
        start_j=w_size+(j-1)*w_size-j*left_size
        if(i==0):
            start_i=0
        if(j==0):
            start_j=0
        print(mem[i][j].shape)
        res[:,:,start_i:start_i+h_size,start_j:start_j+w_size]=mem[i][j]
save_image(res[0], "%s/last.png" % (opt.out_dir), normalize=True)



# # a test program
# fixedArray= np.arange(h_size * w_size).reshape(channels,h_size, w_size)
# fixedTensor=torch.from_numpy(fixedArray)
# #用fixedTensor去尝试填充
# test_res=torch.Tensor(channels,last_h_size,last_w_size)
# for i in range(h):
#     for j in range(w):
#         start_i=h_size+(i-1)*h_size-i*top_size
#         start_j=w_size+(j-1)*w_size-i*left_size
#         test_res[:,start_i:start_i+h_size,start_j:start_j+w_size]=fixedTensor
# save_image(test_res[0], "%s/test.png" % (opt.out_dir), normalize=True)


