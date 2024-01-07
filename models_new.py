import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, channels=3, noise_dim=100, batchsize=128):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.bs = batchsize
        self.noise_dim = noise_dim
        def downsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            return layers

        def upsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=2, padding=1, output_padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            layers.append(nn.ReLU())
            return layers

        self.l1 = nn.Sequential(nn.Linear(self.noise_dim, 512 * 4 ** 2 ))
        self.l2 = nn.Sequential(nn.Linear(self.noise_dim, 512 * 8 ** 2 ))
        self.l3 = nn.Sequential(nn.Linear(self.noise_dim, 256 * 16 ** 2 ))
        self.l4 = nn.Sequential(nn.Linear(self.noise_dim, 128 * 32 ** 2 ))
        self.l5 = nn.Sequential(nn.Linear(self.noise_dim, 64 * 64 ** 2 ))
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False, dropout=False), # ->64, layers=2
            *downsample(64, 64), # ->32, layers=4
            *downsample(64, 128), # ->16, layers=4
            *downsample(128, 256), # ->8, layers=4
            *downsample(256, 512), # ->4, layers=4
            nn.Conv2d(1024, 4000, 1), # -> 4 , layers=1, concat noise first 
            *upsample(4000, 512), # -> 8, layers=4
            *upsample(1024, 256), # -> 16, layers=4, concat noise second
            *upsample(512, 128), # -> 32, layers=4, concat noise third
            *upsample(256, 64), # -> 64, layers=4, concat noise forth
            *upsample(128, 32), # -> 128, layers=4, concat noise fifth 
            nn.Conv2d(32, channels, 3, 1, 1), # -> 128
            nn.Tanh()
        )
    def forward(self, x):
        # 把噪音生成的过程直接放在了model中，原来是外部生成再传入model
        encoder_out = self.model[:18](x)
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], self.noise_dim)))) 
        noise = self.l1(z)
        noise_out = noise.view(noise.shape[0], 512, 4, 4)
        encoder_out = torch.cat((encoder_out, noise_out), dim=1)
        bridge_out=self.model[18:23](encoder_out)

        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], self.noise_dim))))
        noise = self.l2(z)
        noise_out = noise.view(noise.shape[0], 512, 8, 8)
        decoder_out1=torch.cat((bridge_out, noise_out), dim=1)
        decoder_out1=self.model[23:27](decoder_out1)

        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], self.noise_dim))))
        noise = self.l3(z)
        noise_out = noise.view(noise.shape[0], 256, 16, 16)
        decoder_out2=torch.cat((decoder_out1, noise_out), dim=1)
        decoder_out2=self.model[27:31](decoder_out2)

        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], self.noise_dim))))
        noise = self.l4(z)
        noise_out = noise.view(noise.shape[0], 128, 32, 32)
        decoder_out3=torch.cat((decoder_out2, noise_out), dim=1)
        decoder_out3=self.model[31:35](decoder_out3)

        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], self.noise_dim))))
        noise = self.l5(z)
        noise_out = noise.view(noise.shape[0], 64, 64, 64)
        decoder_out4=torch.cat((decoder_out3, noise_out), dim=1)
        decoder_out4=self.model[35:39](decoder_out4)

        out=self.model[39:](decoder_out4)
        return out

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
