import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, channels=3, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.l1 = nn.Sequential(nn.Linear(self.noise_dim, 128 * 4 ** 2 ))
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False), # ->64
            *downsample(64, 64), # ->32
            *downsample(64, 128), # ->16
            *downsample(128, 256), # ->8
            *downsample(256, 512), # ->4
            nn.Conv2d(640, 4000, 1), # -> 4
            *upsample(4000, 512), # -> 8
            *upsample(512, 256), # -> 16
            *upsample(256, 128), # -> 32
            *upsample(128, 64), # -> 64
            *upsample(64, 32), # -> 128
            nn.Conv2d(32, channels, 3, 1, 1), # -> 128
            nn.Tanh()
        )

    def forward(self, x, z):
        noise = self.l1(z)
        noise_out = noise.view(noise.shape[0], 128, 4, 4)
        encoder_out = self.model[:14](x)
        encoder_out = torch.cat((encoder_out, noise_out), dim=1)
        out = self.model[14:](encoder_out)
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
