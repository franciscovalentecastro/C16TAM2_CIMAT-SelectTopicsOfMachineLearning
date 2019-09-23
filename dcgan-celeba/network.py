import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.c, self.h, self.w = image_shape

        self.up_linear1 = nn.Linear(latent_dim, 512)
        self.up_linear2 = nn.Linear(512, self.c * self.h * self.w)

    def generate(self, x):
        h1 = F.relu(self.up_linear1(x))
        h2 = torch.sigmoid(self.up_linear2(h1))
        return h2.view(-1, self.c, self.h, self.w)

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):

    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.c, self.h, self.w = image_shape

        self.down_linear1 = nn.Linear(self.c * self.h * self.w, 512)
        self.down_linear2 = nn.Linear(512, 1)

    def discriminate(self, x):
        h1 = F.relu(self.down_linear1(x.view(-1, self.c * self.h * self.w)))
        h2 = torch.tanh(self.down_linear2(h1))
        return h2.view(-1)

    def forward(self, x):
        return self.discriminate(x)


class ConvolutionalGenerator(nn.Module):

    def __init__(self, latent_dim, image_shape):
        super(ConvolutionalGenerator, self).__init__()
        self.c, self.h, self.w = image_shape

        self.up_linear1 = nn.Linear(latent_dim, 512)
        self.up_linear2 = nn.Linear(512, 64 * self.h // 4 * self.w // 4)

        self.up_conv_block1 = self.up_conv_block(64, 32, 'relu')
        self.up_conv_block2 = self.up_conv_block(32, self.c, 'tanh')

    def up_conv_block(self, in_chan, out_chan, activation='relu', **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan,
                               kernel_size=4, stride=2, padding=1, **kwargs),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan,
                      kernel_size=3, stride=1, padding=1, **kwargs),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU() if activation == 'relu' else nn.Tanh()
        )

    def generate(self, x):
        h1 = F.leaky_relu(self.up_linear1(x))
        h2 = F.leaky_relu(self.up_linear2(h1))
        h2 = h2.view(-1, 64, self.h // 4, self.w // 4)  # Unflatten
        h3 = self.up_conv_block1(h2)
        h4 = self.up_conv_block2(h3)
        return h4

    def forward(self, x):
        return self.generate(x)


class ConvolutionalDiscriminator(nn.Module):

    def __init__(self, image_shape):
        super(ConvolutionalDiscriminator, self).__init__()
        self.c, self.h, self.w = image_shape

        self.down_conv_block1 = self.down_conv_block(self.c, 32)
        self.down_conv_block2 = self.down_conv_block(32, 64)

        self.down_linear1 = nn.Linear(64 * self.h // 4 * self.w // 4, 512)
        self.down_linear2 = nn.Linear(512, 1)

    def down_conv_block(self, in_chan, out_chan, activation='relu', **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=3, stride=2, padding=1, **kwargs),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan,
                      kernel_size=3, stride=1, padding=1, **kwargs),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        )

    def discriminate(self, x):
        h1 = self.down_conv_block1(x)
        h2 = self.down_conv_block2(h1)
        h2 = h2.view(-1, 64 * self.h // 4 * self.w // 4)  # Flatten
        h3 = F.leaky_relu(self.down_linear1(h2))
        h4 = torch.sigmoid(self.down_linear2(h3))
        return h4.view(-1)

    def forward(self, x):
        return self.discriminate(x)
