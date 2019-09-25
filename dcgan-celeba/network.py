import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.c, self.h, self.w = image_shape

        self.up_linear1 = nn.Linear(latent_dim, 1024)
        self.up_linear2 = nn.Linear(1024, self.c * self.h * self.w)

    def generate(self, x):
        h1 = F.leaky_relu(self.up_linear1(x))
        h2 = torch.tanh(self.up_linear2(h1))
        return h2.view(-1, self.c, self.h, self.w)

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):

    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.c, self.h, self.w = image_shape

        self.down_linear1 = nn.Linear(self.c * self.h * self.w, 1024)
        self.down_linear2 = nn.Linear(1024, 1)

    def discriminate(self, x):
        x = x.view(-1, self.c * self.h * self.w)
        h1 = F.leaky_relu(self.down_linear1(x))
        h2 = torch.sigmoid(self.down_linear2(h1))
        return h2.view(-1)

    def forward(self, x):
        return self.discriminate(x)


class ConvolutionalGenerator(nn.Module):

    def __init__(self, latent_dim, image_shape, filters):
        super(ConvolutionalGenerator, self).__init__()
        ld = self.ld = latent_dim
        f = self.f = filters
        self.c, self.h, self.w = image_shape

        # Generator
        self.up_conv_block1 = \
            self.up_conv_block(ld, f * 8, 1, 0, 'relu', False)
        self.up_conv_block2 = \
            self.up_conv_block(f * 8, f * 4, 2, 1, 'relu', True)
        self.up_conv_block3 = \
            self.up_conv_block(f * 4, f * 2, 2, 1, 'relu', True)
        self.up_conv_block4 = \
            self.up_conv_block(f * 2, f, 2, 1, 'relu', True)
        self.up_conv_block5 = \
            self.up_conv_block(f, self.c, 2, 1, 'tanh', False)

        # Intialize weights
        self.apply(self.initialize_weights)

    def up_conv_block(self, in_chan, out_chan, stride=2, padding=1,
                      activation='relu', batch_norm=True, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan,
                               kernel_size=4, stride=stride, padding=padding,
                               **kwargs),
            nn.BatchNorm2d(out_chan) if batch_norm else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.Tanh()
        )

    def generate(self, x):
        x = x.view(-1, self.ld, 1, 1)
        h1 = self.up_conv_block1(x)
        h2 = self.up_conv_block2(h1)
        h3 = self.up_conv_block3(h2)
        h4 = self.up_conv_block4(h3)
        h5 = self.up_conv_block5(h4)
        return h5

    def forward(self, x):
        return self.generate(x)

    def initialize_weights(self, module):
        if type(module) == nn.Conv2d or \
           type(module) == nn.ConvTranspose2d or \
           type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=0.0, std=.02)
            torch.nn.init.zeros_(module.bias)


class ConvolutionalDiscriminator(nn.Module):

    def __init__(self, image_shape, filters):
        super(ConvolutionalDiscriminator, self).__init__()
        self.c, self.h, self.w = image_shape
        f = self.f = filters

        # Discriminator
        self.down_conv_block1 = \
            self.down_conv_block(self.c, f, 2, 1, 'relu', False)
        self.down_conv_block2 = \
            self.down_conv_block(f, f * 2, 2, 1, 'relu', True)
        self.down_conv_block3 = \
            self.down_conv_block(f * 2, f * 4, 2, 1, 'relu', True)
        self.down_conv_block4 = \
            self.down_conv_block(f * 4, f * 8, 2, 1, 'relu', True)
        self.down_conv_block5 = \
            self.down_conv_block(f * 8, 1, 1, 0, 'sigmoid', False)

        # Intialize weights
        self.apply(self.initialize_weights)

    def down_conv_block(self, in_chan, out_chan, stride=2, padding=1,
                        activation='relu', batch_norm=True, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=4, stride=stride, padding=padding,
                      **kwargs),
            nn.BatchNorm2d(out_chan) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == 'relu' else nn.Sigmoid()
        )

    def discriminate(self, x):
        h1 = self.down_conv_block1(x)
        h2 = self.down_conv_block2(h1)
        h3 = self.down_conv_block3(h2)
        h4 = self.down_conv_block4(h3)
        h5 = self.down_conv_block5(h4)
        return h5.view(-1)

    def forward(self, x):
        return self.discriminate(x)

    def initialize_weights(self, module):
        if type(module) == nn.Conv2d or \
           type(module) == nn.ConvTranspose2d or \
           type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=0.0, std=.02)
            torch.nn.init.zeros_(module.bias)
