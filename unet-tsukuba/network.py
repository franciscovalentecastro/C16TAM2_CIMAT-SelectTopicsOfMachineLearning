import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, image_shape, filters=8):
        super(UNet, self).__init__()
        self.c, self.h, self.w = image_shape
        f = self.f = filters

        # Level 1
        self.left_conv_block1 = self.conv_block(self.c, f,
                                                kernel_size=3,
                                                padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block1 = self.conv_block(f * 2, f,
                                                 kernel_size=3,
                                                 padding=1)
        self.conv_output = nn.Conv2d(in_channels=f, out_channels=1,
                                     kernel_size=1, padding=0)

        # Level 2
        self.left_conv_block2 = self.conv_block(f, f * 2,
                                                kernel_size=3,
                                                padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block2 = self.conv_block(f * 4, f * 2,
                                                 kernel_size=3,
                                                 padding=1)
        self.tconv2 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)

        # Level 3
        self.left_conv_block3 = self.conv_block(f * 2, f * 4,
                                                kernel_size=3,
                                                padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block3 = self.conv_block(f * 8, f * 4,
                                                 kernel_size=3,
                                                 padding=1)
        self.tconv3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)

        # Level 4
        self.left_conv_block4 = self.conv_block(f * 4, f * 8,
                                                kernel_size=3,
                                                padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block4 = self.conv_block(f * 16, f * 8,
                                                 kernel_size=3,
                                                 padding=1)
        self.tconv4 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)

        # Level 5 (BottleNeck)
        self.left_conv_block5 = self.conv_block(f * 8, f * 16,
                                                kernel_size=3,
                                                padding=1)
        self.tconv5 = nn.ConvTranspose2d(f * 16, f * 8,
                                         kernel_size=2,
                                         stride=2)

        # Intialize weights
        self.apply(self.initialize_weights)

    def conv_block(self, in_chan, out_chan, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, **kwargs),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, **kwargs),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        )

    def initialize_weights(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            input_dimension = module.in_channels \
                * module.kernel_size[0] \
                * module.kernel_size[1]
            std_dev = math.sqrt(2.0 / float(input_dimension))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)

    def forward(self, x):

        # Level 1
        x1 = self.left_conv_block1(x)
        # Downsample
        x2 = self.pool1(x1)

        # Level 2
        x2 = self.left_conv_block2(x2)
        # Downsample
        x3 = self.pool2(x2)

        # Level 3
        x3 = self.left_conv_block3(x3)
        # Downsample
        x4 = self.pool3(x3)

        # Level 4
        x4 = self.left_conv_block4(x4)
        # Downsample
        x5 = self.pool4(x4)

        # Level 5
        x5 = self.left_conv_block5(x5)
        # Upsample
        x6 = self.tconv5(x5)

        # Level 4
        x6 = torch.cat((x6, x4), 1)
        x6 = self.right_conv_block4(x6)
        # Upsample
        x7 = self.tconv4(x6)

        # Level 3
        x7 = torch.cat((x7, x3), 1)
        x7 = self.right_conv_block3(x7)
        # Upsample
        x8 = self.tconv3(x7)

        # Level 2
        x8 = torch.cat((x8, x2), 1)
        x8 = self.right_conv_block2(x8)
        # Upsample
        x9 = self.tconv2(x8)

        # Level 1
        x9 = torch.cat((x9, x1), 1)
        x9 = self.right_conv_block1(x9)
        x_out = torch.tanh(self.conv_output(x9))

        # print(x.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # print(x5.size())
        # print(x6.size())
        # print(x7.size())
        # print(x8.size())
        # print(x9.size())

        return x_out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
