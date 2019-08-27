import math
import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        # Level 1
        self.left_conv_block1 = self.conv_block(1, 64,
                                                kernel_size=3,
                                                padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block1 = self.conv_block(128, 64,
                                                 kernel_size=3,
                                                 padding=1)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1,
                                     kernel_size=1, padding=0)

        # Level 2
        self.left_conv_block2 = self.conv_block(64, 128,
                                                kernel_size=3,
                                                padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block2 = self.conv_block(256, 128,
                                                 kernel_size=3,
                                                 padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Level 3
        self.left_conv_block3 = self.conv_block(128, 256,
                                                kernel_size=3,
                                                padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block3 = self.conv_block(512, 256,
                                                 kernel_size=3,
                                                 padding=1)
        self.tconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        # Level 4
        self.left_conv_block4 = self.conv_block(256, 512,
                                                kernel_size=3,
                                                padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.right_conv_block4 = self.conv_block(1024, 512,
                                                 kernel_size=3,
                                                 padding=1)
        self.tconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        # Level 5 (BottleNeck)
        self.left_conv_block5 = self.conv_block(512, 1024,
                                                kernel_size=3,
                                                padding=1)
        self.tconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        # Intialize weights
        self.apply(self.initialize_weights)

    def conv_block(self, in_chan, out_chan, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, **kwargs),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, **kwargs),
            nn.ReLU()
        )

    def initialize_weights(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            input_dimension = module.in_channels \
                * module.kernel_size[0] \
                * module.kernel_size[1]
            std_dev = math.sqrt(2.0 / float(input_dimension))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)

    def forward(self, x1):

        # Level 1
        x1 = self.left_conv_block1(x1)
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
        x_out = self.conv_output(x9)

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
